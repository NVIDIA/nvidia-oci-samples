#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# The output-cloning helper and selective FLUX quantization recipe are adapted
# from TorchAO. See THIRD_PARTY_NOTICES.md for the pinned source and license.

import argparse
import json
import statistics
import time
import traceback
from collections import Counter
from functools import wraps
from pathlib import Path

from benchmarks.flux1_schnell.flux_prompt_bank import prompt_digest, request_prompts


MODES = (
    "bf16-regional",
    "nvfp4-regional",
    "nvfp4-regional-cg",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def flux_nvfp4_filter(module, fqn: str) -> bool:
    """Official TorchAO FLUX selective-quantization recipe."""
    import torch

    if not isinstance(module, torch.nn.Linear):
        return False
    if "embed" in fqn:
        return False
    if fqn == "norm_out.linear" or fqn == "proj_out":
        return False
    if module.in_features < 1024 or module.out_features < 1024:
        return False
    return True


def clone_output_wrapper(function):
    """Official workaround for regional compile plus CUDA Graphs."""
    import torch
    from torch.utils._pytree import tree_map_only

    @wraps(function)
    def wrapped(*args, **kwargs):
        outputs = function(*args, **kwargs)
        return tree_map_only(
            torch.Tensor,
            lambda tensor: tensor.clone() if tensor.is_cuda else tensor,
            outputs,
        )

    return wrapped


def apply_regional_compile(transformer, cuda_graph: bool) -> None:
    import torch

    if not cuda_graph:
        transformer.compile_repeated_blocks(fullgraph=True)
        return

    repeated_blocks = getattr(transformer, "_repeated_blocks", None)
    if not repeated_blocks:
        raise ValueError(
            f"_repeated_blocks is not defined on {transformer.__class__.__name__}"
        )
    for submodule in transformer.modules():
        if submodule.__class__.__name__ in repeated_blocks:
            submodule.forward = clone_output_wrapper(
                torch.compile(
                    submodule.forward,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            )


def load_pipeline(model_path: Path, mode: str):
    import torch
    from diffusers import FluxPipeline

    load_started = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Keep text encoding, denoising, and VAE decode in the measured pipeline,
    # while excluding PIL conversion and the GPU-to-host image copy.
    pipe.image_processor.postprocess = (
        lambda image, output_type="pt", do_denormalize=None: image
    )

    selected_fqns = [
        fqn
        for fqn, module in pipe.transformer.named_modules()
        if flux_nvfp4_filter(module, fqn)
    ]
    quantization = None
    if mode.startswith("nvfp4"):
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        from torchao.quantization import quantize_

        quantization = NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=True,
        )
        quantize_(
            pipe.transformer,
            config=quantization,
            filter_fn=flux_nvfp4_filter,
        )

    if mode == "bf16-regional" or mode == "nvfp4-regional":
        apply_regional_compile(pipe.transformer, cuda_graph=False)
    elif mode == "nvfp4-regional-cg":
        apply_regional_compile(pipe.transformer, cuda_graph=True)

    selected_types = Counter()
    module_by_fqn = dict(pipe.transformer.named_modules())
    for fqn in selected_fqns:
        selected_types[type(module_by_fqn[fqn]).__name__] += 1

    torch.cuda.synchronize()
    metadata = {
        "load_seconds": time.perf_counter() - load_started,
        "quantization_config": repr(quantization),
        "selective_filter": {
            "source": "official TorchAO diffusers-blackwell-quants FLUX recipe",
            "linear_min_in_features": 1024,
            "linear_min_out_features": 1024,
            "excluded_name_contains": ["embed"],
            "excluded_exact_fqns": ["norm_out.linear", "proj_out"],
            "selected_count": len(selected_fqns),
            "selected_fqns": selected_fqns,
            "post_quant_module_types": dict(selected_types),
        },
        "regional_compile": mode in (
            "bf16-regional",
            "nvfp4-regional",
            "nvfp4-regional-cg",
        ),
        "cuda_graph_compile_mode": (
            "reduce-overhead" if mode == "nvfp4-regional-cg" else None
        ),
    }
    return pipe, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="Bracket measured iterations with cudaProfilerStart/Stop",
    )
    args = parser.parse_args()

    run_dir = args.output_dir / f"torchao-diffusers-{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    import diffusers
    import mslk
    import torch
    import torchao
    import transformers

    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torchao": torchao.__version__,
        "mslk": mslk.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
    }
    load_result = {
        "status": "error",
        "backend": "hf-diffusers-torchao",
        "mode": args.mode,
        "environment": environment,
    }
    try:
        pipe, metadata = load_pipeline(args.model, args.mode)
        load_result.update({"status": "ok", **metadata})
    except Exception as exc:
        load_result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        (run_dir / "load.json").write_text(json.dumps(load_result, indent=2) + "\n")
        print(json.dumps(load_result, indent=2), flush=True)
        raise

    (run_dir / "load.json").write_text(json.dumps(load_result, indent=2) + "\n")
    print(json.dumps(load_result, indent=2), flush=True)

    for batch in args.batches:
        prompts = request_prompts(batch)
        result = {
            "status": "error",
            "backend": "hf-diffusers-torchao",
            "mode": args.mode,
            "batch_size": batch,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "max_sequence_length": args.max_sequence_length,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "batch_semantics": "request-batch",
            "prompt_count": len(prompts),
            "images_per_prompt": 1,
            "prompt_sha256": prompt_digest(prompts),
            "environment": environment,
        }
        try:

            def generate(seed: int):
                generator = torch.Generator(device="cuda").manual_seed(seed)
                return pipe(
                    prompt=prompts,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=0.0,
                    max_sequence_length=args.max_sequence_length,
                    num_images_per_prompt=1,
                    generator=generator,
                    output_type="pt",
                    return_dict=True,
                ).images

            for index in range(args.warmup):
                image = generate(args.seed + index)
                torch.cuda.synchronize()

            latencies = []
            image = None
            torch.cuda.reset_peak_memory_stats()
            if args.nsys_capture:
                torch.cuda.cudart().cudaProfilerStart()
            try:
                for index in range(args.iterations):
                    started = time.perf_counter()
                    with torch.cuda.nvtx.range(
                        f"torchao_request_batch_b{batch}_iteration_{index}"
                    ):
                        image = generate(args.seed + args.warmup + index)
                        torch.cuda.synchronize()
                    latencies.append(time.perf_counter() - started)
            finally:
                if args.nsys_capture:
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()

            if image is None or image.shape[0] != batch:
                raise RuntimeError(
                    f"Expected {batch} decoded images, got {getattr(image, 'shape', None)}"
                )
            mean_seconds = statistics.mean(latencies)
            result.update(
                {
                    "status": "ok",
                    "image_shape": list(image.shape),
                    "output_count": int(image.shape[0]),
                    "timing_scope": "host_wall_with_cuda_synchronize",
                    "latency_seconds": latencies,
                    "mean_batch_latency_ms": mean_seconds * 1000.0,
                    "median_batch_latency_ms": statistics.median(latencies) * 1000.0,
                    "p90_batch_latency_ms": percentile(latencies, 0.9) * 1000.0,
                    "images_per_second": batch / mean_seconds,
                    "mean_per_image_ms": mean_seconds * 1000.0 / batch,
                    "peak_memory_gib": torch.cuda.max_memory_reserved() / 1024**3,
                }
            )
            if batch == 1:
                sample = image[0].detach().float().cpu()
                sample_path = run_dir / "sample-b1.pt"
                torch.save(sample, sample_path)
                result["sample_tensor"] = str(sample_path)
        except Exception as exc:
            result.update(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

        output_path = run_dir / f"b{batch}.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if result["status"] != "ok":
            raise RuntimeError(f"Batch {batch} failed: {result.get('error')}")


if __name__ == "__main__":
    main()
