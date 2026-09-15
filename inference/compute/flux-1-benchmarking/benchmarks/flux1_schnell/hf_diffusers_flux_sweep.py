#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

from benchmarks.flux1_schnell.flux_prompt_bank import prompt_digest, request_prompts


MODES = ("bf16-compile",)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_pipeline(model_path: Path, mode: str):
    import torch
    from diffusers import FluxPipeline

    load_started = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Keep VAE decode in the measured pipeline while avoiding PIL conversion,
    # file encoding, and a GPU-to-host image copy.
    pipe.image_processor.postprocess = (
        lambda image, output_type="pt", do_denormalize=None: image
    )

    if mode.endswith("compile"):
        pipe.transformer.compile(
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )

    torch.cuda.synchronize()
    return pipe, time.perf_counter() - load_started


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
        "--prompt",
        default="A cinematic photograph of a black forest at sunrise, highly detailed",
    )
    parser.add_argument(
        "--batch-semantics",
        choices=("request-batch", "images-per-prompt"),
        default="request-batch",
        help="request-batch encodes B prompt entries; images-per-prompt encodes one prompt",
    )
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="Bracket measured iterations with cudaProfilerStart/Stop",
    )
    args = parser.parse_args()

    run_dir = args.output_dir / f"hf-diffusers-{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    import diffusers
    import torch
    import transformers

    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
    }
    load_result = {
        "status": "error",
        "backend": "hf-diffusers",
        "mode": args.mode,
        "environment": environment,
    }
    try:
        pipe, load_seconds = load_pipeline(args.model, args.mode)
        load_result.update(
            {
                "status": "ok",
                "load_seconds": load_seconds,
            }
        )
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
        prompts = (
            request_prompts(batch)
            if args.batch_semantics == "request-batch"
            else [args.prompt]
        )
        prompt_input = prompts if args.batch_semantics == "request-batch" else args.prompt
        images_per_prompt = 1 if args.batch_semantics == "request-batch" else batch
        result = {
            "status": "error",
            "backend": "hf-diffusers",
            "mode": args.mode,
            "batch_size": batch,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "max_sequence_length": args.max_sequence_length,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "batch_semantics": args.batch_semantics,
            "prompt_count": len(prompts),
            "images_per_prompt": images_per_prompt,
            "prompt_sha256": prompt_digest(prompts),
            "environment": environment,
        }
        try:
            def generate(seed: int):
                generator = torch.Generator(device="cuda").manual_seed(seed)
                return pipe(
                    prompt=prompt_input,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=0.0,
                    max_sequence_length=args.max_sequence_length,
                    num_images_per_prompt=images_per_prompt,
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
                        f"hf_diffusers_request_batch_b{batch}_iteration_{index}"
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
