#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Derived in part from the BFL FLUX command-line inference flow. See
# THIRD_PARTY_NOTICES.md for the pinned upstream revision and license.

import argparse
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from benchmarks.flux1_schnell.flux_prompt_bank import prompt_digest, request_prompts


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def save_first_image(tensor: torch.Tensor, output_path: Path) -> dict[str, float]:
    image = tensor[0].detach().float().clamp(-1, 1)
    stats = {
        "mean": image.mean().item(),
        "std": image.std().item(),
        "min": image.min().item(),
        "max": image.max().item(),
    }
    pixels = ((image + 1.0) * 127.5).round().to(torch.uint8)
    pixels = pixels.permute(1, 2, 0).cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_path)
    return stats


def custom_onnx_paths(onnx_dir: Path, precision: str) -> str:
    paths = {
        "clip": onnx_dir / "clip.opt/model.onnx",
        "transformer": onnx_dir / f"transformer.opt/{precision}/model.onnx",
        "t5": onnx_dir / "t5.opt/model.onnx",
        "vae": onnx_dir / "vae.opt/model.onnx",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ONNX files: {missing}")
    return ",".join(f"{name}:{path}" for name, path in paths.items())


def load_pytorch(args):
    from flux.modules.conditioner import HFEmbedder
    from flux.util import load_ae, load_flow_model

    os.environ["FLUX_MODEL"] = str(args.native_model_dir / "flux1-schnell.safetensors")
    os.environ["FLUX_AE"] = str(args.native_model_dir / "ae.safetensors")
    for name in ("FLUX_MODEL", "FLUX_AE"):
        if not Path(os.environ[name]).is_file():
            raise FileNotFoundError(f"Missing native checkpoint: {os.environ[name]}")

    device = torch.device("cuda")
    t5_dir = args.native_model_dir.parent / "google_t5-v1_1-xxl"
    clip_dir = args.native_model_dir.parent / "openai_clip-vit-large-patch14"
    if not (t5_dir / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"Missing native T5 checkpoint: {t5_dir}")
    if not (clip_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing native CLIP checkpoint: {clip_dir}")

    # HFEmbedder dispatches CLIP versus T5 from the repository-name prefix.
    # Instantiate from the official names, but temporarily point the local-only
    # Transformers cache at durable checkpoint directories through symlinks.
    alias_root = args.output_dir / "native-model-aliases"
    aliases = {
        alias_root / "google/t5-v1_1-xxl": t5_dir,
        alias_root / "openai/clip-vit-large-patch14": clip_dir,
    }
    for alias, target in aliases.items():
        alias.parent.mkdir(parents=True, exist_ok=True)
        resolved_target = target.resolve()
        if alias.is_symlink():
            if alias.resolve(strict=False) != resolved_target:
                alias.unlink()
                alias.symlink_to(resolved_target, target_is_directory=True)
        elif not alias.exists():
            alias.symlink_to(resolved_target, target_is_directory=True)
    previous_cwd = Path.cwd()
    os.chdir(alias_root)
    try:
        t5 = HFEmbedder(
            "google/t5-v1_1-xxl", max_length=256, torch_dtype=torch.bfloat16
        ).to(device)
        clip = HFEmbedder(
            "openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16
        ).to(device)
    finally:
        os.chdir(previous_cwd)
    model = load_flow_model("flux-schnell", device=device, verbose=False)
    ae = load_ae("flux-schnell", device=device)
    if args.variant == "compile":
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    return None, t5, clip, model, ae


def load_tensorrt(args, batch_size: int):
    # Importing this module installs TensorRT 11 compatibility and optional
    # CUDA Graph patches before TRTManager constructs any engines.
    from benchmarks.flux1_schnell import flux_t2i_trt11  # noqa: F401
    from flux.trt.trt_manager import ModuleName, TRTManager

    if args.variant == "cuda_graph":
        os.environ["FLUX_TRT_CUDA_GRAPH"] = "1"
        os.environ["FLUX_TRT_CUDA_GRAPH_ENGINES"] = "TransformerEngine"
        os.environ["FLUX_TRT_DEDICATED_STREAM"] = "1"
    else:
        os.environ["FLUX_TRT_CUDA_GRAPH"] = "0"
        os.environ["FLUX_TRT_DEDICATED_STREAM"] = "1"
    os.environ["FLUX_TRT_NVTX"] = "1" if args.nsys_capture else "0"

    engine_dir = args.engine_root / f"b{batch_size}"
    engine_dir.mkdir(parents=True, exist_ok=True)
    if not args.build_only:
        plan_dir = engine_dir / "flux-schnell"
        transformer_plans = list(plan_dir.glob(f"transformer_{args.precision}*.plan"))
        if not transformer_plans:
            raise FileNotFoundError(
                f"No prebuilt TensorRT {args.precision} transformer plan for batch "
                f"{batch_size}; see the corresponding build result"
            )
    manager = TRTManager(
        trt_transformer_precision=args.precision,
        trt_t5_precision="bf16",
        max_batch=32,
    )
    engines = manager.load_engines(
        model_name="flux-schnell",
        module_names={
            ModuleName.CLIP,
            ModuleName.TRANSFORMER,
            ModuleName.T5,
            ModuleName.VAE,
        },
        engine_dir=str(engine_dir),
        custom_onnx_paths=custom_onnx_paths(args.onnx_dir, args.precision),
        trt_image_height=args.height,
        trt_image_width=args.width,
        trt_batch_size=batch_size,
        trt_static_batch=True,
        trt_static_shape=True,
        # NVFP4 plans need the full TensorRT tactic-source search for compatible
        # kernels; BF16 retains the upstream constrained tactic selection.
        trt_enable_all_tactics=args.precision == "fp4",
        trt_timing_cache=str(engine_dir / "timing_cache.bin"),
    )
    device = torch.device("cuda")
    return (
        manager,
        engines[ModuleName.T5].to(device),
        engines[ModuleName.CLIP].to(device),
        engines[ModuleName.TRANSFORMER].to(device),
        engines[ModuleName.VAE].to(device),
    )


@torch.inference_mode()
def generate(t5, clip, model, ae, args, batch_size: int, seed: int):
    device = torch.device("cuda")
    noise = get_noise(
        batch_size,
        args.height,
        args.width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
    )
    prompts = (
        request_prompts(batch_size)
        if args.batch_semantics == "request-batch"
        else [args.prompt] * batch_size
    )
    inputs = prepare(t5, clip, noise, prompt=prompts)
    timesteps = get_schedule(args.steps, inputs["img"].shape[1], shift=False)
    latents = denoise(model, **inputs, timesteps=timesteps, guidance=0.0)
    latents = unpack(latents.float(), args.height, args.width)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return ae.decode(latents)


def benchmark_batch(args, batch_size: int) -> dict:
    manager = None
    started = time.time()
    try:
        if args.backend == "pytorch":
            raise RuntimeError("PyTorch models must be supplied by the outer sweep")
        manager, t5, clip, model, ae = load_tensorrt(args, batch_size)
        plans = sorted((args.engine_root / f"b{batch_size}" / "flux-schnell").glob("*.plan"))
        if args.build_only:
            return {
                "status": "built",
                "backend": args.backend,
                "precision": args.precision,
                "variant": args.variant,
                "batch_size": batch_size,
                "plan_count": len(plans),
                "plan_bytes": sum(path.stat().st_size for path in plans),
                "elapsed_seconds": time.time() - started,
            }
        return run_measurements(args, batch_size, t5, clip, model, ae)
    finally:
        if manager is not None:
            manager.stop_runtime()
        gc.collect()
        torch.cuda.empty_cache()


def run_measurements(args, batch_size: int, t5, clip, model, ae) -> dict:
    torch.cuda.synchronize()
    model_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = None
    for index in range(args.warmup):
        output = generate(t5, clip, model, ae, args, batch_size, args.seed + index)
        torch.cuda.synchronize()

    latencies = []
    if args.nsys_capture:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for index in range(args.iterations):
            seed = args.seed + args.warmup + index
            begin = time.perf_counter()
            with torch.cuda.nvtx.range(
                f"flux_request_batch_b{batch_size}_iteration_{index}"
            ):
                output = generate(t5, clip, model, ae, args, batch_size, seed)
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - begin)
    finally:
        if args.nsys_capture:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

    if output is None:
        raise RuntimeError("No output generated")
    if output.ndim < 1 or output.shape[0] != batch_size:
        raise RuntimeError(
            f"Expected {batch_size} decoded images, got {getattr(output, 'shape', None)}"
        )
    prompts = (
        request_prompts(batch_size)
        if args.batch_semantics == "request-batch"
        else [args.prompt]
    )
    mean_seconds = statistics.mean(latencies)
    image_path = args.output_dir / "images" / (
        f"{args.backend}-{args.precision}-{args.variant}-b{batch_size}.png"
    )
    image_stats = save_first_image(output, image_path)
    return {
        "status": "ok",
        "backend": args.backend,
        "precision": args.precision,
        "variant": args.variant,
        "batch_size": batch_size,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "batch_semantics": args.batch_semantics,
        "prompt_count": batch_size if args.batch_semantics == "request-batch" else 1,
        "images_per_prompt": 1 if args.batch_semantics == "request-batch" else batch_size,
        "prompt_sha256": prompt_digest(prompts),
        "output_shape": list(output.shape),
        "timing_scope": "host_wall_with_cuda_synchronize",
        "latency_seconds": latencies,
        "mean_batch_latency_ms": mean_seconds * 1000.0,
        "median_batch_latency_ms": statistics.median(latencies) * 1000.0,
        "p90_batch_latency_ms": percentile(latencies, 0.9) * 1000.0,
        "images_per_second": batch_size / mean_seconds,
        "mean_per_image_ms": mean_seconds * 1000.0 / batch_size,
        "model_memory_gib": model_memory / 2**30,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "sample_image": str(image_path),
        "sample_stats": image_stats,
    }


def write_result(args, result: dict) -> None:
    run_dir = args.output_dir / "runs" / f"{args.backend}-{args.precision}-{args.variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / f"b{result['batch_size']}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("trt", "pytorch"), required=True)
    parser.add_argument("--precision", choices=("bf16", "fp4"), required=True)
    parser.add_argument("--variant", choices=("eager", "cuda_graph", "compile"), default="eager")
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="A cinematic photograph of a black forest at sunrise, highly detailed",
    )
    parser.add_argument(
        "--batch-semantics",
        choices=("request-batch", "images-per-prompt"),
        default="request-batch",
        help="request-batch encodes B prompt entries; images-per-prompt repeats one prompt B times",
    )
    parser.add_argument("--engine-root", type=Path)
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument("--native-model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="Bracket measured iterations with cudaProfilerStart/Stop",
    )
    args = parser.parse_args()

    if args.backend == "pytorch" and args.precision != "bf16":
        parser.error("The BFL native PyTorch path does not provide NVFP4 weights")
    if args.backend == "pytorch" and args.variant != "compile":
        parser.error("Only the compile variant is supported for the PyTorch backend")
    if args.backend == "pytorch" and args.build_only:
        parser.error("--build-only is supported only by the TensorRT backend")
    if args.backend == "trt" and args.variant == "compile":
        parser.error("The compile variant is supported only by the PyTorch backend")
    if args.backend == "pytorch" and args.native_model_dir is None:
        parser.error("--native-model-dir is required for the PyTorch backend")
    if args.backend == "trt" and args.onnx_dir is None:
        parser.error("--onnx-dir is required for the TensorRT backend")
    if args.backend == "trt" and args.engine_root is None:
        parser.error("--engine-root is required for the TensorRT backend")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        import tensorrt

        environment["tensorrt"] = tensorrt.__version__
    except ImportError:
        pass
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )

    if args.backend == "pytorch":
        manager, t5, clip, model, ae = load_pytorch(args)
        had_error = False
        for batch_size in args.batch_sizes:
            try:
                result = run_measurements(args, batch_size, t5, clip, model, ae)
            except Exception as exc:
                had_error = True
                result = {
                    "status": "error",
                    "backend": args.backend,
                    "precision": args.precision,
                    "variant": args.variant,
                    "batch_size": batch_size,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                gc.collect()
                torch.cuda.empty_cache()
            write_result(args, result)
        del manager, t5, clip, model, ae
        gc.collect()
        torch.cuda.empty_cache()
        if had_error:
            raise SystemExit(1)
        return

    had_error = False
    for batch_size in args.batch_sizes:
        try:
            result = benchmark_batch(args, batch_size)
        except Exception as exc:
            had_error = True
            result = {
                "status": "error",
                "backend": args.backend,
                "precision": args.precision,
                "variant": args.variant,
                "batch_size": batch_size,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        write_result(args, result)
    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
