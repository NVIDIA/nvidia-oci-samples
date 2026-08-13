#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Benchmark FLUX.1-schnell through the vLLM Omni offline API."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from benchmarks.flux1_schnell.flux_prompt_bank import prompt_digest, request_prompts


MODE = "bf16-offline"
MODE_DIR = "vllm-omni-bf16-offline"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def nsys_capture_start() -> int:
    """Start the parent-process NVTX capture range used by Nsight Systems."""
    import torch

    return torch.cuda.nvtx.range_start("flux_offline_profile")


def nsys_capture_stop(range_id: int) -> None:
    """Stop the parent-process NVTX capture range used by Nsight Systems."""
    import torch

    torch.cuda.nvtx.range_end(range_id)


def output_images(outputs: list[Any]) -> list[Any]:
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs

    images = []
    for output in outputs:
        images.extend(extract_images_from_outputs(output))
    return images


def run_request_batch(
    omni: Any,
    model_class_name: str | None,
    prompts: list[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[float, list[Any]]:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import build_text_to_image_prompt

    prompt_dicts = [
        build_text_to_image_prompt(
            model_class_name=model_class_name,
            prompt=prompt,
            negative_prompt=None,
            height=args.height,
            width=args.width,
        )
        for prompt in prompts
    ]
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=seed,
        guidance_scale=0.0,
        num_inference_steps=args.steps,
        num_outputs_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
    )

    started = time.perf_counter()
    outputs = omni.generate(
        prompt_dicts,
        sampling_params_list=[sampling_params],
        use_tqdm=False,
    )
    latency = time.perf_counter() - started
    images = output_images(outputs)
    if len(outputs) != len(prompts) or len(images) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} request outputs and images, got "
            f"{len(outputs)} outputs and {len(images)} images"
        )

    return latency, images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--request-batch-max-wait-ms", type=float, default=100.0)
    parser.add_argument("--init-timeout", type=float, default=900.0)
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="Capture warmup and measured offline API calls via an NVTX range",
    )
    args = parser.parse_args()
    if any(batch < 1 for batch in args.batches):
        parser.error("--batches values must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.nsys_capture and len(args.batches) != 1:
        parser.error("--nsys-capture requires exactly one batch size")

    run_dir = args.output_dir / MODE_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.model_extras import get_model_class_name

    runtime_environment = {
        "vllm_omni": importlib.metadata.version("vllm-omni"),
        "container": "vllm/vllm-omni:v0.26.0",
        "precision": "BF16",
        "attention_backend": "platform default (CUDNN_ATTN on GB300)",
        "api": "vllm_omni.entrypoints.omni.Omni",
    }

    load_result = {
        "status": "error",
        "backend": "vllm-omni",
        "mode": MODE,
        "environment": runtime_environment,
        "max_num_seqs": max(args.batches),
        "request_batch_max_wait_ms": args.request_batch_max_wait_ms,
    }
    load_started = time.perf_counter()
    try:
        omni = Omni(
            model=str(args.model),
            mode="text-to-image",
            max_num_seqs=max(args.batches),
            request_batch_max_wait_ms=args.request_batch_max_wait_ms,
            default_sampling_params={
                "0": {"max_sequence_length": args.max_sequence_length}
            },
            log_stats=False,
            init_timeout=int(args.init_timeout),
            stage_init_timeout=int(args.init_timeout),
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

    try:
        try:
            model_class_name = get_model_class_name(omni)
        except Exception as exc:
            load_result.update(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            (run_dir / "load.json").write_text(
                json.dumps(load_result, indent=2) + "\n"
            )
            print(json.dumps(load_result, indent=2), flush=True)
            raise
        load_result.update(
            {
                "status": "ok",
                "api_load_seconds": time.perf_counter() - load_started,
            }
        )
        (run_dir / "load.json").write_text(json.dumps(load_result, indent=2) + "\n")
        print(json.dumps(load_result, indent=2), flush=True)
        for batch in args.batches:
            prompts = request_prompts(batch)
            result = {
                "status": "error",
                "backend": "vllm-omni",
                "mode": MODE,
                "batch_size": batch,
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "max_sequence_length": args.max_sequence_length,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "batch_semantics": "request-batch",
                "request_batch_max_wait_ms": args.request_batch_max_wait_ms,
                "prompt_count": batch,
                "images_per_prompt": 1,
                "prompt_sha256": prompt_digest(prompts),
                "environment": runtime_environment,
            }
            try:
                capture_range_id = None
                try:
                    if args.nsys_capture:
                        capture_range_id = nsys_capture_start()

                    for index in range(args.warmup):
                        run_request_batch(
                            omni,
                            model_class_name,
                            prompts,
                            args,
                            args.seed + index,
                        )

                    latencies = []
                    last_images = None
                    for index in range(args.iterations):
                        latency, images = run_request_batch(
                            omni,
                            model_class_name,
                            prompts,
                            args,
                            args.seed + args.warmup + index,
                        )
                        latencies.append(latency)
                        last_images = images
                finally:
                    if capture_range_id is not None:
                        nsys_capture_stop(capture_range_id)

                mean_seconds = statistics.mean(latencies)
                result.update(
                    {
                        "status": "ok",
                        "output_count": batch,
                        "timing_scope": "offline_api_wall_to_complete_outputs",
                        "latency_seconds": latencies,
                        "api_completed_with_all_outputs": True,
                        "nsys_capture": args.nsys_capture,
                        "nsys_capture_scope": (
                            "warmup_and_measured_offline_api_calls"
                            if args.nsys_capture
                            else None
                        ),
                        "mean_batch_latency_ms": mean_seconds * 1000.0,
                        "median_batch_latency_ms": statistics.median(latencies)
                        * 1000.0,
                        "p90_batch_latency_ms": percentile(latencies, 0.9)
                        * 1000.0,
                        "images_per_second": batch / mean_seconds,
                        "mean_per_image_ms": mean_seconds * 1000.0 / batch,
                    }
                )
                if batch == 1 and last_images:
                    sample_path = run_dir / "sample-b1.png"
                    last_images[0].save(sample_path)
                    result["sample_image"] = str(sample_path)
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
    finally:
        omni.close()


if __name__ == "__main__":
    main()
