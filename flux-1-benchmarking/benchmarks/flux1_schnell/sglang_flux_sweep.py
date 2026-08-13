#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Benchmark FLUX.1-schnell through the SGLang offline API."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from benchmarks.flux1_schnell.flux_prompt_bank import prompt_digest, request_prompts


MODE = "bf16-offline-compile"
MODE_DIR = "sglang-bf16-offline-compile"


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


def build_requests(
    engine: Any,
    prompts: list[str],
    args: argparse.Namespace,
    seed: int,
) -> list[Any]:
    """Build one compatible scheduler batch with one request per prompt."""
    from sglang.multimodal_gen import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    requests = []
    for index, prompt in enumerate(prompts):
        sampling_params = SamplingParams.from_user_sampling_params_args(
            str(args.model),
            server_args=engine.server_args,
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            seed=seed + index,
            num_outputs_per_prompt=1,
            save_output=False,
            return_file_paths_only=False,
            suppress_logs=True,
        )
        sampling_params.diffusers_kwargs = {
            "max_sequence_length": args.max_sequence_length
        }
        requests.append(
            prepare_request(
                server_args=engine.server_args,
                sampling_params=sampling_params,
            )
        )
    return requests


def run_request_batch(
    engine: Any,
    scheduler_client: Any,
    event_loop: asyncio.AbstractEventLoop,
    prompts: list[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[float, list[Any]]:
    requests = build_requests(engine, prompts, args, seed)

    async def send_requests() -> list[Any]:
        return list(
            await asyncio.wait_for(
                asyncio.gather(
                    *(scheduler_client.forward(request) for request in requests)
                ),
                timeout=args.request_timeout_seconds,
            )
        )

    started = time.perf_counter()
    output_batches = event_loop.run_until_complete(send_requests())
    latency = time.perf_counter() - started
    outputs = []
    for index, output_batch in enumerate(output_batches):
        if output_batch.error:
            raise RuntimeError(str(output_batch.error))
        batch_outputs = output_batch.output
        if batch_outputs is None or len(batch_outputs) != 1:
            count = 0 if batch_outputs is None else len(batch_outputs)
            raise RuntimeError(
                f"Expected one decoded output for request {index}, got {count}"
            )
        outputs.append(batch_outputs[0])

    return latency, outputs


def save_sample(value: Any, path: Path) -> bool:
    """Save the B1 API output when it is an image-like value."""
    from PIL import Image

    if isinstance(value, Image.Image):
        value.save(path)
        return True

    try:
        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        if isinstance(value, np.ndarray):
            array = value
            while array.ndim > 3 and array.shape[0] == 1:
                array = array[0]
            if array.ndim == 3 and array.shape[0] in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            if array.dtype != np.uint8:
                array = array.astype(np.float32)
                if array.min() < 0.0:
                    array = array / 2.0 + 0.5
                if array.max() <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
            Image.fromarray(array).save(path)
            return True
    except (TypeError, ValueError):
        return False
    return False


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
    parser.add_argument("--batching-delay-ms", type=float, default=100.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--attention-backend", default="torch_sdpa")
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="Capture warmup and measured offline API calls via an NVTX range",
    )
    args = parser.parse_args()
    if args.nsys_capture and len(args.batches) != 1:
        parser.error("--nsys-capture requires exactly one batch size")

    run_dir = args.output_dir / MODE_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    from sglang.multimodal_gen import DiffGenerator
    from sglang.multimodal_gen.runtime.scheduler_client import (
        async_scheduler_client,
    )

    runtime_environment = {
        "sglang": importlib.metadata.version("sglang"),
        "container": "lmsysorg/sglang:v0.5.12",
        "precision": "BF16 DiT and BF16 VAE",
        "attention_backend": args.attention_backend,
        "api": "sglang.multimodal_gen.DiffGenerator",
    }

    load_result = {
        "status": "error",
        "backend": "sglang",
        "mode": MODE,
        "environment": runtime_environment,
        "batching_max_size": max(args.batches),
        "batching_delay_ms": args.batching_delay_ms,
    }
    load_started = time.perf_counter()
    try:
        engine = DiffGenerator.from_pretrained(
            model_path=str(args.model),
            local_mode=True,
            performance_mode="speed",
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
            vae_cpu_offload=False,
            enable_torch_compile=True,
            attention_backend=args.attention_backend,
            batching_mode="dynamic",
            batching_max_size=max(args.batches),
            batching_delay_ms=args.batching_delay_ms,
            enable_batching_metrics=True,
            dit_precision="bf16",
            vae_precision="bf16",
            output_path=None,
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
    load_result.update(
        {
            "status": "ok",
            "api_load_seconds": time.perf_counter() - load_started,
        }
    )
    (run_dir / "load.json").write_text(json.dumps(load_result, indent=2) + "\n")
    print(json.dumps(load_result, indent=2), flush=True)

    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    async_scheduler_client.initialize(engine.server_args)
    try:
        for batch in args.batches:
            prompts = request_prompts(batch)
            result = {
                "status": "error",
                "backend": "sglang",
                "mode": MODE,
                "batch_size": batch,
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "max_sequence_length": args.max_sequence_length,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "batch_semantics": "request-batch",
                "batching_delay_ms": args.batching_delay_ms,
                "request_timeout_seconds": args.request_timeout_seconds,
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
                            engine,
                            async_scheduler_client,
                            event_loop,
                            prompts,
                            args,
                            args.seed + index * batch,
                        )

                    latencies = []
                    last_outputs = None
                    for index in range(args.iterations):
                        latency, outputs = run_request_batch(
                            engine,
                            async_scheduler_client,
                            event_loop,
                            prompts,
                            args,
                            args.seed + (args.warmup + index) * batch,
                        )
                        latencies.append(latency)
                        last_outputs = outputs
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
                if batch == 1 and last_outputs:
                    sample_path = run_dir / "sample-b1.png"
                    if save_sample(last_outputs[0], sample_path):
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
        try:
            async_scheduler_client.close()
        finally:
            try:
                engine.shutdown()
            finally:
                event_loop.close()


if __name__ == "__main__":
    main()
