#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import copy
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch

from benchmarks.flux1_schnell.flux_prompt_bank import prompt_digest, request_prompts


def install_worker_nsys_capture() -> None:
    """Bracket one DiffusionExecutor request inside the GPU-owning process."""
    if os.environ.get("VISUALGEN_NSYS_WORKER_CAPTURE") != "1":
        return

    import importlib

    import torch
    from tensorrt_llm._torch.visual_gen import executor as executor_module

    DiffusionExecutor = executor_module.DiffusionExecutor

    if os.environ.get("VISUALGEN_NSYS_INLINE_WORKER") == "1":
        master_port = executor_module.find_free_port()

        def detect_inline_launch():
            return 0, 0, 1, "127.0.0.1", master_port

        executor_module._detect_external_launch = detect_inline_launch
        visual_gen_module = importlib.import_module(
            "tensorrt_llm.visual_gen.visual_gen"
        )
        visual_gen_module._detect_external_launch = detect_inline_launch

    if getattr(DiffusionExecutor, "_nsys_worker_capture_installed", False):
        return

    warmup_requests = int(os.environ.get("VISUALGEN_NSYS_WARMUP_REQUESTS", "2"))
    original_process_request = DiffusionExecutor.process_request

    def profiled_process_request(self, req):
        request_index = getattr(self, "_nsys_request_index", 0)
        self._nsys_request_index = request_index + 1
        if request_index != warmup_requests:
            return original_process_request(self, req)

        batch = len(req.prompt)
        marker_path = os.environ.get("VISUALGEN_NSYS_MARKER_PATH")
        if marker_path:
            Path(marker_path).write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "rank": self.rank,
                        "request_id": req.request_id,
                        "request_index": request_index,
                        "batch_size": batch,
                    },
                    indent=2,
                )
                + "\n"
            )
        use_cuda_profiler = (
            os.environ.get("VISUALGEN_NSYS_CAPTURE_MODE", "cudaProfilerApi")
            == "cudaProfilerApi"
        )
        torch.cuda.synchronize()
        if use_cuda_profiler:
            torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(
            f"visualgen_worker_request_b{batch}_id{req.request_id}"
        )
        try:
            result = original_process_request(self, req)
            time.sleep(0.25)
            return result
        finally:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            if use_cuda_profiler:
                torch.cuda.cudart().cudaProfilerStop()
            if os.environ.get("VISUALGEN_NSYS_WORKER_EXIT") == "1":
                time.sleep(0.5)
                os._exit(0)

    DiffusionExecutor.process_request = profiled_process_request
    DiffusionExecutor._nsys_worker_capture_installed = True


install_worker_nsys_capture()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def validate_outputs(raw_output, batch: int, batch_semantics: str):
    if batch_semantics == "request-batch":
        if not isinstance(raw_output, list):
            raise RuntimeError(
                "VisualGen request batching returned a single response instead of a list"
            )
        outputs = raw_output
        if len(outputs) != batch:
            raise RuntimeError(
                f"VisualGen returned {len(outputs)} responses for {batch} prompts"
            )
    else:
        outputs = raw_output if isinstance(raw_output, list) else [raw_output]
        if len(outputs) != 1:
            raise RuntimeError(
                f"VisualGen returned {len(outputs)} responses for one prompt"
            )

    image_shapes = []
    image_count = len(outputs) if batch_semantics == "request-batch" else 0
    for index, output in enumerate(outputs):
        if output.error is not None:
            raise RuntimeError(f"VisualGen response {index} failed: {output.error}")
        if output.metrics is None:
            raise RuntimeError(f"VisualGen response {index} has no timing metrics")
        if output.image is None:
            raise RuntimeError(f"VisualGen response {index} has no image")
        shape = list(output.image.shape)
        if not shape:
            raise RuntimeError(f"VisualGen response {index} has a scalar image")
        image_shapes.append(shape)
        if batch_semantics != "request-batch":
            image_count += shape[0] if len(shape) == 4 else 1

    if image_count != batch:
        raise RuntimeError(
            f"VisualGen produced {image_count} images for requested batch {batch}"
        )
    return outputs, image_shapes, image_count


def main() -> None:
    process_start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "nvfp4"), required=True)
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
        "--batch-semantics",
        choices=("request-batch", "images-per-prompt"),
        default="request-batch",
    )
    parser.add_argument(
        "--prompt",
        default="A cinematic photograph of a black forest at sunrise, highly detailed",
    )
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="Bracket measured iterations with cudaProfilerStart/Stop",
    )
    parser.add_argument(
        "--nsys-fast-exit",
        action="store_true",
        help="Exit after flushing the final result, bypassing VisualGen teardown",
    )
    parser.add_argument(
        "--nsys-measure-not-before",
        type=float,
        default=0.0,
        help="Wait until this many seconds after process start before measurement",
    )
    parser.add_argument(
        "--nsys-post-measure-sleep",
        type=float,
        default=0.0,
        help="Keep the process alive briefly so Nsight can flush",
    )
    args = parser.parse_args()

    if any(batch < 1 for batch in args.batches):
        parser.error("--batches values must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / f"visualgen-{args.precision}"
    run_dir.mkdir(parents=True, exist_ok=True)

    import tensorrt_llm
    from tensorrt_llm import VisualGen, VisualGenArgs

    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    try:
        environment.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "tensorrt_llm": tensorrt_llm.__version__,
            }
        )
    except (AttributeError, RuntimeError) as exc:
        environment["inspection_error"] = repr(exc)

    engine_args = VisualGenArgs.from_yaml(str(args.config))
    had_error = False
    with VisualGen(model=str(args.model), args=engine_args) as visual_gen:
        for batch in args.batches:
            prompts = (
                request_prompts(batch)
                if args.batch_semantics == "request-batch"
                else [args.prompt]
            )
            prompt_input = (
                prompts if args.batch_semantics == "request-batch" else args.prompt
            )
            images_per_prompt = 1 if args.batch_semantics == "request-batch" else batch
            result = {
                "status": "error",
                "backend": "trtllm-visualgen",
                "precision": args.precision,
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
                "config": str(args.config),
            }
            try:
                params = copy.deepcopy(visual_gen.default_params)
                params.height = args.height
                params.width = args.width
                params.num_inference_steps = args.steps
                params.guidance_scale = 0.0
                params.seed = args.seed
                params.max_sequence_length = args.max_sequence_length
                params.num_images_per_prompt = images_per_prompt

                for _ in range(args.warmup):
                    raw_output = visual_gen.generate(inputs=prompt_input, params=params)
                    validate_outputs(raw_output, batch, args.batch_semantics)

                wall_latencies = []
                generation_latencies = []
                pre_denoise = []
                denoise = []
                post_denoise = []
                outputs = None
                image_shapes = None
                image_count = None
                if args.nsys_measure_not_before > 0:
                    wait_seconds = args.nsys_measure_not_before - (
                        time.monotonic() - process_start
                    )
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                torch.cuda.synchronize()
                if args.nsys_capture:
                    torch.cuda.cudart().cudaProfilerStart()
                try:
                    for index in range(args.iterations):
                        params.seed = args.seed + args.warmup + index
                        begin = time.perf_counter()
                        with torch.cuda.nvtx.range(
                            f"visualgen_request_batch_b{batch}_iteration_{index}"
                        ):
                            raw_output = visual_gen.generate(
                                inputs=prompt_input, params=params
                            )
                            torch.cuda.synchronize()
                        wall_latencies.append(time.perf_counter() - begin)
                        outputs, image_shapes, image_count = validate_outputs(
                            raw_output, batch, args.batch_semantics
                        )
                        for output in outputs:
                            generation_latencies.append(output.metrics.generation)
                            pre_denoise.append(output.metrics.pre_denoise)
                            denoise.append(output.metrics.denoise)
                            post_denoise.append(output.metrics.post_denoise)
                finally:
                    if args.nsys_capture:
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()

                if outputs is None or image_shapes is None or image_count is None:
                    raise RuntimeError("VisualGen completed no measured iteration")
                mean_seconds = statistics.mean(wall_latencies)
                result.update(
                    {
                        "status": "ok",
                        "output_count": image_count,
                        "image_shapes": image_shapes,
                        "timing_scope": "client_wall_synchronous_generate",
                        "wall_latency_seconds": wall_latencies,
                        "generation_latency_seconds": generation_latencies,
                        "mean_batch_latency_ms": mean_seconds * 1000.0,
                        "median_batch_latency_ms": statistics.median(
                            wall_latencies
                        )
                        * 1000.0,
                        "p90_batch_latency_ms": percentile(
                            wall_latencies, 0.9
                        )
                        * 1000.0,
                        "images_per_second": batch / mean_seconds,
                        "mean_wall_latency_ms": statistics.mean(wall_latencies)
                        * 1000.0,
                        "mean_generation_latency_ms": statistics.mean(
                            generation_latencies
                        )
                        * 1000.0,
                        "mean_pre_denoise_ms": statistics.mean(pre_denoise)
                        * 1000.0,
                        "mean_denoise_ms": statistics.mean(denoise) * 1000.0,
                        "mean_post_denoise_ms": statistics.mean(post_denoise)
                        * 1000.0,
                    }
                )
                if batch == 1:
                    sample_path = run_dir / "sample-b1.png"
                    outputs[0].save(str(sample_path))
                    result["sample_image"] = str(sample_path)
            except Exception as exc:
                had_error = True
                result["error_type"] = type(exc).__name__
                result["error"] = str(exc)
                result["traceback"] = traceback.format_exc()

            output_path = run_dir / f"b{batch}.json"
            output_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
            if args.nsys_post_measure_sleep > 0:
                time.sleep(args.nsys_post_measure_sleep)
            if (
                args.nsys_fast_exit
                and result.get("status") == "ok"
                and batch == args.batches[-1]
            ):
                os._exit(0)
    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
