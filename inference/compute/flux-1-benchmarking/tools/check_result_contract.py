#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import json
import math
from pathlib import Path

from benchmarks.flux1_schnell.flux_prompt_bank import prompt_digest, request_prompts


def output_count(result: dict) -> int | None:
    explicit_count = result.get("output_count")
    if explicit_count is not None:
        if isinstance(explicit_count, bool) or not isinstance(explicit_count, int):
            return None
        return explicit_count
    for key in ("output_shape", "image_shape"):
        shape = result.get(key)
        if shape is None:
            continue
        if not isinstance(shape, (list, tuple)) or not shape:
            return None
        count = shape[0]
        if isinstance(count, bool) or not isinstance(count, int):
            return None
        return count
    image_shapes = result.get("image_shapes")
    if (
        isinstance(image_shapes, list)
        and image_shapes
        and all(
            isinstance(shape, (list, tuple))
            and shape
            and all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in shape
            )
            for shape in image_shapes
        )
    ):
        return len(image_shapes)
    return None


def contract_errors(
    result: dict,
    expected_batch: int | None = None,
    expected_warmup: int | None = None,
    expected_iterations: int | None = None,
    expected_height: int = 1024,
    expected_width: int = 1024,
    expected_steps: int = 4,
) -> list[str]:
    errors: list[str] = []
    batch = expected_batch if expected_batch is not None else result.get("batch_size")
    if not isinstance(batch, int) or batch <= 0:
        return ["batch_size must be a positive integer"]

    expected_digest = prompt_digest(request_prompts(batch))
    checks = (
        (result.get("status") == "ok", "status must be ok"),
        (result.get("batch_size") == batch, f"batch_size must be {batch}"),
        (result.get("batch_semantics") == "request-batch", "batch_semantics must be request-batch"),
        (result.get("prompt_count") == batch, f"prompt_count must be {batch}"),
        (result.get("images_per_prompt") == 1, "images_per_prompt must be 1"),
        (result.get("prompt_sha256") == expected_digest, "prompt_sha256 does not match the canonical B-prompt bank"),
        (output_count(result) == batch, f"decoded output_count must be {batch}"),
        (result.get("height") == expected_height, f"height must be {expected_height}"),
        (result.get("width") == expected_width, f"width must be {expected_width}"),
        (result.get("steps") == expected_steps, f"steps must be {expected_steps}"),
        (bool(result.get("timing_scope")), "timing_scope must be recorded"),
    )
    for valid, message in checks:
        if not valid:
            errors.append(message)

    completion_flags = (
        "client_completed_after_engine_forward",
        "api_completed_after_engine_forward",
        "api_completed_with_all_outputs",
    )
    for flag in completion_flags:
        if flag in result and result[flag] is not True:
            errors.append("engine API must complete and return every output")

    realized_batches = result.get("realized_request_batch_sizes")
    if realized_batches is not None:
        if not isinstance(realized_batches, list) or any(
            realized != batch for realized in realized_batches
        ):
            errors.append(f"every realized request batch must be {batch}")
        if (
            isinstance(realized_batches, list)
            and isinstance(result.get("iterations"), int)
            and len(realized_batches) != result["iterations"]
        ):
            errors.append("realized request-batch count must match measured iterations")

    if expected_warmup is not None and result.get("warmup") != expected_warmup:
        errors.append(f"warmup must be {expected_warmup}")
    if expected_iterations is not None and result.get("iterations") != expected_iterations:
        errors.append(f"iterations must be {expected_iterations}")

    latency_ms = result.get("mean_batch_latency_ms")
    images_per_second = result.get("images_per_second")
    if not isinstance(latency_ms, (int, float)) or latency_ms <= 0:
        errors.append("mean_batch_latency_ms must be positive")
    if not isinstance(images_per_second, (int, float)) or images_per_second <= 0:
        errors.append("images_per_second must be positive")
    if (
        isinstance(latency_ms, (int, float))
        and latency_ms > 0
        and isinstance(images_per_second, (int, float))
        and images_per_second > 0
    ):
        expected_rate = batch / (latency_ms / 1000.0)
        if not math.isclose(images_per_second, expected_rate, rel_tol=0.02):
            errors.append(
                "images_per_second is inconsistent with batch_size / mean_batch_latency"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether FLUX result metadata follows the benchmark contract"
    )
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--expected-batch", type=int)
    parser.add_argument("--expected-warmup", type=int)
    parser.add_argument("--expected-iterations", type=int)
    parser.add_argument("--expected-height", type=int, default=1024)
    parser.add_argument("--expected-width", type=int, default=1024)
    parser.add_argument("--expected-steps", type=int, default=4)
    args = parser.parse_args()

    has_mismatch = False
    for path in args.paths:
        result = json.loads(path.read_text())
        errors = contract_errors(
            result,
            expected_batch=args.expected_batch,
            expected_warmup=args.expected_warmup,
            expected_iterations=args.expected_iterations,
            expected_height=args.expected_height,
            expected_width=args.expected_width,
            expected_steps=args.expected_steps,
        )
        if errors:
            has_mismatch = True
            print(f"contract mismatch: {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"contract ok: {path}")
    if has_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
