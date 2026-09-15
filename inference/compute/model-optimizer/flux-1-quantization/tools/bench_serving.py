#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Time steady-state generation through TensorRT-LLM VisualGen.

This is the measurement the quality work could never produce. ``mto.restore``
reproduces NVFP4 *numerics* on BF16 storage, so it says nothing about speed;
only the packed ``--hf-ckpt-dir`` export running on real kernels can answer
"what does the quantization buy".

**One arm per process, deliberately.** VisualGen spawns its own worker and holds
the model resident, so loading a BF16 pipeline and an NVFP4 one in the same
process would contend for the same GPU and give both of them a misleading
number. Each invocation appends to the same JSON file instead.

**Warm-up is not optional.** The first generation pays for CUDA context setup,
kernel autotuning and allocator growth. Including it in the average is the
single easiest way to report a speedup that is really a measurement artefact.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def apply_params(params: Any, overrides: dict[str, Any]) -> list[str]:
    """Set only the fields this build of VisualGenParams actually has.

    The params object differs by model family -- video models carry
    ``num_frames``, image models do not -- so setting a field blindly either
    raises or, worse, silently attaches an attribute the pipeline ignores.
    Returns what was applied so the manifest records the real configuration
    rather than the requested one.
    """
    applied = []
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(params, key):
            setattr(params, key, value)
            applied.append(f"{key}={value}")
    return applied


def describe_quantization(visual_gen: Any) -> dict[str, Any]:
    """Record what the runtime believes it is running.

    Worth capturing rather than assuming: a checkpoint that fails to advertise
    itself as quantized will load and generate perfectly well at BF16, and the
    only symptom is a speedup that never arrives.
    """
    found: dict[str, Any] = {}
    for path in ("args", "config", "_args", "_config"):
        obj = getattr(visual_gen, path, None)
        if obj is None:
            continue
        for field in ("quant_config", "quant_algo", "dynamic_weight_quant"):
            value = getattr(obj, field, None)
            if value is not None:
                found[f"{path}.{field}"] = str(value)
    return found


def main() -> int:
    """Time one arm and append the result to the per-model JSON.

    Loads once, discards a warm-up generation, then times the rest and reports
    the median. The warm-up is not optional: the first generation pays for CUDA
    context setup, kernel autotuning and allocator growth, and including it makes
    every arm look slower by an amount that has nothing to do with precision.

    Appends rather than overwrites, since one arm runs per process. The caller
    owns the file and is responsible for truncating it between runs -- otherwise
    a second run stacks on the first and the baseline lookup finds a stale record.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Arm label, e.g. bf16 or nvfp4-static")
    parser.add_argument("--model", required=True, help="Model directory or Hub ID")
    parser.add_argument("--visual-gen-args", help="VisualGenArgs YAML, if the arm needs one")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=3, help="Timed runs after warm-up")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt", default="A weathered wooden shop sign hanging above a "
                        "cobbled street, the words 'OPEN DAILY' carved and painted in white, "
                        "late afternoon light")
    parser.add_argument("--out", type=Path, help="JSON file to append results to")
    parser.add_argument("--save-image", type=Path, help="Write one image, to eyeball the arm")
    args = parser.parse_args()

    # Validated before the model loads. A bad value used to surface minutes
    # later and far from its cause: --iterations 0 leaves an empty latency list
    # and dies in statistics.median, --steps 0 divides by zero working out
    # ms/step. Both after paying for a full model load.
    for flag, value, minimum in (
        ("--steps", args.steps, 1),
        ("--iterations", args.iterations, 1),
        ("--warmup", args.warmup, 0),
        ("--height", args.height, 1),
        ("--width", args.width, 1),
    ):
        if value < minimum:
            raise SystemExit(f"{flag} must be >= {minimum}, got {value}")

    from tensorrt_llm import VisualGen, VisualGenArgs

    extra = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None

    load_start = time.perf_counter()
    visual_gen = VisualGen(model=args.model, args=extra)
    load_s = time.perf_counter() - load_start
    print(f"  loaded {args.name} in {load_s:.1f}s")

    params = visual_gen.default_params
    applied = apply_params(
        params,
        {
            "num_inference_steps": args.steps,
            "height": args.height,
            "width": args.width,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
        },
    )
    print(f"  params applied: {', '.join(applied) or 'none (using model defaults)'}")

    quantization = describe_quantization(visual_gen)
    if quantization:
        for key, value in quantization.items():
            print(f"  {key}: {value}")
    else:
        print("  (could not read a quant config off the runtime object)")

    for index in range(args.warmup):
        start = time.perf_counter()
        visual_gen.generate(inputs=args.prompt, params=params)
        print(f"  warm-up {index + 1}: {time.perf_counter() - start:.2f}s  (discarded)")

    latencies: list[float] = []
    output = None
    for index in range(args.iterations):
        start = time.perf_counter()
        output = visual_gen.generate(inputs=args.prompt, params=params)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        print(f"  run {index + 1}: {elapsed:.2f}s")

    if args.save_image and output is not None:
        args.save_image.parent.mkdir(parents=True, exist_ok=True)
        output.save(str(args.save_image))
        print(f"  wrote {args.save_image}")

    median = statistics.median(latencies)
    record = {
        "arm": args.name,
        "model": args.model,
        "visual_gen_args": args.visual_gen_args,
        "steps": args.steps,
        "resolution": f"{args.height}x{args.width}",
        "load_s": round(load_s, 2),
        "latencies_s": [round(value, 3) for value in latencies],
        "median_s": round(median, 3),
        "mean_s": round(statistics.fmean(latencies), 3),
        "stdev_s": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else None,
        "s_per_step": round(median / args.steps, 4),
        "quantization": quantization,
        "params_applied": applied,
    }

    print(f"\n  {args.name}: median {median:.2f}s over {args.iterations} runs "
          f"({median / args.steps * 1000:.0f} ms/step)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if args.out.is_file():
            try:
                existing = json.loads(args.out.read_text())
            except Exception:  # noqa: BLE001 - a corrupt file should not lose this run
                existing = []
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(record)
        args.out.write_text(json.dumps(existing, indent=2))
        print(f"  appended to {args.out} ({len(existing)} arms recorded)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
