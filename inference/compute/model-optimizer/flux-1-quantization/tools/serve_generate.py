#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generate one arm of the paired quality set through TensorRT-LLM VisualGen.

Every quality figure produced so far came from ``mto.restore``: NVFP4 numerics
executed in BF16, which is the right instrument for "does the recipe cost
quality" and says nothing about the artefact as deployed. This generates the
same prompt set from the **packed 4-bit export on real kernels**, so the score
describes what a customer would actually run.

**One arm per invocation, and both arms through VisualGen.**

Per invocation because VisualGen spawns a worker that holds the model resident;
``del`` does not reap it, so loading a second pipeline in the same process
leaves the first occupying memory. Calling this twice costs one extra model
load and removes the question entirely.

Both arms through VisualGen because reusing the ``verify`` stage's BF16 images
would vary pipeline *and* precision at once, and no difference between them
would be attributable to either.

Records are appended to a shared ``metadata.json``, so the second invocation
joins the first rather than replacing it.

**On pairing.** ``verify`` injects identical starting latents, so a difference
there is precision alone. VisualGen accepts a seed and derives its own latents,
so this set is seed-paired. CMMD is distributional and unaffected; per-image
PSNR is weaker evidence here, and the metadata says so rather than leaving a
reader to assume.

``--determinism-check`` generates one prompt twice and compares the bytes. That
establishes the runtime is deterministic given a seed -- which rules out
run-to-run randomness as the source of a cross-arm difference. It does *not*
establish that the two arms derive identical latents; that would need
instrumentation inside VisualGen, and no claim resting on it should be made
without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from common import images  # noqa: E402

# Fields pushed onto the runtime's params object. Every one of these must match
# across arms or the comparison is measuring more than precision -- and must
# match the verify stage too, or the served figures cannot be set beside the
# mto.restore figures they are meant to replace.
SPEC_FIELDS = (
    "num_inference_steps",
    "height",
    "width",
    "guidance_scale",
    "max_sequence_length",
    "seed",
)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def spec_values(spec: images.GenerationSpec, seed: int) -> dict[str, Any]:
    """Flatten the spec into the parameter names a VisualGen build expects.

    Kept separate from ``apply_spec`` so the intended values can be recorded
    against each image even when the build refuses some of them.
    """
    return {
        "num_inference_steps": spec.steps,
        "height": spec.height,
        "width": spec.width,
        "guidance_scale": spec.guidance_scale,
        "max_sequence_length": spec.max_sequence_length,
        "seed": seed,
    }


def apply_spec(params: Any, spec: images.GenerationSpec, seed: int) -> list[str]:
    """Push the shared generation settings onto whatever params this build has.

    Returns the fields that were **not** settable. A missing field is not
    automatically a problem -- video models carry ``num_frames`` and image
    models do not -- but ``max_sequence_length`` silently defaulting would put
    the served arms on a different text-encoder setting from the verify arms,
    and the comparison those figures exist for would be invalid without
    anything looking wrong.
    """
    skipped = []
    for key, value in spec_values(spec, seed).items():
        if hasattr(params, key):
            setattr(params, key, value)
        else:
            skipped.append(key)
    return skipped


def build(model: str, visual_gen_args: str | None) -> Any:
    """Load a model into a VisualGen engine and report how long it took.

    Omitting ``visual_gen_args`` is how a BF16 arm is requested; passing a YAML is
    how a quantized one is. The YAML must not carry a ``quant_config``, or
    VisualGen takes its own path, defaults ``dynamic`` to true and re-quantizes at
    load time -- scoring the runtime-quantized model while appearing to score the
    checkpoint.
    """
    from tensorrt_llm import VisualGen, VisualGenArgs

    extra = VisualGenArgs.from_yaml(visual_gen_args) if visual_gen_args else None
    start = time.perf_counter()
    engine = VisualGen(model=model, args=extra)
    print(f"  loaded in {time.perf_counter() - start:.1f}s")
    return engine


def determinism_check(engine: Any, spec: images.GenerationSpec,
                      prompt: dict[str, str], out_dir: Path, arm: str) -> dict[str, Any]:
    """Generate the same prompt and seed twice, and compare the bytes."""
    scratch = out_dir / "_determinism"
    scratch.mkdir(parents=True, exist_ok=True)
    digests = []
    for attempt in (1, 2):
        params = engine.default_params
        apply_spec(params, spec, spec.seeds[0])
        output = engine.generate(inputs=prompt["text"], params=params)
        path = scratch / f"{arm}-attempt{attempt}.png"
        output.save(str(path))
        digests.append(file_digest(path))

    identical = digests[0] == digests[1]
    print(f"  determinism: {'identical' if identical else 'DIFFERENT'}  {digests}")
    if not identical:
        print("  WARNING: this runtime is not deterministic given a seed. A difference")
        print("           between arms cannot be attributed to precision.")
    return {"identical": identical, "digests": digests}


def merge_metadata(path: Path, spec: images.GenerationSpec,
                   new_records: list[dict[str, Any]], pairing: str) -> int:
    """Append this arm's records to a manifest the other arm may already own.

    Written as read-modify-write rather than overwrite because the two arms are
    separate processes. Overwriting would leave whichever ran last as the only
    arm in the file, and the quality stage would report "need both arms" — or
    worse, silently score nothing.
    """
    existing: list[dict[str, Any]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text()).get("images", [])
        except json.JSONDecodeError:
            existing = []

    arms_present = {record["arm"] for record in new_records}
    kept = [record for record in existing if record["arm"] not in arms_present]

    payload = {
        "generation": asdict(spec),
        "images": kept + new_records,
        "pairing": pairing,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return len(payload["images"])


def main() -> int:
    """Generate one arm's images through VisualGen and append its metadata.

    One arm per invocation, into a directory shared with the other arms, because
    VisualGen holds a model resident and two pipelines in one process would
    contend for the GPU. Each run merges its records into the directory's
    ``metadata.json`` so the quality stage sees a complete pair set.

    Pairing here is by seed only. VisualGen derives its own latents rather than
    accepting injected ones, so per-image PSNR is weaker evidence than in the
    simulated arms -- which is why the metadata records which pairing was used.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help="Arm label, e.g. bf16 or nvfp4-static-served")
    parser.add_argument("--model", required=True, help="Model directory")
    parser.add_argument("--config", required=True, type=Path, help="Harness config JSON")
    parser.add_argument("--out", required=True, type=Path, help="Shared output directory")
    parser.add_argument("--visual-gen-args", help="VisualGenArgs YAML. Omit for BF16")
    parser.add_argument("--quant-recipe", help="Recorded against each image")
    parser.add_argument("--prompts", type=int, help="Limit the prompt count")
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    spec = images.GenerationSpec.from_config(config)
    # Before the engine loads. --prompts 0 or a negative value produced an empty
    # prompt list, and the failure only surfaced after several minutes of model
    # loading, as an empty output directory with no explanation.
    if args.prompts is not None and args.prompts < 1:
        raise SystemExit(f"--prompts must be 1 or more, got {args.prompts}")

    prompts = images.load_prompts(config, REPOSITORY_ROOT, limit=args.prompts)
    if not prompts:
        raise SystemExit(f"No prompts loaded from {config['prompts_file']}")
    total = len(prompts) * len(spec.seeds)

    print(f"  arm {args.arm}: {len(prompts)} prompts x {len(spec.seeds)} seeds = {total} images")
    print(f"  {spec.steps} steps, {spec.height}x{spec.width}, guidance {spec.guidance_scale}, "
          f"max_seq {spec.max_sequence_length}")

    engine = build(args.model, args.visual_gen_args)
    args.out.mkdir(parents=True, exist_ok=True)

    checks = None
    if args.determinism_check:
        checks = determinism_check(engine, spec, prompts[0], args.out, args.arm)

    records: list[dict[str, Any]] = []
    skipped_reported = False
    index = 0
    for prompt in prompts:
        for seed in spec.seeds:
            index += 1
            params = engine.default_params
            skipped = apply_spec(params, spec, seed)

            if skipped and not skipped_reported:
                skipped_reported = True
                print(f"  NOTE: params object has no {skipped}; the runtime default applies")
                if "max_sequence_length" in skipped:
                    print("        max_sequence_length differing from the verify runs would")
                    print("        make these figures non-comparable with the mto.restore ones")

            output = engine.generate(inputs=prompt["text"], params=params)
            name = images.image_name(prompt["id"], seed, args.arm)
            output.save(str(args.out / name))

            records.append(
                {
                    "prompt_id": prompt["id"],
                    "category": prompt.get("category"),
                    "seed": seed,
                    "arm": args.arm,
                    "file": name,
                    "image_sha256_16": file_digest(args.out / name),
                    "quant_recipe": args.quant_recipe,
                }
            )
            print(f"  [{index}/{total}] {name}")

    written = merge_metadata(
        args.out / "metadata.json", spec, records, images.SEEDED_PAIRING
    )
    print(f"\n  {len(records)} images this arm, {written} in metadata.json")

    if checks is not None:
        path = args.out / "determinism.json"
        combined = {}
        if path.is_file():
            try:
                combined = json.loads(path.read_text())
            except json.JSONDecodeError:
                combined = {}
        combined[args.arm] = checks
        path.write_text(json.dumps(combined, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
