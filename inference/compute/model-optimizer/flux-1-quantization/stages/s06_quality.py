# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Score the paired images.

Metrics are chosen to match how generated images are actually judged in practice,
not what is convenient to compute:

* **CMMD** is the primary distributional metric: CLIP-embedding Maximum Mean
  Discrepancy from "Rethinking FID" (Jayasumana et al., CVPR 2024). It is
  sample-efficient, so a few dozen pairs give a usable reading where FID would
  need tens of thousands.
* **PSNR** is a per-image signal rather than a gate. Values well below 30 dB
  routinely pass human review on generative output, because two images can be
  far apart in pixels without either being worse. See the note this stage prints.
* **CLIP score** covers text alignment, and matters disproportionately: a
  well-formed image with the wrong content fails, and no aggregate perceptual
  score will catch that.

LPIPS and SSIM are deliberately not headline metrics here. They are useful
internal diagnostics — TensorRT-LLM's own regression tests use them — but they
are not how generative output is usually accepted or rejected.

**Always compare against a same-precision control.** Score BF16 against BF16 at
different seeds, at the same sample size, and you learn how much the metric moves
from seed choice alone. Without that reference a distributional number is
uninterpretable: it will produce a figure whatever you feed it.

No metric here replaces human review, which is the actual sign-off in any serious
evaluation. The point of this stage is to produce something worth putting in
front of one.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import images

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

BF16_ARM = "bf16"
# ``-sim`` is mto.restore: NVFP4 numerics on BF16 storage, right for quality and
# useless for speed. ``-served`` is the packed export running on real 4-bit
# kernels through TensorRT-LLM VisualGen -- the artefact as deployed. Keeping
# them as separate arms means a score can never be silently attributed to the
# wrong one.
NVFP4_ARMS = (
    "nvfp4-dynamic",
    "nvfp4-static",
    "nvfp4-static-sim",
    "nvfp4-static-hf",
    "nvfp4-static-served",
)


def _load_pairs(image_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read an image directory's metadata, returning the header and the records.

    Both halves are returned because the caller needs each: the records to pair
    on, and the header to know how they were paired -- injected latents or seed
    alone -- which decides how much a per-image number is worth.
    """
    metadata_path = image_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"No metadata.json in {image_dir}. Run the dynamic stage first.")
    metadata = json.loads(metadata_path.read_text())
    return metadata, metadata["images"]


def _psnr(a, b) -> float:
    """Peak signal-to-noise ratio between two images, in decibels.

    Identical images give infinity, which is a real result and not an error --
    and the reason the report sanitises non-finite floats before writing JSON.
    Computed in float64 so the squared error does not saturate.
    """
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(((a - b) ** 2).mean())
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def _clip_scores(pairs: list[tuple[str, Path]], device: str) -> dict[str, float]:
    """CLIP similarity between each prompt and its image."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    scores: dict[str, float] = {}
    with torch.no_grad():
        for prompt_text, image_path in pairs:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                text=[prompt_text], images=image, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            outputs = model(**inputs)
            image_embed = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embed = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            scores[str(image_path)] = float((image_embed @ text_embed.T).item() * 100)

    del model
    torch.cuda.empty_cache()
    return scores


CMMD_SEARCH = (
    "$CMMD_REPO",
    "$WORKSPACE/src/cmmd-pytorch",
    # A shared checkout beside the shared models directory. On a multi-tenant
    # volume the clone is usually made once for the team, a level above any one
    # person's workspace -- the same place the checkpoints live.
    "$SHARED/src/cmmd-pytorch",
    "/opt/cmmd-pytorch",
    "~/cmmd-pytorch",
)


def _find_cmmd_repo(workspace_root: Path) -> Path | None:
    """Locate a cmmd-pytorch checkout.

    It is a script repository, not a package: there is no setup.py, so
    `pip install` does not work. Clone it and point CMMD_REPO at the checkout.
    """
    # $SHARED is the volume the shared models directory sits on, which is where
    # a team-wide clone normally lives.
    try:
        from common import paths as _paths

        shared = str(_paths.resolve(create=False).models.parent)
    except Exception:  # noqa: BLE001 - a missing workspace must not break scoring
        shared = str(workspace_root.parent)

    for candidate in CMMD_SEARCH:
        expanded = os.path.expandvars(
            candidate.replace("$WORKSPACE", str(workspace_root)).replace("$SHARED", shared)
        )
        path = Path(expanded).expanduser()
        if path.is_dir() and (path / "main.py").exists():
            return path
    return None


def _finite(value: Any) -> Any:
    """Replace non-finite floats with None, recursively.

    NaN and +/-Infinity are valid Python floats and invalid JSON. Identical
    images give an infinite PSNR, so this is reachable on a good run, not only
    a broken one.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _cmmd_one(repo: Path, reference: Path, candidate: Path, counts: dict[str, int]) -> dict[str, Any]:
    """Run the upstream tool over one BF16/NVFP4 directory pair."""
    try:
        completed = subprocess.run(
            [sys.executable, "main.py", str(reference), str(candidate)],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"value": None, "reason": f"{type(error).__name__}: {error}"}

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
        return {"value": None, "reason": "cmmd failed: " + " | ".join(tail)}

    # A labelled value if the tool prints one, else the last decimal number in
    # the output. The bare-integer alternative this replaced (`|\d+`) matched any
    # trailing integer, so an unrelated closing line such as "processed 48" was
    # recorded as the CMMD score.
    text = completed.stdout.strip()
    labelled = re.search(r"CMMD[^\d\-+]*([-+]?\d*\.\d+)", text, re.IGNORECASE)
    if labelled:
        value = labelled.group(1)
    else:
        floats = re.findall(r"[-+]?\d*\.\d+", text)
        if not floats:
            return {"value": None, "reason": f"could not parse output: {text[:200]}"}
        value = floats[-1]
    return {
        "value": round(float(value), 4),
        "pairs": counts,
        "repo": str(repo),
        "note": "Lower is better. Sample-efficient, but still noisy below a few dozen pairs.",
    }


def _cmmd(image_dir: Path, records: list[dict[str, Any]], workspace_root: Path) -> dict[str, Any]:
    """CMMD between the BF16 and NVFP4 image sets, one score per NVFP4 arm.

    Returns a result dict rather than a bare float so a skipped metric is
    visible and explained. A missing metric is recoverable; a silently wrong
    one is not, so we never substitute an approximation.

    The upstream tool compares two directories, so the paired images are split
    into two by symlink first.
    """
    repo = _find_cmmd_repo(workspace_root)
    if repo is None:
        return {
            "value": None,
            "reason": "cmmd-pytorch not found. It is a script repo, not a pip package: "
            "git clone https://github.com/sayakpaul/cmmd-pytorch.git and set CMMD_REPO.",
        }

    staging = image_dir / "_cmmd"
    if staging.exists():
        shutil.rmtree(staging)

    reference = staging / "bf16"
    reference.mkdir(parents=True)
    bf16_count = 0
    for record in records:
        if record["arm"] == BF16_ARM:
            (reference / record["file"]).symlink_to((image_dir / record["file"]).resolve())
            bf16_count += 1

    # One candidate directory per NVFP4 arm, never one shared bucket. Merging
    # them averages a served checkpoint together with a simulated one, or two
    # different exclusion filters, and reports the mixture as a single number
    # attributable to none of them -- the precise confusion the separate names
    # in NVFP4_ARMS exist to prevent. Same-named files across arms would also
    # collide in one directory and silently drop pairs.
    per_arm_counts: dict[str, int] = {}
    for record in records:
        arm = record["arm"]
        if arm not in NVFP4_ARMS:
            continue
        directory = staging / arm
        directory.mkdir(parents=True, exist_ok=True)
        (directory / record["file"]).symlink_to((image_dir / record["file"]).resolve())
        per_arm_counts[arm] = per_arm_counts.get(arm, 0) + 1

    if not bf16_count or not per_arm_counts:
        counts = {"bf16": bf16_count, **per_arm_counts}
        return {"value": None, "reason": f"need both arms, have {counts}"}

    scored = {
        arm: _cmmd_one(repo, reference, staging / arm, {"bf16": bf16_count, arm: count})
        for arm, count in sorted(per_arm_counts.items())
    }

    # A single arm is the normal case, and it keeps the flat shape every existing
    # consumer of quality.json already reads.
    if len(scored) == 1:
        arm, only = next(iter(scored.items()))
        return {**only, "arm": arm}

    return {
        "value": None,
        "reason": (
            f"{len(scored)} NVFP4 arms present ({', '.join(scored)}); scored "
            "separately in by_arm. One value across arms would mix precisions "
            "or exclusion filters."
        ),
        "by_arm": scored,
    }


def run(*, workspace, config_path: Path, manifest, args) -> dict[str, Any]:
    """Score a directory of paired images and write the quality report.

    Driven entirely by the ``metadata.json`` in the image directory, so the same
    code scores the dynamic arm, a restored arm or a served directory without
    knowing which produced it -- pass ``--images`` to pick one. Writes
    ``results/quality.json`` and a named archive copy the next run will not
    overwrite, because a second scoring run has already destroyed a set of
    numbers once. Runs without a GPU.

    Prompt ids absent from the current prompts file are fatal rather than
    skipped. CLIP would otherwise score both arms against an empty caption and
    write a ``clip_delta`` that is indistinguishable from a good result.
    """
    config = json.loads(config_path.read_text())
    image_dir = Path(getattr(args, "images", None) or workspace.images / "dynamic")

    if getattr(args, "dry_run", False):
        print(f"  would score paired images in {image_dir}")
        return {}

    metadata, records = _load_pairs(image_dir)
    # Through images.load_prompts rather than re-reading the file here. The two
    # had already drifted: this copy knew nothing about the category-balanced
    # sampling, so a limited run scored against a different prompt set than the
    # one that generated the images.
    prompts = {p["id"]: p for p in images.load_prompts(config, REPOSITORY_ROOT)}

    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_key[(record["prompt_id"], record["seed"])][record["arm"]] = record

    # Scoring images whose prompt ids are not in the current prompts file would
    # CLIP-score both arms against an empty caption and write a plausible-looking
    # clip_delta. Fatal rather than a warning, because the output is unusable and
    # indistinguishable from a good result.
    unknown = sorted({prompt_id for prompt_id, _ in by_key if prompt_id not in prompts})
    if unknown:
        raise RuntimeError(
            f"{len(unknown)} prompt id(s) in {image_dir} are absent from "
            f"{config['prompts_file']}: {', '.join(unknown[:5])}"
            f"{' ...' if len(unknown) > 5 else ''}. "
            "Score against the prompts file that produced these images."
        )

    import numpy as np
    from PIL import Image

    # INSTALL.md documents the quality stage as runnable without a GPU, so the
    # device cannot be assumed. A hardcoded "cuda" turns that into a hard
    # failure on the CPU-only path this stage is meant to support.
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_pair: list[dict[str, Any]] = []
    clip_inputs: list[tuple[str, Path]] = []

    for (prompt_id, seed), arms in sorted(by_key.items()):
        baseline = arms.get(BF16_ARM)
        if baseline is None:
            continue
        for arm_name in NVFP4_ARMS:
            candidate = arms.get(arm_name)
            if candidate is None:
                continue

            base_img = np.array(Image.open(image_dir / baseline["file"]).convert("RGB"))
            cand_img = np.array(Image.open(image_dir / candidate["file"]).convert("RGB"))
            if base_img.shape != cand_img.shape:
                raise RuntimeError(
                    f"Image sizes differ for {prompt_id} seed {seed}; the pair is not comparable."
                )

            entry = {
                "prompt_id": prompt_id,
                "category": candidate.get("category"),
                "seed": seed,
                "arm": arm_name,
                "psnr_db": round(_psnr(base_img, cand_img), 2),
                "latent_sha256_16": candidate.get("latent_sha256_16"),
            }
            per_pair.append(entry)
            text = prompts.get(prompt_id, {}).get("text", "")
            clip_inputs.append((text, image_dir / baseline["file"]))
            clip_inputs.append((text, image_dir / candidate["file"]))

    if not per_pair:
        raise RuntimeError("No matched BF16/NVFP4 pairs found. Check the dynamic stage output.")

    clip_scores = _clip_scores(clip_inputs, device)
    for entry in per_pair:
        base_file = image_dir / f"{entry['prompt_id']}__seed{entry['seed']}__{BF16_ARM}.png"
        cand_file = image_dir / f"{entry['prompt_id']}__seed{entry['seed']}__{entry['arm']}.png"
        entry["clip_bf16"] = round(clip_scores.get(str(base_file), float("nan")), 3)
        entry["clip_nvfp4"] = round(clip_scores.get(str(cand_file), float("nan")), 3)
        entry["clip_delta"] = round(entry["clip_nvfp4"] - entry["clip_bf16"], 3)

    # Identical pairs give PSNR inf and have to be excluded from the median, but
    # they must still be counted. If every pair is identical the arm produced the
    # same images as the baseline -- quantization was a no-op, or both arms read
    # the same directory -- and dropping them all silently leaves a summary of
    # median: null, below_30db: 0, which reads as a clean pass.
    identical = [e for e in per_pair if e["psnr_db"] == float("inf")]
    psnrs = [e["psnr_db"] for e in per_pair if e["psnr_db"] != float("inf")]
    clip_deltas = [e["clip_delta"] for e in per_pair]

    by_category: dict[str, list[float]] = defaultdict(list)
    for entry in per_pair:
        by_category[entry.get("category") or "uncategorised"].append(entry["clip_delta"])

    summary = {
        "pairs": len(per_pair),
        "psnr_db": {
            "median": round(float(np.median(psnrs)), 2) if psnrs else None,
            "min": round(min(psnrs), 2) if psnrs else None,
            "below_30db": sum(1 for p in psnrs if p < 30),
            "identical_pairs": len(identical),
        },
        "clip_delta": {
            "mean": round(float(np.mean(clip_deltas)), 3),
            "worst": round(min(clip_deltas), 3),
        },
        "clip_delta_by_category": {
            category: round(float(np.mean(values)), 3) for category, values in sorted(by_category.items())
        },
        "cmmd": _cmmd(image_dir, records, workspace.root),
        "notes": [
            "PSNR below 30 dB is a signal, not a failure. Values well below it routinely\n"
            "pass human review on generative output.",
            "CLIP delta is the text-alignment proxy. A well-formed image with the wrong content is a failure no perceptual score will catch.",
            "Human review is the actual sign-off. None of these replace it.",
        ],
    }

    if identical and len(identical) == len(per_pair):
        summary["notes"].insert(
            0,
            f"WARNING: all {len(per_pair)} pairs are byte-identical. The two arms "
            "produced the same images, so either quantization was a no-op or both "
            "arms read the same directory. Do not report this as a quality result.",
        )
    elif identical:
        summary["notes"].insert(
            0,
            f"{len(identical)} of {len(per_pair)} pairs are byte-identical and are "
            "excluded from the PSNR median.",
        )

    # Small samples produce category numbers that look meaningful and are not.
    per_category = min(len(v) for v in by_category.values()) if by_category else 0
    if per_category < 5:
        summary["notes"].insert(
            0,
            f"Only {per_category} pair(s) per category. Treat the category breakdown as "
            "noise, not signal. Scale to a few dozen before drawing conclusions.",
        )

    # allow_nan=False rather than the default. Python emits NaN and Infinity as
    # bare literals, which no strict JSON parser accepts -- so a single
    # non-finite PSNR (identical images give infinite PSNR, and it does happen)
    # produced a quality.json that the notebook and make_figures both refused to
    # load. _finite() maps them to null, which every reader already handles
    # because a missing metric is an expected state here.
    payload = json.dumps(
        _finite({"summary": summary, "pairs": per_pair, "scored": str(image_dir)}),
        indent=2,
        allow_nan=False,
    ) + "\n"

    # Two copies on purpose. `quality.json` is what the notebook, make_figures and
    # anything else already looks for, so it always holds the most recent run.
    # The named copy is what stops a second scoring run destroying the first: we
    # lost a set of numbers that way and had to regenerate them.
    out_path = workspace.results / "quality.json"
    out_path.write_text(payload)

    label = image_dir.name if image_dir.name != "dynamic" else "dynamic"
    archived = workspace.results / f"quality-{label}.json"
    archived.write_text(payload)

    cmmd = summary["cmmd"]
    print(f"  {summary['pairs']} pairs scored")
    if cmmd.get("value") is not None:
        print(f"  CMMD {cmmd['value']}  (lower is better)")
    else:
        print(f"  CMMD not computed: {cmmd.get('reason')}")
    print(f"  PSNR median {summary['psnr_db']['median']} dB, {summary['psnr_db']['below_30db']} below 30 dB")
    print(f"  CLIP delta mean {summary['clip_delta']['mean']}, worst {summary['clip_delta']['worst']}")
    for category, delta in summary["clip_delta_by_category"].items():
        print(f"    {category:<24} {delta:+.3f}")
    for note in summary["notes"]:
        print(f"  note: {note}")
    print(f"  written to {out_path}")
    print(f"  archived as {archived.name}, which the next run will not overwrite")

    return {"report": str(out_path), "archived": str(archived), "summary": summary}
