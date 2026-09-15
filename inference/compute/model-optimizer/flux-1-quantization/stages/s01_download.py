# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Download the BF16 baseline and the published NVFP4 reference.

Two checkpoints, for two different purposes:

* ``baseline_repo`` is the BF16 model. Everything downstream starts here.
* ``reference_repo`` is Black Forest Labs' published NVFP4 checkpoint. It is in
  ComfyUI format and will not load into TRT-LLM VisualGen, so we use it only to
  compare tensor names, shapes, dtypes and scales against our own export.

Resolved revisions are pinned into the manifest. A floating ``main`` makes a run
unreproducible a month later.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from common import paths


def _snapshot(repo: str, target: Path, *, allow_patterns=None) -> dict[str, Any]:
    """Download a repo into a real directory rather than the shared HF cache.

    ``local_dir`` gives actual files at a predictable path, which is what the
    later stages and every documentation example expect. ``allow_patterns`` exists
    for the published reference, where only the tensors are needed and the rest of
    the repo is a few more gigabytes for nothing.
    """
    from huggingface_hub import snapshot_download

    print(f"  downloading {repo}")
    local = snapshot_download(
        repo_id=repo,
        local_dir=target,
        allow_patterns=allow_patterns,
        max_workers=8,
    )
    return {"repo": repo, "path": str(local)}


def _resolved_revision(repo: str) -> str | None:
    """Resolve a repo to the commit actually downloaded, or None.

    Best-effort by design. Pinning the revision is what makes a run reproducible
    a month later, but failing to look it up is not a reason to fail a download
    that has already succeeded.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo).sha
    except Exception:
        return None


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".ckpt")

# Components that legitimately carry no weight files. Everything else named in
# model_index.json must have weights, or the pull was interrupted. Exempting by
# name rather than by "the directory has something in it" matters: an
# interrupted transfer leaves config.json behind long before the weights land,
# so a non-emptiness test passes exactly the case this function exists to catch.
WEIGHTLESS_COMPONENTS = frozenset(
    {"tokenizer", "tokenizer_2", "tokenizer_3", "scheduler", "feature_extractor"}
)


def _is_complete_checkpoint(path: Path) -> tuple[bool, str]:
    """Is this a usable Diffusers checkpoint, or the remains of a failed pull?

    A 34 GB download inside a time-limited allocation gets interrupted often,
    and an interrupted pull leaves a directory that is non-empty but useless:
    metadata and empty component folders, no weights. Treating that as
    "already downloaded" wastes the next run, so check properly.

    Reads model_index.json and confirms every component it names has weights.
    """
    resolved = path.resolve()
    index = resolved / "model_index.json"
    if not index.exists():
        return False, "no model_index.json"

    try:
        manifest = json.loads(index.read_text())
    except json.JSONDecodeError:
        return False, "model_index.json is unreadable"

    missing: list[str] = []
    for component, spec in manifest.items():
        # Component entries look like ["diffusers", "FluxTransformer2DModel"].
        # Anything else is a scalar config value, not a subdirectory.
        if component.startswith("_") or not isinstance(spec, (list, tuple)):
            continue
        directory = resolved / component
        if not directory.is_dir():
            missing.append(component)
            continue
        if component in WEIGHTLESS_COMPONENTS:
            continue
        if not any(f.suffix in WEIGHT_SUFFIXES for f in directory.iterdir() if f.is_file()):
            missing.append(f"{component} (no weights)")

    if missing:
        return False, f"incomplete: missing {', '.join(sorted(missing))}"
    return True, "complete"


def _directory_size_gb(path: Path) -> float:
    """Size on disk, following a symlinked model directory.

    Shared checkpoints are linked rather than copied, and rglob does not
    descend into a symlink, so resolve first or a linked model reports 0 GB.
    """
    resolved = path.resolve()
    if not resolved.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
    return total / (1024**3)


def run(*, workspace, config_path: Path, manifest, args) -> dict[str, Any]:
    """Fetch the BF16 baseline, and the published NVFP4 reference if configured.

    Populates ``models/`` under the workspace -- or symlinks a copy someone has
    already staged -- and records each repo with its resolved revision and size,
    so a run stays reproducible once ``main`` has moved on. Needs a Hugging Face
    token and nothing else.

    An interrupted pull is discarded rather than resumed into. A partial
    checkpoint reads as present, so every later stage would either skip it or
    load half a model, and completeness is therefore re-checked after fetching:
    a failure here is a failure, not a directory that merely exists.
    """
    config = json.loads(config_path.read_text())

    # Keep the Hugging Face cache inside the workspace. Home directories on
    # shared clusters are far too small for a 34 GB model.
    # Also set in quantize.py before any stage imports huggingface_hub, which is
    # the one that takes effect on a normal run. Kept here so this stage is still
    # correct if it is ever driven directly.
    os.environ.setdefault("HF_HOME", str(workspace.hf_home))

    outputs: dict[str, Any] = {"hf_home": str(workspace.hf_home), "downloads": []}

    baseline_dir = workspace.models / config["baseline_dir"]
    if getattr(args, "dry_run", False):
        print(f"  would download {config['baseline_repo']} -> {baseline_dir}")
        if config.get("reference_repo"):
            print(f"  would download {config['reference_repo']}")
        return outputs

    # is_symlink() as well as exists(). exists() follows the link, so a dangling
    # symlink -- a shared cache that moved, a scratch volume unmounted -- reads
    # as "absent", the cleanup branch is skipped, and the later symlink_to()
    # fails with FileExistsError on a path we were told does not exist.
    present = baseline_dir.is_symlink() or baseline_dir.exists()

    complete, reason = (False, "absent")
    if baseline_dir.is_symlink() and not baseline_dir.exists():
        complete, reason = (False, "dangling symlink")
    elif present:
        complete, reason = _is_complete_checkpoint(baseline_dir)

    if complete:
        print(f"  baseline already present at {baseline_dir}")
    else:
        if present:
            # Clear the wreckage of an interrupted pull. Leaving it means every
            # later run either skips a broken checkpoint or resumes into it.
            print(f"  discarding unusable baseline at {baseline_dir} ({reason})")
            if baseline_dir.is_symlink():
                baseline_dir.unlink()
            else:
                shutil.rmtree(baseline_dir)

        if (cached := paths.find_cached_model(config["baseline_repo"])) is not None:
            # Symlink rather than copy. Saves the download and the disk.
            print(f"  found a staged copy at {cached}, linking instead of downloading")
            baseline_dir.parent.mkdir(parents=True, exist_ok=True)
            baseline_dir.symlink_to(cached)
            outputs["used_shared_cache"] = str(cached)
        else:
            _snapshot(config["baseline_repo"], baseline_dir)

        ok, detail = _is_complete_checkpoint(baseline_dir)
        if not ok:
            raise RuntimeError(
                f"Baseline at {baseline_dir} is still unusable after fetching ({detail}). "
                "Re-run this stage; downloads resume rather than starting over."
            )

    outputs["downloads"].append(
        {
            "repo": config["baseline_repo"],
            "revision": _resolved_revision(config["baseline_repo"]),
            "path": str(baseline_dir),
            "size_gb": round(_directory_size_gb(baseline_dir), 1),
            "role": "bf16 baseline",
        }
    )

    reference_repo = config.get("reference_repo")
    if reference_repo:
        reference_dir = workspace.models / config["reference_dir"]
        # A non-empty directory is not a usable checkpoint. An interrupted
        # download leaves .incomplete files behind, which satisfied the old test
        # and then failed much later in the schema diff with no tensors to read.
        # The reference is fetched with allow_patterns, so it holds no config
        # files and cannot use _is_complete_checkpoint -- what it needs is at
        # least one readable safetensors file.
        if reference_dir.exists() and any(reference_dir.rglob("*.safetensors")):
            print(f"  reference already present at {reference_dir}")
        else:
            # Only the tensors are needed for the schema diff, not the whole repo.
            _snapshot(reference_repo, reference_dir, allow_patterns=config.get("reference_patterns"))
            if not any(reference_dir.rglob("*.safetensors")):
                raise RuntimeError(
                    f"Reference at {reference_dir} holds no .safetensors after fetching. "
                    "Check reference_patterns in the config."
                )
        outputs["downloads"].append(
            {
                "repo": reference_repo,
                "revision": _resolved_revision(reference_repo),
                "path": str(reference_dir),
                "size_gb": round(_directory_size_gb(reference_dir), 1),
                "role": "published nvfp4 reference (schema comparison only)",
                "note": "ComfyUI format; does not load into TRT-LLM VisualGen",
            }
        )

    for entry in outputs["downloads"]:
        print(f"  {entry['role']}: {entry['size_gb']} GB at {entry['path']}")

    return outputs
