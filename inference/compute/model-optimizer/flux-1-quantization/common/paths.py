# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Workspace resolution.

Checkpoints are large. FLUX.1-dev BF16 alone is roughly 34 GB, and a full run
with both arms plus exports needs 250 GB or more. Home directories on shared
clusters are far too small, so every stage writes under a single workspace root
chosen here.

Resolution order:

1. ``--workspace`` on the command line.
2. ``FLUX_QUANT_WORKSPACE`` in the environment.
3. Auto-detection: persistent scratch first, then node-local RAID.

Node-local RAID is fast but usually disappears when the allocation ends. When
the workspace looks node-local we say so loudly rather than letting someone
discover it after a four-hour run.
"""

from __future__ import annotations

import getpass
import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Enough for one model arm end to end. A full two-arm run wants ~250 GB.
MIN_FREE_GB = 120
RECOMMENDED_FREE_GB = 250

# Conventional locations for persistent per-user scratch on shared clusters,
# searched in order, first hit wins. These are only a convenience: set
# FLUX_QUANT_WORKSPACE (or pass --workspace) and none of this runs. Add your
# site's convention here if it is not listed.
SCRATCH_GLOBS = (
    "/home/scratch.{user}*",
    "/lustre/fsw/*/{user}",
    "/lustre/{user}",
    "/mnt/shared/*/{user}",
    "/scratch/{user}",
    "/home/{user}/scratch",
)

# Directories whose immediate children are shared team volumes. Used when
# looking for a `models/` directory beside the per-person workspaces, and kept
# next to SCRATCH_GLOBS because the two have to agree: a layout listed above but
# missing here autodetects a workspace and then fails to find its checkpoints.
SHARED_VOLUME_PARENTS = frozenset({"mnt", "shared", "lustre", "fsw", "scratch"})

# Node-local, wiped when the allocation ends. /scratch/local is often the
# largest of these -- around 750 GB has been seen, against far less on /raid --
# so it is searched first.
EPHEMERAL_ROOTS = ("/scratch/local", "/raid", "/scratch.local", "/local", "/tmp")


def export_dir_name(config: dict) -> str:
    """Directory under ``exports/`` for this configuration.

    Normally the Model Optimizer model name, which is also what selects the
    layer-exclusion filter. They are separable on purpose: to measure what a
    filter is worth you need to quantize the *same weights* under a *different*
    filter, and if the two names were tied together the second run would
    overwrite the first.
    """
    return config.get("export_dir") or config["modelopt_model"]

# Read-only caches someone else may have already staged models into. Checking
# before downloading can save tens of gigabytes that are on the box already.
# Site-specific by nature, so it is configured rather than hardcoded: set
# FLUX_QUANT_SHARED_CACHES to a colon-separated list of directories.
SHARED_MODEL_CACHES = tuple(
    p for p in os.environ.get("FLUX_QUANT_SHARED_CACHES", "").split(":") if p
)


@dataclass(frozen=True)
class Workspace:
    """A resolved workspace root and what we know about it.

    Outputs are per-person, because the stages write to fixed paths and two
    people running at once would overwrite each other. Checkpoints are not:
    FLUX.1-dev is 34 GB and there is no reason for a team to hold one copy
    each, so ``models`` can point at a shared directory on the same volume.
    """

    root: Path
    source: str
    ephemeral: bool
    free_gb: float
    models_override: Path | None = None
    models_source: str = "workspace"

    @property
    def models(self) -> Path:
        """Where checkpoints live, which is not necessarily under the workspace.

        The only one of these that can point outside the workspace root. A 34 GB
        baseline is worth sharing across a team; outputs are not, because the
        stages write to fixed paths and two people would overwrite each other.
        """
        return self.models_override or (self.root / "models")

    @property
    def exports(self) -> Path:
        """Quantized checkpoints, one directory per configuration."""
        return self.root / "exports"

    @property
    def images(self) -> Path:
        """Generated images, one subdirectory per arm or per export."""
        return self.root / "images"

    @property
    def results(self) -> Path:
        """Manifests, reports and scores -- everything a reader needs afterwards."""
        return self.root / "results"

    @property
    def hf_home(self) -> Path:
        """Hugging Face cache, kept off the home directory it will not fit in."""
        return self.root / "hf"

    def directories(self) -> tuple[Path, ...]:
        """Every directory a run writes to, for ``ensure`` and for reporting."""
        return (self.models, self.exports, self.images, self.results, self.hf_home)

    def ensure(self) -> None:
        """Create the standard subdirectories. Safe to call repeatedly."""
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)

    def warnings(self) -> list[str]:
        """Things worth saying before a long run, none of them fatal.

        Returned rather than printed so the caller decides where they go -- the
        preflight stage puts them in the manifest as well as on the terminal.
        Both cases cost hours if they are discovered late: node-local storage is
        wiped when the allocation ends, and a volume that is merely large enough
        for one arm fails partway through the second.
        """
        messages: list[str] = []
        if self.ephemeral:
            messages.append(
                f"{self.root} looks node-local. Storage under {', '.join(EPHEMERAL_ROOTS)} "
                "is normally wiped when the allocation ends. Copy exports/ somewhere "
                "persistent before you release the node."
            )
        if self.free_gb < MIN_FREE_GB:
            messages.append(
                f"Only {self.free_gb:.0f} GB free. One model arm needs about "
                f"{MIN_FREE_GB} GB and a full run about {RECOMMENDED_FREE_GB} GB."
            )
        return messages


def _shared_models_root(workspace_root: Path) -> Path | None:
    """Find a team-shared models directory on the same scratch volume.

    Walks up from the workspace looking for a scratch volume root, then checks
    whether someone has created a ``models`` directory beside the per-person
    workspaces. Opting in is a single ``mkdir``; if the directory does not
    exist we leave the per-workspace default alone.

    So this layout works with no configuration:

        /home/scratch.<team>/
            models/              shared checkpoints
            <user>/flux-quant/   per-person outputs
    """
    try:
        resolved = workspace_root.resolve()
    except OSError:
        return None

    # Walk up looking for a shared volume root, then check for a models/ beside
    # the per-person directories. The set of recognised roots has to match
    # SCRATCH_GLOBS above, or a layout the documentation describes will not be
    # detected -- /mnt/shared/<team>/ is exactly that case.
    for parent in resolved.parents:
        if parent.name.startswith("scratch.") or parent.parent.name in SHARED_VOLUME_PARENTS:
            candidate = parent / "models"
            if candidate.is_dir():
                return candidate
        if parent == parent.parent:  # reached the filesystem root
            break
    return None


def find_cached_model(repo: str) -> Path | None:
    """Look for an already-staged copy of a model before downloading it.

    Shared caches on a multi-tenant cluster often already hold checkpoints
    someone else has pulled. A 34 GB download you do not have to repeat is
    worth a two-second check, especially inside a time-boxed allocation.
    """
    leaf = repo.split("/")[-1]
    for cache in SHARED_MODEL_CACHES:
        candidate = Path(cache) / leaf
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def _looks_ephemeral(path: Path) -> bool:
    """Whether a path sits on storage that disappears with the allocation.

    Matched on the resolved path, since scratch is often reached by symlink, and
    with an explicit trailing separator: a bare ``startswith`` would report
    ``/raid2`` as node-local because it begins with ``/raid``.
    """
    resolved = str(path.resolve())
    return any(resolved == root or resolved.startswith(root + "/") for root in EPHEMERAL_ROOTS)


def _autodetect(user: str) -> tuple[Path, str] | None:
    """Guess a workspace root, preferring persistent scratch over node-local disk.

    Order matters more than coverage. Node-local RAID is faster and usually
    larger, but it is wiped when the allocation ends, so it is only offered once
    every persistent candidate has been ruled out. The returned label says which
    kind was found, so the caller can warn rather than silently accept storage
    that will not survive the run.
    """
    # glob.glob rather than manual prefix matching. The hand-rolled version took
    # everything before the first '*' as the base and matched on its trailing
    # segment, so a pattern with a '*' in the middle -- "/lustre/fsw/*/{user}" --
    # resolved to "/lustre/fsw" and silently dropped the {user} segment, handing
    # back the shared volume root instead of a per-person directory.
    for pattern in SCRATCH_GLOBS:
        expanded = pattern.format(user=user)
        matches = sorted(Path(p) for p in glob.glob(expanded))
        for match in matches:
            if match.is_dir() and os.access(match, os.W_OK):
                return match, "autodetected scratch"

    for root in EPHEMERAL_ROOTS:
        candidate = Path(root) / user
        parent = Path(root)
        if parent.is_dir() and os.access(parent, os.W_OK):
            return candidate, f"autodetected node-local {root}"
    return None


def resolve(
    explicit: str | None = None,
    *,
    create: bool = True,
    models_dir: str | None = None,
) -> Workspace:
    """Resolve the workspace root, and where checkpoints live.

    Raises RuntimeError when nothing writable can be found, because every later
    stage depends on this and failing here is much cheaper than failing later.

    Checkpoints resolve separately from outputs: ``models_dir``, then
    ``FLUX_QUANT_MODELS``, then a shared directory on the same scratch volume,
    then ``<workspace>/models``.
    """
    user = getpass.getuser()

    if explicit:
        root, source = Path(explicit).expanduser(), "--workspace"
    elif os.environ.get("FLUX_QUANT_WORKSPACE"):
        root, source = Path(os.environ["FLUX_QUANT_WORKSPACE"]).expanduser(), "FLUX_QUANT_WORKSPACE"
    else:
        detected = _autodetect(user)
        if detected is None:
            raise RuntimeError(
                "No workspace found. Pass --workspace or set FLUX_QUANT_WORKSPACE.\n"
                "This run needs roughly 250 GB on a persistent volume; request "
                "shared scratch from whoever administers the cluster, or point at "
                "node-local storage for a short run you do not need to keep."
            )
        root, source = detected

    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.exists():
        raise RuntimeError(f"Workspace {root} does not exist.")

    if not os.access(root, os.W_OK):
        raise RuntimeError(f"Workspace {root} is not writable by {user}.")

    models_override: Path | None = None
    models_source = "workspace"
    if models_dir:
        models_override, models_source = Path(models_dir).expanduser(), "--models-dir"
    elif os.environ.get("FLUX_QUANT_MODELS"):
        models_override = Path(os.environ["FLUX_QUANT_MODELS"]).expanduser()
        models_source = "FLUX_QUANT_MODELS"
    elif (shared := _shared_models_root(root)) is not None:
        models_override, models_source = shared, "shared volume"

    return Workspace(
        root=root,
        source=source,
        ephemeral=_looks_ephemeral(root),
        free_gb=_free_gb(root),
        models_override=models_override,
        models_source=models_source,
    )
