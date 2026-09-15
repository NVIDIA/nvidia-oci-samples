# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run provenance.

Everything this harness produces is meant to be handed to someone else and
re-run. That only works if we record exactly what produced it: GPU, driver,
container, library versions, model revisions, and the commands as executed.

Each stage appends to ``results/run_manifest.json``. Nothing here needs a GPU,
so the manifest still gets written when a stage fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "run_manifest.json"

# Recorded because a mismatch between these and the reference report is the
# first thing to check when numbers disagree.
TRACKED_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "nvidia-modelopt",
    "tensorrt_llm",
    "safetensors",
    "accelerate",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(cmd: list[str]) -> str | None:
    """Run a command and return its stdout, or None if anything went wrong.

    Every failure collapses to None on purpose -- a missing binary, a non-zero
    exit, a hang past the timeout. This only feeds environment capture, and a
    field the manifest cannot fill is worth far less than a preflight that dies
    because ``nvidia-smi`` was absent.
    """
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def _package_versions() -> dict[str, str | None]:
    """Versions of the packages whose behaviour changes the result.

    Read from installed metadata rather than by importing. Importing to test
    availability is unsafe here: ``tensorrt_llm`` calls MPI_Init at import time,
    which aborts the process when Open MPI was not built against Slurm's PMI.
    A package that is absent records None rather than being omitted, so a reader
    can tell "not installed" from "never checked".
    """
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python < 3.8 only
        return {}
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def gpu_info() -> dict[str, Any]:
    """GPU description without importing torch.

    Uses nvidia-smi so preflight can report something useful even when the
    Python environment is broken.
    """
    query = "name,driver_version,memory.total,compute_cap,uuid"
    raw = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    if not raw:
        return {"available": False}

    devices = []
    for line in raw.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 5:
            continue
        devices.append(
            {
                "name": fields[0],
                "driver": fields[1],
                "memory_total": fields[2],
                "compute_capability": fields[3],
                "uuid": fields[4],
            }
        )

    caps = {d["compute_capability"] for d in devices}
    return {
        "available": bool(devices),
        "count": len(devices),
        "devices": devices,
        # sm_100 is B200, sm_103 is GB300. Serving behaviour differs between
        # them, so a result from one is not evidence for the other.
        "architecture": sorted(caps),
    }


def environment() -> dict[str, Any]:
    """Everything needed to reproduce or diagnose a run."""
    return {
        "captured_at": _utc_now(),
        "hostname": platform.node(),
        "cpu_arch": platform.machine(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu_info(),
        "packages": _package_versions(),
        "container_image": os.environ.get("NVIDIA_PRODUCT_NAME")
        or os.environ.get("SINGULARITY_CONTAINER")
        or os.environ.get("CONTAINER_IMAGE"),
        "slurm": {
            key.lower(): os.environ[key]
            for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION")
            if key in os.environ
        }
        or None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def file_digest(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file. Used for artefacts small enough to be worth hashing."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class Manifest:
    """Append-only record of a run, persisted after every stage."""

    def __init__(self, results_dir: Path):
        """Open, or start, the manifest for a workspace.

        Loads whatever is already there rather than truncating, so a run resumed
        after an allocation ended keeps the record of the stages that succeeded
        before it. The environment is captured once, when the manifest is first
        created, and never refreshed -- it describes the machine the run started
        on, which is the thing a later reader needs.
        """
        self.path = results_dir / MANIFEST_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load the existing manifest, or start a fresh one.

        A file that cannot be parsed is renamed to ``.json.corrupt`` rather than
        deleted, so the run continues and the wreckage is still there to look at.

        Valid JSON is not necessarily a manifest, so the keys the rest of the class
        indexes are restored before returning. Without that a file holding ``{}`` --
        or a manifest truncated to its header -- parses cleanly and then fails with a
        bare KeyError inside ``record``, after a stage has already done its work.
        """
        fresh = {"created_at": _utc_now(), "environment": environment(), "stages": []}
        if not self.path.exists():
            return fresh

        try:
            loaded = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            # A corrupt manifest should not block a run. Keep the old file
            # so it can be inspected, and start a fresh one.
            self.path.rename(self.path.with_suffix(".json.corrupt"))
            return fresh

        # Valid JSON is not necessarily a manifest. A file holding `{}`, a list,
        # or a manifest truncated to its header parses cleanly and then fails
        # much later with a bare KeyError inside record() -- after a stage has
        # already done its work. Restore the keys the rest of the class assumes.
        if not isinstance(loaded, dict):
            self.path.rename(self.path.with_suffix(".json.corrupt"))
            return fresh
        for key, default in fresh.items():
            loaded.setdefault(key, default)
        if not isinstance(loaded["stages"], list):
            loaded["stages"] = []

        # Present is not the same as usable. The keys above can hold anything
        # JSON can express, and a resumed run indexes into both of them --
        # ``stage_status`` reads ``entry["stage"]`` and the validator reads
        # ``environment["gpu"]``. Preserving a string where a dict belongs turns
        # a readable manifest into an AttributeError several stages later.
        if not isinstance(loaded["environment"], dict):
            loaded["environment"] = environment()
        # Malformed entries are dropped rather than raised on. A manifest that is
        # partly readable is still worth resuming from, and the alternative --
        # refusing to load -- costs the record of every stage that did succeed.
        # The field types matter as much as their presence. ``stage`` is used as
        # a dict key by the validator, so a list there is unhashable; ``status``
        # is read positionally by ``stage_status``, so an entry naming a stage
        # and carrying no status raises KeyError on the next resume.
        loaded["stages"] = [
            entry
            for entry in loaded["stages"]
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("stage"), str)
                and isinstance(entry.get("status"), str)
            )
        ]
        return loaded

    def record(
        self,
        stage: str,
        *,
        status: str,
        command: list[str] | None = None,
        outputs: dict[str, Any] | None = None,
        notes: str | None = None,
        duration_s: float | None = None,
    ) -> None:
        """Append one stage outcome and persist immediately.

        Appends rather than replaces, so a re-run leaves the earlier attempt in
        place and the manifest reads as a history rather than a snapshot. Saving
        on every call is deliberate: the point of the file is to survive the run
        that was writing it, and a stage that fails after an hour must still have
        left its record behind.
        """
        self.data["stages"].append(
            {
                "stage": stage,
                "status": status,
                "at": _utc_now(),
                "duration_s": round(duration_s, 1) if duration_s is not None else None,
                "command": command,
                "outputs": outputs or {},
                "notes": notes,
            }
        )
        self.save()

    def save(self) -> None:
        """Write the manifest atomically.

        Written to a temporary file in the same directory and then renamed.
        ``os.replace`` is atomic within a filesystem, so a reader sees either
        the old manifest or the new one and never a half-written file. A direct
        write that is interrupted -- an allocation ending, a node going away --
        truncates the only record of an export that took twelve minutes.
        """
        self.data["updated_at"] = _utc_now()
        payload = json.dumps(self.data, indent=2, sort_keys=False) + "\n"

        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, self.path)

    def stage_status(self, stage: str) -> str | None:
        """Status of the most recent attempt at a stage, or None if never run.

        Reversed, because ``record`` appends: a stage that failed and was then
        re-run successfully must report the success. Iterating forwards would
        make ``--force`` look like it had never worked.
        """
        for entry in reversed(self.data["stages"]):
            if entry["stage"] == stage:
                return entry["status"]
        return None
