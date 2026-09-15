#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate a completed run before anyone quotes it.

Checks that the manifest records what a reader needs to trust or reproduce the
result: which GPU, which library versions, which model revisions, and whether
the stages that matter actually succeeded.

It does not check image quality or whether the recipe is correct. Those need a
human looking at outputs, and this deliberately does not pretend otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_ENVIRONMENT = ("hostname", "cpu_arch", "python", "gpu", "packages")
CRITICAL_PACKAGES = ("torch", "diffusers")
EXPECTED_STAGES = ("preflight", "download", "export", "verify")


def check(manifest_path: Path) -> tuple[list[str], list[str]]:
    """Validate a run manifest, returning errors and warnings separately.

    An error means the manifest cannot support the claims made from it -- no GPU
    recorded, a stage that failed, a package whose version is unknown. A warning
    means the run happened but something about it limits what the numbers mean,
    such as non-Blackwell hardware, where NVFP4 results are not comparable.

    Returns both lists rather than raising, so the caller can print every problem
    at once. Finding one error at a time in a file this size is its own failure.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        return [f"No manifest at {manifest_path}"], []

    # A corrupt manifest is one of the things this tool exists to report, so it
    # should come back as an error alongside the others rather than as a
    # traceback that stops the remaining checks from running at all.
    try:
        data = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return [f"Manifest at {manifest_path} is unreadable: {error}"], []
    if not isinstance(data, dict):
        return [f"Manifest at {manifest_path} is not an object: found {type(data).__name__}"], []

    # Every dereference below is guarded. This is a validator, so a malformed
    # manifest is the input it exists to report on -- crashing on one tells the
    # user less than the file did. Each bad shape becomes an error and the
    # remaining checks still run, so a single invocation surfaces every problem.
    environment = data.get("environment", {})
    if not isinstance(environment, dict):
        errors.append(f"environment is not an object: found {type(environment).__name__}")
        environment = {}

    for field in REQUIRED_ENVIRONMENT:
        if not environment.get(field):
            errors.append(f"environment.{field} is missing")

    gpu = environment.get("gpu", {})
    if not isinstance(gpu, dict):
        errors.append(f"environment.gpu is not an object: found {type(gpu).__name__}")
        gpu = {}
    if not gpu.get("available"):
        errors.append("No GPU recorded. Results cannot be attributed to hardware.")
    else:
        architectures = gpu.get("architecture") or []
        # Blackwell is not one compute capability. Data-centre parts report
        # 10.x -- sm_100 on B200, sm_103 on GB300 -- and consumer Blackwell
        # reports 12.x (sm_120). Matching only "10." warned falsely on hardware
        # that runs NVFP4 perfectly well.
        if not any(str(a).startswith(("10.", "12.")) for a in architectures):
            warnings.append(
                f"GPU architecture {architectures} is not Blackwell. NVFP4 results are "
                "only meaningful on Blackwell."
            )
        if len(architectures) > 1:
            warnings.append(f"Mixed GPU architectures in one run: {architectures}")

    packages = environment.get("packages", {})
    if not isinstance(packages, dict):
        errors.append(f"environment.packages is not an object: found {type(packages).__name__}")
        packages = {}
    for package in CRITICAL_PACKAGES:
        if not packages.get(package):
            errors.append(f"No version recorded for {package}")

    raw_stages = data.get("stages", [])
    if not isinstance(raw_stages, list):
        errors.append(f"stages is not a list: found {type(raw_stages).__name__}")
        raw_stages = []

    # A record with no ``stage`` cannot be attributed to anything, so it is
    # reported and skipped rather than indexed into.
    stages: dict[str, Any] = {}
    for index, entry in enumerate(raw_stages):
        if not isinstance(entry, dict):
            errors.append(f"stages[{index}] is not an object: found {type(entry).__name__}")
        elif "stage" not in entry:
            errors.append(f"stages[{index}] has no 'stage' field")
        elif not isinstance(entry["stage"], str):
            # Used as a dictionary key below, so a list or dict here raises
            # ``unhashable type`` and stops the run before the remaining checks
            # report anything.
            errors.append(
                f"stages[{index}].stage is not a string: found {type(entry['stage']).__name__}"
            )
        else:
            stages[entry["stage"]] = entry

    for name in EXPECTED_STAGES:
        entry = stages.get(name)
        if entry is None:
            warnings.append(f"Stage '{name}' was never run")
        elif entry.get("status") != "ok":
            # .get rather than [] -- an entry naming a stage but carrying no
            # status is malformed, not successful, and saying so beats a KeyError.
            errors.append(f"Stage '{name}' finished with status '{entry.get('status')}'")

    download_entry = stages.get("download") or {}
    download = download_entry.get("outputs") or {}
    if not isinstance(download, dict):
        download = {}
    # Iterated below, so a scalar here raises ``not iterable``. Degrade to an
    # empty list and report it, rather than losing every check that follows.
    records = download.get("downloads", [])
    if not isinstance(records, list):
        errors.append(f"download.downloads is not a list: found {type(records).__name__}")
        records = []
    for record in records:
        if not isinstance(record, dict):
            errors.append(f"A download record is not an object: found {type(record).__name__}")
            continue
        if not record.get("revision"):
            warnings.append(
                f"{record.get('repo')} has no pinned revision. A floating branch makes "
                "this run unreproducible later."
            )

    export = stages.get("export", {}).get("outputs", {})
    if export and not export.get("hf_export"):
        errors.append("Export stage recorded no Diffusers export path")

    preflight = stages.get("preflight", {}).get("outputs", {})
    if preflight.get("ephemeral_workspace"):
        warnings.append(
            "Workspace was node-local. Confirm the export was copied somewhere "
            "persistent before the allocation ended."
        )

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    """Check a manifest and exit non-zero if it holds errors.

    Locates the manifest from the workspace when no path is given, so it can be
    run with no arguments after a run. Warnings go to stdout and errors to
    stderr, which makes this usable as a gate in a script without discarding the
    advisory half.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        help="Path to run_manifest.json. Defaults to $FLUX_QUANT_WORKSPACE/results/run_manifest.json",
    )
    args = parser.parse_args(argv)

    if args.manifest:
        path = Path(args.manifest)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from common import paths

        path = paths.resolve(create=False).results / "run_manifest.json"

    errors, warnings = check(path)

    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} error(s). This run is not ready to quote.", file=sys.stderr)
        return 1
    print(f"\nManifest OK{f', {len(warnings)} warning(s)' if warnings else ''}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
