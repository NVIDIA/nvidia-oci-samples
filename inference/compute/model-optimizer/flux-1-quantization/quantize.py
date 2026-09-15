#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unified entry point for the FLUX.1 NVFP4 quantization stages.

Produces an NVFP4 FLUX.1 checkpoint in Diffusers format using the public
NVIDIA Model Optimizer, confirms it loads and generates, and compares it
against the BF16 baseline.

Run the stages in order:

    python3 quantize.py --list-stages
    python3 quantize.py --stage preflight
    python3 quantize.py --from download --through verify

Every stage appends to ``<workspace>/results/run_manifest.json`` so a run can
be reproduced or diagnosed by someone who was not present for it.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from common import paths  # noqa: E402
from common.manifest import Manifest  # noqa: E402

# Containers verified by the sibling flux-1-benchmarking sample. ModelOpt is
# installed as an overlay; see INSTALL.md.
MODELOPT_CONTAINER = "nvcr.io/nvidia/pytorch:26.07-py3"
TRTLLM_CONTAINER = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22"

SUPPORTED_MODELS = ("flux-dev", "flux-schnell")


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline, and what it needs to run.

    ``container`` and ``needs_gpu`` are recorded rather than enforced. Quantizing
    and serving require different containers -- the Model Optimizer version that
    produces a Diffusers export conflicts with the one TensorRT-LLM pins -- so no
    single environment can run every stage, and the harness says what a stage
    expects instead of pretending it can provide it.
    """

    name: str
    module: str
    summary: str
    container: str | None
    needs_gpu: bool


STAGES: tuple[Stage, ...] = (
    Stage(
        "preflight",
        "stages.s00_preflight",
        "Check GPU, workspace, packages and Hugging Face access before anything expensive",
        None,
        False,
    ),
    Stage(
        "download",
        "stages.s01_download",
        "Fetch the BF16 baseline and the published NVFP4 reference checkpoint",
        None,
        False,
    ),
    Stage(
        "dynamic",
        "stages.s02_dynamic_check",
        "Quick quality read using dynamic NVFP4, no export required",
        TRTLLM_CONTAINER,
        True,
    ),
    Stage(
        "export",
        "stages.s03_static_export",
        "Static PTQ with public ModelOpt, exported to Diffusers format",
        MODELOPT_CONTAINER,
        True,
    ),
    Stage(
        "verify",
        "stages.s04_serve_verify",
        "Load the exported checkpoint and generate images from it",
        TRTLLM_CONTAINER,
        True,
    ),
    Stage(
        "schema",
        "stages.s05_schema_diff",
        "Compare our export against the published checkpoint: names, shapes, dtypes, scales",
        None,
        False,
    ),
    Stage(
        "quality",
        "stages.s06_quality",
        "Score paired BF16 and NVFP4 images: CMMD, PSNR, CLIP",
        None,
        False,
    ),
)

STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


def list_stages() -> None:
    """Print the pipeline order, GPU needs and container per stage.

    Exists so the container split is discoverable without reading the source or
    hitting it as a failure halfway through a run.
    """
    width = max(len(s.name) for s in STAGES)
    print("Stages run in this order:\n")
    for index, stage in enumerate(STAGES, start=1):
        gpu = "GPU" if stage.needs_gpu else "   "
        print(f"  {index}. {stage.name:<{width}}  {gpu}  {stage.summary}")
    print("\nContainers:")
    for stage in STAGES:
        if stage.container:
            print(f"  {stage.name:<{width}}  {stage.container}")
    print("\nStages without a container run anywhere with the harness dependencies.")


def select(args: argparse.Namespace) -> list[Stage]:
    """Resolve the command line into the stages to run, in pipeline order.

    ``--stage`` is a set, not a sequence: the order is always the pipeline's, so
    naming them in the wrong order cannot produce a run that executes an export
    before its download. ``--from``/``--through`` slice inclusively.

    Every rejection here is a ``SystemExit`` with a readable message rather than
    an exception, because the alternative is a traceback for a typo. An inverted
    range is rejected for the same reason -- it slices to nothing and would
    otherwise exit zero having done nothing at all.
    """
    if args.stage:
        unknown = [name for name in args.stage if name not in STAGE_BY_NAME]
        if unknown:
            raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}")
        wanted = set(args.stage)
        return [s for s in STAGES if s.name in wanted]

    names = [s.name for s in STAGES]

    # Validate before indexing. names.index() on a typo raises a bare ValueError
    # traceback, where --stage gives a clean message for the same mistake.
    for flag, value in (("--from", args.from_stage), ("--through", args.through)):
        if value and value not in STAGE_BY_NAME:
            raise SystemExit(f"Unknown stage for {flag}: {value}. Known: {', '.join(names)}")

    start = names.index(args.from_stage) if args.from_stage else 0
    end = names.index(args.through) + 1 if args.through else len(names)

    # `>=` rather than `>`. An inverted range such as `--from export --through
    # dynamic` gives start == end, which slices to [] -- no stages run, nothing
    # printed, exit 0. A run that silently does nothing looks like a run that
    # succeeded.
    if start >= end:
        raise SystemExit(
            f"--from {args.from_stage} comes after --through {args.through}; nothing would run"
        )
    return list(STAGES[start:end])


def build_parser() -> argparse.ArgumentParser:
    """Assemble the command-line interface.

    Separate from ``main`` so the parser can be built and inspected in tests
    without running anything. Help text carries the reasoning behind the defaults
    -- which arm to start with, why checkpoints are shared -- because that is
    where someone reaches when a flag does not do what they expected.
    """
    parser = argparse.ArgumentParser(
        prog="quantize.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-stages", action="store_true", help="Show stages and exit")
    parser.add_argument("--stage", action="append", help="Run only this stage; repeatable")
    parser.add_argument("--from", dest="from_stage", help="First stage to run")
    parser.add_argument("--through", help="Last stage to run")
    parser.add_argument(
        "--model",
        default="flux-dev",
        choices=SUPPORTED_MODELS,
        help="Model arm. flux-dev first: it has a ModelOpt filter entry and a shipped "
        "VisualGen fp4 config. flux-schnell is Apache-2.0 and architecturally identical",
    )
    parser.add_argument("--workspace", help="Workspace root. Defaults to scratch, then node-local RAID")
    parser.add_argument(
        "--models-dir",
        help="Where checkpoints live. Defaults to a shared models/ directory on the same "
        "scratch volume if one exists, otherwise <workspace>/models. Sharing avoids each "
        "person holding their own 34 GB copy",
    )
    parser.add_argument("--config", help="Config JSON. Defaults to configs/<model>.json")
    parser.add_argument("--calib-size", type=int, help="Override calibration set size")
    parser.add_argument("--prompts", type=int, help="Override paired prompt count")
    parser.add_argument(
        "--images",
        help="Quality stage only: directory of paired images to score. Defaults to "
        "<workspace>/images/dynamic. Point it at an images/verify/<export> directory "
        "to score a static arm instead",
    )
    parser.add_argument(
        "--no-exclusions",
        action="store_true",
        help="Dynamic arm only: quantize every linear, ignoring Model Optimizer's "
        "layer exclusions. For measuring what the exclusions are worth, not for "
        "producing a representative result",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what each stage would do")
    parser.add_argument("--force", action="store_true", help="Re-run stages already marked complete")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve the workspace, then run the selected stages in order.

    Returns a process exit code rather than raising: 0 for a clean run, 1 if any
    stage failed, 2 for a problem found before any stage started.

    A failing stage is recorded in the manifest and then stops the run, so the
    file always says which stage broke and what it was doing. Completion is
    tracked per workspace and not per model, which is why running a second arm
    needs ``--force``.
    """
    args = build_parser().parse_args(argv)

    if args.list_stages:
        list_stages()
        return 0

    try:
        workspace = paths.resolve(args.workspace, models_dir=args.models_dir)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        workspace.ensure()
    except OSError as error:
        # A full volume, a read-only mount or a stale NFS handle all land here.
        # Uncaught this printed a traceback that buried the one line worth
        # reading, on the first thing the tool does.
        print(f"error: cannot create the workspace at {workspace.root}: {error}", file=sys.stderr)
        return 2

    # Before any stage imports huggingface_hub. The library reads HF_HOME once,
    # at import time, into module-level constants -- so setting it inside the
    # download stage is too late whenever preflight has already run in the same
    # process, and a 34 GB checkpoint lands in the home directory instead of the
    # workspace. Home directories on shared clusters cannot hold it.
    os.environ.setdefault("HF_HOME", str(workspace.hf_home))

    print(f"Workspace: {workspace.root}  ({workspace.source}, {workspace.free_gb:.0f} GB free)")
    print(f"Models:    {workspace.models}  ({workspace.models_source})")
    for warning in workspace.warnings():
        print(f"warning: {warning}", file=sys.stderr)

    config_path = Path(args.config) if args.config else REPOSITORY_ROOT / "configs" / f"{args.model}.json"
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 2

    manifest = Manifest(workspace.results)
    selected = select(args)

    failures = 0
    for stage in selected:
        if not args.force and manifest.stage_status(stage.name) == "ok":
            print(f"\n[{stage.name}] already complete, skipping. Use --force to re-run.")
            continue

        print(f"\n[{stage.name}] {stage.summary}")
        if stage.container:
            print(f"[{stage.name}] expects container {stage.container}")

        started = time.monotonic()
        try:
            # Imported inside the try because a stage module can fail to import
            # -- a missing optional dependency is the common case. Outside it,
            # the ImportError escaped uncaught and the run ended with no
            # "failed" entry in the manifest, so a later reader saw a stage that
            # had simply never run rather than one that had broken.
            module = importlib.import_module(stage.module)
            outputs = module.run(
                workspace=workspace,
                config_path=config_path,
                manifest=manifest,
                args=args,
            )
        except Exception as error:  # noqa: BLE001 - a failed stage must still be recorded
            manifest.record(
                stage.name,
                status="failed",
                notes=f"{type(error).__name__}: {error}",
                duration_s=time.monotonic() - started,
            )
            print(f"[{stage.name}] failed: {error}", file=sys.stderr)
            failures += 1
            break

        if getattr(args, "dry_run", False):
            # A dry run does no work, so it must not claim any. Recording "ok"
            # here marked every stage complete, and the next real run skipped
            # the lot with "already complete" -- turning a preview into a
            # silently empty pipeline.
            print(f"[{stage.name}] dry run — nothing recorded")
            continue

        manifest.record(
            stage.name,
            status="ok",
            outputs=outputs or {},
            duration_s=time.monotonic() - started,
        )
        print(f"[{stage.name}] ok ({time.monotonic() - started:.0f}s)")

    print(f"\nManifest: {manifest.path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
