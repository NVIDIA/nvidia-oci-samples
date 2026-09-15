#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Report which documented consumers of the packed export are actually available.

Model Optimizer writes ``--hf-ckpt-dir`` for **SGLang, vLLM and TRT-LLM**. A stock
Diffusers pipeline is not on that list and rejects the checkpoint, which is
expected rather than a defect -- but it is easy to stop there and conclude the
checkpoint does not serve. It only means the wrong loader was used.

This script does no GPU work and loads no weights. It answers three questions:

1. Which of the three runtimes can be imported here at all?
2. Does the installed TensorRT-LLM ship anything diffusion-shaped -- a VisualGen
   module, a FLUX model definition, an example config?
3. Does the export on disk look like something a serving stack would accept?

Run it before writing any loader code, because the answers decide what the loader
should even call. Guessing at an API and reporting the resulting ImportError as a
finding wastes everyone's time, including the engineering team's.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path

RUNTIMES = ("tensorrt_llm", "vllm", "sglang")
INTERESTING = ("flux", "visual_gen", "visualgen", "diffus")


def probe_import(name: str) -> dict[str, object]:
    """Import without raising, and report version and location if it worked."""
    if importlib.util.find_spec(name) is None:
        return {"name": name, "available": False, "reason": "not installed"}
    try:
        module = importlib.import_module(name)
    except Exception as error:  # noqa: BLE001 - an import that explodes is the finding
        return {"name": name, "available": False, "reason": f"{type(error).__name__}: {error}"}
    return {
        "name": name,
        "available": True,
        "version": getattr(module, "__version__", "unknown"),
        "path": getattr(module, "__file__", None),
    }


def find_diffusion_surface(package: str, limit: int = 40) -> list[str]:
    """Walk an installed package for modules whose names suggest image generation.

    Cheaper and more honest than guessing import paths: if TensorRT-LLM ships a
    VisualGen entry point, its filename almost certainly contains one of the
    substrings we look for. If nothing matches, that is a real answer too.
    """
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.origin:
        return []
    root = Path(spec.origin).parent
    hits: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for filename in files:
            if not filename.endswith((".py", ".yaml", ".yml")):
                continue
            full = Path(dirpath) / filename
            haystack = str(full.relative_to(root)).lower()
            if any(token in haystack for token in INTERESTING):
                hits.append(str(full.relative_to(root)))
                if len(hits) >= limit:
                    return sorted(hits)
    return sorted(hits)


SKIP_DIRS = {"site-packages", "dist-packages", "node_modules", ".git", "__pycache__"}


def find_example_configs(limit: int = 20, max_depth: int = 6) -> list[str]:
    """Look for shipped example configs outside the package tree.

    Container images usually drop ``examples/`` somewhere on the filesystem rather
    than inside site-packages, so the package walk alone will miss them.

    Bounded deliberately. An unbounded ``rglob`` over ``/usr/local`` crawls every
    installed package and takes minutes in a container this size, which turns a
    ten-second probe into something you interrupt before it answers anything.
    Depth-limited, and skipping the package directories the previous walk already
    covered, keeps it to a couple of seconds.
    """
    roots = [Path("/app"), Path("/workspace"), Path("/opt"), Path("/usr/local"), Path("/examples")]
    hits: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [
                d
                for d in dirs
                if d not in SKIP_DIRS
                and len(Path(dirpath, d).parts) - base_depth <= max_depth
            ]
            for filename in files:
                if not filename.endswith((".yaml", ".yml")):
                    continue
                lowered = filename.lower()
                if "flux" in lowered or "visual" in lowered:
                    hits.append(str(Path(dirpath) / filename))
                    if len(hits) >= limit:
                        return sorted(hits)
    return sorted(hits)


def describe_export(hf_dir: Path) -> dict[str, object]:
    """Summarise the packed export without loading it.

    A serving stack reads ``model_index.json`` and the transformer's config to
    decide how to build the model, so those two files are what a loader will see
    first. ``hf_quant_config.json`` is what tells it the checkpoint is NVFP4 at
    all -- its absence would explain a rejection immediately.
    """
    if not hf_dir.is_dir():
        return {"path": str(hf_dir), "exists": False}

    report: dict[str, object] = {
        "path": str(hf_dir),
        "exists": True,
        "top_level": sorted(p.name for p in hf_dir.iterdir())[:20],
    }
    for candidate in ("model_index.json", "hf_quant_config.json"):
        found = next(hf_dir.rglob(candidate), None)
        if found:
            try:
                report[candidate] = json.loads(found.read_text())
            except Exception as error:  # noqa: BLE001
                report[candidate] = f"unreadable: {error}"
        else:
            report[candidate] = None

    transformer_config = hf_dir / "transformer" / "config.json"
    if transformer_config.is_file():
        try:
            config = json.loads(transformer_config.read_text())
            report["transformer_config_keys"] = sorted(config)[:25]
            for key in ("quantization_config", "quant_config", "_class_name"):
                if key in config:
                    report[f"transformer.{key}"] = config[key]
        except Exception as error:  # noqa: BLE001
            report["transformer_config_keys"] = f"unreadable: {error}"
    return report


def main() -> int:
    """Report which documented consumers of the packed export are importable here.

    Loads no weights and touches no GPU, so it is safe on a login node. Exists to
    separate "this checkpoint does not serve" from "the runtime that reads it is
    not installed in this container" -- two conclusions that look identical from a
    failed load and have entirely different remedies.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-ckpt-dir", type=Path, help="The packed --hf-ckpt-dir export")
    args = parser.parse_args()

    print("=" * 72)
    print("Documented consumers of --hf-ckpt-dir: SGLang, vLLM, TRT-LLM")
    print("=" * 72)
    available = []
    for name in RUNTIMES:
        result = probe_import(name)
        if result["available"]:
            available.append(name)
            print(f"  [yes] {name:<16} {result['version']}")
            print(f"        {result['path']}")
        else:
            print(f"  [no ] {name:<16} {result['reason']}")

    if not available:
        print("\n  None of the three are importable here. Nothing can be concluded")
        print("  about the packed export from this environment.")

    for name in available:
        print(f"\n--- diffusion surface inside {name} ---")
        hits = find_diffusion_surface(name)
        if hits:
            for hit in hits:
                print(f"  {hit}")
        else:
            print(f"  no flux/visual_gen/diffusion modules found in {name}")
            print("  -> this runtime probably cannot serve a diffusion checkpoint at all")

    print("\n--- shipped example configs on the filesystem ---")
    configs = find_example_configs()
    for config in configs:
        print(f"  {config}")
    if not configs:
        print("  none found under /app /workspace /opt /usr/local /examples")

    if args.hf_ckpt_dir:
        print("\n--- the packed export, as a loader would first see it ---")
        print(json.dumps(describe_export(args.hf_ckpt_dir), indent=2)[:4000])

    print("\nNext: whichever runtime shows a diffusion surface is the one to write")
    print("a loader against. If none do, that is itself the answer to take back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
