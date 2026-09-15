# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Preflight.

Runs in seconds and catches the things that otherwise waste an allocation:
no GPU, not enough disk, a missing package, or an unaccepted model licence.

FLUX.1 checkpoints are gated on Hugging Face. You must accept the licence in a
browser and hold a token before anything downloads, and that is the failure
people hit most often.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from common.manifest import environment, gpu_info
from common.paths import MIN_FREE_GB, RECOMMENDED_FREE_GB

# Module name -> (distribution name, what it is needed for). Absence is
# reported, not fatal, because stages run in different containers.
#
# We check with find_spec and read versions from package metadata rather than
# importing. Importing to test availability is unsafe: tensorrt_llm runs
# MPI_Init at import time, which aborts the whole process when Open MPI was not
# built against Slurm's PMI. A preflight check must never be able to kill the
# run it is meant to protect.
OPTIONAL_IMPORTS = {
    "torch": ("torch", "every GPU stage"),
    "diffusers": ("diffusers", "download, export, verify"),
    "transformers": ("transformers", "text encoders"),
    "safetensors": ("safetensors", "schema diff"),
    "modelopt": ("nvidia-modelopt", "static export"),
    "tensorrt_llm": ("tensorrt_llm", "dynamic check and verify"),
    "huggingface_hub": ("huggingface_hub", "download"),
}

# Compute capability -> friendly name. Serving behaviour differs between them,
# so a result on one is not evidence for the other.
KNOWN_ARCHITECTURES = {
    "10.0": "B200 (sm_100)",
    "10.3": "GB300 / B300 (sm_103)",
    "9.0": "H100 (sm_90) - not a Blackwell target",
}


def _check_imports() -> dict[str, str | None]:
    """Report which packages are installed, without importing any of them."""
    import importlib.util
    from importlib import metadata

    found: dict[str, str | None] = {}
    for module_name, (distribution, _) in OPTIONAL_IMPORTS.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            found[module_name] = None
            continue
        try:
            found[module_name] = metadata.version(distribution)
        except Exception:
            found[module_name] = "present"
    return found


def _check_hf_access(repos: list[str]) -> dict[str, Any]:
    """Confirm the token exists and the gated repos are actually readable."""
    result: dict[str, Any] = {"token_present": False, "repos": {}}

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    # Resolve the stored-token path the way huggingface_hub does, rather than
    # assuming the default. This harness deliberately relocates HF_HOME into the
    # workspace, so ~/.cache/huggingface/token is the one place the token
    # frequently is *not*. Hardcoding it fails preflight for a user who is in
    # fact logged in.
    if os.environ.get("HF_TOKEN_PATH"):
        token_file = Path(os.environ["HF_TOKEN_PATH"])
    elif os.environ.get("HF_HOME"):
        token_file = Path(os.environ["HF_HOME"]) / "token"
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
        token_file = Path(cache_home) / "huggingface" / "token"
    result["token_present"] = bool(token) or token_file.exists()
    result["token_file"] = str(token_file)

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception:
        result["checked"] = False
        return result

    api = HfApi()
    result["checked"] = True
    for repo in repos:
        try:
            info = api.model_info(repo, token=token)
        except GatedRepoError:
            result["repos"][repo] = "gated - accept the licence at https://huggingface.co/" + repo
        except RepositoryNotFoundError:
            result["repos"][repo] = "not found"
        except Exception as error:  # noqa: BLE001 - network and auth failures are equally informative
            result["repos"][repo] = f"unavailable: {type(error).__name__}"
        else:
            # `or "unknown"` rather than relying on getattr's default. The
            # default only fires when the attribute is *absent*; a present
            # attribute set to None slices to a TypeError. That is the same
            # trap as the upstream out_channels bug this project reported.
            sha = getattr(info, "sha", None) or "unknown"
            result["repos"][repo] = f"ok (revision {sha[:12]})"
    return result


def run(*, workspace, config_path: Path, manifest, args) -> dict[str, Any]:
    """Check the environment and refuse to continue if the run cannot succeed.

    Writes ``results/environment.json`` and returns the GPU architecture, free
    space and any advisories for the manifest. Depends on no earlier stage.

    Problems raise; advisories only print. That distinction is the whole point of
    the stage: an allocation is expensive, so a missing token or a full volume
    should stop the run in the first few seconds rather than four hours into an
    export, while a node-local workspace is worth saying out loud and not worth
    blocking on.
    """
    config = json.loads(config_path.read_text())
    problems: list[str] = []
    advisories: list[str] = []

    gpu = gpu_info()
    if not gpu.get("available"):
        problems.append("No GPU visible. nvidia-smi returned nothing.")
    else:
        for device in gpu["devices"]:
            label = KNOWN_ARCHITECTURES.get(device["compute_capability"], "unrecognised")
            print(f"  GPU: {device['name']}  cc {device['compute_capability']}  {label}")
        if not any(d["compute_capability"].startswith("10.") for d in gpu["devices"]):
            advisories.append(
                "No Blackwell GPU detected. NVFP4 needs Blackwell; results elsewhere are not meaningful."
            )

    print(f"  Workspace: {workspace.root}  {workspace.free_gb:.0f} GB free")
    if workspace.free_gb < MIN_FREE_GB:
        problems.append(
            f"Only {workspace.free_gb:.0f} GB free, need at least {MIN_FREE_GB} GB "
            f"for one arm and about {RECOMMENDED_FREE_GB} GB for a full run."
        )
    if workspace.ephemeral:
        advisories.append(
            "Workspace looks node-local and is probably wiped when the allocation ends. "
            "Copy exports/ somewhere persistent before releasing the node."
        )

    imports = _check_imports()
    for name, (_, purpose) in OPTIONAL_IMPORTS.items():
        version = imports[name]
        state = version if version else "MISSING"
        print(f"  {name:<18} {state:<12} ({purpose})")
    if not imports["torch"]:
        problems.append("torch is missing; no GPU stage can run.")

    repos = [config["baseline_repo"]]
    if config.get("reference_repo"):
        repos.append(config["reference_repo"])
    hf = _check_hf_access(repos)
    if not hf["token_present"]:
        problems.append(
            "No Hugging Face token found. Run `hf auth login` or set HF_TOKEN. "
            "FLUX.1 checkpoints are gated."
        )
    for repo, state in hf.get("repos", {}).items():
        print(f"  HF {repo}: {state}")
        if "gated" in state or "not found" in state:
            problems.append(f"{repo}: {state}")

    if shutil.which("nvidia-smi") is None:
        advisories.append("nvidia-smi not on PATH; GPU details will be incomplete in the manifest.")

    env_path = workspace.results / "environment.json"
    env_path.write_text(json.dumps(environment(), indent=2) + "\n")
    print(f"  Environment written to {env_path}")

    for advisory in advisories:
        print(f"  note: {advisory}")

    if problems:
        raise RuntimeError(
            "Preflight failed:\n  - " + "\n  - ".join(problems)
        )

    return {
        "environment_file": str(env_path),
        "gpu_architecture": gpu.get("architecture"),
        "free_gb": round(workspace.free_gb, 1),
        "ephemeral_workspace": workspace.ephemeral,
        "advisories": advisories,
    }
