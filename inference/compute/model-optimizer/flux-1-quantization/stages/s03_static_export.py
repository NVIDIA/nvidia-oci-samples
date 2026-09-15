# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static post-training quantization with the public Model Optimizer.

This is the deliverable. Everything else supports it.

We deliberately shell out to the public ``examples/diffusers/quantization/quantize.py``
rather than reimplementing the recipe, because the point is to prove someone
can reproduce this with nothing but the public repository. If it does not work
here, it will not work for them, and that gap is itself the finding.

The output that matters is the ``--hf-ckpt-dir`` export. The published NVFP4
checkpoints are in ComfyUI format and will not load into TRT-LLM VisualGen,
which reads Diffusers format only.

Three flags behave differently from what the documentation shows:

* ``--format fp4`` is correct. There is no ``nvfp4`` value; ``fp4`` selects the
  NVFP4 preset, and for FLUX it selects NVFP4 weights with FP8 attention.
* ``--quantized-torch-ckpt-save-path`` is treated as a *directory*. The upstream
  example still passes a name ending in ``.pt``, which silently creates a
  directory of that name.
* ``--quantize-mha`` is required to quantize attention; without it attention
  stays at higher precision.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from common import paths
from typing import Any

# Where the public repository's diffusers example might live.
MODELOPT_SEARCH = (
    "$MODELOPT_DIFFUSERS_DIR",
    "$WORKSPACE/src/Model-Optimizer/examples/diffusers/quantization",
    "/opt/Model-Optimizer/examples/diffusers/quantization",
    "~/Model-Optimizer/examples/diffusers/quantization",
)

CLONE_HINT = """Model Optimizer's diffusers example was not found.

Clone the public repository into the workspace and point at it:

    git clone https://github.com/NVIDIA/Model-Optimizer.git {workspace}/src/Model-Optimizer
    export MODELOPT_DIFFUSERS_DIR={workspace}/src/Model-Optimizer/examples/diffusers/quantization
    python3 -m pip install 'nvidia-modelopt[hf,onnx]'

Using the public repository is the point: anyone adopting this has to run exactly
this, so nothing here substitutes a private configuration for the public one."""


def _find_modelopt(workspace) -> Path | None:
    """Locate a Model-Optimizer diffusers example directory, or None.

    Checked for ``quantize.py`` rather than just the directory, because a partial
    or wrongly-nested clone is the common mistake and it fails much less clearly
    later. None is returned instead of raising so a dry run can still print the
    command it would have run.
    """
    for candidate in MODELOPT_SEARCH:
        expanded = os.path.expandvars(candidate.replace("$WORKSPACE", str(workspace.root)))
        path = Path(expanded).expanduser()
        if path.is_dir() and (path / "quantize.py").exists():
            return path
    return None


def _share_with_group(root: Path) -> list[str]:
    """Make the export readable by the group that owns the volume.

    umask governs what your shell creates, not what a tool creates with an
    explicit mode. Model Optimizer writes the weights at 0600 -- probably via a
    temp file and rename, which does not inherit the umask -- so on a shared
    volume the one file that matters is the one nobody else can read.

    Returns the files it changed, so the run says what it did rather than
    silently altering permissions.
    """
    changed: list[str] = []
    for path in sorted(root.rglob("*")):
        try:
            mode = path.stat().st_mode & 0o777
        except OSError:
            continue
        # Read access only. Group write on an export directory lets any group
        # member replace or unlink the weights.
        wanted = mode | (0o050 if path.is_dir() else 0o040)
        if wanted != mode:
            try:
                path.chmod(wanted)
            except OSError:
                continue
            changed.append(path.name)
    return changed


def _format_command(command: list[str]) -> str:
    """Render for copy and paste: one flag and its value per line."""
    lines: list[str] = [command[0], command[1]]
    index = 2
    while index < len(command):
        token = command[index]
        if token.startswith("--") and index + 1 < len(command) and not command[index + 1].startswith("--"):
            lines.append(f"{token} {command[index + 1]}")
            index += 2
        else:
            lines.append(token)
            index += 1
    return " \\\n      ".join(lines)


def _build_command(
    *,
    quantize_script: Path,
    config: dict[str, Any],
    model_path: Path,
    torch_ckpt_dir: Path,
    hf_ckpt_dir: Path,
    calib_size: int,
) -> list[str]:
    quant = config["quantization"]
    command = [
        sys.executable,
        str(quantize_script),
        "--model",
        config["modelopt_model"],
        "--override-model-path",
        str(model_path),
        "--model-dtype",
        quant.get("model_dtype", "BFloat16"),
        "--format",
        "fp4",
        "--batch-size",
        str(quant.get("batch_size", 2)),
        "--calib-size",
        str(calib_size),
        "--n-steps",
        str(quant.get("calib_steps", 20)),
        "--collect-method",
        quant.get("collect_method", "default"),
        # Directory, not a file. See module docstring.
        "--quantized-torch-ckpt-save-path",
        str(torch_ckpt_dir),
        # The artefact we actually need: Diffusers format, servable by VisualGen.
        "--hf-ckpt-dir",
        str(hf_ckpt_dir),
    ]
    if quant.get("quantize_mha", True):
        command.append("--quantize-mha")
    if quant.get("quant_algo"):
        command += ["--quant-algo", quant["quant_algo"]]
    return command


def materialize_out_channels(hf_ckpt_dir: Path) -> dict[str, Any]:
    """Write ``out_channels`` explicitly into the exported transformer config.

    **Why this is needed.** Every FLUX config in the wild ships
    ``"out_channels": null``. Diffusers' own ``FluxTransformer2DModel`` resolves
    it internally with ``out_channels or in_channels``, so the null is harmless
    there and nobody has ever had to think about it. Model Optimizer copies the
    field through unchanged, which is correct.

    TensorRT-LLM VisualGen's *pre-quantized* loading path reimplements the
    transformer and multiplies the raw value::

        patch_size * patch_size * self.out_channels
        TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'

    Its dynamic path does not hit this, which is why the shipped
    ``flux1-dev-fp4-1gpu.yaml`` example works on stock weights while a real
    static export fails. Confirmed on TensorRT-LLM 1.3.0rc22 against both the
    stock BF16 checkpoint and our export -- identical configs, one loads.

    **Why editing here is safe.** ``out_channels = in_channels`` is exactly what
    Diffusers computes, so this changes nothing about the model. It only spells
    out a value the loader declined to infer.

    **Why it is recorded rather than silent.** This is a deviation from what
    Model Optimizer produced. An undocumented edit inside an export directory is
    how a reproducible artefact stops being reproducible, so the return value
    goes into the run manifest and a ``.orig`` backup stays alongside. Remove
    this function once the upstream fallback lands.

    Idempotent: re-running an export, or calling this twice, is a no-op.
    """
    config_path = hf_ckpt_dir / "transformer" / "config.json"
    if not config_path.is_file():
        return {"edit": "out_channels", "changed": False, "reason": f"no {config_path}"}

    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        return {"edit": "out_channels", "changed": False, "reason": f"unreadable: {error}"}

    if config.get("out_channels") is not None:
        return {
            "edit": "out_channels",
            "changed": False,
            "reason": "already set",
            "out_channels": config["out_channels"],
        }

    in_channels = config.get("in_channels")
    if in_channels is None:
        return {"edit": "out_channels", "changed": False, "reason": "in_channels is also null"}

    backup = config_path.with_suffix(".json.orig")
    if not backup.exists():
        backup.write_text(config_path.read_text())

    config["out_channels"] = in_channels
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    return {
        "edit": "out_channels",
        "changed": True,
        "was": None,
        "out_channels": in_channels,
        "file": str(config_path),
        "backup": str(backup),
        "reason": (
            "TRT-LLM VisualGen's pre-quantized path multiplies out_channels without "
            "the `or in_channels` fallback Diffusers applies, so a null fails to load. "
            "Semantically a no-op; remove once fixed upstream."
        ),
    }


def run(*, workspace, config_path: Path, manifest, args) -> dict[str, Any]:
    """Run Model Optimizer's own quantize.py and produce the deliverable.

    Writes two artefacts with different consumers: ``exports/<name>/hf``, the
    packed Diffusers export that actually serves, and ``exports/<name>/torch``,
    the fake-quantized checkpoint ``mto.restore`` reads. Needs the download stage
    and a Model-Optimizer checkout reachable from ``MODELOPT_DIFFUSERS_DIR``.
    Expect 30 minutes to 2 hours.

    A zero exit code is not success on its own. The stage fails if the Diffusers
    export is missing or empty, because that directory is the only artefact worth
    having and Model Optimizer can exit cleanly without producing it. Two repairs
    are applied afterwards and recorded in the manifest -- group read on weights
    written at 0600, and the ``out_channels`` value TRT-LLM VisualGen needs.
    """
    config = json.loads(config_path.read_text())
    quant = config["quantization"]

    model_path = workspace.models / config["baseline_dir"]
    export_root = workspace.exports / paths.export_dir_name(config)
    torch_ckpt_dir = export_root / "torch"
    hf_ckpt_dir = export_root / "hf"
    calib_size = getattr(args, "calib_size", None) or quant.get("calib_size", 128)

    dry_run = getattr(args, "dry_run", False)
    quantize_script_dir = _find_modelopt(workspace)
    if quantize_script_dir is None:
        if not dry_run:
            raise RuntimeError(CLONE_HINT.format(workspace=workspace.root))
        # A dry run should still show the command, so someone can see what
        # would happen before installing anything.
        quantize_script_dir = Path("$MODELOPT_DIFFUSERS_DIR")
        print("  note: Model Optimizer not found; showing the command with a placeholder path")

    command = _build_command(
        quantize_script=quantize_script_dir / "quantize.py",
        config=config,
        model_path=model_path,
        torch_ckpt_dir=torch_ckpt_dir,
        hf_ckpt_dir=hf_ckpt_dir,
        calib_size=calib_size,
    )

    if config["modelopt_model"] == "flux-schnell":
        print(
            "  note: flux-schnell has no entry in Model Optimizer's filter map, so it "
            "falls back to a generic default. schnell and dev are architecturally "
            "identical, so the dev exclusions should apply, but confirm the layers "
            "actually excluded before trusting the result."
        )

    print("  command:")
    print("    " + _format_command(command))

    if dry_run:
        return {"command": command, "dry_run": True}

    if not model_path.exists():
        raise RuntimeError(f"Baseline model not found at {model_path}. Run the download stage first.")

    export_root.mkdir(parents=True, exist_ok=True)
    log_path = workspace.results / "modelopt_quantize.log"

    print(f"  calibration size {calib_size}, {quant.get('calib_steps', 20)} steps")
    print(f"  log: {log_path}")
    print("  this typically takes 30 minutes to 2 hours")

    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            cwd=quantize_script_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if process.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-25:])
        raise RuntimeError(
            f"Model Optimizer exited {process.returncode}. Last lines:\n{tail}\n\n"
            f"Full log: {log_path}\n"
            "If this is a gap in the public tool rather than our setup, that is a "
            "reportable finding: raise it with the ModelOpt team."
        )

    if not hf_ckpt_dir.exists() or not any(hf_ckpt_dir.iterdir()):
        raise RuntimeError(
            f"Model Optimizer succeeded but produced no Diffusers export at {hf_ckpt_dir}. "
            "That export is the artefact worth having; without it this stage has failed."
        )

    export_size = sum(f.stat().st_size for f in hf_ckpt_dir.rglob("*") if f.is_file()) / (1024**3)
    tensor_files = sorted(p.name for p in hf_ckpt_dir.rglob("*.safetensors"))
    shared = _share_with_group(hf_ckpt_dir)
    if shared:
        print(f"  made {len(shared)} file(s) group-readable; Model Optimizer writes weights at 0600")

    serving_fix = materialize_out_channels(hf_ckpt_dir)
    if serving_fix.get("changed"):
        print(
            f"  set transformer out_channels to {serving_fix['out_channels']} "
            f"(was null) so the export loads in TRT-LLM VisualGen"
        )

    print(f"  Diffusers export: {hf_ckpt_dir}  ({export_size:.1f} GB, {len(tensor_files)} safetensors)")
    if workspace.ephemeral:
        print("  reminder: this workspace is node-local. Copy the export off before releasing the node.")

    return {
        "command": command,
        "hf_export": str(hf_ckpt_dir),
        "torch_checkpoint": str(torch_ckpt_dir),
        "export_size_gb": round(export_size, 2),
        "safetensors_files": tensor_files,
        "calib_size": calib_size,
        "log": str(log_path),
        "modelopt_example_dir": str(quantize_script_dir),
        "post_export_edits": [serving_fix] if serving_fix.get("changed") else [],
    }
