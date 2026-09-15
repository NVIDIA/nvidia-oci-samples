# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Load the quantized checkpoint and generate from it.

The point of this stage is to produce images from the *static* export so its
quality can be compared against BF16 and against the dynamic arm. Producing a
checkpoint is not the same as producing something usable.

**Use the documented path.** Model Optimizer writes two artefacts and the
`examples/diffusers` README sends them to different places:

* ``--quantized-torch-ckpt-save-path`` — restore into PyTorch with
  ``modelopt.torch.opt.restore()``, on top of the ordinary BF16 pipeline.
* ``--hf-ckpt-dir`` — a Hugging Face checkpoint for SGLang, vLLM or TRT-LLM.

Plain ``FluxPipeline.from_pretrained()`` on the Hugging Face export is on
neither list. It fails, and that failure is not a defect: it is an unsupported
loader. This stage therefore tries ``mto.restore`` first and treats the
Diffusers attempt as a secondary datapoint.

Images use the same prompts, seeds and injected latents as the dynamic stage, so
the two arms are directly comparable and can be scored by the same quality
stage.

**Why the arm is called ``nvfp4-static-sim``.** ``mto.restore`` rebuilds the
quantized module structure and loads the calibrated scales, but the checkpoint
stores weights at full precision — 23.8 GB, the same as BF16. Each forward pass
rounds a weight onto the NVFP4 grid using its calibrated scale, dequantizes, and
performs the matmul in BF16.

The arithmetic is therefore exactly NVFP4 while the execution is not. That makes
it the right instrument for a quality question and the wrong one for a
performance question, and the name says so rather than leaving a reader to infer
that images came off a 4-bit kernel. Calling it ``nvfp4-static`` invites exactly
that misreading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import images, paths

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _generate(pipe, spec, prompts, out_dir: Path, arm: str, torch_mod) -> list[dict[str, Any]]:
    """Generate one image per prompt and seed, recorded exactly as the dynamic stage does.

    Returns metadata records rather than filenames so this stage can write its own
    ``metadata.json``. Without one the quality stage has nothing to pair on and
    silently scores whatever happens to be in ``images/dynamic`` instead — which
    looks like a result and is not.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for prompt in prompts:
        for seed in spec.seeds:
            latents = images.make_latents(spec, seed, device="cuda", dtype=torch_mod.bfloat16)
            digest = images.latent_digest(latents)
            image = pipe(
                prompt=prompt["text"],
                height=spec.height,
                width=spec.width,
                num_inference_steps=spec.steps,
                guidance_scale=spec.guidance_scale,
                max_sequence_length=spec.max_sequence_length,
                latents=latents,
            ).images[0]
            name = images.image_name(prompt["id"], seed, arm)
            image.save(out_dir / name)
            records.append(
                {
                    "prompt_id": prompt["id"],
                    "category": prompt.get("category"),
                    "seed": seed,
                    "arm": arm,
                    "file": name,
                    "latent_sha256_16": digest,
                    "quant_recipe": "modelopt static PTQ, restored with mto.restore",
                }
            )
            print(f"  {arm}: {name}")
    return records


def _generate_baseline(baseline_dir: Path, spec, prompts, out_dir: Path) -> list[dict[str, Any]]:
    """Generate the BF16 arm alongside, from the same weights and latents.

    The quality stage pairs a baseline against a candidate inside one directory.
    Reusing the dynamic stage's BF16 images would work only when both ran with the
    same prompts, seeds and model — an assumption that quietly breaks the moment a
    second model arm or a second exclusion filter is introduced. Forty-eight
    images cost about four minutes, which is a small price for a self-contained
    comparison.
    """
    import torch
    from diffusers import FluxPipeline

    print(f"  generating the BF16 baseline from {baseline_dir.name}")
    pipe = FluxPipeline.from_pretrained(baseline_dir, torch_dtype=torch.bfloat16).to("cuda")
    records = _generate(pipe, spec, prompts, out_dir, "bf16", torch)
    for record in records:
        record["quant_recipe"] = None

    del pipe
    torch.cuda.empty_cache()
    return records


def resolve_torch_checkpoint(path: Path) -> Path:
    """Find the checkpoint file, which may be inside a directory of that name.

    ``--quantized-torch-ckpt-save-path`` is treated as a *directory* by the
    export script even though the upstream example passes a name ending in
    ``.pt``. ``mto.restore`` wants the file, so hand it a directory and it fails
    with ``IsADirectoryError`` rather than anything that explains itself.

    Accept either, and when given a directory take the largest checkpoint-shaped
    file in it — the weights dwarf any metadata sitting alongside them.
    """
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"No checkpoint at {path}")

    candidates = [
        p
        for pattern in ("*.pt", "*.pth", "*.bin", "*.safetensors")
        for p in path.rglob(pattern)
        if p.is_file()
    ]
    if not candidates:
        contents = sorted(p.name for p in path.iterdir())[:10]
        raise FileNotFoundError(
            f"{path} is a directory with no checkpoint file in it. Contains: {contents}"
        )
    return max(candidates, key=lambda p: p.stat().st_size)


def _try_modelopt_restore(
    baseline_dir: Path, torch_ckpt: Path, spec, prompts, out_dir: Path
) -> dict[str, Any]:
    """The documented route: restore the quantized state onto the BF16 pipeline.

    ``mto.restore`` rebuilds the quantizer modules and loads the calibrated
    scales, so the pipeline that comes out is the quantized model rather than a
    BF16 one wearing a quantized name.
    """
    import torch
    import modelopt.torch.opt as mto
    from diffusers import FluxPipeline

    checkpoint = resolve_torch_checkpoint(torch_ckpt)
    size_gb = checkpoint.stat().st_size / 1e9
    print(f"  restoring {checkpoint.name} ({size_gb:.2f} GB) onto {baseline_dir.name}")

    pipe = FluxPipeline.from_pretrained(baseline_dir, torch_dtype=torch.bfloat16)
    mto.restore(pipe.transformer, str(checkpoint))
    pipe.to("cuda")

    records = _generate(pipe, spec, prompts, out_dir, "nvfp4-static-sim", torch)

    del pipe
    torch.cuda.empty_cache()
    return {
        "backend": "modelopt-restore",
        "checkpoint": str(checkpoint),
        "images": [r["file"] for r in records],
        "records": records,
    }


def _try_diffusers_hf(hf_ckpt_dir: Path, spec, prompts, out_dir: Path) -> dict[str, Any]:
    """Load the Hugging Face export with a stock Diffusers pipeline.

    Recorded because it is what an adopter might reach for first, not because it
    is supported. A failure here is information about ergonomics, not a defect.
    """
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(hf_ckpt_dir, torch_dtype=torch.bfloat16).to("cuda")
    records = _generate(pipe, spec, prompts, out_dir, "nvfp4-static-hf", torch)

    del pipe
    torch.cuda.empty_cache()
    # The records travel with the result, not just their filenames. Returning
    # only ``images`` left the nvfp4-static-hf arm out of metadata.json, so any
    # image it produced was on disk but invisible to scoring and to the figures
    # -- and on the rare run where this path succeeds, that is the arm a reader
    # would most want to compare.
    return {
        "backend": "diffusers-hf-export",
        "images": [r["file"] for r in records],
        "records": records,
    }


def _serving_note(hf_ckpt_dir: Path) -> str:
    return (
        "The Hugging Face export is documented for SGLang, vLLM and TRT-LLM rather\n"
        "than for a stock Diffusers pipeline. To try TRT-LLM by hand:\n"
        "  1. Copy examples/visual_gen/configs/flux1-dev-fp4-1gpu.yaml\n"
        "  2. Set quant_config: {quant_algo: NVFP4, dynamic: false}\n"
        f"  3. Point --model at {hf_ckpt_dir}\n"
        "Record whatever it produces; nobody has reported testing that flow."
    )


def run(*, workspace, config_path: Path, manifest, args) -> dict[str, Any]:
    """Restore the quantized checkpoint and generate images from it.

    Writes a BF16 arm alongside every NVFP4 arm that loaded into
    ``images/verify/<export>``, with a ``metadata.json``, so the directory is a
    self-contained comparison the quality stage can score without reference to
    anything else. Needs the export stage, plus the BF16 baseline that
    ``mto.restore`` sits on top of.

    Every load attempt is recorded, successful or not, and only the documented
    ``mto.restore`` path failing is fatal. A stock Diffusers pipeline refusing the
    Hugging Face export is an observation about an unsupported loader rather than
    a defect, and treating it as one would misreport the result.
    """
    config = json.loads(config_path.read_text())
    spec = images.GenerationSpec.from_config(config)
    limit = getattr(args, "prompts", None) or config.get("verify_prompt_count")
    prompts = images.load_prompts(config, REPOSITORY_ROOT, limit=limit)

    export_root = workspace.exports / paths.export_dir_name(config)
    torch_ckpt = export_root / "torch"
    hf_ckpt_dir = export_root / "hf"
    baseline_dir = workspace.models / config["baseline_dir"]

    # Namespaced by export, not a shared "verify" directory. Image names are keyed
    # on prompt, seed and arm only, so a second model -- or the same weights under
    # a different exclusion filter -- would silently overwrite the first run's
    # output and leave two experiments indistinguishable on disk.
    out_dir = workspace.images / "verify" / paths.export_dir_name(config)

    if getattr(args, "dry_run", False):
        print(f"  would restore {torch_ckpt} onto {baseline_dir}")
        print(f"  and generate {len(prompts)} prompts x {len(spec.seeds)} seeds")
        return {}

    if not torch_ckpt.exists() and not hf_ckpt_dir.exists():
        raise RuntimeError(f"No export under {export_root}. Run the export stage first.")

    attempts: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None

    # The documented path first.
    if torch_ckpt.exists():
        if not baseline_dir.exists():
            raise RuntimeError(
                f"mto.restore needs the BF16 pipeline at {baseline_dir}. Run download first."
            )
        try:
            result = _try_modelopt_restore(baseline_dir, torch_ckpt, spec, prompts, out_dir)
            attempts.append({**result, "loaded": True})
            manifest_note = result.get("checkpoint")
            if manifest_note:
                print(f"  restored from {manifest_note}")
            print(f"\n  restored and generated {len(result['images'])} images into {out_dir}")
        except Exception as error:  # noqa: BLE001 - the failure mode is the finding
            attempts.append(
                {
                    "backend": "modelopt-restore",
                    "loaded": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"  mto.restore failed: {error}")
    else:
        print(f"  no PyTorch checkpoint at {torch_ckpt}; skipping the documented path")

    # Secondary: what an adopter might try first, and whether it works.
    if hf_ckpt_dir.exists():
        try:
            hf_result = _try_diffusers_hf(hf_ckpt_dir, spec, prompts, out_dir)
            attempts.append({**hf_result, "loaded": True})
            print("  the Hugging Face export also loads in stock Diffusers")
        except Exception as error:  # noqa: BLE001
            attempts.append(
                {
                    "backend": "diffusers-hf-export",
                    "loaded": False,
                    "error": f"{type(error).__name__}: {error}",
                    "note": "Expected. Stock Diffusers is not a documented consumer of "
                    "--hf-ckpt-dir; this is an ergonomics observation, not a defect.",
                }
            )
            print(f"  stock Diffusers did not load the HF export ({type(error).__name__}) — expected")

    if result is None:
        print("\n" + _serving_note(hf_ckpt_dir))
        raise RuntimeError(
            "The quantized checkpoint did not load through the documented PyTorch path. "
            "Record the error: that one is worth raising, unlike the Diffusers attempt."
        )

    # Generate the matching BF16 arm and write metadata, so this directory is a
    # self-contained comparison the quality stage can score on its own.
    baseline_records = _generate_baseline(baseline_dir, spec, prompts, out_dir)

    # Every arm that actually produced images, not only the documented one. When
    # the Diffusers path succeeds it writes files into the same directory, and
    # omitting them from metadata.json left images on disk that scoring and the
    # figures could not see. Quality scores each NVFP4 arm separately, so an
    # extra arm here adds a column rather than contaminating one.
    generated = list(baseline_records)
    for attempt in attempts:
        generated.extend(attempt.get("records") or [])

    images.write_metadata(out_dir / "metadata.json", spec, generated)
    print(f"  wrote metadata.json covering {len(generated)} images")
    print(f"  score it with: python3 quantize.py --stage quality --images {out_dir}")

    print(
        "\n  These images use the same prompts, seeds and injected latents as the "
        "dynamic arm, so the quality stage can compare them directly."
    )

    return {
        "torch_checkpoint": str(torch_ckpt),
        "hf_export": str(hf_ckpt_dir),
        "image_dir": str(out_dir),
        "attempts": attempts,
        "next": "Run the quality stage to score static against BF16 and against dynamic.",
    }
