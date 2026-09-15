# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamic NVFP4 quality check.

Run this before the static export. Dynamic quantization computes the scaling
factors at runtime from the activations, so it needs no calibration pass and no
exported checkpoint: point the runtime at the ordinary BF16 model and generate.

The reason this is worth doing first is that dynamic and static NVFP4 should
land at the same accuracy. The GEMM path is identical; only the source of the
scale differs. So if quality holds here it will hold for the static export, and
we learn that in an hour rather than after a full calibration and export cycle.

Two backends:

* ``diffusers`` uses Hugging Face Diffusers with TorchAO NVFP4. Fewer moving
  parts, and it is the path the sibling benchmarking sample already exercises.
* ``visualgen`` uses TensorRT-LLM VisualGen, which is the runtime we expect to
  serve from. Its exact invocation should be confirmed against the container in
  use; the shipped config is ``examples/visual_gen/configs/flux1-dev-fp4-1gpu.yaml``.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from common import images

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _weight_fingerprint(module) -> tuple:
    """Cheap signature of a module's parameter dtypes and classes.

    Used to prove quantization actually took effect. A silent no-op produces
    two identical image sets, which looks like a flawless result rather than a
    failed one -- the worst way for this to go wrong.

    Restricted to parameters quantization can actually change, and not
    truncated. Sampling the first N parameters lands almost entirely in the
    embedders and normalisation layers at the front of the module -- exactly the
    layers the exclusion filter protects -- so the fingerprint would compare
    equal on a perfectly successful run and report a no-op that did not happen.
    Matrices of rank 2 or higher are the Linear weights the recipe converts;
    1-D norms and biases stay at BF16 either way and only add noise.
    """
    return tuple(
        (name, str(param.dtype), type(param).__name__)
        for name, param in module.named_parameters()
        if param.ndim >= 2
    )


def _nvfp4_config() -> tuple[Any, str]:
    """Resolve TorchAO's NVFP4 config across versions.

    This lives under torchao.prototype and the name has already changed once:
    ``NVFP4InferenceConfig`` became ``NVFP4DynamicActivationNVFP4WeightConfig``
    in 0.15. Resolve by trying the known names rather than pinning to one, and
    record which was used so a later run can be compared honestly.

    Note the deliberate exclusion of ``NVFP4WeightOnlyConfig``. Weight-only
    leaves activations at BF16, which is not what the served path does, and
    would make quality look better than it will be in practice.
    """
    from torchao.prototype import mx_formats

    for name in (
        "NVFP4DynamicActivationNVFP4WeightConfig",  # torchao >= 0.15
        "NVFP4InferenceConfig",  # earlier releases
    ):
        config = getattr(mx_formats, name, None)
        if config is not None:
            return config(), name

    available = [n for n in dir(mx_formats) if "NVFP4" in n]
    raise RuntimeError(
        "No usable NVFP4 config found in torchao.prototype.mx_formats. "
        f"Available NVFP4 symbols: {available or 'none'}. "
        "If only a weight-only config is present, do not substitute it: "
        "it leaves activations at BF16 and overstates quality."
    )


# Layers Model Optimizer's FLUX filter keeps at higher precision, mirroring
# filter_func_flux_dev in examples/diffusers/quantization. These are embedders
# and output projections — the recipe protects them because quantizing them is
# what turns a working result into a visibly degraded one.
MODELOPT_EXCLUDED_ROOTS = (
    "proj_out",
    "time_text_embed",
    "context_embedder",
    "x_embedder",
    "norm_out",
    "time_guidance_embed",
    "stream_modulation",
)


def is_quantizable_path(fqn: str) -> bool:
    """Does Model Optimizer's recipe allow this module path to be quantized?

    Matching is on the *root* segment, so ``proj_out`` at the model level is
    protected along with everything beneath it, while
    ``single_transformer_blocks.N.proj_out`` — an ordinary large GEMM that merely
    shares a leaf name — is not. Collapsing those two would either protect 38
    healthy layers or expose one sensitive one.
    """
    if not fqn:
        return False
    return fqn.split(".")[0] not in MODELOPT_EXCLUDED_ROOTS


def modelopt_filter_fn(module: Any, fqn: str) -> bool:
    """Should torchao quantize this module? Mirrors Model Optimizer's exclusions.

    Without this, ``quantize_`` converts *every* ``nn.Linear`` in the transformer,
    including the ten layers the shipped recipe deliberately protects. That is a
    more aggressive recipe than anything NVIDIA ships, and it made our dynamic
    arm score 4.8x further from BF16 than the static export — a difference we
    initially read as dynamic-versus-static when it was really a difference in
    which layers were touched.
    """
    import torch.nn as nn

    return isinstance(module, nn.Linear) and is_quantizable_path(fqn)


def _generate_diffusers(
    *,
    model_path: Path,
    spec: images.GenerationSpec,
    prompts: list[dict[str, str]],
    out_dir: Path,
    arm: str,
    quantize: bool,
    apply_exclusions: bool = True,
) -> list[dict[str, Any]]:
    """Generate one arm with Diffusers, optionally quantizing to NVFP4 in place."""
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    quant_recipe = None
    if quantize:
        # TorchAO applies NVFP4 to the transformer's linear layers at load time.
        # Text encoders and the VAE stay at BF16, which is deliberate: those are
        # the layers where low precision costs quality rather than buying speed.
        from torchao.quantization import quantize_

        config, quant_recipe = _nvfp4_config()
        before = _weight_fingerprint(pipe.transformer)

        if apply_exclusions:
            quantize_(pipe.transformer, config, filter_fn=modelopt_filter_fn)
            quant_recipe = f"{quant_recipe} + modelopt exclusions"
        else:
            quantize_(pipe.transformer, config)
            quant_recipe = f"{quant_recipe} (no exclusions)"

        after = _weight_fingerprint(pipe.transformer)

        if before == after:
            raise RuntimeError(
                f"quantize_ with {quant_recipe} left the transformer unchanged. "
                "The comparison would show two identical image sets and read as a "
                "perfect result when nothing was quantized at all."
            )

        # Count only Linear layers. Walking every module sweeps up containers
        # that were never quantization candidates -- time_text_embed itself, its
        # sub-embedders, activation layers -- and inflates the figure to around
        # 21, which then disagrees with the schema stage's 10 and undermines
        # confidence in both.
        import torch.nn as nn

        protected = [
            name
            for name, module in pipe.transformer.named_modules()
            if isinstance(module, nn.Linear) and not is_quantizable_path(name)
        ]
        print(f"  {arm}: transformer quantized via {quant_recipe}")
        if apply_exclusions:
            print(f"  {arm}: {len(protected)} linear layers held at high precision by the filter")
            for name in sorted(protected):
                print(f"    {name}")
        else:
            print(
                f"  {arm}: WARNING every linear quantized, including "
                f"{len(protected)} the recipe protects"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for prompt in prompts:
        for seed in spec.seeds:
            latents = images.make_latents(spec, seed, device="cuda", dtype=torch.bfloat16)
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
                    "quant_recipe": quant_recipe,
                }
            )
            print(f"  {arm}: {name}")

    del pipe
    torch.cuda.empty_cache()
    return records


def run(*, workspace, config_path: Path, manifest, args) -> dict[str, Any]:
    """Generate BF16 and dynamic-NVFP4 images from the BF16 baseline.

    Writes both arms into ``images/dynamic`` with the ``metadata.json`` the
    quality stage pairs on. Needs the download stage and nothing else -- no
    export, which is the point: dynamic quantization has no calibration step, so
    this answers the quality question in an hour instead of a full export cycle.

    Both arms are given identical injected latents, and the run aborts if the
    latent digests disagree for any prompt and seed. Without that check a broken
    pairing still produces a complete set of plausible numbers, which is the one
    failure here that cannot be caught by looking at the images.
    """
    config = json.loads(config_path.read_text())
    spec = images.GenerationSpec.from_config(config)
    limit = getattr(args, "prompts", None) or config.get("dynamic_prompt_count")
    prompts = images.load_prompts(config, REPOSITORY_ROOT, limit=limit)

    model_path = workspace.models / config["baseline_dir"]
    out_dir = workspace.images / "dynamic"

    if getattr(args, "dry_run", False):
        print(f"  would generate {len(prompts)} prompts x {len(spec.seeds)} seeds, both arms")
        print(f"  would write to {out_dir}")
        return {}

    if not model_path.exists():
        raise RuntimeError(f"Baseline model not found at {model_path}. Run the download stage first.")

    records: list[dict[str, Any]] = []
    records += _generate_diffusers(
        model_path=model_path,
        spec=spec,
        prompts=prompts,
        out_dir=out_dir,
        arm="bf16",
        quantize=False,
    )
    records += _generate_diffusers(
        model_path=model_path,
        spec=spec,
        prompts=prompts,
        out_dir=out_dir,
        arm="nvfp4-dynamic",
        quantize=True,
        apply_exclusions=not getattr(args, "no_exclusions", False),
    )

    metadata = out_dir / "metadata.json"
    images.write_metadata(metadata, spec, records)

    # Confirm the pairing actually held. If the latent hashes differ between
    # arms for the same prompt and seed, the comparison downstream is invalid.
    by_key: dict[tuple[str, int], set[str]] = {}
    for record in records:
        by_key.setdefault((record["prompt_id"], record["seed"]), set()).add(record["latent_sha256_16"])
    mismatched = [key for key, digests in by_key.items() if len(digests) != 1]
    if mismatched:
        raise RuntimeError(
            f"Initial latents differ between arms for {len(mismatched)} prompt/seed pairs. "
            "The paired comparison would be meaningless; investigate before scoring."
        )

    print(f"  {len(records)} images, {len(by_key)} matched pairs")
    print(f"  metadata: {metadata}")
    print("  next: run the quality stage to score these, or go straight to export")

    recipes = {r["quant_recipe"] for r in records if r.get("quant_recipe")}
    return {
        "image_dir": str(out_dir),
        "metadata": str(metadata),
        "pairs": len(by_key),
        "images": len(records),
        "quant_recipe": sorted(recipes),
        "note": "dynamic NVFP4 should match static accuracy; scale source differs, GEMM path does not",
    }
