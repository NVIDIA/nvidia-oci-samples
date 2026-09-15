# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Paired image generation.

For a BF16 versus NVFP4 comparison to mean anything, both sides must start from
the same noise. The same integer seed is not sufficient: different runtimes
consume the random stream differently, so two stacks given seed 42 can begin
from different latents.

We therefore generate the initial latents once, hash them, save them, and inject
the same tensors into both sides. The hash goes in the metadata so a reviewer
can confirm the pairing rather than take it on trust.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationSpec:
    """Everything that must match between the two arms."""

    height: int
    width: int
    steps: int
    guidance_scale: float
    max_sequence_length: int
    seeds: tuple[int, ...]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GenerationSpec":
        """Build the spec from a model config's ``generation`` block.

        Every key is read without a default, so a config missing one fails here
        rather than silently generating at a different resolution or step count
        than the arm it is about to be compared against. Nothing about a paired
        comparison survives the two arms being generated differently.
        """
        gen = config["generation"]
        return cls(
            height=gen["height"],
            width=gen["width"],
            steps=gen["steps"],
            guidance_scale=gen["guidance_scale"],
            max_sequence_length=gen["max_sequence_length"],
            seeds=tuple(gen["seeds"]),
        )


def latent_shape(spec: GenerationSpec, *, batch: int = 1) -> tuple[int, ...]:
    """FLUX packs 2x2 patches of an 8x-downsampled latent into 64 channels."""
    return (batch, (spec.height // 16) * (spec.width // 16), 64)


def make_latents(spec: GenerationSpec, seed: int, *, device: str = "cpu", dtype=None):
    """Deterministic initial latents for one seed, generated on CPU.

    CPU generation keeps the values identical across GPU architectures, which
    matters when the two arms run on different nodes.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(latent_shape(spec), generator=generator, dtype=torch.float32)
    if dtype is not None:
        latents = latents.to(dtype)
    return latents.to(device)


def latent_digest(latents) -> str:
    """Short hash of a latent tensor, for proving two arms started identically.

    Recorded per image so a paired comparison can be checked rather than
    assumed. If the digests differ for the same prompt and seed, the two arms
    began from different noise and every per-image number downstream is
    meaningless -- while still looking entirely plausible.

    Copied to CPU and cast to float32 first, so the digest is stable across
    devices and dtypes and can be compared between runs.
    """
    import numpy as np

    array = latents.detach().to("cpu", copy=True).float().numpy().astype(np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def load_prompts(config: dict[str, Any], repository_root: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Prompts with a category tag, sampled across categories when limited.

    Categories matter because acceptance usually turns on specific failure modes
    rather than on an average: image deformation and poor text alignment. The
    prompt set deliberately over-samples text rendering, counting, spatial
    relations and anatomy so those surface rather than averaging away.

    That same weighting makes a naive ``prompts[:limit]`` misleading: a small
    run would be entirely text and counting, which FLUX struggles with even at
    BF16. You would be looking at the model's weakest cases with no baseline
    for comparison, and could easily blame quantization for it.

    So when limiting, take one from each category in turn.
    """
    prompt_file = repository_root / config["prompts_file"]
    prompts = json.loads(prompt_file.read_text())["prompts"]
    if not limit or limit >= len(prompts):
        return prompts

    by_category: dict[str, list[dict[str, str]]] = {}
    for prompt in prompts:
        by_category.setdefault(prompt.get("category", "uncategorised"), []).append(prompt)

    selected: list[dict[str, str]] = []
    while len(selected) < limit:
        added = False
        for bucket in by_category.values():
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
                added = True
        if not added:
            break
    return selected


def image_name(prompt_id: str, seed: int, arm: str) -> str:
    """Filename encoding the prompt, seed and arm that produced an image.

    All three are in the name because pairing is done on them, and a directory of
    images has to remain readable after it has been copied somewhere without its
    ``metadata.json``. The arm in particular: two files differing only by
    precision are otherwise indistinguishable on disk.
    """
    return f"{prompt_id}__seed{seed}__{arm}.png"


INJECTED_LATENT_PAIRING = "identical injected initial latents, hashed per seed"
SEEDED_PAIRING = (
    "same seed per pair; latents derived by the runtime rather than injected. "
    "Distributional metrics remain valid; per-image PSNR is weaker evidence than "
    "under injected latents."
)


def write_metadata(
    path: Path,
    spec: GenerationSpec,
    records: list[dict[str, Any]],
    *,
    pairing: str = INJECTED_LATENT_PAIRING,
) -> None:
    """Write the manifest the quality stage pairs on.

    ``pairing`` is explicit because it is a claim about evidence, not a label.
    Injecting identical latents makes any difference attributable to precision
    alone. A runtime that only accepts a seed derives its own latents, and a
    reader who assumes otherwise will over-read a per-image PSNR. Recording
    which one produced a directory keeps that distinction with the data instead
    of in someone's memory.
    """
    payload = {
        "generation": asdict(spec),
        "images": records,
        "pairing": pairing,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
