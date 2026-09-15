#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Build BF16-vs-NVFP4 comparison figures from a completed run.

Reads the paired images written by the ``dynamic`` stage and the scores written
by ``quality``, and emits one side-by-side figure per pair plus a contact sheet.

Figures are written to ``docs/figures/`` and are deliberately **not** committed.
FLUX.1-dev is under a non-commercial licence and this repository is public, so
images generated from the ``flux-dev`` arm must not be pushed. Regenerate them
locally instead::

    make figures

Anything published or shared should be generated from the ``flux-schnell`` arm,
which is Apache-2.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PANEL = 560
HEADER = 50
LABEL = 34
GUTTER = 8
MARGIN = 14

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
)
FONT_CANDIDATES_REGULAR = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
)


@dataclass(frozen=True)
class Pair:
    """One BF16/NVFP4 comparison at a single prompt and seed."""

    prompt_id: str
    category: str
    seed: int
    prompt_text: str
    bf16: Path
    nvfp4: Path
    psnr_db: float | None
    clip_delta: float | None
    latent_digest: str | None
    # Which NVFP4 arm the right-hand panel came from. A figure labelled only
    # "NVFP4" is ambiguous the moment a run holds more than one arm -- a served
    # checkpoint and a simulated one look identical on the page and mean
    # different things.
    arm: str = "nvfp4"


def _load_font(size: int, *, bold: bool = True):
    """First usable font from the candidate list, falling back to PIL's default.

    Font paths differ across the containers and login nodes this runs on, and a
    figure with ugly default text is still a usable figure. Raising here would
    lose the whole render over a cosmetic detail.
    """
    from PIL import ImageFont

    for path in FONT_CANDIDATES if bold else FONT_CANDIDATES_REGULAR:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_prompt_text(config_dir: Path) -> dict[str, str]:
    """Map prompt id to prompt text, or an empty map if there is no prompts file.

    Accepts both the wrapped ``{"prompts": [...]}`` shape and a bare list, since
    both are in circulation. Absent prompts only cost the figures their captions,
    so this returns empty rather than failing the render.
    """
    prompts_file = config_dir / "prompts.json"
    if not prompts_file.is_file():
        return {}
    payload = json.loads(prompts_file.read_text())
    entries = payload["prompts"] if isinstance(payload, dict) else payload
    return {entry["id"]: entry.get("text", "") for entry in entries}


def _collect_pairs(images_dir: Path, results_dir: Path, config_dir: Path) -> list[Pair]:
    """Match BF16 and NVFP4 renders, and attach scores where they exist."""
    metadata_file = images_dir / "metadata.json"
    if not metadata_file.is_file():
        raise SystemExit(f"no metadata.json in {images_dir}; run the dynamic stage first")

    metadata = json.loads(metadata_file.read_text())
    prompt_text = _load_prompt_text(config_dir)

    # Keyed by arm as well as prompt and seed. Without the arm, a quality.json
    # holding more than one nvfp4 arm -- nvfp4-static-sim beside
    # nvfp4-static-hf, say -- silently labelled a figure with the other arm's
    # PSNR and CLIP delta. The numbers looked plausible and belonged to a
    # different image.
    scores: dict[tuple[str, int, str], dict] = {}
    quality_file = results_dir / "quality.json"
    if quality_file.is_file():
        for record in json.loads(quality_file.read_text()).get("pairs", []):
            arm = record.get("arm") or record.get("nvfp4_arm") or ""
            scores[(record["prompt_id"], record["seed"], arm)] = record

    by_key: dict[tuple[str, int], dict[str, dict]] = {}
    for entry in metadata.get("images", []):
        key = (entry["prompt_id"], entry["seed"])
        by_key.setdefault(key, {})[entry["arm"]] = entry

    pairs: list[Pair] = []
    for (prompt_id, seed), arms in sorted(by_key.items()):
        baseline = arms.get("bf16")
        # sorted() rather than dict order, so the arm chosen for a given prompt
        # is the same on every run and across machines.
        quantized_name = next(
            (name for name in sorted(arms) if name.startswith("nvfp4")), None
        )
        if baseline is None or quantized_name is None:
            continue
        quantized = arms[quantized_name]

        # Prefer the score recorded for this exact arm. Fall back to an
        # unlabelled record, which is what older quality.json files contain.
        score = scores.get((prompt_id, seed, quantized_name)) or scores.get(
            (prompt_id, seed, ""), {}
        )
        pairs.append(
            Pair(
                prompt_id=prompt_id,
                category=baseline.get("category", ""),
                seed=seed,
                prompt_text=prompt_text.get(prompt_id, ""),
                bf16=images_dir / baseline["file"],
                nvfp4=images_dir / quantized["file"],
                psnr_db=score.get("psnr_db"),
                clip_delta=score.get("clip_delta"),
                latent_digest=baseline.get("latent_sha256_16"),
                arm=quantized_name,
            )
        )
    return pairs


def _wrap(draw, text: str, font, max_width: int) -> str:
    """Shorten text to a single line that fits, with an ellipsis if needed."""
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "...", font=font) > max_width:
        text = text[:-1]
    return text.rstrip() + "..."


def _render_pair(pair: Pair, destination: Path) -> Path:
    """Draw one BF16/NVFP4 comparison to a PNG and return its path.

    Both panels are resized to the same square, so a composition difference reads
    as a difference rather than as a layout artefact. Prompt, seed and scores are
    drawn onto the image because these files get separated from the run that
    produced them almost immediately.
    """
    from PIL import Image, ImageDraw

    title_font = _load_font(20)
    body_font = _load_font(17, bold=False)
    label_font = _load_font(19)

    left = Image.open(pair.bf16).convert("RGB").resize((PANEL, PANEL), Image.LANCZOS)
    right = Image.open(pair.nvfp4).convert("RGB").resize((PANEL, PANEL), Image.LANCZOS)

    width = PANEL * 2 + GUTTER + MARGIN * 2
    height = HEADER + LABEL + PANEL + MARGIN * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    heading = f"{pair.prompt_id}  ({pair.category})  seed {pair.seed}"
    draw.text((MARGIN, MARGIN), heading, fill="#111111", font=title_font)
    if pair.prompt_text:
        subtitle = _wrap(draw, f'"{pair.prompt_text}"', body_font, width - MARGIN * 2)
        draw.text((MARGIN, MARGIN + 22), subtitle, fill="#555555", font=body_font)

    top = MARGIN + HEADER + LABEL
    canvas.paste(left, (MARGIN, top))
    canvas.paste(right, (MARGIN + PANEL + GUTTER, top))

    label_y = MARGIN + HEADER + 8
    draw.text((MARGIN, label_y), "BF16 baseline", fill="#111111", font=label_font)

    metrics = []
    if pair.psnr_db is not None:
        metrics.append(f"PSNR {pair.psnr_db:.2f} dB")
    if pair.clip_delta is not None:
        metrics.append(f"CLIP {pair.clip_delta:+.2f}")
    # Name the arm, so a sheet is still readable once separated from the run
    # that produced it.
    arm_label = (pair.arm or "nvfp4").replace("nvfp4-", "NVFP4 ").replace("nvfp4", "NVFP4")
    right_label = arm_label + ("   " + "   ".join(metrics) if metrics else "")
    draw.text((MARGIN + PANEL + GUTTER, label_y), right_label, fill="#111111", font=label_font)

    draw.rectangle(
        [MARGIN, top, MARGIN + PANEL - 1, top + PANEL - 1], outline="#dddddd", width=1
    )
    draw.rectangle(
        [MARGIN + PANEL + GUTTER, top, MARGIN + PANEL * 2 + GUTTER - 1, top + PANEL - 1],
        outline="#dddddd",
        width=1,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    return destination


def _render_contact_sheet(pairs: list[Pair], destination: Path, seed: int | None) -> Path | None:
    """One sheet with every prompt at a single seed, for a slide."""
    from PIL import Image, ImageDraw

    selected = [p for p in pairs if seed is None or p.seed == seed]
    if not selected:
        return None

    cell = 300
    title_font = _load_font(20)
    small_font = _load_font(15)

    columns = 2
    rows = len(selected)
    width = cell * columns + GUTTER + MARGIN * 2
    height = MARGIN + 30 + rows * (cell + 26) + MARGIN

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    caption = "BF16 (left) vs NVFP4 (right)" + (f" — seed {seed}" if seed else "")
    draw.text((MARGIN, MARGIN), caption, fill="#111111", font=title_font)

    y = MARGIN + 30
    for pair in selected:
        left = Image.open(pair.bf16).convert("RGB").resize((cell, cell), Image.LANCZOS)
        right = Image.open(pair.nvfp4).convert("RGB").resize((cell, cell), Image.LANCZOS)
        canvas.paste(left, (MARGIN, y))
        canvas.paste(right, (MARGIN + cell + GUTTER, y))
        note = pair.prompt_id
        if pair.psnr_db is not None:
            note += f"   PSNR {pair.psnr_db:.2f} dB"
        if pair.clip_delta is not None:
            note += f"   CLIP {pair.clip_delta:+.2f}"
        draw.text((MARGIN, y + cell + 5), note, fill="#444444", font=small_font)
        y += cell + 26

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    return destination


def main() -> int:
    """Render side-by-side figures and a contact sheet from a scored image set.

    Reads the image directory's ``metadata.json`` for pairing and ``quality.json``
    for the per-pair scores, so a figure carries the same numbers as the report
    rather than a second, separately computed set.

    Licence matters here in a way it does not elsewhere in the harness: figures
    are the output most likely to be pasted into a document and shared, and
    ``flux-dev`` images are non-commercial. Generate anything shareable from the
    ``flux-schnell`` arm.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, required=True, help="directory of paired renders with metadata.json"
    )
    parser.add_argument(
        "--results", type=Path, help="directory holding quality.json (defaults to --images)"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs" / "figures", help="where to write figures"
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs", help="directory holding prompts.json"
    )
    parser.add_argument(
        "--sheet-seed", type=int, help="seed to use for the contact sheet; omit for all pairs"
    )
    args = parser.parse_args()

    try:
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit("Pillow is required: pip install pillow")

    results = args.results or args.images
    pairs = _collect_pairs(args.images, results, args.config)
    if not pairs:
        raise SystemExit("no BF16/NVFP4 pairs found; both arms must have run")

    for pair in pairs:
        written = _render_pair(
            pair, args.out / f"{pair.prompt_id}__seed{pair.seed}__compare.png"
        )
        print(f"wrote {written.relative_to(REPO_ROOT) if written.is_relative_to(REPO_ROOT) else written}")

    sheet = _render_contact_sheet(pairs, args.out / "contact-sheet.png", args.sheet_seed)
    if sheet is not None:
        print(f"wrote {sheet.relative_to(REPO_ROOT) if sheet.is_relative_to(REPO_ROOT) else sheet}")

    print(f"\n{len(pairs)} pairs rendered to {args.out}")
    print("These are not committed: flux-dev output is non-commercial and this repo is public.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
