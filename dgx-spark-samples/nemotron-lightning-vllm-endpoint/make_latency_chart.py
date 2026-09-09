#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Chart the latency probe from results/latency.json.

    ./make_latency_chart.py                    # -> results/latency.png
    ./make_latency_chart.py --compact          # shorter, for slides

Two panels:
  left   time to first token, median with the full min-max range drawn
  right  decode rate, median with range

The range is drawn deliberately and at the same weight as the median. On a
shared gateway the median can look healthy while the spread runs to seconds,
and a chart showing only medians would hide exactly the thing a reader needs
to see before trusting the number. Where the whisker is long, the honest
reading is "this endpoint is queueing", not "this model is slow".

Endpoints that were skipped -- usually a missing API key -- are drawn as an
explicit row saying so, never as a zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
GREEN, GREY, RED, INK = "#76B900", "#B9BEC4", "#C4453B", "#141618"


def acquired(rows, key, header_val, unknown="varying by endpoint"):
    """How to describe an acquisition setting shared -- or not -- by the rows.

    The file header records the invocation that wrote it, but `--resume`
    deliberately carries across endpoints measured under different settings.
    Captioning the figure from the header therefore claims every bar was taken
    at the last run's --samples, which for the retained rows is false:
    `--resume --samples 11` would caption a five-sample row as eleven. The rows
    know what they were measured at; the header does not.

    Four cases, in order:

      * No row carries the setting -- a latency.json written before rows
        recorded their own. The header is all there is, and for such a file it
        is accurate, because that version could not merge settings either.
      * Every row agrees. Say the number, and say it exactly as the caption
        always did, so a committed figure stays reproducible from its JSON.
      * They disagree but all are known. Give the range.
      * Some row carries no setting at all. That is a disagreement, not a
        match on the header, because what the older row used is unknown -- so
        say so rather than pick a number.

    `unknown` is per call site because the phrase has to read correctly in the
    sentence it lands in, and the two slots are not the same shape.
    """
    vals = [r.get(key) for r in rows]
    if not vals or all(v is None for v in vals):
        return str(header_val)
    uniq = set(vals)
    if len(uniq) == 1:
        return f"{uniq.pop()}"
    if None in uniq:
        return unknown
    return f"{min(uniq):g}–{max(uniq):g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", default=str(HERE / "results" / "latency.json"))
    ap.add_argument("--out", default=str(HERE / "results" / "latency.png"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--compact", action="store_true",
                    help="shorter figure with no footnote — for embedding in slides")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.latency).read_text())
    rows = data["endpoints"]
    ok = [r for r in rows if not r.get("skipped")]
    skipped = [r for r in rows if r.get("skipped")]

    is_local = lambda r: "local" in (r.get("location") or "").lower()
    ok.sort(key=lambda r: (0 if is_local(r) else 1, r["ttft_ms"]["median"]))
    names = [r["name"] for r in ok] + [r["name"] for r in skipped]

    height = 3.4 if args.compact else 4.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, height), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    y = list(range(len(names)))[::-1]

    def panel(ax, key, fmt, label, title):
        # Axis range first, so the value label can be offset by a fixed fraction
        # of the axis rather than a fraction of its own value. A local TTFT of
        # 70 ms on an axis running to 5 s would otherwise land on top of its own
        # whisker cap, which is precisely the row a reader looks at first.
        # Fall back only when there is nothing to plot. An earlier version wrote
        # min([...] + [1]), which pinned the low end at 1 whatever the data said
        # and forced a log scale onto the decode panel -- flattening a real 2x
        # difference into two bars that looked alike. A chart that understates
        # its own finding is the same failure as a number that overstates one.
        highs = [r[key]["max"] for r in ok]
        lows = [r[key]["min"] for r in ok]
        hi = max(highs) if highs else 1
        lo = min(lows) if lows else 1

        # A queued endpoint can be four orders of magnitude off a local one --
        # 58 ms against 213 seconds, observed. On a linear axis that renders as
        # one long bar and four invisible ones, which is not a chart. Switch to
        # log when the spread demands it and say so on the axis, because a log
        # scale read as linear understates the difference enormously.
        log = hi / max(lo, 1e-9) > 50
        if log:
            ax.set_xscale("log")
            label += "  ·  log scale"
        top = hi * (3.0 if log else 1.30)
        pad = top * 0.015
        floor = max(lo * 0.5, 1e-9) if log else 0
        for i, r in enumerate(ok):
            v, c, yy = r[key], GREEN if is_local(r) else GREY, y[i]
            ax.barh(yy, v["median"] - floor, left=floor, height=0.55, color=c, zorder=2)
            if v["max"] > v["min"]:
                ax.plot([v["min"], v["max"]], [yy, yy], color=INK, lw=1.4,
                        zorder=3, solid_capstyle="butt")
                ax.plot([v["min"], v["max"]], [yy, yy], "|", color=INK, ms=7, zorder=3)
            # A decode figure the server did not vouch for gets a tilde. It is
            # still plotted -- hiding it would be its own distortion -- but the
            # reader is told not to quote it.
            approx = key == "decode_tok_s" and not r.get("decode_exact", True)
            ax.text(max(v["max"], v["median"]) + pad, yy,
                    ("~" if approx else "") + fmt.format(v["median"]),
                    va="center", fontsize=9.5,
                    color="#7A8188" if approx else INK, zorder=4)
        for j, r in enumerate(skipped):
            ax.text(0.02, y[len(ok) + j], f"skipped — {r['skipped'][:34]}",
                    va="center", fontsize=9, style="italic", color=RED,
                    transform=ax.get_yaxis_transform())
        ax.set_yticks(y)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(title, fontsize=11.5, color=INK, pad=10, loc="left")
        ax.grid(axis="x", color="#E6E8EA", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(floor, top)

    panel(ax1, "ttft_ms", "{:.0f} ms", "Time to first token (ms) — lower is better",
          "How long before anything comes back?")
    panel(ax2, "decode_tok_s", "{:.0f} tok/s", "Decode rate (tokens/s) — higher is better",
          "How fast does it write after that?")
    ax1.set_yticklabels(names, fontsize=10)
    ax2.set_yticklabels([])

    for ax in (ax1, ax2):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#D8DBDE")
        ax.tick_params(length=0)

    if not args.compact:
        # Read from the rows, falling back to the header. See acquired().
        n = acquired(ok, "samples_requested", data.get("samples_requested", "?"),
                     unknown="A varying number of")
        mt = acquired(ok, "max_tokens", data.get("max_tokens", "?"))
        fig.text(0.008, -0.02,
                 f"{n} sequential samples per endpoint, concurrency 1, max_tokens "
                 f"{mt}, one warmup discarded. Bars are medians; "
                 f"whiskers are the full min-max range. Green = local on one DGX Spark. "
                 f"Hosted figures are end-to-end and include network and gateway queueing, "
                 f"which cannot be separated from model time."
                 + (" A tilde marks a decode rate inferred from stream chunks because the "
                    "endpoint returned no token count; a gateway that batches chunks inflates it."
                    if any(not r.get("decode_exact", True) for r in ok) else "")
                 + f"  Run: {data.get('generated','')}",
                 fontsize=8.2, color="#7A8188")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
