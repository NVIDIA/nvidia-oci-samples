#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Chart When2Call results from results/summary.json.

    ./make_chart.py                       # -> results/when2call.png
    ./make_chart.py --out /tmp/chart.png

Two panels:
  left   decision accuracy with 95% Wilson intervals
  right  over-call rate (lower is better) -- the metric that actually separates

Models that returned no data are drawn as an explicit "no data" row, never as
a zero score. A rate-limited model is a missing measurement, not a bad one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
GREEN, GREY, RED, INK = "#76B900", "#B9BEC4", "#C4453B", "#141618"


def _short_reason(entry: dict) -> str:
    """A chart-width summary of why an endpoint produced no data.

    `sweep` writes a full sentence into `reason`; a bar label has room for a
    few words. Classify the common causes and fall back to a truncation rather
    than to a guess, because the wrong guess here -- "rate limited" for every
    failure -- makes a claim about a third-party service that the run did not
    observe.
    """
    reason = (entry.get("reason") or "").strip()
    if not reason:
        return "not run"
    low = reason.lower()
    if "api_key" in low or "api key" in low or "not set" in low:
        return "no API key"
    # Before the cause-specific checks: sweep's "all N requests failed" message
    # quotes the first error, so matching on "timeout" first would relabel a
    # wholesale failure as a timeout on the strength of one example.
    if "all " in low and "failed" in low:
        return "all requests failed"
    if "429" in low or "rate" in low or "quota" in low:
        return "rate limited"
    if "model" in low and ("not found" in low or "unknown" in low or "404" in low):
        return "model not found"
    if "timeout" in low or "timed out" in low or "queued" in low:
        return "timed out"
    return reason if len(reason) <= 34 else reason[:31].rstrip() + "..."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(HERE / "results" / "summary.json"))
    ap.add_argument("--out", default=str(HERE / "results" / "when2call.png"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--compact", action="store_true",
                    help="shorter figure with no footnote — for embedding in slides, "
                         "where the caption lives on the slide instead")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.summary).read_text())
    models = data["models"]
    failed = data.get("failed", [])

    # local model first, then by accuracy
    models.sort(key=lambda m: (0 if "local" in m.get("location", "").lower() else 1,
                               -m["decision_accuracy"]["rate"]))
    names = [m["name"] for m in models] + [f["name"] for f in failed]

    height = 3.6 if args.compact else 4.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, height), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    y = list(range(len(names)))[::-1]

    def is_local(m):
        return "local" in m.get("location", "").lower()

    # ---- left: decision accuracy -----------------------------------------
    for i, m in enumerate(models):
        d = m["decision_accuracy"]
        c = GREEN if is_local(m) else GREY
        yy = y[i]
        ax1.barh(yy, 100 * d["rate"], height=0.55, color=c, zorder=2)
        ax1.plot([100 * d["lo"], 100 * d["hi"]], [yy, yy],
                 color=INK, lw=1.4, zorder=3, solid_capstyle="butt")
        ax1.plot([100 * d["lo"], 100 * d["hi"]], [yy, yy], "|",
                 color=INK, ms=7, zorder=3)
        # place the value clear of the upper CI whisker, not on top of it
        ax1.text(100 * d["hi"] + 1.8, yy, f"{100*d['rate']:.1f}%",
                 va="center", fontsize=9.5, color=INK, zorder=4)
    for j, f in enumerate(failed):
        yy = y[len(models) + j]
        # Say why it actually failed. `sweep` records a reason -- a missing API
        # key, an unresolved model id, every request failing -- and hardcoding
        # "rate limited" charted a model skipped for a missing credential as a
        # throttled one. That is a claim about someone else's service that the
        # data does not support.
        ax1.text(2, yy, f"no data — {_short_reason(f)}", va="center", fontsize=9.5,
                 style="italic", color=RED)

    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlim(0, 100); ax1.set_xlabel("Decision accuracy (%)", fontsize=10)
    ax1.set_title("Did it call a tool exactly when it should have?",
                  fontsize=11.5, color=INK, pad=10, loc="left")
    ax1.grid(axis="x", color="#E6E8EA", zorder=0)
    ax1.set_axisbelow(True)

    # ---- right: over-call rate -------------------------------------------
    for i, m in enumerate(models):
        o = m["over_call_rate"]
        c = GREEN if is_local(m) else GREY
        yy = y[i]
        ax2.barh(yy, 100 * o["rate"], height=0.55, color=c, zorder=2)
        ax2.text(100 * o["rate"] + 0.6, yy, f"{100*o['rate']:.1f}%",
                 va="center", fontsize=9.5, color=INK, zorder=4)
    for j, f in enumerate(failed):
        yy = y[len(models) + j]
        ax2.text(0.6, yy, "no data", va="center", fontsize=9.5,
                 style="italic", color=RED)

    ax2.set_yticks(y); ax2.set_yticklabels([])
    top = max([100 * m["over_call_rate"]["rate"] for m in models] + [10]) * 1.35
    ax2.set_xlim(0, top)
    ax2.set_xlabel("Over-call rate (%) — lower is better", fontsize=10)
    ax2.set_title("How often did it fire a tool it shouldn't have?",
                  fontsize=11.5, color=INK, pad=10, loc="left")
    ax2.grid(axis="x", color="#E6E8EA", zorder=0)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#D8DBDE")
        ax.tick_params(length=0)

    # `n_examples` is what the sweep asked for; a model that errored on some
    # examples is scored on fewer, and its Wilson interval is computed from
    # that smaller n. Report what was actually scored, and say so as a range
    # when the endpoints disagree -- the caption should not claim a sample size
    # no model was measured at.
    scored = sorted({m.get("n_scored") for m in models if m.get("n_scored")})
    if not scored:
        n_text = f"{data.get('n_examples', '?')} examples requested per model"
    elif len(scored) == 1:
        n_text = f"{scored[0]} examples scored per model"
    else:
        n_text = (f"{scored[0]}–{scored[-1]} examples scored per model "
                  f"({data.get('n_examples', '?')} requested)")
    if not args.compact:
        fig.text(0.008, -0.02,
                 f"nvidia/When2Call, {n_text}, identical prompts and tool "
                 f"schemas. Error bars are 95% Wilson intervals. Green = running locally "
                 f"on one DGX Spark.  Run: {data.get('generated','')}",
                 fontsize=8.2, color="#7A8188")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
