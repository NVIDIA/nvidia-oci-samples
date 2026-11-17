#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Summarize a NeMo Relay ATOF event log: routing decisions, per-call usage, and USD cost.

Usage: summarize_events.py [logs/events.jsonl]
"""
import collections
import json
import sys


def load(path):
    events = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip("\x00\n")
            if line:
                events.append(json.loads(line))
    return events


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/events.jsonl"
    events = load(path)

    print("Switchyard routing decisions")
    print(f"{'time':12s} {'algorithm':20s} {'evidence':34s} {'selected -> served'}")
    for e in events:
        if e.get("kind") == "mark" and e.get("name") == "switchyard.routing.decision":
            d = e.get("data") or {}
            ev = d.get("evidence") or {}
            evidence = ", ".join(f"{k}={v}" for k, v in ev.items()) or "-"
            print(f"{e['timestamp'][11:23]:12s} {d.get('algorithm', ''):20s} {evidence:34s} "
                  f"{d.get('selected_model')} -> {d.get('served_model')}")
        if e.get("kind") == "mark" and e.get("name") == "switchyard.routing.error":
            d = e.get("data") or {}
            print(f"{e['timestamp'][11:23]:12s} {'ERROR':20s} {d.get('category')}/{d.get('phase'):26s} target={d.get('target')}")

    print("\nModel calls recorded by Relay")
    print(f"{'time':12s} {'api':24s} {'requested':14s} {'served model':42s} {'in':>6s} {'out':>6s} {'cached':>6s} {'USD':>10s} {'status'}")
    totals = collections.defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "usd": 0.0, "unpriced": 0})
    for e in events:
        if e.get("category") != "llm" or e.get("scope_category") != "end":
            continue
        profile = e.get("category_profile") or {}
        resp = profile.get("annotated_response") or {}
        usage = resp.get("usage") or {}
        cost = usage.get("cost") or {}
        served = resp.get("model") or profile.get("model_name") or "?"
        requested = profile.get("model_name") or "?"
        usd = cost.get("total")
        row = totals[served]
        row["calls"] += 1
        row["in"] += usage.get("prompt_tokens") or 0
        row["out"] += usage.get("completion_tokens") or 0
        if usd is None:
            row["unpriced"] += 1
        else:
            row["usd"] += usd
        status = (e.get("metadata") or {}).get("otel.status_code") or "-"
        print(f"{e['timestamp'][11:23]:12s} {e.get('name', ''):24s} {requested:14s} {served:42s} "
              f"{usage.get('prompt_tokens', 0) or 0:6d} {usage.get('completion_tokens', 0) or 0:6d} "
              f"{usage.get('cache_read_tokens', 0) or 0:6d} {('%.6f' % usd) if usd is not None else '-':>10s} {status}")

    print("\nTotals by served model")
    print(f"{'served model':42s} {'calls':>5s} {'in':>7s} {'out':>7s} {'USD':>10s}")
    grand = 0.0
    for model, row in sorted(totals.items(), key=lambda kv: -kv[1]["usd"]):
        grand += row["usd"]
        note = f"  ({row['unpriced']} call(s) without a catalog price)" if row["unpriced"] else ""
        print(f"{model:42s} {row['calls']:5d} {row['in']:7d} {row['out']:7d} {row['usd']:10.6f}{note}")
    print(f"{'total':42s} {'':5s} {'':7s} {'':7s} {grand:10.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
