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
        for number, raw in enumerate(handle, 1):
            text = raw.strip("\x00\n")
            if not text:
                continue
            try:
                events.append(json.loads(text))
            except json.JSONDecodeError:
                # A partial final line is normal while the gateway is still writing.
                print(f"warning: skipping malformed JSON on line {number} of {path}", file=sys.stderr)
    return events


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/events.jsonl"
    events = load(path)

    agent_tool = [e for e in events if e.get("kind") == "scope" and e.get("scope_category") == "end" and e.get("category") in ("agent", "tool")]
    if agent_tool:
        print("Agent and tool scopes")
        for e in agent_tool:
            print(f"{e['timestamp'][11:23]:12s} {e.get('category', ''):6s} {e.get('name', '')}")
        print()
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
            where = f"{d.get('category') or '-'}/{d.get('phase') or '-'}"
            print(f"{e['timestamp'][11:23]:12s} {'ERROR':20s} {where:34s} target={d.get('target') or '-'}")

    print("\nModel calls recorded by Relay")
    print(f"{'time':12s} {'api':24s} {'requested':14s} {'served model':42s} {'in':>6s} {'out':>6s} {'cached':>6s} {'USD':>10s} {'status'}")
    totals = collections.defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "cached": 0, "usd": 0.0, "unpriced": 0})
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
        row["cached"] += usage.get("cache_read_tokens") or 0
        if usd is None:
            row["unpriced"] += 1
        else:
            row["usd"] += usd
        status = (e.get("metadata") or {}).get("otel.status_code") or "-"
        print(f"{e['timestamp'][11:23]:12s} {e.get('name', ''):24s} {requested:14s} {served:42s} "
              f"{usage.get('prompt_tokens', 0) or 0:6d} {usage.get('completion_tokens', 0) or 0:6d} "
              f"{usage.get('cache_read_tokens', 0) or 0:6d} {('%.6f' % usd) if usd is not None else '-':>10s} {status}")

    # Judge calls happen inside the Switchyard plugin and are recorded as marks, not LLM scopes,
    # so they never carry a catalog price. Report them so route totals are read as partial.
    judge = collections.defaultdict(lambda: {"calls": 0, "input": 0, "output": 0})
    for e in events:
        if e.get("kind") != "mark":
            continue
        d = e.get("data") or {}
        if e.get("name") == "switchyard.routing.llm_call" and d.get("call_role") == "routing":
            judge[d.get("selected_model") or "?"]["calls"] += 1
        if e.get("name") == "switchyard.routing.llm_tokens":
            for m in d.get("measurements") or []:
                attrs = m.get("attributes") or {}
                if attrs.get("call_role") == "routing" and attrs.get("token_type") in ("input", "output"):
                    judge[attrs.get("target_model") or "?"][attrs["token_type"]] += int(m.get("value") or 0)
    if judge:
        print("\nSwitchyard judge calls (routing decisions; not priced by the catalog)")
        print(f"{'judge model':42s} {'calls':>5s} {'in':>7s} {'out':>7s}")
        for model, row in judge.items():
            print(f"{model:42s} {row['calls']:5d} {row['input']:7d} {row['output']:7d}")

    print("\nTotals by served model (judge calls excluded)" if judge else "\nTotals by served model")
    print(f"{'served model':42s} {'calls':>5s} {'in':>7s} {'cached':>7s} {'out':>7s} {'USD':>10s}")
    grand = 0.0
    for model, row in sorted(totals.items(), key=lambda kv: -kv[1]["usd"]):
        grand += row["usd"]
        note = f"  ({row['unpriced']} call(s) without a catalog price)" if row["unpriced"] else ""
        print(f"{model:42s} {row['calls']:5d} {row['in']:7d} {row['cached']:7d} {row['out']:7d} {row['usd']:10.6f}{note}")
    print(f"{'total':42s} {'':5s} {'':7s} {'':7s} {'':7s} {grand:10.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
