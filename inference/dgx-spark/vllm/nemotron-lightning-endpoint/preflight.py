#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Check every endpoint in endpoints.json before spending ten minutes on a sweep.

    ./preflight.py                  # all endpoints
    ./preflight.py --only kimi-k3   # one of them

Why this exists
---------------
A When2Call sweep is 600 requests. Finding out afterwards that one endpoint
401'd, or served a different model than you configured, or cannot emit
structured tool calls at all, costs the whole run. Worse, two of those three
failures produce a *plausible-looking number* rather than an error:

  * A model that describes a tool call in prose instead of returning
    `tool_calls` scores as "decided not to call" on every single example. That
    is not a low score, it is a broken measurement, and it looks the same.

  * A budget too small for a reasoning model is spent thinking, so the call
    never arrives. Recall is understated and tool-selection accuracy is
    flattered, because fewer calls were attempted.

By default it sends one synthetic request per endpoint. That is fast but it
tests a hand-written, spec-compliant tool -- which is exactly what the real
payload is not. When2Call carries BFCL tool names with dots in them and
parameter schemas using Python type names, and a gateway that rejects those
will pass this probe and then 400 on all 120 examples.

So use --real, which sends actual dataset examples through the same
build_messages/to_openai_tools path the sweep uses. A few minutes across every
endpoint, and it validates the payload that will actually be sent.

    ./preflight.py --real 2        # 2 examples per label, per endpoint

It never prints an API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# A prompt where calling is unambiguously correct: the tool exists and every
# required argument is present. Anything other than a tool call is a finding.
PROBE_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string",
                                    "description": "City name"}},
            "required": ["city"],
        },
    },
}]
PROBE_PROMPT = "What is the weather in Austin, Texas right now?"

OK, WARN, FAIL, SKIP = "ok", "warn", "FAIL", "skip"


# A sweep is 3 x n examples per endpoint, run `concurrency` at a time. If one
# request takes t seconds, that endpoint alone costs roughly
# (3n / concurrency) * t. At n=40 and concurrency 2 a 30 s request is a
# half-hour; a 264 s request is most of a day.
SLOW_REQUEST_S = 25.0
SWEEP_N_PER_LABEL = 40


def check_real(spec: dict, examples: list, client, model_id: str) -> dict:
    """Send real dataset examples through the exact path the sweep uses.

    Reports per-example outcomes rather than aggregate accuracy -- with a
    handful of examples the rates are meaningless, but "was the payload
    accepted" and "did a structured call come back" are not.
    """
    import when2call as w2c
    from concurrent.futures import ThreadPoolExecutor

    budget = spec.get("max_tokens", 3072)
    per_timeout = spec.get("timeout", 180)
    workers = min(spec.get("concurrency", 2), len(examples))

    def one(ex):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=w2c.build_messages(ex),
                tools=w2c.to_openai_tools(ex["tools"]), tool_choice="auto",
                max_tokens=budget, temperature=0.0, timeout=per_timeout,
                **({"extra_body": spec["extra_body"]} if spec.get("extra_body") else {}),
            )
        except Exception as e:                                  # noqa: BLE001
            return {"label": ex["label"], "error": f"{type(e).__name__}: {e}",
                    "elapsed": time.perf_counter() - t0}
        m = resp.choices[0].message
        called = bool(m.tool_calls)
        return {"label": ex["label"], "error": None,
                "called": called,
                "tool": m.tool_calls[0].function.name if called else None,
                "gold": ex.get("gold_tool"),
                "finish": resp.choices[0].finish_reason,
                "tokens": getattr(resp.usage, "completion_tokens", None),
                "elapsed": time.perf_counter() - t0}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return {"results": list(pool.map(one, examples))}


def check(spec: dict, timeout: int | None = None,
          examples: list | None = None) -> dict:
    """One endpoint. Returns a verdict dict."""
    from openai import OpenAI

    name = spec["name"]
    r = {"name": name, "location": spec.get("location", ""),
         "verdict": OK, "notes": []}

    key_env = spec.get("api_key_env")
    if key_env:
        api_key = os.environ.get(key_env, "")
        if not api_key:
            r["verdict"] = SKIP
            r["notes"].append(f"{key_env} is not set in this environment")
            return r
    else:
        api_key = "not-needed"

    timeout = timeout if timeout is not None else spec.get("timeout", 120)
    client = OpenAI(base_url=spec["base_url"], api_key=api_key, timeout=timeout)
    configured = spec.get("model")

    # --- 1. can we reach it, and is the configured model actually there? ----
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as e:                                      # noqa: BLE001
        r["verdict"] = FAIL
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            r["notes"].append(
                f"401 Unauthorized. Is {key_env} a key for {spec['base_url']}? "
                "Keys are not portable between the public catalog and an "
                "internal inference service.")
        else:
            r["notes"].append(f"cannot list models: {type(e).__name__}: {msg[:110]}")
        return r

    r["catalogue"] = len(served)
    if configured and configured not in served:
        if len(served) == 1:
            r["notes"].append(f"serves one model, '{served[0]}' — using that")
            configured = served[0]
        else:
            r["verdict"] = FAIL
            near = [m for m in served if configured.split("/")[-1] in m]
            hint = f" Closest: {', '.join(near[:3])}." if near else ""
            r["notes"].append(
                f"'{configured}' not offered ({len(served)} models).{hint}")
            return r

    # --- 2a. real dataset examples, if asked for ----------------------------
    if examples:
        got = check_real(spec, examples, client, configured)["results"]
        errs = [g for g in got if g["error"]]
        ok_ = [g for g in got if not g["error"]]
        r["n_real"] = len(got)
        r["n_ok"] = len(ok_)
        r["latency_s"] = round(sum(g["elapsed"] for g in got) / len(got), 2)

        if len(errs) == len(got):
            r["verdict"] = FAIL
            first = errs[0]["error"]
            r["notes"].append(f"all {len(got)} real examples failed: {first[:150]}")
            if "400" in first or "invalid" in first.lower():
                r["notes"].append(
                    "a 400 on the real payload but not on the synthetic probe means "
                    "the dataset's tool schemas are being rejected, not the model.")
            return r
        if errs:
            r["verdict"] = WARN
            r["notes"].append(f"{len(errs)}/{len(got)} failed: {errs[0]['error'][:110]}")
            if any("429" in e["error"] for e in errs):
                r["notes"].append(
                    f"429s at concurrency {spec.get('concurrency', 2)}. This probe does "
                    f"not retry; run_benchmark.py does, with exponential backoff, so the "
                    f"sweep may still complete -- more slowly. Lower concurrency if most "
                    f"requests are being rejected.")

        called = [g for g in ok_ if g["called"]]
        r["called"] = f"{len(called)}/{len(ok_)}"
        gold_pool = [g for g in called if g["label"] == "tool_call" and g["gold"]]
        if gold_pool:
            hits = sum(g["tool"] == g["gold"] for g in gold_pool)
            r["gold_match"] = f"{hits}/{len(gold_pool)}"
            if hits == 0:
                r["verdict"] = FAIL
                g = gold_pool[0]
                r["notes"].append(
                    f"tool names never match gold: returned {g['tool']!r} vs gold "
                    f"{str(g['gold'])[:60]!r}. Tool-selection accuracy would read 0% "
                    f"for every model.")
        truncated = [g for g in ok_ if g["finish"] == "length"]
        if truncated:
            r["verdict"] = WARN if r["verdict"] == OK else r["verdict"]
            r["notes"].append(
                f"{len(truncated)}/{len(ok_)} hit the {spec.get('max_tokens', 3072)}-token "
                f"budget. Raise max_tokens for this endpoint.")

        lat = r["latency_s"]
        rounds = (3 * SWEEP_N_PER_LABEL) / max(1, spec.get("concurrency", 2))
        mins = rounds * lat / 60
        r["sweep_estimate"] = (f"{mins/60:.1f}h" if mins > 90 else f"{mins:.0f}m")
        return r

    # --- 2b. synthetic probe ------------------------------------------------
    budget = spec.get("max_tokens", 3072)
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=configured,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            tools=PROBE_TOOL, tool_choice="auto",
            max_tokens=budget, temperature=0.0,
            **({"extra_body": spec["extra_body"]} if spec.get("extra_body") else {}),
        )
    except Exception as e:                                      # noqa: BLE001
        r["verdict"] = FAIL
        msg = str(e)
        if "429" in msg:
            r["notes"].append("429 rate limited on a single request — lower "
                              "concurrency will not save the sweep")
        elif "Timeout" in type(e).__name__:
            r["notes"].append(
                f"timed out after {timeout}s. Raise \"timeout\" for this "
                f"endpoint in endpoints.json -- free-tier endpoints queue, and "
                f"a timeout here is a queue length, not a broken model.")
        elif "tool" in msg.lower():
            r["notes"].append(f"rejected the tools parameter: {msg[:110]}")
        else:
            r["notes"].append(f"{type(e).__name__}: {msg[:110]}")
        return r

    r["latency_s"] = round(time.perf_counter() - t0, 2)
    choice = resp.choices[0]
    msg = choice.message
    used = getattr(resp.usage, "completion_tokens", None)
    r["model"] = resp.model
    r["tokens"] = used
    r["finish"] = choice.finish_reason

    reasoning = (getattr(msg, "reasoning", None)
                 or getattr(msg, "reasoning_content", None) or "")
    r["reasons"] = bool(reasoning)

    if msg.tool_calls:
        r["tool"] = msg.tool_calls[0].function.name
        try:
            json.loads(msg.tool_calls[0].function.arguments)
        except Exception:                                       # noqa: BLE001
            r["verdict"] = WARN
            r["notes"].append("tool_calls present but arguments are not valid JSON")
    else:
        r["verdict"] = FAIL
        if choice.finish_reason == "length":
            suggest = budget * 2
            r["notes"].append(
                f"budget of {budget} exhausted before any tool call "
                f"({used} tokens, all reasoning). Try max_tokens {suggest} "
                f"for this endpoint in endpoints.json.")
        else:
            body = (msg.content or "").strip().replace("\n", " ")[:70]
            r["notes"].append(
                "no structured tool_calls — every When2Call example would "
                f"score as 'decided not to call'. Returned text instead: {body!r}")

    # --- 3. latency, projected onto the full sweep --------------------------
    # A slow endpoint is not a broken one, but it can make the sweep
    # impractical, and that is worth knowing before you start rather than
    # forty minutes in.
    lat = r["latency_s"]
    if lat >= SLOW_REQUEST_S:
        rounds = (3 * SWEEP_N_PER_LABEL) / max(1, spec.get("concurrency", 2))
        mins = rounds * lat / 60
        r["verdict"] = WARN if r["verdict"] == OK else r["verdict"]
        unit = f"{mins/60:.1f} hours" if mins > 90 else f"{mins:.0f} minutes"
        r["notes"].append(
            f"{lat:.0f}s for a trivial request. At n={SWEEP_N_PER_LABEL} and "
            f"concurrency {spec.get('concurrency', 2)} this endpoint alone "
            f"would take about {unit}.")
        if used and used < 60:
            r["notes"].append(
                f"only {used} tokens generated, so that time is queueing on a "
                f"shared endpoint, not compute. Raising concurrency overlaps "
                f"the waits; lowering max_tokens will not help.")

    # --- 4. budget headroom -------------------------------------------------
    if r["verdict"] == OK and used and used > 0.8 * budget:
        r["verdict"] = WARN
        r["notes"].append(
            f"used {used} of {budget} tokens on a trivial prompt — little "
            f"headroom for the harder When2Call examples")

    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "endpoints.json"))
    ap.add_argument("--only", action="append", help="check only these (repeatable)")
    ap.add_argument("--real", type=int, default=0, metavar="N",
                    help="send N real When2Call examples per label through the "
                         "exact path the sweep uses, instead of a synthetic "
                         "probe. This is what catches a gateway rejecting the "
                         "dataset's tool schemas.")
    ap.add_argument("--seed", type=int, default=99,
                    help="example seed. Deliberately not the sweep's default, "
                         "so preflight and the sweep do not share examples.")
    args = ap.parse_args()

    specs = json.loads(Path(args.config).read_text())["endpoints"]
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]
    if not specs:
        print("No endpoints selected.", file=sys.stderr)
        return 2

    examples = None
    if args.real:
        import when2call as w2c
        examples = w2c.load_examples(n_per_label=args.real, seed=args.seed)
        print(f"Preflight: {len(specs)} endpoint(s), {len(examples)} real "
              f"When2Call examples each.\n")
    else:
        print(f"Preflight: {len(specs)} endpoint(s), one synthetic request each.")
        print("  (--real N sends actual dataset examples; a synthetic tool cannot "
              "catch\n   a gateway rejecting the dataset's own schemas.)\n")

    results = []
    for spec in specs:
        print(f"  {spec['name']:<18} ...", end=" ", flush=True)
        r = check(spec, examples=examples)
        results.append(r)
        bits = []
        if r.get("n_real"):
            bits.append(f"{r['n_ok']}/{r['n_real']} ok")
        if r.get("called"):
            bits.append(f"called {r['called']}")
        if r.get("gold_match"):
            bits.append(f"gold {r['gold_match']}")
        if r.get("sweep_estimate"):
            bits.append(f"sweep ~{r['sweep_estimate']}")
        if r.get("tool"):
            bits.append(f"tool_calls -> {r['tool']}")
        if r.get("tokens") is not None:
            bits.append(f"{r['tokens']} tok")
        if r.get("reasons"):
            bits.append("reasons")
        if r.get("latency_s"):
            bits.append(f"{r['latency_s']}s")
        print(f"{r['verdict']:<5} {'  '.join(bits)}")
        for n in r["notes"]:
            print(f"      {n}")

    print()
    bad = [r for r in results if r["verdict"] == FAIL]
    skipped = [r for r in results if r["verdict"] == SKIP]
    warned = [r for r in results if r["verdict"] == WARN]

    if bad:
        print(f"  {len(bad)} endpoint(s) would produce meaningless data: "
              f"{', '.join(r['name'] for r in bad)}")
        print("  Fix these before sweeping — a broken measurement looks like a low score.")
    if skipped:
        print(f"  {len(skipped)} skipped for missing credentials: "
              f"{', '.join(r['name'] for r in skipped)}")
    if warned:
        print(f"  {len(warned)} with warnings — usable, but read the notes.")
    if not bad and not skipped:
        print("  All endpoints ready. Run:  ./run_benchmark.py --config "
              f"{Path(args.config).name} --n 40 --save-raw")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
