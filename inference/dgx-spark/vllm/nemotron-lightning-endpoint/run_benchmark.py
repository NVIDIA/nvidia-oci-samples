#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Run the When2Call agentic tool-calling benchmark against one or more
OpenAI-compatible endpoints and write a summary you can chart.

    ./run_benchmark.py --config endpoints.json --n 40
    ./run_benchmark.py --only spark --n 12          # quick local slice
    ./run_benchmark.py --config endpoints.json --n 40 --concurrency 1

Why the backoff matters
-----------------------
An earlier sweep issued 960 requests back-to-back at concurrency 4 and tripped
the hosted gateway's rate limit: four of eight models returned HTTP 429 on
*every* call and produced no data at all. Retry-with-exponential-backoff plus a
low default concurrency for remote endpoints is the fix, and it is the reason
this script defaults to 2 workers rather than something faster.

Endpoints are configured in endpoints.json. API keys come from the environment
so nothing secret is ever committed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import when2call as w2c

HERE = Path(__file__).parent
RESULTS = HERE / "results"


# --------------------------------------------------------------------------
# Retrying client call
# --------------------------------------------------------------------------

def _retry_after(exc) -> float | None:
    """Seconds the server asked us to wait, if it said so.

    A gateway that returns Retry-After knows its own limits better than our
    backoff curve does. Honouring it converges faster and annoys the service
    less than guessing.
    """
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-after"):
        val = headers.get(key) if hasattr(headers, "get") else None
        if val:
            try:
                return max(0.0, float(val))
            except (TypeError, ValueError):
                return None
    return None


def call_with_backoff(client, *, model, messages, tools, max_tokens,
                      extra_body=None,
                      max_attempts=8, base=2.0, cap=90.0, timeout=180):
    """One chat completion, retrying 429 and 5xx with exponential backoff.

    Defaults are deliberately patient. Free-tier catalog endpoints rate-limit
    on request count, so a sweep that gives up early reports a working model as
    "no data" -- which is honest but useless. Eight attempts with a 90 s cap and
    full jitter costs wall-clock time and buys complete rows.

    Returns (message, finish_reason, error_str). The error string names the
    attempt count so an exhausted retry budget is distinguishable from a
    first-try failure.
    """
    last = "unknown"
    attempts = 0
    for attempt in range(max_attempts):
        attempts = attempt + 1
        try:
            kwargs = dict(model=model, messages=messages,
                          max_tokens=max_tokens, temperature=0.0,
                          timeout=timeout)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            return choice.message, choice.finish_reason, None
        except Exception as e:                                  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            status = getattr(e, "status_code", None)
            text = str(e).lower()
            # A 400 is a malformed request: the schema, the model id, the
            # parameters. Retrying it just wastes the budget eight times over.
            if status == 400 or "invalid" in text and "400" in text:
                return None, None, f"{last} (not retryable)"
            retryable = status in (408, 409, 429, 500, 502, 503, 504) or \
                        "429" in text or "rate" in text or "timeout" in text
            if not retryable or attempt == max_attempts - 1:
                break
            asked = _retry_after(e)
            if asked is not None:
                sleep = min(cap, asked)
            else:
                sleep = random.uniform(0, min(cap, base * (2 ** attempt)))
            time.sleep(sleep)
    return None, None, f"{last} (after {attempts} attempts)"


# --------------------------------------------------------------------------
# One endpoint
# --------------------------------------------------------------------------

def resolve_model(client, configured: str | None, name: str) -> str | None:
    """Return a model id the endpoint will actually accept.

    A server started with --served-model-name advertises a short alias rather
    than the full Hugging Face path, so a hard-coded id 404s. Ask the endpoint
    what it serves and prefer that.
    """
    try:
        served = [m.id for m in client.models.list().data]
    except Exception:
        return configured                      # can't ask; try what we were given
    if not served:
        return configured
    if configured in served:
        return configured

    # Only auto-pick when there is genuinely no choice. A single-model server
    # (a local vLLM with --served-model-name) advertises one id and picking it
    # is right. A hosted catalogue advertises hundreds, and picking served[0]
    # would silently benchmark an arbitrary model under the configured name --
    # a wrong number is far worse than no number.
    if len(served) == 1:
        print(f"    note: '{configured}' is not served by {name}; "
              f"using '{served[0]}', the only model this endpoint offers")
        return served[0]

    near = [m for m in served if configured and configured.split("/")[-1] in m]
    hint = f" Closest matches: {', '.join(near[:3])}." if near else ""
    print(f"    '{configured}' is not offered by {name} "
          f"({len(served)} models available).{hint}")
    return None


def run_endpoint(spec: dict, examples: list[dict], max_tokens: int,
                 concurrency: int | None, verbose: bool = True):
    from openai import OpenAI

    name = spec["name"]
    key_env = spec.get("api_key_env")
    api_key = os.environ.get(key_env, "") if key_env else "not-needed"
    if key_env and not api_key:
        return None, f"{key_env} is not set"

    client = OpenAI(base_url=spec["base_url"], api_key=api_key or "not-needed")

    model_id = resolve_model(client, spec.get("model"), name)
    if not model_id:
        return None, "no model id: endpoint served none and none configured"
    spec = {**spec, "model": model_id}
    workers = concurrency if concurrency is not None else spec.get("concurrency", 2)

    # A reasoning model spends part of its budget thinking before it can emit a
    # tool call, and runs out sooner than a non-reasoning one on the same
    # number. Truncation scores identically to deciding not to call, so a
    # single global budget quietly understates recall for the models that
    # think. Let an endpoint raise its own.
    budget = spec.get("max_tokens", max_tokens)
    extra = spec.get("extra_body")
    # Free-tier endpoints queue. One endpoint measured 264 s for a 17-token
    # reply -- that is wait, not generation -- so a fixed 180 s timeout would
    # fail it on every example and report a working model as dead.
    per_req_timeout = spec.get("timeout", 180)
    attempts = spec.get("max_attempts", 8)

    def one(ex):
        # Everything here, not just the API call. build_messages, the tool
        # conversion and observe() all run per example and can raise on one
        # malformed record; uncaught, that exception surfaces at fut.result()
        # and takes down the whole endpoint, discarding every example that had
        # already completed and leaving no trace in `failed` either. One bad
        # example is worth one error row, not a lost run.
        try:
            msg, finish, err = call_with_backoff(
                client, model=spec["model"], messages=w2c.build_messages(ex),
                tools=w2c.to_openai_tools(ex["tools"]), max_tokens=budget,
                extra_body=extra, timeout=per_req_timeout,
                max_attempts=attempts)
            if err:
                return w2c.Record(label=ex["label"], called=False, tool_name=None,
                                  gold_tool=ex.get("gold_tool"), has_text=False,
                                  finish_reason=None, error=err)
            rec = w2c.observe(msg, finish)
            rec.label = ex["label"]
            rec.gold_tool = ex.get("gold_tool")
            return rec
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            return w2c.Record(label=ex["label"], called=False, tool_name=None,
                              gold_tool=ex.get("gold_tool"), has_text=False,
                              finish_reason=None,
                              error=f"{type(exc).__name__}: {exc}")

    t0 = time.perf_counter()
    records: list[w2c.Record] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, ex): ex for ex in examples}
        for i, fut in enumerate(as_completed(futures), 1):
            # `one` catches its own failures, so this is the last resort: a
            # worker killed by something outside it still costs one record
            # rather than the endpoint.
            try:
                records.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                ex = futures[fut]
                records.append(w2c.Record(
                    label=ex["label"], called=False, tool_name=None,
                    gold_tool=ex.get("gold_tool"), has_text=False,
                    finish_reason=None,
                    error=f"worker failed: {type(exc).__name__}: {exc}"))
            if verbose and i % 10 == 0:
                print(f"    {name}: {i}/{len(examples)}", flush=True)
    elapsed = time.perf_counter() - t0

    errs = [r for r in records if r.error]
    if len(errs) == len(records):
        return None, f"all {len(records)} requests failed — first: {errs[0].error[:120]}"

    summary = w2c.score(records)
    summary.update({"name": name, "model": spec["model"],
                    "location": spec.get("location", ""),
                    "elapsed_s": round(elapsed, 1),
                    "concurrency": workers,
                    "max_tokens": budget})
    return (summary, records), None


# --------------------------------------------------------------------------

def sweep(config=None, n_per_label=40, only=None, concurrency=None,
          max_tokens=3072, seed=0, out=None, save_raw=False, resume=False,
          verbose=True):
    """Run the sweep and return the summary dict, also writing it to `out`.

    Factored out of main() so a notebook can call it directly rather than
    shelling out. `main()` is a thin wrapper over this.

    Endpoints whose api_key_env is unset are skipped with a reason rather than
    scored as failures -- a missing credential is not a model behaviour.
    """
    config = Path(config or (HERE / "endpoints.json"))
    out = Path(out or (RESULTS / "summary.json"))

    specs = json.loads(config.read_text())["endpoints"]
    if only:
        only = [only] if isinstance(only, str) else list(only)
        specs = [s for s in specs if s["name"] in only]
    if not specs:
        raise ValueError(f"No endpoints selected from {config}")

    if verbose:
        print(f"Loading When2Call ({n_per_label} per label = "
              f"{3 * n_per_label} examples)...")
    examples = w2c.load_examples(n_per_label=n_per_label, seed=seed)
    if verbose:
        print(f"  {len(examples)} examples\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    # `--out` may point anywhere, including a directory that does not exist yet.
    # Creating it here rather than at write time means checkpoint() cannot lose
    # a completed endpoint to a missing parent after the requests are spent.
    out.parent.mkdir(parents=True, exist_ok=True)
    # `max_tokens_default` is the CLI fallback, not what anything actually ran
    # at -- endpoints.json overrides it per endpoint, and the budget each model
    # was really given is recorded on that model's own entry. Naming this field
    # "max_tokens" invited exactly the wrong reading: that every model shared
    # one budget, when in practice they did not and the comparison depends on
    # knowing which one had what.
    result = {"n_examples": len(examples), "max_tokens_default": max_tokens,
              "seed": seed, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "dataset": dict(w2c.LAST_DATASET),
              "models": [], "failed": []}

    # Resume: an endpoint already recorded in `out` is not re-run. A sweep can
    # take hours when a hosted endpoint is queueing, and losing four completed
    # models because the fifth timed out -- or because an SSH session dropped
    # -- is the expensive failure here, not the fifth model itself.
    if resume and out.exists():
        prior = json.loads(out.read_text())
        done = {m["name"] for m in prior.get("models", [])} | \
               {f["name"] for f in prior.get("failed", [])}
        if done:
            result["models"] = prior.get("models", [])
            result["failed"] = prior.get("failed", [])
            before = len(specs)
            specs = [s for s in specs if s["name"] not in done]
            if verbose:
                print(f"  resuming: {before - len(specs)} endpoint(s) already "
                      f"in {out.name}, {len(specs)} to go\n")

    def checkpoint():
        """Write what we have so far. Called after every endpoint, so an
        interrupted sweep still leaves usable results on disk."""
        out.write_text(json.dumps(result, indent=2))

    for spec in specs:
        if verbose:
            print(f"  {spec['name']} ({spec.get('location','')}) ...", flush=True)
        got, err = run_endpoint(spec, examples, max_tokens, concurrency,
                                verbose=verbose)
        if err:
            if verbose:
                print(f"    SKIPPED — {err}\n")
            result["failed"].append({"name": spec["name"], "reason": err})
            checkpoint()
            continue
        summary, records = got
        result["models"].append(summary)
        if verbose:
            print(w2c.summarise(spec["name"], summary))
        if save_raw:
            raw = RESULTS / f"raw-{spec['name']}.jsonl"
            raw.write_text("\n".join(json.dumps(asdict(r)) for r in records))
        checkpoint()

    checkpoint()
    if verbose:
        print(f"\nWrote {out}")
        if result["failed"]:
            print("\nEndpoints that produced no data:")
            for f in result["failed"]:
                print(f"  {f['name']}: {f['reason'][:100]}")
            print("\nA model with no data is reported as 'no data', never as a low score.")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "endpoints.json"))
    ap.add_argument("--n", type=int, default=40,
                    help="examples per label (3 labels, so total = 3n). Default 40 -> 120.")
    ap.add_argument("--only", action="append",
                    help="run only these endpoint names (repeatable)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="override per-endpoint concurrency")
    ap.add_argument("--max-tokens", type=int, default=3072,
                    help="Nemotron's median completion is ~768 tokens with a tail to 4096; "
                         "too low and it exhausts the budget while reasoning and never emits "
                         "a call, which scores identically to deciding not to call.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "summary.json"))
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip endpoints already present in --out. Results are "
                         "checkpointed after every endpoint, so an interrupted "
                         "sweep picks up where it stopped.")
    args = ap.parse_args()

    try:
        sweep(config=args.config, n_per_label=args.n, only=args.only,
              concurrency=args.concurrency, max_tokens=args.max_tokens,
              seed=args.seed, out=args.out, save_raw=args.save_raw,
              resume=args.resume)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
