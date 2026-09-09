#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Measure end-to-end latency to every endpoint in endpoints.json.

    ./latency.py                    # all endpoints, 5 samples each
    ./latency.py --only spark       # just the local one, no key needed
    ./latency.py --samples 11       # tighter medians, longer run
    ./latency.py --resume --deadline 240    # retry only what was queued, patiently

What this measures, precisely
-----------------------------
**Time to first token** and **decode rate**, one request at a time, from the
machine you run it on. Not throughput: concurrency is 1 throughout, because a
single stream is what an interactive tool feels like.

Those two metrics are chosen because they compare across models fairly and
total wall time does not. Total depends on how many tokens a model decided to
emit, and a reasoning model that thinks for 900 tokens is not "slower" than one
that answers in 40 -- it did more work. TTFT is unaffected by output length,
and decode rate is per-token by construction.

What it does NOT isolate
------------------------
For a hosted endpoint this is **end-to-end from where you are sitting**: your
network path, the gateway's routing and admission control, any queueing behind
other tenants, and only then the model. Those cannot be separated from outside,
so this script does not pretend to. A hosted figure here is what a developer on
this network would actually experience at this time of day -- which is a useful
number, and an honest one, as long as it is not relabelled "model speed".

Two consequences worth taking seriously before quoting a result:

  * **It is a snapshot.** Shared free-tier gateways vary by hour. Re-run it
    close to when the number will be used, and record the timestamp -- which
    is why results/latency.json carries one.

  * **A slow third-party model here is probably a statement about the free
    gateway, not about the model.** The pair worth trusting is the same model
    served in two places, because everything except the serving path is held
    constant. Treat the rest as context.

The local endpoint needs no API key. Hosted endpoints read NVIDIA_API_KEY from
the environment, and are skipped with a reason if it is not set.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# Short, fixed, and deliberately dull. A prompt that invites a long answer would
# make total time depend on the model's verbosity rather than the serving path,
# and we cap output anyway.
PROMPT = "In one sentence, what is a GPU?"
MAX_TOKENS = 128



def _is_timeout(e: BaseException) -> bool:
    """Did this exception mean "it did not answer in time"?

    Checked by name rather than by class. httpx raises ReadTimeout, the OpenAI
    SDK wraps it as APITimeoutError, and neither derives from the builtin
    TimeoutError -- so `except TimeoutError` silently misses both, which is how
    a 60s deadline turned into a six-minute wait.
    """
    return isinstance(e, TimeoutError) or "timeout" in type(e).__name__.lower()


def _is_stream_options_error(e: BaseException) -> bool:
    """Did this exception mean "I do not accept stream_options"?

    Matched by name and message rather than by class, for the same reason
    _is_timeout is: the SDK's exception types are not the builtins', and a
    gateway in front of the model may surface a rejected parameter as any of
    several of them. Both halves of the test matter. The first narrows it to
    the client's own request being wrong; the message is then what
    distinguishes *this* parameter from every other 400 -- a bad key, a
    malformed body, a model id that does not exist. Retrying one of those
    without usage would often succeed, and the sample would be recorded as
    though the first attempt had never failed.
    """
    name = type(e).__name__.lower()
    status = getattr(e, "status_code", None)
    if not ("badrequest" in name or "unprocessable" in name
            or status in (400, 422)):
        return False
    msg = str(e).lower()
    return "stream_options" in msg or "include_usage" in msg


def one_request(client, model: str, extra_body: dict | None, timeout: int,
                deadline: float = 60.0):
    """Stream one completion.

    Returns (ttft_s, decode_tok_s, n_tokens, total_s, exact) or None.

    `exact` is False when the token count had to be inferred from the number of
    streamed chunks. That distinction matters more than it looks:

    Counting chunks and calling them tokens is correct against local vLLM, which
    emits one token per chunk. It is wrong against a gateway that packs several
    tokens into a chunk or flushes a burst at the end -- and it fails *upward*,
    because the burst lands in a fraction of a second. An early version of this
    script reported 1303 tok/s single-stream for a hosted model, which is not a
    fast model, it is a broken measurement that looks like a fast model.

    So ask the server for its own count via stream_options.include_usage, and
    use that. Endpoints that reject the parameter fall back to chunk counting
    and are marked inexact, so the caller can refuse to report the number rather
    than quietly publishing it.
    """
    def _run(with_usage: bool):
        t0 = time.perf_counter()
        ttft, n_chunks, reported = None, 0, None
        kwargs = dict(
            model=model, messages=[{"role": "user", "content": PROMPT}],
            max_tokens=MAX_TOKENS, temperature=0.0, stream=True, timeout=timeout,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        if with_usage:
            kwargs["stream_options"] = {"include_usage": True}
        for chunk in client.chat.completions.create(**kwargs):
            # Second line of defence only. This fires when chunks keep arriving
            # but the response never ends -- a gateway sending SSE keepalives
            # while a request is queued. It cannot help while we are still
            # waiting for the *first* chunk, because create() blocks and this
            # loop body has not run yet. That case is handled by giving the HTTP
            # client a read timeout equal to the deadline, below.
            if time.perf_counter() - t0 > deadline:
                raise TimeoutError(f"no completion within {deadline:.0f}s")
            usage = getattr(chunk, "usage", None)
            if usage is not None and getattr(usage, "completion_tokens", None):
                reported = usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Reasoning tokens count. They are tokens the server generated and
            # streamed; excluding them would understate a thinking model's
            # decode rate, which is the opposite of what this measures.
            if (getattr(delta, "content", None)
                    or getattr(delta, "reasoning", None)
                    or getattr(delta, "reasoning_content", None)):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_chunks += 1
        return ttft, n_chunks, reported, time.perf_counter() - t0

    try:
        ttft, n_chunks, reported, total = _run(with_usage=True)
    except Exception as e:                                      # noqa: BLE001
        # This retry exists for exactly one condition: a gateway that rejects
        # stream_options. Everything else must propagate and be counted as a
        # failed sample by probe().
        #
        # A timeout must not fall through. The endpoint is queued, not fussy
        # about stream_options, and retrying would spend a second full deadline
        # learning the same thing. That check is redundant against the one
        # below and is kept anyway, because it is the case a reader comes here
        # looking for and it should not depend on a message match.
        #
        # Nor must anything else. An auth failure, a rate limit or a 5xx
        # retried here can succeed on the second attempt and be recorded as a
        # clean sample -- taken under conditions the other samples were not,
        # with the first failure invisible -- and, because the retry cannot ask
        # for usage, it marks the whole endpoint decode_exact: false for a
        # reason that has nothing to do with token counting.
        if _is_timeout(e) or not _is_stream_options_error(e):
            raise
        ttft, n_chunks, reported, total = _run(with_usage=False)

    n = reported if reported else n_chunks
    if ttft is None or n < 2 or total <= ttft:
        return None
    return ttft, (n - 1) / (total - ttft), n, total, bool(reported)


def probe(spec: dict, samples: int, deadline: float = 60.0,
          verbose: bool = True) -> dict | None:
    from openai import OpenAI

    name = spec["name"]
    key_env = spec.get("api_key_env")
    if key_env and not os.environ.get(key_env):
        return {"name": name, "skipped": f"{key_env} is not set"}

    # The HTTP read timeout is what actually bounds a silently queued endpoint.
    # create() blocks until the first byte of the response, so no amount of
    # in-loop checking helps there; httpx raising ReadTimeout is what rescues us.
    # One endpoint sat 5.6 minutes before its first token with the default 120s
    # in place, which is how this was found.
    #
    # max_retries=0 matters just as much. The SDK retries a timed-out request
    # twice by default, so a 60s deadline would quietly cost 180s per sample.
    client = OpenAI(base_url=spec["base_url"],
                    api_key=os.environ.get(key_env, "not-needed") if key_env else "not-needed",
                    timeout=deadline, max_retries=0)
    model = spec["model"]
    extra = spec.get("extra_body")

    # One throwaway request first. A cold connection pays TLS setup, and a cold
    # server pays cache and graph warmup; neither is what anyone means by
    # latency, and both land entirely on the first sample.
    try:
        one_request(client, model, extra, deadline, deadline)
    except Exception as e:                                      # noqa: BLE001
        # A read timeout and our own wall clock mean the same thing here: the
        # endpoint did not answer in the time we were willing to wait. Report it
        # as queued rather than as a very large latency, because that is what it
        # is -- a fact about the gateway, not a measurement of the model.
        if _is_timeout(e):
            return {"name": name,
                    "skipped": f"queued longer than {deadline:.0f}s — not measurable right now"}
        return {"name": name, "skipped": f"{type(e).__name__}: {str(e)[:90]}"}

    ttfts, decodes, errors, exact_all = [], [], 0, True
    for i in range(samples):
        try:
            got = one_request(client, model, extra, deadline, deadline)
        except Exception as e:                                  # noqa: BLE001
            errors += 1
            if verbose:
                print(f"    {name}: sample {i+1} failed — {type(e).__name__}", flush=True)
            continue
        if got is None:
            errors += 1
            continue
        ttfts.append(got[0] * 1000)
        decodes.append(got[1])
        exact_all &= got[4]
        if verbose:
            mark = "" if got[4] else "  (token count inferred)"
            print(f"    {name}: {i+1}/{samples}  ttft {got[0]*1000:6.0f} ms  "
                  f"decode {got[1]:5.1f} tok/s{mark}", flush=True)

    if not ttfts:
        return {"name": name, "skipped": f"all {samples} samples failed"}

    q = lambda xs: {"median": round(statistics.median(xs), 1),
                    "min": round(min(xs), 1), "max": round(max(xs), 1)}
    # The acquisition settings are recorded on the row, not only in the file
    # header. --resume deliberately carries across endpoints measured under
    # *different* settings -- giving one queued endpoint a longer deadline
    # while keeping the healthy measurements is the whole reason the flag
    # exists -- and a header written by the current invocation then describes
    # rows that were never taken that way. A chart captioned from the header
    # would state eleven samples for a row measured with five: a number that
    # looks like a measurement and is not, which is the same failure as the
    # 1303 tok/s above. A row that carries its own settings cannot lie about
    # them however it is later merged.
    return {"name": name, "location": spec.get("location", ""), "model": model,
            "samples": len(ttfts), "errors": errors,
            "samples_requested": samples, "deadline_s": deadline,
            "max_tokens": MAX_TOKENS,
            "ttft_ms": q(ttfts), "decode_tok_s": q(decodes),
            "decode_exact": exact_all}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "endpoints.json"))
    ap.add_argument("--only", action="append", help="endpoint name; repeatable")
    ap.add_argument("--samples", type=int, default=5,
                    help="sequential samples per endpoint, after one discarded "
                         "warmup. Five is enough for a median when the finding is "
                         "an order of magnitude; raise it to tighten one that is not.")
    ap.add_argument("--deadline", type=float, default=60.0,
                    help="seconds to wait for one completion before giving up on a "
                         "sample. A queued free-tier endpoint will otherwise hold the "
                         "connection open indefinitely.")
    ap.add_argument("--out", default=str(RESULTS / "latency.json"))
    ap.add_argument("--resume", action="store_true",
                    help="keep endpoints already measured in --out and retry only the "
                         "ones that were skipped. Lets a queued endpoint be given a "
                         "longer deadline without re-measuring the healthy ones.")
    args = ap.parse_args()

    specs = json.loads(Path(args.config).read_text())["endpoints"]
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]
        if not specs:
            print(f"no endpoint matched {args.only}", file=sys.stderr)
            return 2

    print(f"Latency probe — {args.samples} sequential samples, max_tokens "
          f"{MAX_TOKENS}, concurrency 1, {args.deadline:.0f}s deadline\n")
    # These header fields describe *this* invocation. Under --resume the file
    # can also hold rows measured under other settings, so the header is not a
    # safe caption for the figure -- each row carries its own copy, and
    # make_latency_chart.py reads those. The header is kept because a
    # latency.json written before per-row settings existed has nothing else,
    # and because "what was the last run" is worth recording on its own.
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "prompt": PROMPT, "max_tokens": MAX_TOKENS,
           "samples_requested": args.samples, "deadline_s": args.deadline,
           "endpoints": []}

    # Write after every endpoint, not once at the end. A probe against shared
    # gateways is exactly the thing you interrupt -- and losing four good
    # measurements because the fifth was queueing is the expensive failure here,
    # not the fifth endpoint. run_benchmark.py checkpoints for the same reason.
    outfile = Path(args.out)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # --resume keeps whatever already measured cleanly and retries only the rest.
    # A shared gateway that was queueing at one hour may not be at the next, and
    # re-measuring the healthy endpoints to find that out both wastes minutes and
    # replaces good samples with ones taken under different conditions.
    keep = {}
    if args.resume and outfile.exists():
        prior = json.loads(outfile.read_text())
        prior_endpoints = prior.get("endpoints", [])
        keep = {r["name"]: r for r in prior_endpoints if not r.get("skipped")}
        # Endpoints the current selection does not cover have to be carried
        # across explicitly, because save() rewrites the whole file. The
        # documented way to use --resume is with --only, to give one queued
        # endpoint a longer deadline:
        #
        #     ./latency.py --resume --only gemma-4-31b --deadline 300
        #
        # With --only, `specs` holds one endpoint, so without this the file is
        # rewritten containing that endpoint alone and the four healthy
        # measurements the flag exists to preserve are deleted -- by the run
        # that was meant to protect them.
        selected = {s["name"] for s in specs}
        carried = [r for r in prior_endpoints if r["name"] not in selected]
        out["endpoints"].extend(carried)
        if carried:
            print(f"  resuming — carrying {', '.join(r['name'] for r in carried)} "
                  f"from the previous run")
        if keep:
            print(f"  resuming — keeping {', '.join(keep)}\n")

    def save():
        outfile.write_text(json.dumps(out, indent=2) + "\n")

    for spec in specs:
        if spec["name"] in keep:
            out["endpoints"].append(keep[spec["name"]])
            print(f"  {spec['name']} — kept from the previous run")
            save()
            continue
        print(f"  {spec['name']} ...", flush=True)
        try:
            out["endpoints"].append(probe(spec, args.samples, args.deadline))
        except KeyboardInterrupt:
            print(f"\n  interrupted during {spec['name']} — keeping "
                  f"{len(out['endpoints'])} completed endpoint(s)")
            out["interrupted_during"] = spec["name"]
            save()
            break
        save()

    if not out["endpoints"]:
        print("\n  nothing measured.")
        return 1
    print(f"\n  {'endpoint':<18}{'TTFT median':>14}{'range':>18}{'decode':>16}")
    print("  " + "-" * 66)
    for r in out["endpoints"]:
        if r.get("skipped"):
            print(f"  {r['name']:<18}  skipped — {r['skipped'][:44]}")
            continue
        t, d = r["ttft_ms"], r["decode_tok_s"]
        rng = f"{t['min']:.0f}–{t['max']:.0f} ms"
        mark = "" if r.get("decode_exact", True) else " ~"
        print(f"  {r['name']:<18}{t['median']:>11.0f} ms{rng:>18}"
              f"{d['median']:>11.1f} tok/s{mark}")

    if any(not r.get("decode_exact", True) for r in out["endpoints"] if not r.get("skipped")):
        print("\n  ~ decode rate inferred from stream chunks because the endpoint did not\n"
              "    return a token count. Treat those as indicative only; a gateway that\n"
              "    batches chunks inflates this figure.")

    save()
    print(f"\nWrote {args.out}")
    print("\n  Hosted figures are end-to-end from this machine and include network and\n"
          "  gateway queueing, which cannot be separated from model time. The pair worth\n"
          "  trusting is the same model in two places; read the rest as context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
