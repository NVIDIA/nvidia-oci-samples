# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Scoring for the nvidia/When2Call agentic tool-calling benchmark.

When2Call asks a harder question than most function-calling benchmarks: not
"was the tool call correct" but "should the model have called anything at all".
Three labels:

    tool_call         every required argument is available -> call the tool
    request_for_info  a required argument was withheld     -> ask a follow-up
    cannot_answer     no available tool covers the request -> decline

Scoring is deterministic. We observe what the server actually did -- no LLM
judge -- so the numbers are reproducible on your own hardware.

Metrics:
    decision accuracy             did it call exactly when it should have?
    actionable decision accuracy  correct decision AND it either called a tool
                                  or produced real text. Guards against a model
                                  scoring well by saying nothing.
    tool-selection accuracy       when it correctly called, was it the right tool?
    over-call rate                how often it called when it should not have.
                                  Lower is better. This is the number that
                                  predicts agent misbehaviour in production.

Dataset: https://huggingface.co/datasets/nvidia/When2Call (CC-BY-4.0)
"""

from __future__ import annotations

import json
import re
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any

LABELS = ("tool_call", "request_for_info", "cannot_answer")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the tools listed below. "
    "Use a tool only when the request needs it and every required argument is "
    "available. If a required argument is missing, ask the user for it. If no "
    "available tool covers the request, say so plainly. Always reply with "
    "something useful -- never return an empty response."
)


# --------------------------------------------------------------------------
# Wilson score interval
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point, lo, hi) as proportions. Wilson interval, not normal
    approximation -- at n=120 the difference matters."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

# OpenAI's function-name grammar. The dataset carries names straight from BFCL
# like "api_token_api.APITokenApi.get_api_tokens" and "cmd_controller.exe",
# which contain dots and fail validation on stricter gateways:
#     Validation: Function at index 0 has an invalid name: "cmd_controller.exe"
# vLLM accepts them, so the local endpoint scores fine while hosted ones reject
# every request -- an apples-to-oranges comparison hiding as a model result.
_NAME_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# JSON Schema type names, versus the Python type names the dataset sometimes
# uses. Strict validators reject the latter:
#     Tool 0 function has invalid 'parameters' schema: 'dict' is not valid
_TYPE_FIXES = {"dict": "object", "list": "array", "tuple": "array",
               "str": "string", "int": "integer", "float": "number",
               "double": "number", "bool": "boolean", "none": "null",
               "any": "string"}


def sanitise_name(name: str | None) -> str | None:
    """Make a function name satisfy ^[A-Za-z0-9_-]{1,64}$, or return it as-is.

    Applied to both the tools we send and the gold tool we score against, so
    the two still compare equal.
    """
    if not name:
        return None
    if _NAME_OK.match(name):
        return name
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or None


def normalise_schema(node: Any) -> Any:
    """Recursively map Python type names onto JSON Schema type names."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = _TYPE_FIXES.get(v.strip().lower(), v)
            elif k == "type" and isinstance(v, list):
                out[k] = [_TYPE_FIXES.get(str(t).strip().lower(), t) for t in v]
            else:
                out[k] = normalise_schema(v)
        return out
    if isinstance(node, list):
        return [normalise_schema(v) for v in node]
    return node


def tool_name_of(value: Any) -> str | None:
    """Pull a bare function name out of whatever the dataset stored.

    `target_tool` follows the same convention as `tools`: it may be a bare
    name, a JSON string of the whole function spec, or a dict. Comparing the
    raw value against the name a model returned gives 0% tool selection for
    every model, which reads as a finding and is not one.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return (value.get("name")
                or (value.get("function") or {}).get("name"))
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("{"):
            try:
                d = json.loads(v)
            except json.JSONDecodeError:
                return v or None
            return (d.get("name") or (d.get("function") or {}).get("name") or None)
        return v or None
    return None


def parse_tools(tools: Any) -> list[dict] | None:
    """Normalise the dataset's tool spec to a list of dicts.

    Three shapes have appeared across revisions and all three are accepted:
      * a list of JSON strings, one per function   <- current
      * a JSON string containing a list
      * a plain list of dicts
    Returns None when nothing usable comes out, so the caller can drop the row
    rather than send the model an empty toolset -- with no tools to call, every
    example would look like 'cannot_answer' regardless of the model.
    """
    if tools is None:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None
    if not isinstance(tools, (list, tuple)):
        return None
    out: list[dict] = []
    for t in tools:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except json.JSONDecodeError:
                continue
        if isinstance(t, dict):
            out.append(t)
    return out or None


def normalise_row(row: dict) -> dict | None:
    """One dataset row -> the shape the harness uses, or None if unusable.

    Field names have moved across revisions, so every known spelling is tried.
    The current test/mcq and test/llm_judge splits carry the class in
    `correct_answer`, the prompt in `question`, and the gold tool in
    `target_tool`; earlier revisions used `label`, `query` and `gold_tool`.
    """
    label = (row.get("label") or row.get("correct_answer")
             or row.get("answer") or row.get("category"))
    query = (row.get("query") or row.get("question")
             or row.get("user_query"))
    gold = tool_name_of(row.get("gold_tool") or row.get("target_tool")
                        or row.get("target_function") or row.get("function_name"))
    tools = parse_tools(row.get("tools") or row.get("functions")
                        or row.get("available_tools"))

    if label not in LABELS or not query or not tools:
        return None

    # Repair the tool specs once, here, so every endpoint is sent the same
    # valid schemas and the gold name still matches what a model will return.
    repaired = []
    for t in tools:
        t = normalise_schema(dict(t))
        if "function" in t and isinstance(t["function"], dict):
            t["function"]["name"] = sanitise_name(t["function"].get("name"))
        elif t.get("name"):
            t["name"] = sanitise_name(t["name"])
        repaired.append(t)

    return {"label": label, "tools": repaired, "query": query,
            "gold_tool": sanitise_name(gold)}


# Filled in by load_examples() so a run can record exactly what it scored.
# The Hub layout has already changed once under this sample -- what was config
# "mcq" with a "test" split is now a config named "test" -- and accuracy numbers
# do not carry across that any more than they carry across a weights change.
LAST_DATASET: dict = {}


def load_examples(n_per_label: int = 40, seed: int = 0,
                  split: str | None = None, config: str | None = None,
                  verbose: bool = True) -> list[dict]:
    """Load When2Call, stratified equally across the three labels.

    Both the Hub layout and the field names have moved between revisions, so
    resolve the config and split from what the dataset actually exposes rather
    than hard-coding names, then normalise fields -- and say out loud which
    config and split were used, because that is part of the result.
    """
    from datasets import (load_dataset, get_dataset_config_names,
                          get_dataset_split_names)

    repo = "nvidia/When2Call"
    available = get_dataset_config_names(repo)
    if not available:
        raise RuntimeError(f"{repo} exposes no configs")

    # Caller's choice first, then the historical name, then anything that is
    # not obviously a training split.
    order = [c for c in (config, "mcq", "test") if c and c in available]
    order += [c for c in available if c not in order and "train" not in c]
    order += [c for c in available if c not in order]

    ds = chosen = None
    last_err: Exception | None = None
    for cfg in order:
        try:
            splits = get_dataset_split_names(repo, cfg)
        except Exception as e:                                  # noqa: BLE001
            last_err = e
            continue
        preferred = [s for s in (split, "mcq", "test", "validation", "train")
                     if s and s in splits]
        for sp in preferred + [s for s in splits if s not in preferred]:
            try:
                ds = load_dataset(repo, cfg, split=sp)
                chosen = (cfg, sp)
                break
            except Exception as e:                              # noqa: BLE001
                last_err = e
        if ds is not None:
            break

    if ds is None:
        raise RuntimeError(
            f"Could not load {repo}. Configs offered: {available}. "
            f"Last error: {last_err}")

    rows = [r for r in (normalise_row(dict(x)) for x in ds) if r]
    if not rows:
        cols = list(ds.features) if hasattr(ds, "features") else "unknown"
        raise RuntimeError(
            f"Loaded {repo} config={chosen[0]!r} split={chosen[1]!r} "
            f"({len(ds)} rows) but none survived normalisation. Columns are "
            f"{cols}; extend normalise_row() in when2call.py to match.")

    counts = {lab: sum(1 for r in rows if r["label"] == lab) for lab in LABELS}
    LAST_DATASET.clear()
    LAST_DATASET.update({"repo": repo, "config": chosen[0], "split": chosen[1],
                         "usable_rows": len(rows), "by_label": counts})
    if verbose:
        print(f"  {repo} config={chosen[0]!r} split={chosen[1]!r} — "
              f"{len(rows)} usable rows {counts}")

    short = [f"{lab} has {n}" for lab, n in counts.items() if n < n_per_label]
    if short and verbose:
        print(f"  NOTE: fewer rows than requested for {'; '.join(short)}. "
              f"Strata will be uneven, so read the counts, not just the rates.")

    rng = random.Random(seed)
    out: list[dict] = []
    for lab in LABELS:
        pool = [r for r in rows if r["label"] == lab]
        rng.shuffle(pool)
        out.extend(pool[:n_per_label])
    rng.shuffle(out)
    return out


def to_openai_tools(tools: Any) -> list[dict]:
    """Convert the dataset's tool spec into OpenAI tools format."""
    converted = []
    for t in tools or []:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except json.JSONDecodeError:
                continue
        if t.get("type") == "function" and "function" in t:
            converted.append(t)
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": t.get("name", "unknown"),
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or t.get("input_schema")
                              or {"type": "object", "properties": {}},
            },
        })
    return converted


def build_messages(example: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["query"]}]


# --------------------------------------------------------------------------
# Observing what the server did
# --------------------------------------------------------------------------

@dataclass
class Record:
    label: str
    called: bool
    tool_name: str | None
    gold_tool: str | None
    has_text: bool
    finish_reason: str | None
    reasoned: bool = False
    error: str | None = None

    @property
    def predicted(self) -> str:
        return "tool_call" if self.called else "no_call"

    @property
    def decision_correct(self) -> bool:
        want_call = self.label == "tool_call"
        return self.called == want_call

    @property
    def actionable(self) -> bool:
        """Correct decision AND it actually did something."""
        return self.decision_correct and (self.called or self.has_text)


def observe(message: Any, finish_reason: str | None = None) -> Record:
    """Turn one chat completion message into a Record. Label is attached by
    the caller; this only reads what the server returned."""
    tool_calls = getattr(message, "tool_calls", None) or []
    content = (getattr(message, "content", None) or "").strip()

    # Reasoning models return their thinking on a separate field -- `reasoning`
    # on vLLM 0.26+, `reasoning_content` on the NVIDIA API catalog and older
    # builds. It is not an answer, so it must not count as one for
    # actionable accuracy. But a model that produced a long trace and no
    # content did not "stay silent" either; it ran out of budget mid-thought.
    # Record that separately so the two are distinguishable in the raw data.
    reasoning = (getattr(message, "reasoning", None)
                 or getattr(message, "reasoning_content", None) or "").strip()

    name = tool_calls[0].function.name if tool_calls else None
    return Record(label="", called=bool(tool_calls), tool_name=name,
                  gold_tool=None, has_text=bool(content),
                  reasoned=bool(reasoning),
                  finish_reason=finish_reason)


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------

def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value for two proportions differing, pooled z-test.

    Overlapping confidence intervals are the usual eyeball test and they are
    *conservative*: two intervals can overlap while the difference is real. So
    do not decide "tie" from the picture alone.
    """
    if not n1 or not n2:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def holm_pairwise(models: list[dict], metric: str, alpha: float = 0.05) -> list[dict]:
    """Every pairwise comparison on one metric, Holm-Bonferroni corrected.

    Five models is ten comparisons per metric. Testing all ten at 0.05 would
    expect about one false positive from noise alone, so an uncorrected
    "significant" here is worth very little. Holm controls that without being
    as brutal as plain Bonferroni.

    Returns the comparisons sorted by p, each with the threshold it had to beat
    and whether it did.
    """
    pairs = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            ka, na = a[metric]["k"], a[metric]["n"]
            kb, nb = b[metric]["k"], b[metric]["n"]
            pairs.append({"a": a["name"], "b": b["name"],
                          "p": two_proportion_p(ka, na, kb, nb)})
    pairs.sort(key=lambda x: x["p"])
    m = len(pairs)
    # Holm is a step-down procedure, and the stepping down is the whole point.
    # Walk the pairs in ascending p and stop at the first one that fails its
    # threshold: everything after it is non-significant regardless of its own
    # threshold. Testing each pair independently against alpha/(m-i) is not
    # Holm -- the thresholds grow as you descend, so a later pair with a larger
    # p can clear a larger threshold and be called significant after an earlier,
    # stronger comparison has already failed. That inflates the family-wise
    # error rate this correction exists to control.
    still_rejecting = True
    for i, pair in enumerate(pairs):
        pair["threshold"] = alpha / (m - i)
        if still_rejecting and pair["p"] < pair["threshold"]:
            pair["significant"] = True
        else:
            still_rejecting = False
            pair["significant"] = False
    return pairs


def score(records: list[Record]) -> dict:
    ok = [r for r in records if r.error is None]
    n = len(ok)

    dec_k = sum(r.decision_correct for r in ok)
    act_k = sum(r.actionable for r in ok)

    # tool selection: only cases it correctly decided to call are eligible
    sel_pool = [r for r in ok if r.label == "tool_call" and r.called and r.gold_tool]
    sel_k = sum(r.tool_name == r.gold_tool for r in sel_pool)

    # over-call: fired a tool when it should not have
    over_pool = [r for r in ok if r.label != "tool_call"]
    over_k = sum(r.called for r in over_pool)

    def block(k, m):
        p, lo, hi = wilson(k, m)
        return {"k": k, "n": m, "rate": p, "lo": lo, "hi": hi}

    per_label = {}
    for lab in LABELS:
        pool = [r for r in ok if r.label == lab]
        per_label[lab] = block(sum(r.decision_correct for r in pool), len(pool))

    return {
        "n_scored": n,
        "n_errors": len(records) - n,
        "decision_accuracy": block(dec_k, n),
        "actionable_decision_accuracy": block(act_k, n),
        "tool_selection_accuracy": block(sel_k, len(sel_pool)),
        "over_call_rate": block(over_k, len(over_pool)),
        "per_label": per_label,
    }


def summarise(name: str, s: dict) -> str:
    d, a = s["decision_accuracy"], s["actionable_decision_accuracy"]
    t, o = s["tool_selection_accuracy"], s["over_call_rate"]
    return (
        f"  {name}\n"
        f"    decision accuracy      {d['k']:>3}/{d['n']:<3} {pct(d['rate']):>7}"
        f"   [{pct(d['lo'])} – {pct(d['hi'])}]\n"
        f"    actionable             {a['k']:>3}/{a['n']:<3} {pct(a['rate']):>7}\n"
        f"    tool selection         {t['k']:>3}/{t['n']:<3} {pct(t['rate']):>7}\n"
        f"    over-call rate         {o['k']:>3}/{o['n']:<3} {pct(o['rate']):>7}"
        f"   (lower is better)\n"
    )


def overlaps(a: dict, b: dict) -> bool:
    """True if two Wilson intervals overlap -- i.e. the difference is not
    separable at this sample size. Used to state ties honestly."""
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])
