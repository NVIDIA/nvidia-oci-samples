# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate the Nemotron 3.5 Lightning endpoint: models, chat, structured tool call, streaming, reasoning field.

Run with a port-forward open: kubectl -n lightning port-forward svc/lightning-router-service 8000:80
Override the base URL with ENDPOINT (default http://127.0.0.1:8000/v1).
"""

import json
import os
import sys
import time

import requests

BASE = os.environ.get("ENDPOINT", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record a failed expectation; the script exits nonzero if any check fails."""
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL: {message}")


def post(payload: dict, **kw):
    return requests.post(f"{BASE}/chat/completions", json={"model": MODEL, **payload}, timeout=300, **kw)


models = requests.get(f"{BASE}/models", timeout=30).json()
served = [m["id"] for m in models["data"]]
print("models:", served)
check(MODEL in served, f"{MODEL} is not served")

t0 = time.time()
r = post({"max_tokens": 200, "temperature": 0.0, **NO_THINK,
          "messages": [{"role": "system", "content": "Be terse."},
                       {"role": "user", "content": "Reply exactly: LIGHTNING-OK"}]}).json()
msg = r["choices"][0]["message"]
usage = {k: r["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
print(f"chat ({time.time() - t0:.1f}s): content={msg.get('content')!r} finish={r['choices'][0]['finish_reason']} usage={usage}")
check("LIGHTNING-OK" in (msg.get("content") or ""), "chat reply does not contain LIGHTNING-OK")

t0 = time.time()
r = post({"max_tokens": 300, "temperature": 0.0, **NO_THINK,
          "messages": [{"role": "user", "content": "What is the weather in Phoenix right now? Use the tool."}],
          "tools": [{"type": "function", "function": {
              "name": "get_weather", "description": "Current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}).json()
ch = r["choices"][0]
print(f"tools ({time.time() - t0:.1f}s): finish={ch['finish_reason']} tool_calls={json.dumps(ch['message'].get('tool_calls'))}")
tool_calls = ch["message"].get("tool_calls") or []
check(any(c["function"]["name"] == "get_weather" for c in tool_calls), "no structured get_weather tool call returned")

t0 = time.time()
chunks, text = 0, ""
with post({"max_tokens": 80, "temperature": 0.0, "stream": True, **NO_THINK,
           "messages": [{"role": "user", "content": "Count from 1 to 10, comma separated."}]}, stream=True) as s:
    for line in s.iter_lines():
        if line and line.startswith(b"data: ") and line != b"data: [DONE]":
            chunks += 1
            text += json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
print(f"stream ({time.time() - t0:.1f}s): {chunks} chunks, text={text.strip()!r}")
check(chunks > 1 and "1" in text and "10" in text, "streamed response is incomplete")

t0 = time.time()
r = post({"max_tokens": 600, "temperature": 0.0,
          "messages": [{"role": "user", "content": "Is 91 prime? Answer yes or no with one sentence of justification."}]}).json()
msg = r["choices"][0]["message"]
reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
usage = {k: r["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
print(f"reasoning-on ({time.time() - t0:.1f}s): finish={r['choices'][0]['finish_reason']} reasoning={len(reasoning)} chars "
      f"content={str(msg.get('content'))[:80]!r} usage={usage}")
check(len(reasoning) > 0, "no reasoning trace returned with thinking enabled")

if FAILURES:
    print(f"\nVALIDATION FAILED: {len(FAILURES)} check(s) failed")
    sys.exit(1)
print("\nVALIDATION PASSED")
