# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional: record two endpoint calls as an ATIF trajectory with NVIDIA NeMo Relay.

Relay recognizes the OpenAI chat payload shape automatically, so the serving stack is untouched.
Requires `pip install nemo-relay>=0.8.4 requests` and a port-forward as for validate.py.
"""

import asyncio
import json
import os

import requests

import nemo_relay

ENDPOINT = os.environ.get("ENDPOINT", "http://127.0.0.1:8000/v1") + "/chat/completions"
MODEL = os.environ.get("MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def post_vllm(payload: dict) -> dict:
    r = requests.post(ENDPOINT, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


async def main() -> None:
    exporter = nemo_relay.AtifExporter("oke-lightning", "lightning-probe", "1.0.0", model_name=MODEL)
    exporter.register("atif_lightning")

    async def call_vllm(request: nemo_relay.LLMRequest):
        return await asyncio.to_thread(post_vllm, request.content)

    chat = {"model": MODEL, "max_tokens": 200, "temperature": 0.0, **NO_THINK,
            "messages": [{"role": "system", "content": "Be terse."},
                         {"role": "user", "content": "Reply exactly: SELF-HOSTED-OK"}]}
    tools = {"model": MODEL, "max_tokens": 300, "temperature": 0.0, **NO_THINK,
             "messages": [{"role": "user", "content": "Call the get_utc_time tool to find the current UTC time, then report it."}],
             "tools": [{"type": "function", "function": {"name": "get_utc_time", "description": "Return the current UTC time",
                        "parameters": {"type": "object", "properties": {}, "required": []}}}]}

    with nemo_relay.scope.scope("lightning-probe", nemo_relay.ScopeType.Agent):
        result = await nemo_relay.llm.execute(MODEL, nemo_relay.LLMRequest({}, chat), call_vllm)
        print("reply:", json.dumps(result["choices"][0]["message"]["content"]))
        tool_result = await nemo_relay.llm.execute(MODEL, nemo_relay.LLMRequest({}, tools), call_vllm)
        print("tool finish_reason:", tool_result["choices"][0]["finish_reason"])
        print("tool_calls:", json.dumps(tool_result["choices"][0]["message"].get("tool_calls")))

    trajectory = exporter.export()
    out = os.environ.get("TRAJECTORY_OUT", "relay-trajectory.json")
    with open(out, "w") as f:
        json.dump(trajectory, f, indent=2)
    print("final_metrics:", json.dumps(trajectory.get("final_metrics")))
    print(f"trajectory written to {out}")


asyncio.run(main())
