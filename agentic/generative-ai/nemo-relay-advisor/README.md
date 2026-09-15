<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Agent observability on OCI Generative AI with NeMo Relay

This sample runs a small LangChain agent on [OCI Generative AI](https://www.oracle.com/artificial-intelligence/generative-ai/) and uses [NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) to record and govern every model call and tool call, then export the whole run as a portable [ATIF](https://github.com/NVIDIA/NeMo-Relay) trajectory.

The agent is an ordinary LangChain `create_agent` with `ChatOCIGenAI` as its model. The only NeMo Relay line it needs is one middleware entry:

```python
agent = create_agent(
    model=llm,
    tools=[list_genai_models, gpu_capacity],
    middleware=[NemoRelayMiddleware()],   # <- the only line Relay adds
    system_prompt=SYSTEM,
)
```

Relay observes the run in-process and writes a JSON trajectory with the model name, per-step token usage, and each structured tool call. It does not change what the agent does; it records and can govern it.

It is intentionally small and external-safe:

- No API keys, credentials, customer data, or internal content. Everything is parameterized through environment variables you set from your own tenancy.
- All components are publicly available open-source releases.
- The two tools issue read-only `oci` CLI queries against your own tenancy (the managed model catalog and a Compute capacity report). Nothing is created or changed.

## What the Sample Shows

The agent answers a real deployment question: *can we run NVIDIA Nemotron on this OCI tenancy, and how?* To answer it, the agent:

1. Searches the managed OCI Generative AI catalog for `nemotron` and for `llama` models.
2. Checks real host capacity for the `VM.GPU.A10.2` shape per availability domain (via an OCI Compute capacity report: `AVAILABLE` vs `OUT_OF_HOST_CAPACITY`), the signal that reflects whether the shape can actually be launched, not just the configured service limit.
3. Recommends a path: use a managed catalog model as-is, or self-host an open-weights model such as NVIDIA Nemotron on OKE (see the [Nemotron Lightning on OKE sample](../../../inference/oke/vllm/nemotron-lightning-endpoint/)).

Every model and tool call is captured by NeMo Relay and exported to [`results/advisor-trajectory.json`](./results/advisor-trajectory.json).

## Why Observe The Agent With NeMo Relay

An agent's real behavior is the sequence of model and tool calls it actually made, with their inputs, outputs, and token costs. NeMo Relay records that sequence in-process as a standard ATIF trajectory you can store, diff, replay, or feed to any observability backend, without instrumenting the agent by hand. The same middleware can also enforce guardrails and redact sensitive fields from what gets exported. Here it runs in observe-only mode.

## Requirements

- An OCI tenancy with access to [OCI Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm) in a supported region (the default is `us-chicago-1`).
- The `oci` CLI configured for your tenancy (the tools shell out to it), and Python 3.11 or newer.
- Python packages from [`requirements.txt`](./requirements.txt): `langchain`, `langchain-oci`, and `nemo-relay`.

NeMo Relay 0.9.0 or newer lets you pass `ChatOCIGenAI` to the middleware directly. On the 0.8.x line the sample uses a tiny `_OCIChatCompat` shim that drops an unsupported request header ([NVIDIA/NeMo-Relay#1012](https://github.com/NVIDIA/NeMo-Relay/pull/1012)); delete it once you are on 0.9.0+.

## Run It

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..."   # your compartment (or tenancy) OCID
export OCI_REGION="us-chicago-1"                         # a region with OCI Generative AI
export OCI_PROFILE="DEFAULT"                             # a profile in ~/.oci/config
export OCI_AUTH_TYPE="API_KEY"                           # or SECURITY_TOKEN / INSTANCE_PRINCIPAL / RESOURCE_PRINCIPAL
# export ADVISOR_MODEL="meta.llama-3.3-70b-instruct"     # any managed chat model in the catalog

python3 advisor.py
```

## Expected Output

The agent makes three tool calls (catalog search for `nemotron`, catalog search for `llama`, `VM.GPU.A10.2` host capacity), then gives a recommendation. NeMo Relay exports a five-step trajectory. A captured transcript is in [`results/sample-run.txt`](./results/sample-run.txt) and the trajectory in [`results/advisor-trajectory.json`](./results/advisor-trajectory.json):

| Metric | Value |
| --- | --- |
| Trajectory steps | 5 (1 user turn, 4 model turns, 3 nested tool calls) |
| Prompt tokens | 3,079 |
| Completion tokens | 137 |
| Model | `meta.llama-3.3-70b-instruct` (managed OCI Generative AI) |

Numbers come from the run recorded in [`results/`](./results/) on 2026-09-15. `langchain-oci` prints a `GenericProvider could not extract text` warning on turns where the model returns only tool calls; it is harmless.

## Files

| File | Purpose |
| --- | --- |
| `advisor.py` | The LangChain agent, its two read-only OCI tools, and the one-line NeMo Relay middleware wiring. |
| `requirements.txt` | `langchain`, `langchain-oci`, `nemo-relay`. |
| `results/` | A captured run: the transcript and the exported ATIF trajectory. |
