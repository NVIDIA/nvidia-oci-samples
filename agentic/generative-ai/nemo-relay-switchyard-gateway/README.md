<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Route coding agents across OCI Generative AI models with NeMo Relay and NeMo Switchyard

This sample puts the [NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) gateway in front of [OCI Generative AI](https://www.oracle.com/artificial-intelligence/generative-ai/) and shows three things working together on OCI's OpenAI-compatible endpoint:

1. **Observability for any OpenAI-compatible client.** Chat Completions and Responses API traffic to OCI Generative AI is recorded as [ATOF](https://docs.nvidia.com/nemo/relay/) lifecycle events, with per-call token usage.
2. **Model routing with [NVIDIA NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard).** The Switchyard plugin for Relay routes each request between an efficient and a capable OCI model, either on a judge model's forecast or on the agent's own tool evidence. Fixed routes are included for comparison.
3. **Cost on every call.** A pricing catalog built from OCI's public price list lets Relay attach a USD estimate to each recorded model call, so routing policies can be compared on real cost, not just token counts.

A coding-agent run is included: [Codex](https://github.com/openai/codex) through Relay against OCI's Responses API, with Relay replacing Codex's credential by an OCI Generative AI API key.

Everything here is external-safe: no keys, OCIDs, or tenancy details. The API key is read from the environment, Relay writes to a sample-local configuration directory, and all components are public releases.

## What the Sample Shows

```text
 OpenAI SDK / curl ─┐                          ┌─ Switchyard plugin ───────────────────────┐
 Codex (Responses) ─┼─▶ NeMo Relay gateway ───▶│ oci/efficient  → openai.gpt-oss-120b        │──▶ OCI Generative AI
                    │   127.0.0.1:4040         │ oci/capable    → xai.grok-4.7               │    OpenAI-compatible
                    │   • ATOF events          │ oci/auto       → tool evidence decides      │    endpoint, one region
                    │   • pricing catalog      │ oci/task       → judge (Llama 4 Maverick)   │
                    │                          └───────────────────────────────────────────┘
                    └─▶ direct model id (e.g. openai.gpt-oss-20b) → forwarded as-is ─────────▶
```

- Clients select a route by requesting its id as the model name. Anything else is forwarded to OCI unchanged.
- `oci/task` asks a judge model whether the efficient model can satisfy the request and escalates on a low score. `oci/auto` starts on the efficient model and escalates on tool evidence, with no judge call.
- Relay records every call, the Switchyard decision marks (algorithm, evidence, selected and served model), and the USD estimate from the catalog in `logs/events.jsonl`.

## Requirements

- An OCI tenancy with [OCI Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm) in a region that serves the models in [`relay/plugins.toml.in`](./relay/plugins.toml.in) (default region `us-chicago-1`), and an [OCI Generative AI API key](https://docs.oracle.com/en-us/iaas/Content/generative-ai/api-keys.htm) whose compartment policy allows those models. Create the policy before the key.
- The [NeMo Relay CLI](https://docs.nvidia.com/nemo/relay/) 0.11.1 or newer on `PATH` (`nemo-relay --version`), and the [GitHub CLI](https://cli.github.com/) logged in, or `GH_TOKEN` set: `nemo-relay plugins install` downloads the Switchyard bundle from GitHub releases.
- Python 3.10 or newer with the `openai` package for `scripts/route_demo.py` (`pip install openai`).
- Optional: [Codex CLI](https://github.com/openai/codex) for `scripts/codex_demo.sh`.

Tested on 2026-11-17 with NeMo Relay 0.11.1, the Switchyard plugin `switchyard-plugin-0.3.2`, Codex 0.166.0, and the OpenAI Python SDK 3.52, on macOS arm64.

## Run It

```bash
export OCI_GENAI_API_KEY="sk-..."          # OCI Generative AI API key, never written to disk by the sample
export OCI_GENAI_REGION="us-chicago-1"      # a region that serves the configured models

scripts/setup.sh                            # render Relay config, install + validate the Switchyard plugin
scripts/gateway.sh start                    # standalone gateway on 127.0.0.1:4040

python3 scripts/route_demo.py               # the same prompts through every route
python3 scripts/summarize_events.py         # decisions, usage, and USD per call from logs/events.jsonl

scripts/gateway.sh stop
```

`scripts/setup.sh` writes everything under `.relay/` inside this directory (a sample-local `XDG_CONFIG_HOME`), so your real `~/.config/nemo-relay` is untouched. Re-run it after editing the templates in `relay/`.

To run the coding agent, set `CODEX_MODEL` to a model your tenancy serves through the Responses API (see [Codex and model choice](#codex-and-model-choice)) and run:

```bash
CODEX_MODEL="<model id>" scripts/codex_demo.sh
python3 scripts/summarize_events.py
```

## Expected Output

`scripts/route_demo.py` sends an easy, a medium, and a hard prompt through the routes. A captured run is in [`results/route-demo.txt`](./results/route-demo.txt):

```text
route          prompt  served model                 in    out      ms  answer
oci/efficient  easy    openai.gpt-oss-120b          86     55    1100  The Quick Brown Fox Jumps Over the Lazy Dog
oci/capable    easy    xai.grok-4.7               1261     10    3272  The Quick Brown Fox Jumps Over The Lazy Dog
oci/auto       medium  openai.gpt-oss-120b          79    145    1271  TCP's three-way handshake lets both ends verify ...
oci/task       easy    openai.gpt-oss-120b          86     76     930  The Quick Brown Fox Jumps Over the Lazy Dog
oci/task       medium  openai.gpt-oss-120b          79    160    1742  TCP uses a three-way handshake to reliably ...
oci/task       hard    xai.grok-4.7               1288    600   15672  A practical C11 design is a power-of-two ring ...
(direct)       easy    openai.gpt-oss-20b           86    200     829  The Quick Brown Fox Jumps Over the Lazy Dog
```

`scripts/summarize_events.py` reads the ATOF log and shows why each request went where it went, and what it cost ([`results/event-summary.txt`](./results/event-summary.txt)):

```text
Switchyard routing decisions
time         algorithm            evidence                                         selected -> served
16:32:02.292 stage_router         source=fall_open                                 openai.gpt-oss-120b -> openai.gpt-oss-120b
16:32:03.223 llm_task_classifier  score=0.99, source=llm-classifier, threshold=0.5 openai.gpt-oss-120b -> openai.gpt-oss-120b
16:32:20.639 llm_task_classifier  score=0.0, source=llm-classifier, threshold=0.7  xai.grok-4.7 -> xai.grok-4.7

Totals by served model
served model            calls      in     out        USD
xai.grok-4.7                2    2549     610   0.005110
openai.gpt-oss-120b         4     330     436   0.000311
openai.gpt-oss-20b          1      86     200   0.000066
total                                            0.005487
```

The judge rated the easy and medium prompts as within the efficient model's reach (scores 0.99 and 0.9) and the lock-free queue design as not (score 0.0), so only that request paid for Grok 4.7. The two Grok calls account for 93% of the run's cost. The `status` column carries Relay's per-call status; all seven calls are `OK`.

The Codex run creates a file through a tool call and answers, with both Responses API calls recorded with their token usage ([`results/codex-run.txt`](./results/codex-run.txt), [`results/codex-event-summary.txt`](./results/codex-event-summary.txt)). The model used there is not in the public price list, so those two calls carry no cost.

## How It Works

### Credentials

Relay never stores the OCI key. Three paths carry it:

| Caller | How the key reaches OCI |
|---|---|
| Switchyard routes (`oci/*`) | The plugin reads `OCI_GENAI_API_KEY` from the gateway's environment (`api_key_env` in the deployment). The client's own credential is not used. |
| Direct model ids through the gateway | Relay keeps the client's `Authorization` header, so the client sends the OCI key as its API key. Relay only injects a gateway-held credential for clients that authenticate to Relay. |
| Managed Codex run (`nemo-relay codex`) | Relay authenticates the launched agent and replaces its credential with `NEMO_RELAY_OPENAI_AUTH_HEADER`, which `scripts/codex_demo.sh` sets to `Bearer $OCI_GENAI_API_KEY`. |

### Routing

[`relay/plugins.toml.in`](./relay/plugins.toml.in) is a Switchyard version-1 deployment inside the Relay plugin configuration. One `openai_chat` client points at the region's OpenAI-compatible base URL. Targets are plain OCI model ids. Routes:

| Route id | Type | Behaviour |
|---|---|---|
| `oci/efficient` | `passthrough` | Always `openai.gpt-oss-120b`. |
| `oci/capable` | `passthrough` | Always `xai.grok-4.7`, with `reasoning_effort = "low"` to keep the demo interactive. |
| `oci/auto` | `auto` | Switchyard's stage router: efficient first, escalates on tool-call evidence. No judge call. |
| `oci/task` | `llm_classifier` (capability mode) | Llama 4 Maverick forecasts whether the efficient model can satisfy the task; low scores go to the capable model. |

Swap model ids for any others your region serves. The Switchyard [TOML schema](https://github.com/NVIDIA-NeMo/Switchyard/blob/release/0.3/docs/reference/toml_schema.md) documents the other route types.

### Pricing

[`pricing/oci-genai-pricing.json`](./pricing/oci-genai-pricing.json) is a Relay pricing catalog with OCI's public on-demand list prices for the token-priced chat models: `openai.gpt-oss-120b`, `openai.gpt-oss-20b`, `xai.grok-4.7`, `xai.grok-4.6`, `xai.grok-4.3`, `xai.grok-4.2`, and the Gemini 2.5 family. Grok and Gemini 2.5 Pro use Relay's `prompt_token_threshold` schedule for their short- and long-context rates. Prices were taken from the [OCI price list](https://www.oracle.com/cloud/price-list/#genai) on 2026-11-17; check them before relying on the estimates.

Requests arrive on Relay's OpenAI-compatible route, so Relay prices them as provider `openai` with the OCI model id; the catalog also carries `oci/...` and `oci_genai/...` aliases for calls recorded through Relay's native OCI codec. Models that OCI bills per 10,000 transactions (Llama, Cohere) are not in the catalog, so their calls show no cost. Validate and inspect the catalog with:

```bash
nemo-relay model-pricing validate pricing/oci-genai-pricing.json
nemo-relay model-pricing resolve xai.grok-4.7 --provider openai --prompt-tokens 250000 --completion-tokens 1000
```

### Codex and model choice

`scripts/codex_demo.sh` runs `nemo-relay codex -- --model "$CODEX_MODEL" exec ...` with a minimal Codex home ([`codex/config.toml`](./codex/config.toml)). Relay starts a temporary gateway for the run, injects Codex hooks, and forwards the Responses API traffic to OCI.

Codex needs a model that OCI serves through the Responses API with a multi-turn tool loop, and OCI's Responses API accepts `function`, `web_search_preview`, `code_interpreter`, and `mcp` tools. The minimal Codex configuration therefore turns off Codex's optional tool namespaces (apps, multi-agent, goals, web search). What we observed on 2026-11-17:

- `openai.gpt-oss-120b` answers the first turn and issues the tool call, but OCI rejects the follow-up turn with `reasoning_text is not supported by model openai.gpt-oss-120b`.
- xAI models reject Codex's Responses requests with `Non-OpenAI models require 'OpenAI-Project' or 'opc-conversation-store-id' header for the Responses API`, a header Codex does not send.
- The run completed with an OpenAI GPT-5-series model enabled in our tenancy, which is why the script requires `CODEX_MODEL` instead of defaulting to a model.

Codex also calls `GET /v1/models`, which OCI does not serve; Codex logs the 404 and continues with fallback model metadata.

Relay records Responses API calls to OCI with their full content and usage, but Relay 0.11.1 sometimes marks them `ERROR` with `stream ended before its final response event`, even though Codex received the complete answer and the recorded response carries `completed_at`. OCI's Responses stream sends the event type only inside each `data:` payload, with no SSE `event:` lines, which is the likely trigger. Chat Completions calls are not affected.

## Notes

- The ATOF file sink path in `plugins.toml` is relative to the gateway's working directory, so the scripts start the gateway from this directory and events land in `logs/events.jsonl`. Protect that file: it contains prompts and answers.
- Relay's gateway routes are Chat Completions and Responses; Switchyard routes are exercised here over Chat Completions. Add an `openai_responses` client to the deployment to route Responses traffic.
- The Switchyard bundle ships a digest but no signature, so the deployment accepts it with `attestation = "integrity_only"`, as in the Relay documentation.
- To export the same events to an OpenTelemetry collector or as ATIF trajectories, add exporters to the observability component; see [NeMo Relay observability](https://docs.nvidia.com/nemo/relay/).

## Files

| Path | Purpose |
|---|---|
| `scripts/setup.sh` | Renders `relay/*.in` for `OCI_GENAI_REGION` into `.relay/`, installs and validates the Switchyard plugin, validates the catalog. |
| `scripts/gateway.sh` | Starts, stops, or checks the standalone gateway (`RELAY_BIND`, default `127.0.0.1:4040`). |
| `scripts/route_demo.py` | Sends easy, medium, and hard prompts through each route, plus one direct model call. |
| `scripts/summarize_events.py` | Prints routing decisions, per-call usage and USD, and totals from an ATOF event log. |
| `scripts/codex_demo.sh` | Runs one Codex task through Relay against OCI's Responses API. |
| `relay/config.toml.in` | Relay upstream (OCI OpenAI-compatible base URL) and the Codex agent entry. |
| `relay/plugins.toml.in` | Switchyard deployment, observability (ATOF file sink), and pricing component. |
| `pricing/oci-genai-pricing.json` | OCI Generative AI list prices in Relay's catalog format. |
| `codex/config.toml` | Minimal Codex configuration used as `CODEX_HOME`. |
| `results/` | Captured runs: route demo output, event summaries, and the Codex transcript. |

## References

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay/) and [repository](https://github.com/NVIDIA/NeMo-Relay)
- [NeMo Relay Plugins (Switchyard bundle)](https://github.com/NVIDIA/NeMo-Relay-Plugins)
- [NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)
- [OCI Generative AI OpenAI-compatible endpoints](https://docs.oracle.com/en-us/iaas/Content/generative-ai/openai-compatible-api.htm), [Responses API](https://docs.oracle.com/en-us/iaas/Content/generative-ai/oci-openai.htm), and [API keys](https://docs.oracle.com/en-us/iaas/Content/generative-ai/api-keys.htm)
- [Agent observability on OCI Generative AI with NeMo Relay](../nemo-relay-advisor), the in-process (LangChain) companion to this gateway sample
