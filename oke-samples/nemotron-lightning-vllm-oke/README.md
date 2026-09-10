<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Nemotron 3.5 Lightning vLLM Endpoint on OKE

This sample serves [NVIDIA Nemotron 3.5 Lightning 30B-A3B (NVFP4)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) as a private, OpenAI-compatible endpoint on [Oracle Container Engine for Kubernetes (OKE)](https://www.oracle.com/cloud/cloud-native/container-engine-kubernetes/), on a single `VM.GPU.A10.2` worker (two NVIDIA A10, 24 GB each), using vLLM through the [vLLM Production Stack](https://github.com/vllm-project/production-stack) Helm chart.

It is the OKE counterpart of the [DGX Spark sample](../../dgx-spark-samples/nemotron-lightning-vllm-endpoint/): same model, same OpenAI-compatible surface, running on Oracle Cloud GPU shapes you can provision in minutes.

It is intentionally small and external-safe:

- No API keys, credentials, customer data, or internal content. Everything is parameterized through environment variables you set from your own tenancy.
- All model and inference components are publicly available open-source or open-weights releases.
- The validation script uses deterministic prompts and a synthetic tool. Nothing reaches the network except your own cluster.

The point of the sample is the same single line as on DGX Spark:

```python
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",   # <- the only line that changes
    api_key="not-needed",                  #    no key: it is your cluster
)
```

## What the Sample Shows

1. Creates a GPU node pool on an existing OKE cluster with the one cloud-init line that most OKE GPU deployments get wrong (see [Pitfalls](#pitfalls)).
2. Deploys Nemotron 3.5 Lightning with vLLM tensor-parallel across two A10 GPUs using one Helm command.
3. Confirms the endpoint serves the model, answers a chat request, returns **structured tool calls**, streams tokens, and exposes the **reasoning trace as a field**.
4. Optionally records every call as an [ATIF](https://github.com/NVIDIA/NeMo-Relay) trajectory with NVIDIA NeMo Relay, with no change to the serving stack.
5. Tears everything down so the GPUs stop billing.

## Why A10 Works For A 30B Model

Nemotron 3.5 Lightning is a 30B-parameter mixture-of-experts model with 3B active parameters. The NVFP4 checkpoint is 21.6 GB on disk. A10 GPUs are Ampere and have no FP4 tensor cores, but the model card lists **"NVIDIA Ampere via W4A16"**: vLLM serves the 4-bit expert weights through Marlin weight-only kernels and computes in 16-bit. Two A10s hold the weights with room for a large KV cache. Measured on this deployment:

| Metric | Value |
| --- | --- |
| Shape | `VM.GPU.A10.2` (2 x NVIDIA A10 24 GB), tensor parallel 2 |
| vLLM image | `vllm/vllm-openai:v0.27.1` |
| GPU KV cache | 1,336,934 tokens (fp8 KV cache) |
| Max concurrency at 65,536 tokens per request | 20.4x |
| Chat completion, 6 output tokens, thinking off | 0.7 s |
| Structured tool call | 0.4 s |
| Streaming, 31 chunks | 0.4 s |

Numbers come from the run recorded in [`results/`](./results/) on 2026-09-09. They are a single-node reference point, not a benchmark.

## Everything Here Is Open

| Component | License |
| --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) and [vLLM Production Stack](https://github.com/vllm-project/production-stack) | Apache 2.0 |
| [Nemotron 3.5 Lightning 30B-A3B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | OpenMDW-1.1 |
| [NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) (optional) | Apache 2.0 |
| This sample | Apache 2.0 |

No NGC subscription or support contract is required.

## Requirements

- An OCI tenancy with quota for `VM.GPU.A10.2` in at least one availability domain of your region. The service limit is `gpu-a10-count`; two GPUs are consumed.
- An existing OKE cluster (Basic or Enhanced, Kubernetes 1.33 or newer) whose API endpoint you can reach with `kubectl`. The cluster's worker subnet must allow outbound internet access (NAT or Internet Gateway) to pull the vLLM image and the model weights.
- `oci` CLI configured for your tenancy, plus `kubectl`, `helm` 3, `python3` 3.10 or newer, and `jq`.
- About 40 minutes end to end: 10 for the node to join, 3 to pull the 10 GB image, 5 to download 21.6 GB of weights, a few for kernel warm-up.
- A Hugging Face token only if you hit download rate limits (`HF_TOKEN`, passed through the chart's `hf_token` value).

Cost: `VM.GPU.A10.2` is billed per hour while the node exists. Run [`cleanup.sh`](./cleanup.sh) when you are done.

## Architecture

```text
Your laptop                          OCI region
  kubectl port-forward  ──────────►  OKE cluster (existing)
  OpenAI SDK / curl                    │
                                       ├── namespace: lightning
                                       │     ├── router Deployment (vllm-stack)  ── Service :80
                                       │     └── vLLM engine Deployment ─────────── Service :80
                                       │           model: Nemotron 3.5 Lightning NVFP4
                                       │           --tensor-parallel-size 2
                                       │
                                       └── node pool: 1 x VM.GPU.A10.2 (created by this sample)
                                             cloud-init: oci-growfs + OKE bootstrap
```

## Quickstart

Every script reads its inputs from environment variables. Set them once:

```bash
export OCI_REGION="us-phoenix-1"                       # a region with A10 capacity
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..."   # compartment of the cluster
export OKE_CLUSTER_ID="ocid1.cluster.oc1.phx...."      # your existing OKE cluster
export WORKER_SUBNET_ID="ocid1.subnet.oc1.phx...."     # the cluster's worker subnet
export OCI_CLI_PROFILE="DEFAULT"                       # optional; your ~/.oci/config profile
export KUBECONFIG="$HOME/.kube/config"                  # must point at the cluster
```

Then:

```bash
./preflight.sh          # tools, A10.2 capacity per availability domain, cluster version
./create-node-pool.sh   # 1 x VM.GPU.A10.2 with the grow-root-filesystem cloud-init; waits for ACTIVE
./deploy.sh             # helm install; waits for the engine pod to be Ready
```

In a second terminal, keep a port-forward open:

```bash
kubectl -n lightning port-forward svc/lightning-router-service 8000:80
```

Back in the first terminal:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 validate.py                    # models, chat, tool call, streaming, reasoning field
python3 relay_probe.py                 # optional: record the calls as an ATIF trajectory
./cleanup.sh                            # helm uninstall + delete the node pool
```

Expected `validate.py` output is in [`results/validate-2026-09-09.txt`](./results/validate-2026-09-09.txt).

## Files

| File | Purpose |
| --- | --- |
| `preflight.sh` | Checks tools, reports `VM.GPU.A10.2` availability per availability domain, reads the cluster's Kubernetes version. |
| `cloud-init.sh` | Node bootstrap: `/usr/libexec/oci-growfs -y`, then the standard OKE init script. Passed as `--node-metadata user_data`. |
| `create-node-pool.sh` | Finds the matching GPU node image for the cluster version, creates the node pool labeled `nvidia-oci-samples/pool=<name>`, waits for that pool's node. |
| `values.yaml` | vLLM Production Stack values: model, image, TP=2, Ampere-friendly backends, tool and reasoning parsers, scheduling pinned to the sample's node pool. |
| `deploy.sh` | `helm upgrade --install`, waits for rollout, prints the vLLM startup summary. |
| `validate.py` | Five checks against the OpenAI-compatible endpoint; exits nonzero if any expectation fails. |
| `relay_probe.py` | Optional. Sends two requests through NeMo Relay's managed execution and writes an ATIF trajectory. |
| `cleanup.sh` | Removes the Helm release, the namespace (only if `deploy.sh` created it), and, after confirmation, the node pool. |
| `results/` | Outputs captured from the 2026-09-09 run. |

## Serving Configuration

The chart renders this `vllm serve` command:

```text
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --moe-backend marlin --mamba-backend flashinfer --kv-cache-dtype fp8 \
  --enable-prefix-caching --max-num-seqs 32
```

- `--moe-backend marlin` selects the W4A16 kernels that run NVFP4 expert weights on Ampere. Hopper-specific backends such as `humming` are not used.
- `--mamba-backend flashinfer` is the backend NVIDIA's A100 recipe uses for the Mamba-2 layers.
- `--kv-cache-dtype fp8` doubles KV capacity; drop it if your vLLM build rejects fp8 KV on your GPUs.
- `--max-model-len 65536` keeps the sample predictable. The model supports up to 1M tokens; raise the value if you have KV headroom.
- The router Deployment comes from the chart's defaults. The vLLM engine image is pinned to `v0.27.1`; the router image upstream publishes only as `latest`, so pin it by digest in `values.yaml` if your environment requires immutable references.

### Reasoning Is On By Default

vLLM 0.27 returns the model's thinking in the `reasoning` field of the message. With a small `max_tokens`, the budget can be consumed by reasoning and `content` comes back empty. For deterministic tool-calling demos, pass `"chat_template_kwargs": {"enable_thinking": false}` as `validate.py` does, or give the request a larger `max_tokens`.

## Observing The Endpoint With NeMo Relay (Optional)

[NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) is an in-process runtime that records and controls model and tool calls. Because the endpoint speaks the OpenAI chat format, Relay recognizes it automatically; `relay_probe.py` wraps two requests in a Relay scope and exports an [ATIF](https://github.com/NVIDIA/NeMo-Relay) trajectory with the model name, per-step token usage, and the structured tool call. The recorded trajectory from the reference run is [`results/relay-trajectory-2026-09-09.json`](./results/relay-trajectory-2026-09-09.json). Requires `nemo-relay>=0.8.4`.

## Pitfalls

**Root filesystem is ~30 GB regardless of boot volume size.** OKE node images do not grow the root partition to the requested boot volume unless cloud-init runs `/usr/libexec/oci-growfs`. Without it the node reports about 30 GiB of ephemeral storage, the kubelet raises `DiskPressure` while pulling the 10 GB vLLM image, and the engine pod is evicted with `The node was low on resource: ephemeral-storage`. `cloud-init.sh` runs the grow step before the OKE bootstrap, exactly as Console-created node pools do. If you created the pool without it, recreate it; expanding in place also requires a node reset so the kubelet re-reads capacity.

**Capacity is per availability domain.** `VM.GPU.A10.2` may be available in one AD and out of host capacity in another. `preflight.sh` runs a capacity report so you can pick the AD before creating the pool.

**The model needs to be told to use tools.** Like most instruction-tuned models, Lightning answers directly when a question could be answered without a tool. The tool checks in `validate.py` and `relay_probe.py` say so explicitly.

## Cleanup

```bash
./cleanup.sh
```

This removes the `lightning` Helm release, deletes the namespace only if `deploy.sh` created it, and asks before deleting the node pool. The OKE cluster, VCN, and anything else you already had are left untouched.
