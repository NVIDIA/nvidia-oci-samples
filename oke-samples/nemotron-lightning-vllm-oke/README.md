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

1. Optionally creates a small OKE cluster from an empty compartment, then creates a GPU node pool with the one cloud-init line that most OKE GPU deployments get wrong (see [Pitfalls](#pitfalls)).
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
| Chat completion, 6 output tokens, thinking off | 0.6 s |
| Structured tool call | 0.5 s |
| Streaming, 31 chunks | 0.5 s |
| Reasoning on, 586 output tokens | 3.4 s |

Numbers come from the run recorded in [`results/`](./results/) on 2026-09-10, produced by the scripts in this directory on a freshly created cluster. They are a single-node reference point, not a benchmark.

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
- An OKE cluster (Basic or Enhanced, Kubernetes 1.33 or newer) whose API endpoint you can reach with `kubectl`. The cluster's worker subnet must allow outbound internet access (NAT or Internet Gateway) to pull the vLLM image and the model weights. If you do not have one, [`create-cluster.sh`](./create-cluster.sh) builds a VCN and a Basic cluster that meet these requirements; a Basic cluster with no nodes has no charge.
- Somewhere for CoreDNS to run. OKE GPU nodes carry a `nvidia.com/gpu` NoSchedule taint, so on a GPU-only cluster CoreDNS stays Pending and pods cannot resolve names. `deploy.sh` detects that and adds the GPU toleration to CoreDNS and its autoscaler, the same fix as the Nemotron cookbook; if you would rather not touch `kube-system`, add a small CPU node pool first.
- `oci` CLI configured for your tenancy, plus `kubectl`, `helm` 3.13 or newer, `python3` 3.10 or newer, and `jq`.
- About 40 minutes end to end: 10 for the node to join, 3 to pull the 10 GB image, 5 to download 21.6 GB of weights, a few for kernel warm-up.
- A Hugging Face token only if you hit download rate limits (`HF_TOKEN`, passed through the chart's `hf_token` value).

Cost: `VM.GPU.A10.2` is billed per hour while the node exists. Run [`cleanup.sh`](./cleanup.sh) when you are done.

## Architecture

```text
Your laptop                          OCI region
  kubectl port-forward  ──────────►  OKE cluster (existing)
  OpenAI SDK / curl                    │
                                       ├── namespace: lightning
                                       │     └── vLLM engine Deployment ── Service :80 (OpenAI-compatible)
                                       │           model: Nemotron 3.5 Lightning NVFP4
                                       │           --tensor-parallel-size 2
                                       │
                                       └── node pool: 1 x VM.GPU.A10.2 (created by this sample)
                                             cloud-init: oci-growfs + OKE bootstrap
```

The cluster can be one you already have or one created by `create-cluster.sh` (VCN with Internet, NAT, and Service gateways; public API endpoint; Flannel).

## Quickstart

### Starting from an empty compartment (optional)

If you have no OKE cluster yet, create one first. This takes about 10 minutes and creates nothing that bills while idle:

```bash
export OCI_REGION="us-phoenix-1"                       # a region with A10 capacity
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..."   # where the VCN and cluster go
export OCI_CLI_PROFILE="DEFAULT"                       # optional; your ~/.oci/config profile
./create-cluster.sh                                    # VCN + Basic OKE cluster + ./kubeconfig; prints the exports below
```

It picks the newest Kubernetes version that also has an OKE GPU node image, so the next step can find one; set `KUBERNETES_VERSION` to choose. Copy the three `export` lines it prints (`OKE_CLUSTER_ID`, `WORKER_SUBNET_ID`, `KUBECONFIG`) and continue with the steps below. The Kubernetes API is reachable from `API_ALLOWED_CIDR` (default `0.0.0.0/0`, the same as the Console quick-create; requests still need a signed OCI token). Set it to your own egress range if you know it. When you are done, `./delete-cluster.sh` removes the cluster and the VCN; see [Cleanup](#cleanup).

### Deploying on a cluster

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
kubectl -n lightning port-forward svc/lightning-nemotron-35-lightning-engine-service 8000:80
```

Back in the first terminal:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 validate.py                    # models, chat, tool call, streaming, reasoning field
python3 relay_probe.py                 # optional: record the calls as an ATIF trajectory
./cleanup.sh                            # helm uninstall + delete the node pool
```

Expected `validate.py` output is in [`results/validate-2026-09-10.txt`](./results/validate-2026-09-10.txt).

## Files

| File | Purpose |
| --- | --- |
| `create-cluster.sh` | Optional. Creates a VCN (Internet, NAT, and Service gateways; API, worker, and load-balancer subnets) and a Basic OKE cluster with a public endpoint, both tagged `nvidia-oci-samples-owner=nemotron-lightning-vllm-oke`; writes a kubeconfig and prints the exports for the next steps. |
| `preflight.sh` | Checks tools, reports `VM.GPU.A10.2` availability per availability domain, reads the cluster's Kubernetes version. |
| `cloud-init.sh` | Node bootstrap: `/usr/libexec/oci-growfs -y`, then the standard OKE init script. Passed as `--node-metadata user_data`. |
| `create-node-pool.sh` | Finds the matching GPU node image for the cluster version, creates the node pool labeled `nvidia-oci-samples/pool=<name>`, waits for that pool's node with a 20-minute deadline. |
| `values.yaml` | vLLM Production Stack values: model, image, TP=2, Ampere-friendly backends, tool and reasoning parsers, scheduling pinned to the sample's node pool, router disabled. |
| `deploy.sh` | Creates and labels the namespace, makes CoreDNS schedulable on GPU-only clusters, refuses to overwrite a release it did not create, `helm upgrade --install` pinned to the pool from `NODE_POOL_NAME`, waits until the engine is Available, prints the vLLM startup summary. |
| `validate.py` | Five checks against the OpenAI-compatible endpoint; exits nonzero if any expectation fails. |
| `relay_probe.py` | Optional. Sends two requests through NeMo Relay's managed execution and writes an ATIF trajectory. |
| `cleanup.sh` | After confirmation, removes the Helm release (only if it matches the ownership record `deploy.sh` wrote), the namespace (only if `deploy.sh` created it), and the node pool. `ASSUME_YES=1` skips the prompts. |
| `delete-cluster.sh` | Optional. Deletes a cluster created by `create-cluster.sh` and then its VCN. Refuses clusters and VCNs without the ownership tag, refuses while node pools remain, and keeps the VCN if another cluster still uses it. |
| `results/` | Outputs captured on 2026-09-10 by running these scripts end to end: `validate.py` output, selected vLLM startup log lines, and the Relay trajectory. |

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
- The chart's router is disabled. A single engine needs no routing layer, and the router image carries no toleration for the `nvidia.com/gpu` taint that OKE GPU nodes have, so on a GPU-only pool it stays Pending. Clients talk to the engine Service directly; enable `routerSpec.enableRouter` only if you add a CPU node pool for it.

### Reasoning Is On By Default

vLLM 0.27 returns the model's thinking in the `reasoning` field of the message. With a small `max_tokens`, the budget can be consumed by reasoning and `content` comes back empty. For deterministic tool-calling demos, pass `"chat_template_kwargs": {"enable_thinking": false}` as `validate.py` does, or give the request a larger `max_tokens`.

## Observing The Endpoint With NeMo Relay (Optional)

[NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) is an in-process runtime that records and controls model and tool calls. Because the endpoint speaks the OpenAI chat format, Relay recognizes it automatically; `relay_probe.py` wraps two requests in a Relay scope and exports an [ATIF](https://github.com/NVIDIA/NeMo-Relay) trajectory with the model name, per-step token usage, and the structured tool call. Requires `nemo-relay>=0.8.4`.

The recorded trajectory from the reference run is four steps: two user prompts and two model turns. The second model turn is the structured tool call, recorded with its token usage. Excerpt from [`results/relay-trajectory-2026-09-10.json`](./results/relay-trajectory-2026-09-10.json):

```json
{
  "schema_version": "ATIF-v1.7",
  "agent": {"name": "lightning-probe", "model_name": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"},
  "final_metrics": {"total_steps": 4, "total_prompt_tokens": 293, "total_completion_tokens": 24},
  "steps": [
    {"step_id": 3, "source": "user", "message": "Call the get_utc_time tool to find the current UTC time, then report it."},
    {"step_id": 4, "source": "agent",
     "model_name": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
     "tool_calls": [{"tool_call_id": "chatcmpl-tool-b1db46801ef76be6", "function_name": "get_utc_time", "arguments": {}}],
     "metrics": {"prompt_tokens": 264, "completion_tokens": 16}}
  ]
}
```

Nothing in the serving stack changed to get this: the probe calls the same endpoint the OpenAI SDK would, through Relay's `llm.execute`, and Relay's exporter writes the file.

## Pitfalls

**Root filesystem is ~30 GB regardless of boot volume size.** OKE node images do not grow the root partition to the requested boot volume unless cloud-init runs `/usr/libexec/oci-growfs`. Without it the node reports about 30 GiB of ephemeral storage, the kubelet raises `DiskPressure` while pulling the 10 GB vLLM image, and the engine pod is evicted with `The node was low on resource: ephemeral-storage`. `cloud-init.sh` runs the grow step before the OKE bootstrap, exactly as Console-created node pools do. If you created the pool without it, recreate it; expanding in place also requires a node reset so the kubelet re-reads capacity.

**GPU-only clusters leave system pods Pending.** The `nvidia.com/gpu` taint on OKE GPU nodes is respected by everything without a matching toleration: CoreDNS, its autoscaler, and the chart's router. Symptoms are `Temporary failure in name resolution` from the engine and pods stuck in `Pending` with `untolerated taint`. `deploy.sh` tolerates the taint for CoreDNS when needed, and the router is disabled; a CPU node pool avoids the issue entirely.

**Capacity is per availability domain.** `VM.GPU.A10.2` may be available in one AD and out of host capacity in another. `preflight.sh` runs a capacity report so you can pick the AD before creating the pool.

**The model needs to be told to use tools.** Like most instruction-tuned models, Lightning answers directly when a question could be answered without a tool. The tool checks in `validate.py` and `relay_probe.py` say so explicitly.

## Cleanup

```bash
./cleanup.sh
```

This uninstalls the Helm release only if the sample's ownership record matches the live release (its Helm `firstDeployed` timestamp) and you confirm, deletes the namespace only if `deploy.sh` created it, and asks again before deleting the node pool. Set `ASSUME_YES=1` for unattended runs. The OKE cluster, VCN, and anything else you already had are left untouched.

If `create-cluster.sh` created the cluster, remove it and its VCN afterwards:

```bash
./delete-cluster.sh
```

It deletes only a cluster and VCN that carry the tag `nvidia-oci-samples-owner=nemotron-lightning-vllm-oke`, refuses while the cluster still has node pools (run `cleanup.sh` first), and leaves the VCN in place if another cluster uses it. `ASSUME_YES=1` skips the prompt.
