<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# DGX Spark Samples

Samples that run on [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) — the GB10 Grace Blackwell desktop system with 128 GB of unified CPU+GPU memory.

Unlike the other directories in this repository, **these samples do not require an OCI account or cloud GPU shapes.** They run entirely on hardware on your desk. They are included here because DGX Spark is commonly used as the local development and prototyping half of a hybrid workflow: build and validate locally, deploy to OCI.

## Platform

| | |
|---|---|
| Chip | GB10 Grace Blackwell Superchip |
| Compute capability | `sm_121` |
| Memory | 128 GB unified LPDDR5x (CPU+GPU coherent) |
| Memory bandwidth | 273 GB/s |
| Architecture | `aarch64` |
| CUDA | 13.x |

Two units can be linked over 200 Gb/s ConnectX-7 for 256 GB of pooled memory.

## Samples

- [`nemotron-lightning-vllm-endpoint/`](./nemotron-lightning-vllm-endpoint) — Serve NVIDIA Nemotron 3.5 Lightning (30B-A3B, NVFP4) on a single DGX Spark as a private OpenAI-compatible endpoint, then drive it from the OpenAI SDK and run a multi-step tool-calling agent. Includes a presentable Jupyter notebook.

## A note on what DGX Spark is good at

DGX Spark is **memory-rich and bandwidth-bound**. At 273 GB/s it will not win a throughput contest against datacentre parts, and a single unit is a development box rather than a shared production endpoint. What it does uniquely well:

- **Capacity** — runs models and fine-tuning jobs that do not fit on a 32 GB consumer GPU at any speed
- **Data residency** — no prompt, document, or customer record leaves the machine
- **Zero marginal cost per token** — relevant to inner-loop developer tooling and high-volume batch work
- **Deterministic, isolated capacity** — no shared-tenant queueing and no rate limits

Size your expectations accordingly, and see each sample's README for measured numbers.

## Getting started

Each sample directory has its own README with prerequisites and run instructions. Most assume:

- A DGX Spark running DGX OS with the NVIDIA stack preinstalled
- Docker configured for your user (`sudo gpasswd -a $(whoami) docker`, then log out and back in)
- Python 3.10+
- A Hugging Face account and token for gated or large model downloads

## Related

- [DGX Spark playbooks](https://build.nvidia.com/spark) — official step-by-step workflows
- [NVIDIA/dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) — playbook source
- [DGX Spark developer forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)
