#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Deploy Nemotron 3.5 Lightning (NVFP4) with the vLLM Production Stack chart into namespace `lightning`,
# then wait for the engine to finish loading. Requires KUBECONFIG pointing at the cluster.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RELEASE="${RELEASE:-lightning}"; NAMESPACE="${NAMESPACE:-lightning}"; CHART_VERSION="${CHART_VERSION:-0.1.12}"

helm repo add vllm https://vllm-project.github.io/production-stack >/dev/null 2>&1 || true
helm repo update vllm >/dev/null
helm upgrade --install "$RELEASE" vllm/vllm-stack --version "$CHART_VERSION" \
  --namespace "$NAMESPACE" --create-namespace -f "$HERE/values.yaml"

DEPLOY="${RELEASE}-nemotron-35-lightning-deployment-vllm"
echo "waiting for $DEPLOY (image pull ~3 min, 21.6 GB of weights ~5 min, kernel warm-up)..."
kubectl -n "$NAMESPACE" rollout status deploy/"$DEPLOY" --timeout=40m

echo "vLLM startup summary:"
kubectl -n "$NAMESPACE" logs deploy/"$DEPLOY" --tail=400 | grep -E 'Mamba SSU backend|GPU KV cache size|Maximum concurrency|Application startup complete' || true
echo
echo "Next: kubectl -n $NAMESPACE port-forward svc/${RELEASE}-router-service 8000:80   (in another terminal), then python3 validate.py"
