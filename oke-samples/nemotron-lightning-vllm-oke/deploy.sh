#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Deploy Nemotron 3.5 Lightning (NVFP4) with the vLLM Production Stack chart into namespace `lightning`,
# then wait for the engine to finish loading. Requires KUBECONFIG pointing at the cluster.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RELEASE="${RELEASE:-lightning}"; NAMESPACE="${NAMESPACE:-lightning}"; CHART_VERSION="${CHART_VERSION:-0.1.12}"

OWNER="nemotron-lightning-vllm-oke"
OWNER_LABEL="nvidia-oci-samples/owner=$OWNER"
MARKER="nvidia-oci-samples-owner"            # ConfigMap recording the release this sample created
NODE_POOL_NAME="${NODE_POOL_NAME:-nemotron-lightning-a10x2}"   # must match create-node-pool.sh

if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
  # Create the namespace with the ownership label in one request so cleanup.sh can recognize it.
  kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml \
    | kubectl label --local -f - "$OWNER_LABEL" -o yaml \
    | kubectl apply -f -
else
  echo "namespace $NAMESPACE already exists; installing into it (cleanup.sh will leave the namespace in place)"
fi

# Never overwrite a release this sample did not create.
if helm status "$RELEASE" -n "$NAMESPACE" >/dev/null 2>&1; then
  if [ "$(kubectl -n "$NAMESPACE" get configmap "$MARKER" -o jsonpath='{.data.release}' 2>/dev/null)" != "$RELEASE" ]; then
    echo "ERROR: a Helm release named $RELEASE already exists in $NAMESPACE and was not created by this sample." >&2
    echo "       Choose another RELEASE/NAMESPACE, or remove it yourself first." >&2
    exit 1
  fi
fi

helm repo add vllm https://vllm-project.github.io/production-stack >/dev/null 2>&1 || true
helm repo update vllm >/dev/null
helm upgrade --install "$RELEASE" vllm/vllm-stack --version "$CHART_VERSION" \
  --namespace "$NAMESPACE" -f "$HERE/values.yaml" \
  --set-string "servingEngineSpec.modelSpec[0].nodeSelectorTerms[0].matchExpressions[0].values[0]=$NODE_POOL_NAME"
kubectl -n "$NAMESPACE" create configmap "$MARKER" --from-literal=release="$RELEASE" --from-literal=owner="$OWNER" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

DEPLOY="${RELEASE}-nemotron-35-lightning-deployment-vllm"
echo "waiting for $DEPLOY (image pull ~3 min, 21.6 GB of weights ~5 min, kernel warm-up)..."
kubectl -n "$NAMESPACE" rollout status deploy/"$DEPLOY" --timeout=40m

echo "vLLM startup summary:"
kubectl -n "$NAMESPACE" logs deploy/"$DEPLOY" --tail=400 | grep -E 'Mamba SSU backend|GPU KV cache size|Maximum concurrency|Application startup complete' || true
echo
echo "Next: kubectl -n $NAMESPACE port-forward svc/${RELEASE}-router-service 8000:80   (in another terminal), then python3 validate.py"
