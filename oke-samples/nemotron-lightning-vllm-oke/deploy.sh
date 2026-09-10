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
elif kubectl get namespace "$NAMESPACE" -o jsonpath='{.metadata.labels.nvidia-oci-samples/owner}' | grep -q "^$OWNER$"; then
  echo "namespace $NAMESPACE exists and was created by this sample"
else
  echo "namespace $NAMESPACE already exists and is not owned by this sample; installing into it (cleanup.sh will leave it in place)"
fi

# GPU-only clusters: OKE GPU nodes carry the nvidia.com/gpu NoSchedule taint, so CoreDNS and its
# autoscaler cannot schedule when no untainted node exists, and without DNS the engine cannot reach
# Hugging Face. Tolerate the taint on those two deployments only when CoreDNS is not running.
if ! kubectl -n kube-system get pods -l k8s-app=kube-dns -o jsonpath='{.items[*].status.phase}' 2>/dev/null | grep -q Running; then
  echo "CoreDNS has no schedulable node (GPU-only cluster); adding the nvidia.com/gpu toleration to coredns and kube-dns-autoscaler"
  TOLERATION='[{"op":"add","path":"/spec/template/spec/tolerations/-","value":{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}}]'
  for d in coredns kube-dns-autoscaler; do
    kubectl -n kube-system patch deployment "$d" --type=json -p "$TOLERATION" >/dev/null 2>&1 || echo "  (could not patch $d; continuing)"
  done
  kubectl -n kube-system rollout status deploy/coredns --timeout=5m
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
# A re-run after fixing cluster DNS may find the engine crash-looping on the old failure; restart it.
if kubectl -n "$NAMESPACE" get pods -l model=nemotron-35-lightning -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}' 2>/dev/null | grep -q CrashLoopBackOff; then
  echo "engine pod is crash-looping from an earlier attempt; restarting it"
  kubectl -n "$NAMESPACE" rollout restart deploy/"$DEPLOY" >/dev/null
fi
echo "waiting for $DEPLOY to become Available (image pull ~3 min, 21.6 GB of weights ~5 min, kernel warm-up)..."
kubectl -n "$NAMESPACE" wait --for=condition=Available deploy/"$DEPLOY" --timeout=40m

echo "vLLM startup summary:"
kubectl -n "$NAMESPACE" logs deploy/"$DEPLOY" --tail=400 | grep -E 'Mamba SSU backend|GPU KV cache size|Maximum concurrency|Application startup complete' || true
echo
echo "Next: kubectl -n $NAMESPACE port-forward svc/${RELEASE}-nemotron-35-lightning-engine-service 8000:80   (in another terminal), then python3 validate.py"
