#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Deploy Nemotron 3.5 Lightning (NVFP4) with the vLLM Production Stack chart into namespace `lightning`,
# then wait for the engine to finish loading. Requires KUBECONFIG pointing at the cluster.
set -euo pipefail
# Requires kubectl, helm 3.13 or newer (helm get metadata), and jq.
HERE="$(cd "$(dirname "$0")" && pwd)"
RELEASE="${RELEASE:-lightning}"; NAMESPACE="${NAMESPACE:-lightning}"; CHART_VERSION="${CHART_VERSION:-0.1.12}"

OWNER="nemotron-lightning-vllm-oke"
OWNER_LABEL="nvidia-oci-samples/owner=$OWNER"
MARKER="nvidia-oci-samples-owner"            # ConfigMap recording the release this sample created
NODE_POOL_NAME="${NODE_POOL_NAME:-nemotron-lightning-a10x2}"   # must match create-node-pool.sh

if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
  # Create the namespace with the ownership label in one request so cleanup.sh can recognize it.
  # `kubectl create` (not apply) so a namespace that appears concurrently is never adopted.
  if ! kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml \
      | kubectl label --local -f - "$OWNER_LABEL" -o yaml \
      | kubectl create -f -; then
    echo "ERROR: namespace $NAMESPACE appeared while this script was creating it; not adopting it. Re-run to install into it." >&2
    exit 1
  fi
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

# Ownership marker. It records the release name and, once installed, Helm's own firstDeployed
# timestamp for that release instance, so a stale marker cannot claim an unrelated release that
# later reuses the same (namespace, release) pair. helm get metadata needs Helm 3.13 or newer.
marker_field() { kubectl -n "$NAMESPACE" get configmap "$MARKER" -o jsonpath="{.data.$1}" 2>/dev/null || true; }
release_first_deployed() { helm get metadata "$RELEASE" -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.firstDeployed // empty'; }

NEW_INSTALL=0
if helm status "$RELEASE" -n "$NAMESPACE" >/dev/null 2>&1; then
  # Never overwrite a release this sample did not create, or one that replaced ours.
  if [ "$(marker_field release)" != "$RELEASE" ] || [ -z "$(marker_field first_deployed)" ] \
     || [ "$(marker_field first_deployed)" != "$(release_first_deployed)" ]; then
    echo "ERROR: a Helm release named $RELEASE already exists in $NAMESPACE and does not match this sample's ownership record." >&2
    echo "       Choose another RELEASE/NAMESPACE, or remove it yourself first." >&2
    exit 1
  fi
else
  # Persist ownership before Helm mutates anything; remove it again if the install fails.
  NEW_INSTALL=1
  kubectl -n "$NAMESPACE" create configmap "$MARKER" --from-literal=release="$RELEASE" --from-literal=owner="$OWNER" \
    --from-literal=first_deployed="" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
fi

helm repo add vllm https://vllm-project.github.io/production-stack >/dev/null 2>&1 || true
helm repo update vllm >/dev/null
if ! helm upgrade --install "$RELEASE" vllm/vllm-stack --version "$CHART_VERSION" \
    --namespace "$NAMESPACE" -f "$HERE/values.yaml" \
    --set-string "servingEngineSpec.modelSpec[0].nodeSelectorTerms[0].matchExpressions[0].values[0]=$NODE_POOL_NAME"; then
  [ "$NEW_INSTALL" = 1 ] && kubectl -n "$NAMESPACE" delete configmap "$MARKER" --ignore-not-found >/dev/null
  echo "ERROR: helm upgrade --install failed" >&2
  exit 1
fi
if [ "$NEW_INSTALL" = 1 ]; then
  kubectl -n "$NAMESPACE" patch configmap "$MARKER" --type merge -p "{\"data\":{\"first_deployed\":\"$(release_first_deployed)\"}}" >/dev/null
fi

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
