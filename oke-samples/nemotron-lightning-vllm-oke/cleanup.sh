#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Remove what this sample created: the Helm release, the namespace (only if deploy.sh created it), then the
# GPU node pool (after confirmation).
# The OKE cluster, VCN, and anything you already had are left untouched.
# Requires OCI_REGION, OCI_COMPARTMENT_ID, OKE_CLUSTER_ID, kubectl, helm 3.13+, jq. Optional: NODE_POOL_NAME,
# OCI_CLI_PROFILE, OCI_CLI_AUTH, ASSUME_YES=1 to skip the two confirmation prompts.
set -euo pipefail
: "${OCI_REGION:?}" "${OCI_COMPARTMENT_ID:?}" "${OKE_CLUSTER_ID:?}"
OCI=(oci --region "$OCI_REGION" ${OCI_CLI_PROFILE:+--profile "$OCI_CLI_PROFILE"} ${OCI_CLI_AUTH:+--auth "$OCI_CLI_AUTH"})
RELEASE="${RELEASE:-lightning}"; NAMESPACE="${NAMESPACE:-lightning}"; NODE_POOL_NAME="${NODE_POOL_NAME:-nemotron-lightning-a10x2}"

MARKER="nvidia-oci-samples-owner"
marker_field() { kubectl -n "$NAMESPACE" get configmap "$MARKER" -o jsonpath="{.data.$1}" 2>/dev/null || true; }
release_first_deployed() { helm get metadata "$RELEASE" -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.firstDeployed // empty'; }
confirm() { # confirm "<question>" ; honors ASSUME_YES=1 for unattended runs
  if [ "${ASSUME_YES:-0}" = 1 ]; then return 0; fi
  read -r -p "$1 [y/N] " ANSWER; [ "$ANSWER" = "y" ] || [ "$ANSWER" = "Y" ]
}

# The marker lives in the namespace, so it is only a hint; the release is uninstalled only when the
# marker matches the live release's Helm firstDeployed timestamp AND you confirm.
if helm status "$RELEASE" -n "$NAMESPACE" >/dev/null 2>&1; then
  if [ "$(marker_field release)" = "$RELEASE" ] && [ -n "$(release_first_deployed)" ] \
     && [ "$(marker_field first_deployed)" = "$(release_first_deployed)" ]; then
    if confirm "Uninstall Helm release $RELEASE in namespace $NAMESPACE (first deployed $(release_first_deployed))?"; then
      helm uninstall "$RELEASE" -n "$NAMESPACE"
      kubectl -n "$NAMESPACE" delete configmap "$MARKER" --ignore-not-found >/dev/null
      echo "release $RELEASE uninstalled"
    else
      echo "release $RELEASE kept"
    fi
  else
    echo "release $RELEASE in $NAMESPACE does not match this sample's ownership record; not uninstalled"
  fi
else
  echo "no Helm release named $RELEASE in $NAMESPACE"
  kubectl -n "$NAMESPACE" delete configmap "$MARKER" --ignore-not-found >/dev/null 2>&1 || true
fi
if kubectl get namespace "$NAMESPACE" -o jsonpath='{.metadata.labels.nvidia-oci-samples/owner}' 2>/dev/null | grep -q '^nemotron-lightning-vllm-oke$'; then
  if [ -n "$(helm list -n "$NAMESPACE" -q 2>/dev/null)" ]; then
    echo "namespace $NAMESPACE was created by this sample but still holds Helm releases; left in place"
  else
    kubectl delete namespace "$NAMESPACE" --wait=false
    echo "namespace $NAMESPACE deleted (created by this sample)"
  fi
else
  echo "namespace $NAMESPACE was not created by this sample; left in place"
fi

POOL_ID=$("${OCI[@]}" ce node-pool list --cluster-id "$OKE_CLUSTER_ID" --compartment-id "$OCI_COMPARTMENT_ID" \
  --query "data[?name=='$NODE_POOL_NAME' && \"lifecycle-state\"!='DELETED'].id | [0]" --raw-output)
if [ -n "$POOL_ID" ] && [ "$POOL_ID" != "null" ]; then
  if confirm "Delete node pool $NODE_POOL_NAME ($POOL_ID)? This terminates the GPU node."; then
    "${OCI[@]}" ce node-pool delete --node-pool-id "$POOL_ID" --force --wait-for-state SUCCEEDED --wait-for-state FAILED --max-wait-seconds 1800 >/dev/null
    echo "node pool deleted"
  else
    echo "node pool kept (still billing)"
  fi
else
  echo "no node pool named $NODE_POOL_NAME found"
fi
