#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Remove what this sample created: the Helm release, the namespace (only if deploy.sh created it), then the
# GPU node pool (after confirmation).
# The OKE cluster, VCN, and anything you already had are left untouched.
# Requires OCI_REGION, OCI_COMPARTMENT_ID, OKE_CLUSTER_ID, kubectl, helm 3.13+, jq. Optional: NODE_POOL_NAME,
# OCI_CLI_PROFILE, OCI_CLI_AUTH, STATE_DIR (where deploy.sh keeps its ownership receipts), ASSUME_YES=1 to skip
# the confirmation prompts. Unattended uninstall of the Helm release additionally requires the local receipt
# deploy.sh wrote on the machine that installed it; the in-cluster marker alone only allows an interactive
# uninstall, because anyone who can write ConfigMaps in the namespace could have written the marker.
set -euo pipefail
: "${OCI_REGION:?}" "${OCI_COMPARTMENT_ID:?}" "${OKE_CLUSTER_ID:?}"
OCI=(oci --region "$OCI_REGION" ${OCI_CLI_PROFILE:+--profile "$OCI_CLI_PROFILE"} ${OCI_CLI_AUTH:+--auth "$OCI_CLI_AUTH"})
RELEASE="${RELEASE:-lightning}"; NAMESPACE="${NAMESPACE:-lightning}"; NODE_POOL_NAME="${NODE_POOL_NAME:-nemotron-lightning-a10x2}"
STATE_DIR="${STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/nvidia-oci-samples/nemotron-lightning-vllm-oke}"
FAILED=0

MARKER="nvidia-oci-samples-owner"
marker_field() { kubectl -n "$NAMESPACE" get configmap "$MARKER" -o jsonpath="{.data.$1}" 2>/dev/null || true; }
release_first_deployed() { helm get metadata "$RELEASE" -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.firstDeployed // empty'; }
digest() { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; }
receipt_path() { # one receipt per (cluster API server, namespace, release), written by deploy.sh
  local server; server=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}' 2>/dev/null || true)
  printf '%s/%s.first_deployed' "$STATE_DIR" "$(printf '%s|%s|%s' "$server" "$NAMESPACE" "$RELEASE" | digest | cut -c1-32)"
}
confirm() { # confirm "<question>" ; honors ASSUME_YES=1 for unattended runs
  if [ "${ASSUME_YES:-0}" = 1 ]; then return 0; fi
  read -r -p "$1 [y/N] " ANSWER; [ "$ANSWER" = "y" ] || [ "$ANSWER" = "Y" ]
}
uninstall_release() {
  helm uninstall "$RELEASE" -n "$NAMESPACE"
  kubectl -n "$NAMESPACE" delete configmap "$MARKER" --ignore-not-found >/dev/null
  rm -f "$(receipt_path)"
  echo "release $RELEASE uninstalled"
}

if helm status "$RELEASE" -n "$NAMESPACE" >/dev/null 2>&1; then
  LIVE=$(release_first_deployed)
  RECEIPT=$(cat "$(receipt_path)" 2>/dev/null || true)
  if [ -n "$LIVE" ] && [ "$RECEIPT" = "$LIVE" ]; then
    # The local receipt was written by the deploy.sh run that installed this release instance. Nothing in the
    # cluster can forge it, so it is enough for an unattended uninstall.
    if confirm "Uninstall Helm release $RELEASE in namespace $NAMESPACE (first deployed $LIVE)?"; then uninstall_release
    else echo "release $RELEASE kept"; fi
  elif [ -n "$LIVE" ] && [ "$(marker_field release)" = "$RELEASE" ] && [ "$(marker_field first_deployed)" = "$LIVE" ]; then
    # Only the in-cluster marker matches. It lives in the namespace, so it is a hint, never an authorization.
    if [ "${ASSUME_YES:-0}" = 1 ]; then
      echo "release $RELEASE matches the in-cluster marker, but this machine has no ownership receipt from the deploy.sh run that installed it; unattended uninstall refused. Run without ASSUME_YES to confirm interactively."
    elif confirm "Uninstall Helm release $RELEASE in namespace $NAMESPACE (first deployed $LIVE; in-cluster marker matches, no local receipt)?"; then uninstall_release
    else echo "release $RELEASE kept"; fi
  else
    echo "release $RELEASE in $NAMESPACE does not match this sample's ownership record; not uninstalled"
  fi
else
  echo "no Helm release named $RELEASE in $NAMESPACE"
  kubectl -n "$NAMESPACE" delete configmap "$MARKER" --ignore-not-found >/dev/null 2>&1 || true
  rm -f "$(receipt_path)"
fi
if kubectl get namespace "$NAMESPACE" -o jsonpath='{.metadata.labels.nvidia-oci-samples/owner}' 2>/dev/null | grep -q '^nemotron-lightning-vllm-oke$'; then
  # helm list shows only deployed and failed releases by default; include the in-progress states too, and
  # fail closed: if the lookup itself fails, keep the namespace.
  if ! RELEASES=$(helm list -n "$NAMESPACE" -q --deployed --failed --pending --uninstalling); then
    echo "ERROR: could not list Helm releases in namespace $NAMESPACE; namespace retained" >&2; FAILED=1
  elif [ -n "$RELEASES" ]; then
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
exit "$FAILED"
