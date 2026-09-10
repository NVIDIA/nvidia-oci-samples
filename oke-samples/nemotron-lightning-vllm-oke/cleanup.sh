#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Remove what this sample created: the Helm release, the namespace (only if deploy.sh created it), then the
# GPU node pool (after confirmation).
# The OKE cluster, VCN, and anything you already had are left untouched.
# Requires OCI_REGION, OCI_COMPARTMENT_ID, OKE_CLUSTER_ID. Optional: NODE_POOL_NAME, OCI_CLI_PROFILE, OCI_CLI_AUTH.
set -euo pipefail
: "${OCI_REGION:?}" "${OCI_COMPARTMENT_ID:?}" "${OKE_CLUSTER_ID:?}"
OCI=(oci --region "$OCI_REGION" ${OCI_CLI_PROFILE:+--profile "$OCI_CLI_PROFILE"} ${OCI_CLI_AUTH:+--auth "$OCI_CLI_AUTH"})
RELEASE="${RELEASE:-lightning}"; NAMESPACE="${NAMESPACE:-lightning}"; NODE_POOL_NAME="${NODE_POOL_NAME:-nemotron-lightning-a10x2}"

helm uninstall "$RELEASE" -n "$NAMESPACE" 2>/dev/null || true
if kubectl get namespace "$NAMESPACE" -o jsonpath='{.metadata.labels.nvidia-oci-samples/owner}' 2>/dev/null | grep -q '^nemotron-lightning-vllm-oke$'; then
  kubectl delete namespace "$NAMESPACE" --wait=false
  echo "namespace $NAMESPACE deleted (created by this sample)"
else
  echo "namespace $NAMESPACE was not created by this sample; left in place"
fi

POOL_ID=$("${OCI[@]}" ce node-pool list --cluster-id "$OKE_CLUSTER_ID" --compartment-id "$OCI_COMPARTMENT_ID" \
  --query "data[?name=='$NODE_POOL_NAME' && \"lifecycle-state\"!='DELETED'].id | [0]" --raw-output)
if [ -n "$POOL_ID" ] && [ "$POOL_ID" != "null" ]; then
  read -r -p "Delete node pool $NODE_POOL_NAME ($POOL_ID)? This terminates the GPU node. [y/N] " ANSWER
  if [ "$ANSWER" = "y" ] || [ "$ANSWER" = "Y" ]; then
    "${OCI[@]}" ce node-pool delete --node-pool-id "$POOL_ID" --force --wait-for-state SUCCEEDED --wait-for-state FAILED --max-wait-seconds 1800 >/dev/null
    echo "node pool deleted"
  else
    echo "node pool kept (still billing)"
  fi
else
  echo "no node pool named $NODE_POOL_NAME found"
fi
