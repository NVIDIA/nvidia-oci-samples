#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Create a 1 x VM.GPU.A10.2 node pool on an existing OKE cluster, with the cloud-init that grows the
# root filesystem (cloud-init.sh). Waits until the node is ACTIVE and Ready in Kubernetes.
# Requires OCI_REGION, OCI_COMPARTMENT_ID, OKE_CLUSTER_ID, WORKER_SUBNET_ID. Optional: AVAILABILITY_DOMAIN,
# NODE_POOL_NAME (default nemotron-lightning-a10x2; must match values.yaml's nodeSelectorTerms), BOOT_VOLUME_GB
# (default 250), OCI_CLI_PROFILE, OCI_CLI_AUTH. Nodes are labeled nvidia-oci-samples/pool=<NODE_POOL_NAME>.
set -euo pipefail
: "${OCI_REGION:?}" "${OCI_COMPARTMENT_ID:?}" "${OKE_CLUSTER_ID:?}" "${WORKER_SUBNET_ID:?}"
OCI=(oci --region "$OCI_REGION" ${OCI_CLI_PROFILE:+--profile "$OCI_CLI_PROFILE"} ${OCI_CLI_AUTH:+--auth "$OCI_CLI_AUTH"})
NODE_POOL_NAME="${NODE_POOL_NAME:-nemotron-lightning-a10x2}"
BOOT_VOLUME_GB="${BOOT_VOLUME_GB:-250}"
HERE="$(cd "$(dirname "$0")" && pwd)"

K8S_VERSION=$("${OCI[@]}" ce cluster get --cluster-id "$OKE_CLUSTER_ID" --query 'data."kubernetes-version"' --raw-output)
echo "cluster Kubernetes version: $K8S_VERSION"

# Newest GPU node image built for exactly this Kubernetes version. The trailing dash keeps
# OKE-1.33.1- from matching OKE-1.33.10-.
GPU_IMAGE_ID=$("${OCI[@]}" ce node-pool-options get --node-pool-option-id all --compartment-id "$OCI_COMPARTMENT_ID" \
  --query "data.sources[?contains(\"source-name\", 'GPU') && contains(\"source-name\", 'OKE-${K8S_VERSION#v}-')] | sort_by(@, &\"source-name\") | reverse(@) | [0].\"image-id\"" --raw-output)
[ -n "$GPU_IMAGE_ID" ] && [ "$GPU_IMAGE_ID" != "null" ] || { echo "no GPU node image found for $K8S_VERSION"; exit 1; }
echo "GPU node image: $GPU_IMAGE_ID"

if [ -z "${AVAILABILITY_DOMAIN:-}" ]; then
  for AD in $("${OCI[@]}" iam availability-domain list --compartment-id "$OCI_COMPARTMENT_ID" --query 'data[*].name' --raw-output | jq -r '.[]'); do
    STATUS=$("${OCI[@]}" compute compute-capacity-report create --compartment-id "$OCI_COMPARTMENT_ID" --availability-domain "$AD" \
      --shape-availabilities '[{"instanceShape":"VM.GPU.A10.2"}]' --query 'data."shape-availabilities"[0]."availability-status"' --raw-output)
    if [ "$STATUS" = "AVAILABLE" ]; then AVAILABILITY_DOMAIN="$AD"; break; fi
  done
  [ -n "${AVAILABILITY_DOMAIN:-}" ] || { echo "no availability domain currently has VM.GPU.A10.2 capacity in $OCI_REGION"; exit 1; }
fi
echo "availability domain: $AVAILABILITY_DOMAIN"

USER_DATA=$(base64 < "$HERE/cloud-init.sh" | tr -d '\n')

echo "creating node pool $NODE_POOL_NAME (1 x VM.GPU.A10.2, ${BOOT_VOLUME_GB} GB boot volume)"
POOL_ID=$("${OCI[@]}" ce node-pool create \
  --cluster-id "$OKE_CLUSTER_ID" --compartment-id "$OCI_COMPARTMENT_ID" \
  --name "$NODE_POOL_NAME" --kubernetes-version "$K8S_VERSION" \
  --node-shape VM.GPU.A10.2 --size 1 \
  --node-image-id "$GPU_IMAGE_ID" --node-boot-volume-size-in-gbs "$BOOT_VOLUME_GB" \
  --node-metadata "{\"user_data\": \"$USER_DATA\"}" \
  --placement-configs "[{\"availabilityDomain\": \"$AVAILABILITY_DOMAIN\", \"subnetId\": \"$WORKER_SUBNET_ID\"}]" \
  --initial-node-labels "[{\"key\": \"nvidia-oci-samples/pool\", \"value\": \"$NODE_POOL_NAME\"}]" \
  --wait-for-state SUCCEEDED --wait-for-state FAILED --max-wait-seconds 1800 \
  --query 'data.resources[?"entity-type"==`nodepool`].identifier | [0]' --raw-output)
echo "node pool: $POOL_ID"

echo "waiting for the pool's node to become ACTIVE (about 10 minutes)..."
until [ "$("${OCI[@]}" ce node-pool get --node-pool-id "$POOL_ID" --query 'data.nodes[0]."lifecycle-state"' --raw-output 2>/dev/null)" = "ACTIVE" ]; do sleep 20; done
NODE_IP=$("${OCI[@]}" ce node-pool get --node-pool-id "$POOL_ID" --query 'data.nodes[0]."private-ip"' --raw-output)
echo "node $NODE_IP is ACTIVE; waiting for it to be Ready in Kubernetes with both GPUs advertised..."
until [ "$(kubectl get node "$NODE_IP" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)" = "True" ] \
   && [ "$(kubectl get node "$NODE_IP" -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null)" = "2" ]; do sleep 15; done
kubectl get node "$NODE_IP" \
  -o custom-columns='NODE:.metadata.name,POOL:.metadata.labels.nvidia-oci-samples/pool,READY:.status.conditions[?(@.type=="Ready")].status,GPUS:.status.capacity.nvidia\.com/gpu,EPHEMERAL:.status.capacity.ephemeral-storage'
echo "Expected EPHEMERAL close to the boot volume size (about 238Gi for 250 GB). If it reads ~30Gi, the cloud-init did not run; see README Pitfalls."
