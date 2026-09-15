#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Preflight: tools, VM.GPU.A10.2 capacity per availability domain, and the cluster's Kubernetes version.
# Read-only. Requires OCI_REGION, OCI_COMPARTMENT_ID, OKE_CLUSTER_ID.
set -euo pipefail
: "${OCI_REGION:?set OCI_REGION}" "${OCI_COMPARTMENT_ID:?set OCI_COMPARTMENT_ID}" "${OKE_CLUSTER_ID:?set OKE_CLUSTER_ID}"
OCI=(oci --region "$OCI_REGION" ${OCI_CLI_PROFILE:+--profile "$OCI_CLI_PROFILE"} ${OCI_CLI_AUTH:+--auth "$OCI_CLI_AUTH"})

for t in oci kubectl helm python3 jq; do command -v "$t" >/dev/null || { echo "missing tool: $t"; exit 1; }; done
echo "tools: ok"

echo "cluster:"
"${OCI[@]}" ce cluster get --cluster-id "$OKE_CLUSTER_ID" \
  --query 'data.{name:name,version:"kubernetes-version",state:"lifecycle-state",type:type}' --output table

echo "kubectl:"
kubectl get nodes -o wide --request-timeout=20s | head -5

echo "VM.GPU.A10.2 capacity per availability domain (informational capacity report):"
for AD in $("${OCI[@]}" iam availability-domain list --compartment-id "$OCI_COMPARTMENT_ID" --query 'data[*].name' --raw-output | jq -r '.[]'); do
  "${OCI[@]}" compute compute-capacity-report create --compartment-id "$OCI_COMPARTMENT_ID" --availability-domain "$AD" \
    --shape-availabilities '[{"instanceShape":"VM.GPU.A10.2"}]' \
    --query 'data."shape-availabilities"[0].{ad:`'"$AD"'`,status:"availability-status",count:"available-count"}' --output table | grep -v '^$'
done
echo "Pick an AD with status AVAILABLE and export AVAILABILITY_DOMAIN=<name> before running create-node-pool.sh (or let it pick the first available)."
