#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Optional: delete a cluster created by create-cluster.sh, then the VCN it built. Deletes only resources
# that carry the freeform tag nvidia-oci-samples-owner=nemotron-lightning-vllm-oke, refuses a cluster that
# still has node pools (run cleanup.sh first), leaves the VCN alone if another cluster still uses it, and
# refuses to touch a VCN that contains anything without the tag.
# Requires OCI_REGION, OCI_COMPARTMENT_ID, OKE_CLUSTER_ID. Optional: ASSUME_YES=1 (skip the prompt),
# OCI_CLI_PROFILE, OCI_CLI_AUTH.
set -euo pipefail
: "${OCI_REGION:?set OCI_REGION}" "${OCI_COMPARTMENT_ID:?set OCI_COMPARTMENT_ID}" "${OKE_CLUSTER_ID:?set OKE_CLUSTER_ID}"
OCI=(oci --region "$OCI_REGION" ${OCI_CLI_PROFILE:+--profile "$OCI_CLI_PROFILE"} ${OCI_CLI_AUTH:+--auth "$OCI_CLI_AUTH"})
OWNER_KEY=nvidia-oci-samples-owner; OWNER_VALUE=nemotron-lightning-vllm-oke
C=$OCI_COMPARTMENT_ID
log(){ echo "[$(date +%H:%M:%S)] $*"; }
confirm(){ [ "${ASSUME_YES:-0}" = "1" ] && return 0; read -r -p "$1 [y/N] " a; [ "$a" = "y" ] || [ "$a" = "Y" ]; }
ids(){ jq -r '.[]?' 2>/dev/null || true; }

CLUSTER=$("${OCI[@]}" ce cluster get --cluster-id "$OKE_CLUSTER_ID" \
  --query "data.{name:name,state:\"lifecycle-state\",vcn:\"vcn-id\",owner:\"freeform-tags\".\"$OWNER_KEY\"}" --output json)
NAME=$(jq -r .name <<<"$CLUSTER"); STATE=$(jq -r .state <<<"$CLUSTER"); VCN_ID=$(jq -r .vcn <<<"$CLUSTER"); OWNER=$(jq -r '.owner // empty' <<<"$CLUSTER")
if [ "$OWNER" != "$OWNER_VALUE" ]; then
  echo "refusing: cluster $NAME ($STATE) does not carry the tag $OWNER_KEY=$OWNER_VALUE, so create-cluster.sh did not create it. Delete it yourself if that is what you want." >&2
  exit 1
fi
POOLS=$("${OCI[@]}" ce node-pool list --compartment-id "$C" --cluster-id "$OKE_CLUSTER_ID" \
  --query 'data[?"lifecycle-state"!=`DELETED`].name' --raw-output | ids)
if [ -n "$POOLS" ]; then
  echo "refusing: cluster $NAME still has node pools ($(tr '\n' ' ' <<<"$POOLS")). Run ./cleanup.sh first so the GPU node is removed deliberately." >&2
  exit 1
fi

confirm "Delete OKE cluster $NAME ($STATE) and, if nothing else uses it, its VCN?" || { echo "aborted"; exit 1; }
log "deleting cluster $NAME (a few minutes)"
"${OCI[@]}" ce cluster delete --cluster-id "$OKE_CLUSTER_ID" --force --wait-for-state SUCCEEDED --wait-for-state FAILED --max-wait-seconds 1800 >/dev/null
log "cluster deleted"

VCN_OWNER=$("${OCI[@]}" network vcn get --vcn-id "$VCN_ID" --query "data.\"freeform-tags\".\"$OWNER_KEY\"" --raw-output 2>/dev/null || true)
if [ "$VCN_OWNER" != "$OWNER_VALUE" ]; then log "VCN is not tagged $OWNER_KEY=$OWNER_VALUE; leaving it in place"; exit 0; fi
OTHERS=$("${OCI[@]}" ce cluster list --compartment-id "$C" --query "data[?\"vcn-id\"=='$VCN_ID' && \"lifecycle-state\"!='DELETED'].name" --raw-output | ids)
if [ -n "$OTHERS" ]; then log "VCN is still used by cluster(s): $(tr '\n' ' ' <<<"$OTHERS"); leaving it in place"; exit 0; fi

DEFAULT_RT=$("${OCI[@]}" network vcn get --vcn-id "$VCN_ID" --query 'data."default-route-table-id"' --raw-output)
DEFAULT_SL=$("${OCI[@]}" network vcn get --vcn-id "$VCN_ID" --query 'data."default-security-list-id"' --raw-output)
DEFAULT_DHCP=$("${OCI[@]}" network vcn get --vcn-id "$VCN_ID" --query 'data."default-dhcp-options-id"' --raw-output)
OWNED_Q="data[?\"freeform-tags\".\"$OWNER_KEY\"=='$OWNER_VALUE'].id"

# The VCN tag proves ownership of the VCN only. Every child create-cluster.sh made carries the same tag, so
# anything inside the VCN without it (a subnet, route table, gateway, NSG, peering, or DRG attachment someone
# added later) means the VCN is shared. Refuse before changing anything; the OCI-created defaults are exempt.
log "checking that everything inside the VCN carries the ownership tag"
UNOWNED=""
check_owned() { # <label> <network list subcommand>
  local label=$1 all owned id; shift
  all=$("${OCI[@]}" network "$@" --compartment-id "$C" --vcn-id "$VCN_ID" --query 'data[*].id' --raw-output | ids)
  owned=$("${OCI[@]}" network "$@" --compartment-id "$C" --vcn-id "$VCN_ID" --query "$OWNED_Q" --raw-output | ids)
  for id in $all; do
    case "$id" in "$DEFAULT_RT"|"$DEFAULT_SL"|"$DEFAULT_DHCP") continue;; esac
    grep -qx "$id" <<<"$owned" || UNOWNED+="  $label $id"$'\n'
  done
}
check_owned "route table" route-table list
check_owned "security list" security-list list
check_owned "subnet" subnet list
check_owned "internet gateway" internet-gateway list
check_owned "NAT gateway" nat-gateway list
check_owned "service gateway" service-gateway list
check_owned "network security group" nsg list
check_owned "local peering gateway" local-peering-gateway list
check_owned "DRG attachment" drg-attachment list
if [ -n "$UNOWNED" ]; then
  echo "refusing to touch VCN $VCN_ID: it contains resources create-cluster.sh did not create:" >&2
  printf '%s' "$UNOWNED" >&2
  echo "The cluster is deleted. Remove those resources yourself, or delete the VCN from the Console." >&2
  exit 1
fi

log "deleting the VCN and everything create-cluster.sh put in it"
# Route rules reference the gateways, so empty the owned route tables before deleting anything.
for RT in $("${OCI[@]}" network route-table list --compartment-id "$C" --vcn-id "$VCN_ID" --query "$OWNED_Q" --raw-output | ids); do
  "${OCI[@]}" network route-table update --rt-id "$RT" --route-rules '[]' --force >/dev/null
done
for S in $("${OCI[@]}" network subnet list --compartment-id "$C" --vcn-id "$VCN_ID" --query "$OWNED_Q" --raw-output | ids); do
  "${OCI[@]}" network subnet delete --subnet-id "$S" --force --wait-for-state TERMINATED --max-wait-seconds 600 && log "  subnet deleted"
done
for G in $("${OCI[@]}" network internet-gateway list --compartment-id "$C" --vcn-id "$VCN_ID" --query "$OWNED_Q" --raw-output | ids); do
  "${OCI[@]}" network internet-gateway delete --ig-id "$G" --force --wait-for-state TERMINATED && log "  internet gateway deleted"
done
for G in $("${OCI[@]}" network nat-gateway list --compartment-id "$C" --vcn-id "$VCN_ID" --query "$OWNED_Q" --raw-output | ids); do
  "${OCI[@]}" network nat-gateway delete --nat-gateway-id "$G" --force --wait-for-state TERMINATED && log "  NAT gateway deleted"
done
for G in $("${OCI[@]}" network service-gateway list --compartment-id "$C" --vcn-id "$VCN_ID" --query "$OWNED_Q" --raw-output | ids); do
  "${OCI[@]}" network service-gateway delete --service-gateway-id "$G" --force --wait-for-state TERMINATED && log "  service gateway deleted"
done
for RT in $("${OCI[@]}" network route-table list --compartment-id "$C" --vcn-id "$VCN_ID" --query "$OWNED_Q" --raw-output | ids); do
  "${OCI[@]}" network route-table delete --rt-id "$RT" --force --wait-for-state TERMINATED
done
for SL in $("${OCI[@]}" network security-list list --compartment-id "$C" --vcn-id "$VCN_ID" --query "$OWNED_Q" --raw-output | ids); do
  "${OCI[@]}" network security-list delete --security-list-id "$SL" --force --wait-for-state TERMINATED
done
"${OCI[@]}" network vcn delete --vcn-id "$VCN_ID" --force --wait-for-state TERMINATED --max-wait-seconds 600
log "VCN deleted. Nothing from create-cluster.sh remains."
