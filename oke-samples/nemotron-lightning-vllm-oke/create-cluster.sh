#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Optional: create a small OKE cluster so the sample can run from an empty compartment. Builds a VCN
# (Internet, NAT, and Service gateways; API, worker, and load-balancer subnets), a Basic OKE cluster with a
# public API endpoint and Flannel networking, and a kubeconfig. Nothing GPU-specific happens here; run
# create-node-pool.sh next. These are the commands that created the cluster used for the results/ run.
# Every resource, including each subnet, route table, security list, and gateway, carries the freeform tag
# nvidia-oci-samples-owner=nemotron-lightning-vllm-oke; delete-cluster.sh deletes only resources that carry it.
#
# Requires OCI_REGION, OCI_COMPARTMENT_ID. Optional: CLUSTER_NAME (default nemotron-lightning),
# KUBERNETES_VERSION (default: the newest version OKE offers that also has a GPU node image, so create-node-pool.sh
# can find one; the results/ run used v1.35.2), VCN_CIDR (default 10.0.0.0/16; must be a private /16 not overlapping
# 10.244.0.0/16 or 10.96.0.0/16; the subnets are carved from its first three /24s), API_ALLOWED_CIDR (source range allowed to reach the
# Kubernetes API on 6443; default 0.0.0.0/0 like the Console quick-create, the API still requires signed OCI
# tokens; tighten it to your egress range if you know it), KUBECONFIG_OUT (default ./kubeconfig),
# OCI_CLI_PROFILE, OCI_CLI_AUTH. Prints the exports the other scripts need.
set -euo pipefail
: "${OCI_REGION:?set OCI_REGION}" "${OCI_COMPARTMENT_ID:?set OCI_COMPARTMENT_ID}"
OCI=(oci --region "$OCI_REGION" ${OCI_CLI_PROFILE:+--profile "$OCI_CLI_PROFILE"} ${OCI_CLI_AUTH:+--auth "$OCI_CLI_AUTH"})
CLUSTER_NAME="${CLUSTER_NAME:-nemotron-lightning}"
K8S="${KUBERNETES_VERSION:-}"
VCN_CIDR="${VCN_CIDR:-10.0.0.0/16}"
API_ALLOWED_CIDR="${API_ALLOWED_CIDR:-0.0.0.0/0}"
KCFG="${KUBECONFIG_OUT:-./kubeconfig}"
PODS_CIDR=10.244.0.0/16; SERVICES_CIDR=10.96.0.0/16
OWNER_TAGS='{"nvidia-oci-samples-owner": "nemotron-lightning-vllm-oke"}'
C=$OCI_COMPARTMENT_ID
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# Every resource is recorded as it is created, so a failure part-way through prints what exists and the
# exact commands that remove it, newest first (the order OCI dependencies require).
CREATED=()
note() { CREATED+=("$1 $2"); }
delete_hint() { # <kind> <id>
  local o="oci --region $OCI_REGION${OCI_CLI_PROFILE:+ --profile $OCI_CLI_PROFILE}${OCI_CLI_AUTH:+ --auth $OCI_CLI_AUTH}"
  case "$1" in
    cluster)          echo "$o ce cluster delete --cluster-id $2 --force";;
    subnet)           echo "$o network subnet delete --subnet-id $2 --force";;
    security-list)    echo "$o network security-list delete --security-list-id $2 --force";;
    route-table)      echo "$o network route-table delete --rt-id $2 --force";;
    service-gateway)  echo "$o network service-gateway delete --service-gateway-id $2 --force";;
    nat-gateway)      echo "$o network nat-gateway delete --nat-gateway-id $2 --force";;
    internet-gateway) echo "$o network internet-gateway delete --ig-id $2 --force";;
    vcn)              echo "$o network vcn delete --vcn-id $2 --force";;
  esac
}
on_exit() {
  local rc=$? i id
  [ "$rc" = 0 ] && return 0
  echo >&2; echo "ERROR: create-cluster.sh failed (exit $rc)." >&2
  # A cluster request that did not end ACTIVE can still leave a cluster (CREATING or FAILED) behind.
  if [ -n "${VCN_ID:-}" ] && [ -z "${CLUSTER_ID:-}" ]; then
    for id in $("${OCI[@]}" ce cluster list --compartment-id "$C" --query "data[?\"vcn-id\"=='$VCN_ID' && \"lifecycle-state\"!='DELETED'].id" --raw-output 2>/dev/null | jq -r '.[]?'); do note cluster "$id"; done
  fi
  if [ "${#CREATED[@]}" = 0 ]; then echo "Nothing was created." >&2; return 0; fi
  echo "Resources created so far, newest first. Remove them in this order (re-run a command if OCI reports a dependency that is still terminating):" >&2
  for ((i=${#CREATED[@]}-1; i>=0; i--)); do
    set -- ${CREATED[$i]}
    echo "  $1  $2" >&2; echo "    $(delete_hint "$1" "$2")" >&2
  done
  echo "If a cluster is listed, you can instead wait for it to leave CREATING, export OKE_CLUSTER_ID=<its id>, and run ./delete-cluster.sh, which removes the cluster and the tagged VCN." >&2
}
trap on_exit EXIT

# The VCN must be a private /16 (A.B.0.0/16) so the fixed subnets below fit inside it, and it must not overlap
# the pod or service ranges. Checked before any OCI resource is created.
if ! [[ "$VCN_CIDR" =~ ^(10\.([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])|172\.(1[6-9]|2[0-9]|3[01])|192\.168)\.0\.0/16$ ]]; then
  echo "ERROR: VCN_CIDR must be a private /16 such as 10.0.0.0/16, 172.16.0.0/16, or 192.168.0.0/16 (got $VCN_CIDR)" >&2; exit 1
fi
case "$VCN_CIDR" in "${PODS_CIDR%.*.*}".0.0/16|"${SERVICES_CIDR%.*.*}".0.0/16)
  echo "ERROR: VCN_CIDR $VCN_CIDR overlaps the pod range $PODS_CIDR or the service range $SERVICES_CIDR" >&2; exit 1;; esac
# A.B.0.0/16 -> A.B.0.0/28 (API endpoint), A.B.10.0/24 (workers), A.B.20.0/24 (load balancers).
BASE=${VCN_CIDR%.*.*}
API_SUBNET_CIDR="$BASE.0.0/28"; WORKER_SUBNET_CIDR="$BASE.10.0/24"; LB_SUBNET_CIDR="$BASE.20.0/24"
# VCN DNS labels: letters and digits only, at most 15 characters.
DNS_LABEL=$(printf '%s' "${CLUSTER_NAME//[^a-zA-Z0-9]/}" | cut -c1-15)

if [ -z "$K8S" ]; then
  # Newest first, compared numerically so v1.36.10 sorts above v1.36.2.
  for V in $("${OCI[@]}" ce cluster-options get --cluster-option-id all --query 'data."kubernetes-versions"' --raw-output \
             | jq -r 'sort_by(ltrimstr("v") | split(".") | map(tonumber)) | reverse | .[]'); do
    N=$("${OCI[@]}" ce node-pool-options get --node-pool-option-id all --compartment-id "$C" \
      --query "length(data.sources[?contains(\"source-name\", 'GPU') && contains(\"source-name\", 'OKE-${V#v}-')])" --raw-output)
    if [ "${N:-0}" -gt 0 ]; then K8S=$V; break; fi
  done
  [ -n "$K8S" ] || { echo "ERROR: no offered Kubernetes version has a GPU node image in $OCI_REGION; set KUBERNETES_VERSION explicitly" >&2; exit 1; }
fi
log "Kubernetes $K8S"

log "VCN $CLUSTER_NAME-vcn ($VCN_CIDR) with Internet, NAT, and Service gateways"
VCN_ID=$("${OCI[@]}" network vcn create --compartment-id "$C" --display-name "$CLUSTER_NAME-vcn" --cidr-blocks "[\"$VCN_CIDR\"]" \
  --dns-label "$DNS_LABEL" --freeform-tags "$OWNER_TAGS" --wait-for-state AVAILABLE --query data.id --raw-output)
note vcn "$VCN_ID"
IGW_ID=$("${OCI[@]}" network internet-gateway create --compartment-id "$C" --vcn-id "$VCN_ID" --display-name "$CLUSTER_NAME-igw" --freeform-tags "$OWNER_TAGS" \
  --is-enabled true --wait-for-state AVAILABLE --query data.id --raw-output)
note internet-gateway "$IGW_ID"
NAT_ID=$("${OCI[@]}" network nat-gateway create --compartment-id "$C" --vcn-id "$VCN_ID" --display-name "$CLUSTER_NAME-nat" --freeform-tags "$OWNER_TAGS" \
  --wait-for-state AVAILABLE --query data.id --raw-output)
note nat-gateway "$NAT_ID"
SVC_ID=$("${OCI[@]}" network service list --query "data[?contains(name, 'All') && contains(name, 'Services')].id | [0]" --raw-output)
SVC_CIDR=$("${OCI[@]}" network service list --query "data[?contains(name, 'All') && contains(name, 'Services')].\"cidr-block\" | [0]" --raw-output)
SGW_ID=$("${OCI[@]}" network service-gateway create --compartment-id "$C" --vcn-id "$VCN_ID" --display-name "$CLUSTER_NAME-sgw" --freeform-tags "$OWNER_TAGS" \
  --services "[{\"serviceId\": \"$SVC_ID\"}]" --wait-for-state AVAILABLE --query data.id --raw-output)
note service-gateway "$SGW_ID"

log "route tables and security list"
PRIV_RT=$("${OCI[@]}" network route-table create --compartment-id "$C" --vcn-id "$VCN_ID" --display-name "$CLUSTER_NAME-private-rt" --freeform-tags "$OWNER_TAGS" \
  --route-rules "[{\"cidrBlock\": \"0.0.0.0/0\", \"networkEntityId\": \"$NAT_ID\"},{\"destination\": \"$SVC_CIDR\", \"destinationType\": \"SERVICE_CIDR_BLOCK\", \"networkEntityId\": \"$SGW_ID\"}]" \
  --wait-for-state AVAILABLE --query data.id --raw-output)
note route-table "$PRIV_RT"
PUB_RT=$("${OCI[@]}" network route-table create --compartment-id "$C" --vcn-id "$VCN_ID" --display-name "$CLUSTER_NAME-public-rt" --freeform-tags "$OWNER_TAGS" \
  --route-rules "[{\"cidrBlock\": \"0.0.0.0/0\", \"networkEntityId\": \"$IGW_ID\"}]" --wait-for-state AVAILABLE --query data.id --raw-output)
note route-table "$PUB_RT"
# Ingress: Kubernetes API from API_ALLOWED_CIDR; everything from the VCN, pod, and service ranges; path-MTU ICMP.
SL_ID=$("${OCI[@]}" network security-list create --compartment-id "$C" --vcn-id "$VCN_ID" --display-name "$CLUSTER_NAME-sl" --freeform-tags "$OWNER_TAGS" \
  --egress-security-rules '[{"destination": "0.0.0.0/0", "protocol": "all", "isStateless": false}]' \
  --ingress-security-rules "[
    {\"source\": \"$API_ALLOWED_CIDR\", \"protocol\": \"6\", \"isStateless\": false, \"tcpOptions\": {\"destinationPortRange\": {\"min\": 6443, \"max\": 6443}}},
    {\"source\": \"$VCN_CIDR\", \"protocol\": \"all\", \"isStateless\": false},
    {\"source\": \"$PODS_CIDR\", \"protocol\": \"all\", \"isStateless\": false},
    {\"source\": \"$SERVICES_CIDR\", \"protocol\": \"all\", \"isStateless\": false},
    {\"source\": \"0.0.0.0/0\", \"protocol\": \"1\", \"isStateless\": false, \"icmpOptions\": {\"type\": 3, \"code\": 4}}]" \
  --wait-for-state AVAILABLE --query data.id --raw-output)
note security-list "$SL_ID"

log "subnets: API $API_SUBNET_CIDR (public), workers $WORKER_SUBNET_CIDR (private), load balancers $LB_SUBNET_CIDR (public)"
API_SUBNET=$("${OCI[@]}" network subnet create --compartment-id "$C" --vcn-id "$VCN_ID" --display-name "$CLUSTER_NAME-api" --freeform-tags "$OWNER_TAGS" --cidr-block "$API_SUBNET_CIDR" \
  --route-table-id "$PUB_RT" --security-list-ids "[\"$SL_ID\"]" --dns-label kubeapi --wait-for-state AVAILABLE --query data.id --raw-output)
note subnet "$API_SUBNET"
WORKER_SUBNET=$("${OCI[@]}" network subnet create --compartment-id "$C" --vcn-id "$VCN_ID" --display-name "$CLUSTER_NAME-workers" --freeform-tags "$OWNER_TAGS" --cidr-block "$WORKER_SUBNET_CIDR" \
  --route-table-id "$PRIV_RT" --security-list-ids "[\"$SL_ID\"]" --dns-label workers --prohibit-public-ip-on-vnic true --wait-for-state AVAILABLE --query data.id --raw-output)
note subnet "$WORKER_SUBNET"
LB_SUBNET=$("${OCI[@]}" network subnet create --compartment-id "$C" --vcn-id "$VCN_ID" --display-name "$CLUSTER_NAME-lb" --freeform-tags "$OWNER_TAGS" --cidr-block "$LB_SUBNET_CIDR" \
  --route-table-id "$PUB_RT" --security-list-ids "[\"$SL_ID\"]" --dns-label lb --wait-for-state AVAILABLE --query data.id --raw-output)
note subnet "$LB_SUBNET"

log "OKE cluster $CLUSTER_NAME (Basic, $K8S, public endpoint, Flannel); about 10 minutes"
"${OCI[@]}" ce cluster create --compartment-id "$C" --name "$CLUSTER_NAME" --vcn-id "$VCN_ID" --kubernetes-version "$K8S" --type BASIC_CLUSTER \
  --endpoint-subnet-id "$API_SUBNET" --endpoint-public-ip-enabled true --service-lb-subnet-ids "[\"$LB_SUBNET\"]" \
  --pods-cidr "$PODS_CIDR" --services-cidr "$SERVICES_CIDR" --cluster-pod-network-options '[{"cniType":"FLANNEL_OVERLAY"}]' \
  --freeform-tags "$OWNER_TAGS" \
  --wait-for-state SUCCEEDED --wait-for-state FAILED --max-wait-seconds 1800 >/dev/null
CLUSTER_ID=$("${OCI[@]}" ce cluster list --compartment-id "$C" --name "$CLUSTER_NAME" --lifecycle-state ACTIVE \
  --query "data[?\"vcn-id\"=='$VCN_ID'].id | [0]" --raw-output)
[ "$CLUSTER_ID" != "null" ] || CLUSTER_ID=""
[ -n "$CLUSTER_ID" ] || { echo "ERROR: cluster $CLUSTER_NAME did not reach ACTIVE; check the work request in the Console" >&2; exit 1; }
note cluster "$CLUSTER_ID"

log "kubeconfig -> $KCFG"
"${OCI[@]}" ce cluster create-kubeconfig --cluster-id "$CLUSTER_ID" --file "$KCFG" --token-version 2.0.0 --kube-endpoint PUBLIC_ENDPOINT --overwrite
if [ -n "${OCI_CLI_PROFILE:-}" ] || [ -n "${OCI_CLI_AUTH:-}" ]; then
  # The generated exec credential runs `oci ce cluster generate-token` with the default profile; pass yours through.
  USER_NAME=$(kubectl --kubeconfig "$KCFG" config view -o jsonpath='{.users[0].name}')
  kubectl --kubeconfig "$KCFG" config set-credentials "$USER_NAME" --exec-api-version=client.authentication.k8s.io/v1beta1 --exec-command=oci \
    --exec-arg=ce --exec-arg=cluster --exec-arg=generate-token --exec-arg=--cluster-id --exec-arg="$CLUSTER_ID" --exec-arg=--region --exec-arg="$OCI_REGION" \
    ${OCI_CLI_PROFILE:+--exec-arg=--profile --exec-arg="$OCI_CLI_PROFILE"} ${OCI_CLI_AUTH:+--exec-arg=--auth --exec-arg="$OCI_CLI_AUTH"} >/dev/null
fi
# The public endpoint can take a minute after ACTIVE before it answers.
for _ in $(seq 1 18); do KUBECONFIG="$KCFG" kubectl get --raw /version >/dev/null 2>&1 && break; sleep 10; done
if KUBECONFIG="$KCFG" kubectl get --raw /version >/dev/null 2>&1; then log "kubectl reaches the API"
else echo "WARN: the API endpoint is not answering yet. If this persists, check API_ALLOWED_CIDR against your egress address." >&2; fi

log "done. Export these before the next steps:"
cat <<ENV
export OKE_CLUSTER_ID=$CLUSTER_ID
export WORKER_SUBNET_ID=$WORKER_SUBNET
export KUBECONFIG=$(cd "$(dirname "$KCFG")" && pwd)/$(basename "$KCFG")
ENV
