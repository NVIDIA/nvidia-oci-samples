<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deploy AI-Q on Oracle Cloud (OCI)

Two steps:

1. **Clone this repo** and run `terraform apply` to create the OCI infrastructure (OKE cluster, VCN, Load Balancer, Vault).
2. **`helm pull`** the pre-built AI-Q chart from NGC and install it onto the cluster.

No image builds, no clone of the AI-Q repo. All container images come straight from `nvcr.io/nvidia/blueprint`.

## Architecture

```text
Internet
   │
   ▼
OCI Load Balancer (port 80)              ← Terraform
   │  health-checks & forwards to
   ▼  NodePort 30080
┌─────────────────────────────────────┐
│  OKE Cluster                        │  ← Terraform
│                                     │
│  ┌────────────┐   ┌──────────────┐  │
│  │ Frontend   │   │ Backend      │  │  ← Helm (NGC chart)
│  │ (Next.js)  │   │ (FastAPI)    │  │     images: nvcr.io
│  └────────────┘   └──────┬───────┘  │
│                          │          │
│                   ┌──────▼───────┐  │
│                   │ PostgreSQL   │  │  ← Helm (NGC chart)
│                   └──────┬───────┘  │
│                          │          │
└──────────────────────────┼──────────┘
                           │
                  OCI Block Volume        ← dynamically provisioned
                                            by OKE CSI driver
```

Terraform owns all OCI resources. Helm owns only Kubernetes workloads. `terraform destroy` removes everything — the OKE cluster goes away and Kubernetes resources go with it.

## Prerequisites

| Tool      | Version | Install                                                                                       |
| --------- | ------- | --------------------------------------------------------------------------------------------- |
| OCI CLI   | 3.x     | `brew install oci-cli` or [docs](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm) |
| Terraform | 1.5+    | `brew install terraform`                                                                      |
| kubectl   | 1.28+   | `brew install kubectl`                                                                        |
| Helm      | 3.x     | `brew install helm`                                                                           |

You also need:

- OCI CLI configured (`~/.oci/config`) with a valid API key
- An OCI tenancy with a compartment for the project
- An **NGC API key** from [build.nvidia.com](https://build.nvidia.com/) — used both as the NVIDIA inference key *and* to authenticate to the NGC Helm/container registry
- A **Tavily API key** from [tavily.com](https://tavily.com/) — used for web search

## Step 1 — Provision OCI infrastructure with Terraform

Clone the repo and configure variables:

```bash
git clone https://github.com/NVIDIA/nvidia-oci-samples.git
cd nvidia-oci-samples/agentic/oke/aiq/aiq-2.0/deploy/terraform

cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

| Variable             | Where to find it                                                  |
| -------------------- | ----------------------------------------------------------------- |
| `tenancy_ocid`       | OCI Console → Profile → Tenancy                                   |
| `user_ocid`          | OCI Console → Profile → My profile                                |
| `fingerprint`        | From your API key in `~/.oci/config`                              |
| `private_key_path`   | Path to your OCI API private key (e.g., `~/.oci/oci_api_key.pem`) |
| `region`             | e.g., `us-ashburn-1`, `eu-frankfurt-1`                            |
| `compartment_id`     | OCI Console → Identity → Compartments                             |
| `db_admin_password`  | Choose one — avoid `@` `!` `:` `/` (they break DB URIs)           |
| `nvidia_api_key`     | NGC API key from build.nvidia.com                                 |
| `tavily_api_key`     | From tavily.com                                                   |

Provision:

```bash
terraform init
terraform plan      # review
terraform apply
```

This creates a VCN, an OKE cluster, an OCI Load Balancer pointing at NodePort 30080, an OCI Vault for secret storage, and supporting resources. **It takes 10–15 minutes** — the OKE cluster is the slowest part.

When it finishes, capture the cluster ID and LB IP for the next steps:

```bash
export OKE_CLUSTER_ID="$(terraform output -raw oke_cluster_id)"
export LB_PUBLIC_IP="$(terraform output -raw lb_public_ip)"
```

## Step 2 — Install AI-Q from the NGC Helm chart

### 2a. Configure kubectl for the OKE cluster

```bash
oci ce cluster create-kubeconfig \
  --cluster-id "$OKE_CLUSTER_ID" \
  --file ~/.kube/config \
  --region <your-region> \
  --token-version 2.0.0 \
  --kube-endpoint PUBLIC_ENDPOINT

kubectl get nodes        # sanity check
```

### 2b. Create the namespace and secrets

> **These values must match what you set in `terraform.tfvars` in Step 1.**
> Terraform already stored them in OCI Vault (`<prefix>-db-user-password`,
> `<prefix>-nvidia-api-key`, `<prefix>-tavily-api-key`). The Kubernetes secret
> below must carry the *same* values, or the backend won't be able to reach
> PostgreSQL and the API keys won't work. Set the shell variables to the exact
> `db_admin_password`, `nvidia_api_key`, and `tavily_api_key` you configured:

```bash
# Must match terraform.tfvars (Step 1)
export DB_USER_PASSWORD="<same as db_admin_password in terraform.tfvars>"
export NGC_API_KEY="<same as nvidia_api_key in terraform.tfvars>"
export TAVILY_API_KEY="<same as tavily_api_key in terraform.tfvars>"

kubectl create namespace ns-aiq --dry-run=client -o yaml | kubectl apply -f -

# API credentials consumed by the application
kubectl create secret generic aiq-credentials -n ns-aiq \
  --from-literal=NVIDIA_API_KEY="$NGC_API_KEY" \
  --from-literal=TAVILY_API_KEY="$TAVILY_API_KEY" \
  --from-literal=DB_USER_NAME="aiq" \
  --from-literal=DB_USER_PASSWORD="$DB_USER_PASSWORD"

# Image pull secret for nvcr.io (NGC container registry)
kubectl create secret docker-registry ngc-secret -n ns-aiq \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"
```

### 2c. Pull and install the chart from NGC

```bash
helm pull https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq2-web-2.0.0.tgz \
  --username='$oauthtoken' \
  --password="$NGC_API_KEY"

# (optional) inspect what you're about to install
helm show chart aiq2-web-2.0.0.tgz

# Install with the OCI overlay (NodePort 30080 + ngc-secret pull secret)
helm upgrade --install aiq aiq2-web-2.0.0.tgz \
  -n ns-aiq --create-namespace \
  --wait --timeout 10m \
  -f deploy/helm/values-oci-ngc.yaml
```

The overlay is intentionally tiny — it only pins the frontend service to NodePort 30080 (the port the OCI Load Balancer health-checks) and names the image pull secret. Everything else (image repositories, postgres init SQL, dynamic-provisioned 10Gi block volume PVC) comes from the chart's own defaults.

### 2d. Verify

```bash
kubectl get pods -n ns-aiq
```

Expected:

```text
NAME                            READY   STATUS    RESTARTS   AGE
aiq-backend-xxx                 1/1     Running   0          1m
aiq-frontend-xxx                1/1     Running   0          1m
aiq-postgres-xxx                1/1     Running   0          1m
```

### 2e. Access the application

```bash
echo "http://$LB_PUBLIC_IP"     # frontend UI through the OCI Load Balancer
```

The OCI LB forwards port 80 → NodePort 30080 → frontend pod. Backend API docs are available via port-forward: `kubectl port-forward -n ns-aiq svc/aiq-backend 8000:8000` then `http://localhost:8000/docs`.

## Tear down

```bash
cd deploy/terraform
terraform destroy
```

Destroying the OKE cluster removes all Helm-installed Kubernetes resources with it (pods, PVCs, services, the dynamically-provisioned block volume). If the destroy fails on the first run due to OCI async cleanup (NSG or subnet conflicts), wait a minute and retry.

## Troubleshooting

### `ImagePullBackOff` on backend or frontend

The `ngc-secret` is missing, has a stale token, or doesn't reference `nvcr.io`. Re-create it:

```bash
kubectl delete secret ngc-secret -n ns-aiq
kubectl create secret docker-registry ngc-secret -n ns-aiq \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"
kubectl rollout restart deployment -n ns-aiq aiq-backend aiq-frontend
```

### LB shows backends as "Critical"

The frontend Service must be on NodePort **30080** — the LB module's health check is hardcoded to that port. Confirm:

```bash
kubectl get svc -n ns-aiq aiq-frontend -o jsonpath='{.spec.ports[0].nodePort}'
# → 30080
```

If you see a different port, your overlay didn't apply — re-run `helm upgrade` with `-f deploy/helm/values-oci-ngc.yaml`.

### Backend `CrashLoopBackOff` with "failed to resolve host"

The DB password in `aiq-credentials` contains a character (`@`, `!`, `:`, `/`) that breaks PostgreSQL URI parsing. Re-create the secret with a safer password and restart the backend.

### PVC stuck in `Pending`

OKE's default storage class needs to be `oci-bv`. Check:

```bash
kubectl get storageclass
kubectl describe pvc -n ns-aiq aiq-postgres-data
```

## File layout

```text
deploy/
├── README.md                                ← this file
└── oci/
    ├── helm/
    │   └── values-oci-ngc.yaml              ← OCI overlay applied in step 2
    ├── blog/
    │   ├── part1-aiq-on-oci.md              ← blog post
    │   └── aiq-agent-architecture.png
    ├── oci-aiq-architecture-diagram.png     ← rendered from .mmd via mmdc
    ├── oci-aiq-architecture.mmd             ← Mermaid source for the diagram
    └── terraform/
        ├── main.tf                          ← root module
        ├── variables.tf
        ├── outputs.tf
        ├── versions.tf
        ├── terraform.tfvars.example
        └── modules/
            ├── network/                     ← VCN, subnets, NSGs, gateways
            ├── oke/                         ← OKE cluster + node pool
            ├── loadbalancer/                ← OCI LB → NodePort 30080
            └── vault/                       ← OCI Vault + secrets
```
