# Oracle Database Integration

This sample adds Oracle AI Database 26ai vector storage and Oracle Private AI
Services (PAI)/NVIDIA cuVS index construction to the stock NVIDIA RAG Blueprint
2.6.0. It is distributed as one Kubernetes deployment file:
[`oracle26ai-rag-extension.yaml`](./oracle26ai-rag-extension.yaml).

The extension prepares the Oracle connection, deploys a private PAI endpoint,
verifies a real GPU index build, and adds Oracle runtime wiring to the stock
`rag-server` and `ingestor-server`. The stock RAG chart continues to own its
models, ingestion services, object storage and application topology. AI-Q and
VSS are installed independently and can consume the Oracle-backed RAG API;
this file does not install those applications or change VSS's video storage.

## Before applying

- Prepare an OCI OKE cluster with NVIDIA GPU/CUDA support, NIM Operator APIs,
  sufficient resources for the chosen stock RAG profile, and one additional
  GPU for PAI.
- Install the official [NVIDIA RAG Blueprint 2.6.0](https://github.com/NVIDIA-AI-Blueprints/rag/tree/v2.6.0)
  using its stock Helm chart and prerequisites.
- Assign an Oracle Base Database Service 26ai database in the same regional
  VCN as the cluster. The tested workshop DB uses `VM.BaseDB.x86`, four ECPUs,
  8 GB memory and 256 GB LVM data storage. Verify a positive PDB `SGA_TARGET`.
- Allow private cluster-to-DB TCP 22/1521 and DB-to-PAI TCP 8443, plus required
  DNS, OCI API and registry access. BYO mode does not create the DB's NSG.
- Obtain an entitled Oracle Container Registry credential for
  `database/private-ai`. The image is pulled directly from Oracle; this
  repository does not redistribute it.
- Prepare the bootstrap input Secret described below. Use disposable,
  least-privilege session OCI and DB-host SSH credentials.

Use a separate DB system/CDB for each concurrent workshop participant: the
bootstrap updates the CDB SSL-wallet configuration for its PAI endpoint.

## Use an existing database

The following example uses the default `rag` namespace. Create its
`oracle-registry` image-pull Secret with your Oracle Container Registry
credential, then provide the existing DB inputs. The adapter reads the config
and signing key from this Secret and rewrites `key_file` to the mounted key;
no workstation-path edit is required.

The manifest defaults to create-new mode, which can provision billable database
and networking resources. Keep `ORACLE26AI_DB_MODE=existing` for a preprovisioned
workshop database.

```bash
kubectl -n rag create secret generic oracle26ai-bootstrap-inputs \
  --from-file=config='<session OCI config path>' \
  --from-file=oci_api_key.pem='<session OCI signing-key path>' \
  --from-file=BYO_SSH_PRIVATE_KEY='<assigned DB-host SSH private-key path>' \
  --from-literal=OCI_PROFILE=DEFAULT \
  --from-literal=ORACLE26AI_DB_MODE=existing \
  --from-literal=BYO_DB_SYSTEM_ID='<assigned DB system OCID>' \
  --from-literal=BYO_ALLOW_DB_CHANGES=true \
  --from-literal=DB_PARALLEL_DEGREE=2 \
  --from-literal=DB_SCHEMA='<unique uppercase schema>'

printf '%s  %s\n' \
  '648187c50ce1d416062d36cbb71143ea1d81e5f4b97446b55a31d5ddf4f4a000' \
  oracle26ai-rag-extension.yaml | sha256sum -c -
kubectl -n rag apply -f oracle26ai-rag-extension.yaml
```

The adapter discovers the DB and PDB names. Successful bootstrap retires the
input Secret. A running Pod alone is insufficient: require the adapter's
`proofStatus=verified`, a valid offloaded HNSW index, and successful document
ingestion/retrieval through RAG.

The full file includes shared cluster RBAC whose default binding targets
`rag`. For participant namespaces, the workshop host prepares the shared RBAC
and a namespaced session copy first. cuVS uses a 13-object subset and the
workshop's tested PAI image override; follow the corresponding host and
participant guides rather than applying this full file from a scoped lab.
Deleting Kubernetes resources does not delete an existing BYO database.

## Validation and contents

This is the unchanged adapter used in the Osaka A100 40GB workshop rehearsals:
two concurrent cuVS participant flows, plus the VSS Base/Oracle document path
with one first cached startup and two service restarts. The workshop-specific
image/profile settings are supplied by the guides. These results do not
qualify 40 simultaneous participants, a newly built tenancy, or every VSS
mode.

The YAML embeds NVIDIA RAG runtime Python source and the `python-oracledb`
3.4.2 CPython 3.13 Linux x86-64 wheel to avoid runtime package downloads.
The RAG source retains its Apache-2.0 headers. The
[Oracle python-oracledb](https://github.com/oracle/python-oracledb/tree/v3.4.2)
wheel retains its license, notice and third-party license files; its metadata
identifies UPL-1.0 OR Apache-2.0. The embedded wheel's SHA-256 is
`b26a10f9c790bd141ffc8af68520803ed4a44a9258bf7d1eea9bfdd36bd6df7f`.
PAI and NVIDIA model containers are external registry dependencies subject to
their respective terms.

SHA-256 of this file:
`648187c50ce1d416062d36cbb71143ea1d81e5f4b97446b55a31d5ddf4f4a000`.

This sample follows the repository's [support policy](../../README.md#support-level)
and [license](../../LICENSE).
