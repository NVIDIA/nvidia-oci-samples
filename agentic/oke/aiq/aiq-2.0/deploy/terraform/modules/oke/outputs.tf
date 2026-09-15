# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

output "cluster_id" {
  description = "OKE cluster OCID"
  value       = oci_containerengine_cluster.this.id
}

output "cluster_endpoint" {
  description = "OKE cluster Kubernetes API endpoint"
  value       = oci_containerengine_cluster.this.endpoints[0].public_endpoint
}

output "node_pool_id" {
  description = "OKE node pool OCID"
  value       = oci_containerengine_node_pool.workers.id
}

output "availability_domain" {
  description = "Availability domain used by the node pool"
  value       = data.oci_identity_availability_domains.this.availability_domains[0].name
}
