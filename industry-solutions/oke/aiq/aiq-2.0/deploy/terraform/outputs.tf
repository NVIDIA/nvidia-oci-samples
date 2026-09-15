# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

output "vcn_id" {
  description = "VCN OCID"
  value       = module.network.vcn_id
}

# -----------------------------------------------------------------------------
# OKE
# -----------------------------------------------------------------------------

output "oke_cluster_id" {
  description = "OKE cluster OCID"
  value       = module.oke.cluster_id
}

output "oke_cluster_endpoint" {
  description = "OKE Kubernetes API endpoint"
  value       = module.oke.cluster_endpoint
}

output "kubeconfig_command" {
  description = "Command to configure kubectl for this OKE cluster"
  value       = "oci ce cluster create-kubeconfig --cluster-id ${module.oke.cluster_id} --region ${var.region} --token-version 2.0.0 --kube-endpoint PUBLIC_ENDPOINT"
}

# -----------------------------------------------------------------------------
# Vault
# -----------------------------------------------------------------------------

output "vault_id" {
  description = "OCI Vault OCID"
  value       = module.vault.vault_id
}

# -----------------------------------------------------------------------------
# Load Balancer
# -----------------------------------------------------------------------------

output "lb_public_ip" {
  description = "Public IP of the Terraform-managed load balancer"
  value       = module.loadbalancer.lb_public_ip
}
