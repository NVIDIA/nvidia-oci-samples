# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

output "vcn_id" {
  description = "VCN OCID"
  value       = oci_core_vcn.this.id
}

output "public_subnet_id" {
  description = "Public subnet OCID (load balancer)"
  value       = oci_core_subnet.public.id
}

output "oke_subnet_id" {
  description = "Private OKE worker subnet OCID"
  value       = oci_core_subnet.oke_workers.id
}

output "nsg_lb_id" {
  description = "NSG OCID for the load balancer"
  value       = oci_core_network_security_group.lb.id
}

output "nsg_oke_workers_id" {
  description = "NSG OCID for OKE worker nodes"
  value       = oci_core_network_security_group.oke_workers.id
}

output "nsg_oke_api_id" {
  description = "NSG OCID for the K8s API endpoint"
  value       = oci_core_network_security_group.oke_api.id
}
