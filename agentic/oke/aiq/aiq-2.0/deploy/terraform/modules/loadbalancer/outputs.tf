# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

output "lb_id" {
  description = "Load balancer OCID"
  value       = oci_load_balancer_load_balancer.this.id
}

output "lb_public_ip" {
  description = "Load balancer public IP address"
  value       = oci_load_balancer_load_balancer.this.ip_address_details[0].ip_address
}
