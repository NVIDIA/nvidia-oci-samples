# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

variable "compartment_id" {
  description = "OCI compartment OCID"
  type        = string
}

variable "label_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "public_subnet_id" {
  description = "Public subnet OCID for the load balancer"
  type        = string
}

variable "nsg_lb_id" {
  description = "NSG OCID for the load balancer"
  type        = string
}

variable "node_pool_id" {
  description = "OKE node pool OCID (used to discover worker node IPs)"
  type        = string
}

variable "frontend_node_port" {
  description = "Fixed NodePort for the frontend service"
  type        = number
  default     = 30080
}

variable "shape" {
  description = "Load balancer shape"
  type        = string
  default     = "flexible"
}

variable "shape_min_mbps" {
  description = "Minimum bandwidth in Mbps (flexible shape)"
  type        = number
  default     = 10
}

variable "shape_max_mbps" {
  description = "Maximum bandwidth in Mbps (flexible shape)"
  type        = number
  default     = 100
}

variable "freeform_tags" {
  description = "Freeform tags for all resources"
  type        = map(string)
  default     = {}
}
