# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

variable "compartment_id" {
  description = "OCI compartment OCID"
  type        = string
}

variable "label_prefix" {
  description = "Prefix for resource names"
  type        = string
  default     = "aiq"
}

variable "vcn_cidr" {
  description = "CIDR block for the VCN"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR for the public (load balancer) subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "oke_subnet_cidr" {
  description = "CIDR for the private OKE worker subnet"
  type        = string
  default     = "10.0.16.0/20"
}

variable "api_allowed_cidr" {
  description = "CIDR allowed to reach the Kubernetes API endpoint publicly (kubectl). Restrict to office/VPN ranges for production."
  type        = string
  default     = "0.0.0.0/0"
}

variable "freeform_tags" {
  description = "Freeform tags for all network resources"
  type        = map(string)
  default     = {}
}
