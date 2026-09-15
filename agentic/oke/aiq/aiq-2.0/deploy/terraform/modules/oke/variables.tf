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

variable "kubernetes_version" {
  description = "Kubernetes version for the OKE cluster"
  type        = string
  default     = "v1.34.2"
}

variable "vcn_id" {
  description = "VCN OCID"
  type        = string
}

variable "oke_subnet_id" {
  description = "Subnet OCID for OKE worker nodes"
  type        = string
}

variable "public_subnet_id" {
  description = "Public subnet OCID for the K8s API endpoint"
  type        = string
}

variable "nsg_oke_workers_id" {
  description = "NSG OCID for OKE worker nodes"
  type        = string
}

variable "nsg_oke_api_id" {
  description = "NSG OCID for the K8s API endpoint"
  type        = string
}

variable "node_pool_size" {
  description = "Number of nodes in the node pool (1 = single-node dev/test, 3+ = HA production)"
  type        = number
  default     = 1
}

variable "node_shape" {
  description = "Compute shape for worker nodes"
  type        = string
  default     = "VM.Standard.E4.Flex"
}

variable "node_ocpus" {
  description = "Number of OCPUs per worker node (flex shapes)"
  type        = number
  default     = 2
}

variable "node_memory_gb" {
  description = "Memory in GB per worker node (flex shapes)"
  type        = number
  default     = 16
}

variable "node_image_id" {
  description = "Custom image OCID for worker nodes. If empty, uses the latest OKE-optimized image."
  type        = string
  default     = ""
}

variable "ssh_public_key" {
  description = "SSH public key for worker node access (optional)"
  type        = string
  default     = ""
}

variable "freeform_tags" {
  description = "Freeform tags for OKE resources"
  type        = map(string)
  default     = {}
}
