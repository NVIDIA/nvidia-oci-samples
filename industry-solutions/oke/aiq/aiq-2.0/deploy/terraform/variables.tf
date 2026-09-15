# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# OCI Provider
# -----------------------------------------------------------------------------

variable "tenancy_ocid" {
  description = "OCI tenancy OCID"
  type        = string
}

variable "user_ocid" {
  description = "OCI user OCID (leave empty for instance principal auth)"
  type        = string
  default     = ""
}

variable "fingerprint" {
  description = "API key fingerprint (leave empty for instance principal auth)"
  type        = string
  default     = ""
}

variable "private_key_path" {
  description = "Path to the OCI API private key (leave empty for instance principal auth)"
  type        = string
  default     = ""
}

variable "region" {
  description = "OCI region (e.g., us-ashburn-1, eu-frankfurt-1)"
  type        = string
}

variable "compartment_id" {
  description = "OCI compartment OCID where all resources will be created"
  type        = string
}

# -----------------------------------------------------------------------------
# General
# -----------------------------------------------------------------------------

variable "label_prefix" {
  description = "Prefix for all resource names"
  type        = string
  default     = "aiq"
}

variable "freeform_tags" {
  description = "Freeform tags applied to all resources"
  type        = map(string)
  default = {
    "project"   = "aiq-blueprint"
    "managedBy" = "terraform"
  }
}

# -----------------------------------------------------------------------------
# OKE
# -----------------------------------------------------------------------------

variable "kubernetes_version" {
  description = "Kubernetes version for the OKE cluster"
  type        = string
  default     = "v1.34.2"
}

variable "node_pool_size" {
  description = "Number of OKE worker nodes (1 = single-node dev/test, 3+ = HA production)"
  type        = number
  default     = 1
}

variable "node_shape" {
  description = "Compute shape for OKE worker nodes"
  type        = string
  default     = "VM.Standard.E4.Flex"
}

variable "node_ocpus" {
  description = "OCPUs per worker node"
  type        = number
  default     = 2
}

variable "node_memory_gb" {
  description = "Memory in GB per worker node"
  type        = number
  default     = 16
}

variable "ssh_public_key" {
  description = "SSH public key for OKE worker node access (optional)"
  type        = string
  default     = ""
}

variable "api_allowed_cidr" {
  description = "CIDR allowed to reach the Kubernetes API endpoint publicly (kubectl). Restrict to office/VPN ranges for production."
  type        = string
  default     = "0.0.0.0/0"
}

# -----------------------------------------------------------------------------
# Vault secrets
# -----------------------------------------------------------------------------

variable "db_admin_password" {
  description = "PostgreSQL password (stored in OCI Vault and consumed by the Helm chart's in-cluster PostgreSQL)"
  type        = string
  sensitive   = true
}

variable "nvidia_api_key" {
  description = "NVIDIA / NGC API key (will be stored in OCI Vault)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "tavily_api_key" {
  description = "Tavily API key (will be stored in OCI Vault)"
  type        = string
  sensitive   = true
  default     = ""
}

# -----------------------------------------------------------------------------
# Load Balancer
# -----------------------------------------------------------------------------

variable "frontend_node_port" {
  description = "Fixed NodePort for the frontend K8s service (LB routes port 80 here)"
  type        = number
  default     = 30080
}
