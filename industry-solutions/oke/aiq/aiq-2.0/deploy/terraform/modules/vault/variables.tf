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

variable "vault_type" {
  description = "OCI Vault type: DEFAULT or VIRTUAL_PRIVATE"
  type        = string
  default     = "DEFAULT"

  validation {
    condition     = contains(["DEFAULT", "VIRTUAL_PRIVATE"], var.vault_type)
    error_message = "vault_type must be DEFAULT or VIRTUAL_PRIVATE."
  }
}

variable "key_shape_algorithm" {
  description = "Master encryption key algorithm"
  type        = string
  default     = "AES"
}

variable "key_shape_length" {
  description = "Master encryption key length in bytes (16, 24, or 32)"
  type        = number
  default     = 32

  validation {
    condition     = contains([16, 24, 32], var.key_shape_length)
    error_message = "key_shape_length must be 16, 24, or 32."
  }
}

variable "secrets" {
  description = "Map of secret names to their base64-encoded content. Keys are resource identifiers (not secret), values are the secret payloads."
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "freeform_tags" {
  description = "Freeform tags"
  type        = map(string)
  default     = {}
}
