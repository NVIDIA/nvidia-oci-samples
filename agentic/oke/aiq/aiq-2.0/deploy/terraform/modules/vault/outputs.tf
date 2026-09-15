# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

output "vault_id" {
  description = "OCI Vault OCID"
  value       = oci_kms_vault.this.id
}

output "master_key_id" {
  description = "Master encryption key OCID"
  value       = oci_kms_key.master.id
}

output "secret_ids" {
  description = "Map of secret name to OCID"
  value       = { for k, v in oci_vault_secret.this : k => v.id }
}

output "vault_management_endpoint" {
  description = "Vault management endpoint URL"
  value       = oci_kms_vault.this.management_endpoint
}
