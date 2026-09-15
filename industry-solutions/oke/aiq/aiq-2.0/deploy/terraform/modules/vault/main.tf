# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# OCI Vault for secrets management
# -----------------------------------------------------------------------------

resource "oci_kms_vault" "this" {
  compartment_id = var.compartment_id
  display_name   = "${var.label_prefix}-vault"
  vault_type     = var.vault_type
  freeform_tags  = var.freeform_tags
}

resource "oci_kms_key" "master" {
  compartment_id      = var.compartment_id
  display_name        = "${var.label_prefix}-master-key"
  management_endpoint = oci_kms_vault.this.management_endpoint
  freeform_tags       = var.freeform_tags

  key_shape {
    algorithm = var.key_shape_algorithm
    length    = var.key_shape_length
  }
}

# Create a secret for each entry in var.secrets
resource "oci_vault_secret" "this" {
  for_each       = nonsensitive(toset(keys(var.secrets)))
  compartment_id = var.compartment_id
  vault_id       = oci_kms_vault.this.id
  key_id         = oci_kms_key.master.id
  secret_name    = "${var.label_prefix}-${each.key}"
  freeform_tags  = var.freeform_tags

  secret_content {
    content_type = "BASE64"
    content      = var.secrets[each.key]
  }
}
