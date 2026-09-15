# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# OCI Provider
# -----------------------------------------------------------------------------

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid != "" ? var.user_ocid : null
  fingerprint      = var.fingerprint != "" ? var.fingerprint : null
  private_key_path = var.private_key_path != "" ? var.private_key_path : null
  region           = var.region
}

# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

module "network" {
  source = "./modules/network"

  compartment_id   = var.compartment_id
  label_prefix     = var.label_prefix
  freeform_tags    = var.freeform_tags
  api_allowed_cidr = var.api_allowed_cidr
}

# -----------------------------------------------------------------------------
# OKE Cluster
# -----------------------------------------------------------------------------

module "oke" {
  source = "./modules/oke"

  compartment_id     = var.compartment_id
  label_prefix       = var.label_prefix
  kubernetes_version = var.kubernetes_version
  vcn_id             = module.network.vcn_id
  oke_subnet_id      = module.network.oke_subnet_id
  public_subnet_id   = module.network.public_subnet_id
  nsg_oke_workers_id = module.network.nsg_oke_workers_id
  nsg_oke_api_id     = module.network.nsg_oke_api_id
  node_pool_size     = var.node_pool_size
  node_shape         = var.node_shape
  node_ocpus         = var.node_ocpus
  node_memory_gb     = var.node_memory_gb
  ssh_public_key     = var.ssh_public_key
  freeform_tags      = var.freeform_tags
}

# -----------------------------------------------------------------------------
# OCI Vault
# -----------------------------------------------------------------------------

module "vault" {
  source = "./modules/vault"

  compartment_id = var.compartment_id
  label_prefix   = var.label_prefix
  freeform_tags  = var.freeform_tags

  secrets = {
    nvidia-api-key   = var.nvidia_api_key != "" ? base64encode(var.nvidia_api_key) : base64encode("PLACEHOLDER")
    tavily-api-key   = var.tavily_api_key != "" ? base64encode(var.tavily_api_key) : base64encode("PLACEHOLDER")
    db-user-name     = base64encode("aiq")
    db-user-password = base64encode(var.db_admin_password)
  }
}

# -----------------------------------------------------------------------------
# Load Balancer (forwards port 80 to the frontend NodePort)
# -----------------------------------------------------------------------------

module "loadbalancer" {
  source = "./modules/loadbalancer"

  compartment_id     = var.compartment_id
  label_prefix       = var.label_prefix
  public_subnet_id   = module.network.public_subnet_id
  nsg_lb_id          = module.network.nsg_lb_id
  node_pool_id       = module.oke.node_pool_id
  frontend_node_port = var.frontend_node_port
  freeform_tags      = var.freeform_tags
}
