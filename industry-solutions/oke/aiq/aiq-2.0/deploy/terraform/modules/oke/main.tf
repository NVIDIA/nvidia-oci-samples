# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Lookup OKE-optimized Oracle Linux images compatible with the node shape
data "oci_containerengine_node_pool_option" "this" {
  node_pool_option_id = "all"
  compartment_id      = var.compartment_id
}

locals {
  # major.minor version for image matching (e.g. "1.34" from "v1.34.2")
  k8s_minor = join(".", slice(split(".", replace(var.kubernetes_version, "v", "")), 0, 2))

  # Filter for standard x86_64 Oracle Linux 8 OKE images (exclude ARM and GPU variants)
  compatible_images = [
    for src in data.oci_containerengine_node_pool_option.this.sources :
    src.image_id
    if length(regexall("Oracle-Linux-8", src.source_name)) > 0
    && length(regexall(local.k8s_minor, src.source_name)) > 0
    && length(regexall("aarch64", src.source_name)) == 0
    && length(regexall("GPU", src.source_name)) == 0
  ]

  node_image_id = var.node_image_id != "" ? var.node_image_id : local.compatible_images[0]
}

resource "terraform_data" "validate_node_image" {
  lifecycle {
    precondition {
      condition     = var.node_image_id != "" || length(local.compatible_images) > 0
      error_message = "No compatible OKE node image found for Kubernetes ${var.kubernetes_version} (filter: Oracle-Linux-8 + ${local.k8s_minor}, non-ARM, non-GPU). Set var.node_image_id explicitly."
    }
  }
}

# -----------------------------------------------------------------------------
# OKE Cluster
# -----------------------------------------------------------------------------

resource "oci_containerengine_cluster" "this" {
  compartment_id     = var.compartment_id
  kubernetes_version = var.kubernetes_version
  name               = "${var.label_prefix}-oke"
  vcn_id             = var.vcn_id
  type               = "ENHANCED_CLUSTER"
  freeform_tags      = var.freeform_tags

  cluster_pod_network_options {
    cni_type = "OCI_VCN_IP_NATIVE"
  }

  endpoint_config {
    is_public_ip_enabled = true
    subnet_id            = var.public_subnet_id
    nsg_ids              = [var.nsg_oke_api_id]
  }

  options {
    service_lb_subnet_ids = [var.public_subnet_id]

    kubernetes_network_config {
      pods_cidr     = "10.244.0.0/16"
      services_cidr = "10.96.0.0/16"
    }

    persistent_volume_config {
      freeform_tags = var.freeform_tags
    }
  }
}

# -----------------------------------------------------------------------------
# Node Pool
# -----------------------------------------------------------------------------

resource "oci_containerengine_node_pool" "workers" {
  compartment_id     = var.compartment_id
  cluster_id         = oci_containerengine_cluster.this.id
  kubernetes_version = var.kubernetes_version
  name               = "${var.label_prefix}-nodepool"
  freeform_tags      = var.freeform_tags

  node_shape = var.node_shape

  node_shape_config {
    ocpus         = var.node_ocpus
    memory_in_gbs = var.node_memory_gb
  }

  node_source_details {
    source_type = "IMAGE"
    image_id    = local.node_image_id
  }

  node_config_details {
    size                                = var.node_pool_size
    is_pv_encryption_in_transit_enabled = true
    freeform_tags                       = var.freeform_tags

    dynamic "placement_configs" {
      for_each = data.oci_identity_availability_domains.this.availability_domains
      content {
        availability_domain = placement_configs.value.name
        subnet_id           = var.oke_subnet_id
      }
    }

    node_pool_pod_network_option_details {
      cni_type       = "OCI_VCN_IP_NATIVE"
      pod_subnet_ids = [var.oke_subnet_id]
      pod_nsg_ids    = [var.nsg_oke_workers_id]
    }

    nsg_ids = [var.nsg_oke_workers_id]
  }

  ssh_public_key = var.ssh_public_key != "" ? var.ssh_public_key : null
}

data "oci_identity_availability_domains" "this" {
  compartment_id = var.compartment_id
}
