# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# OCI Load Balancer — routes HTTP traffic to the frontend NodePort on OKE workers
# -----------------------------------------------------------------------------

data "oci_containerengine_node_pool" "this" {
  node_pool_id = var.node_pool_id
}

resource "oci_load_balancer_load_balancer" "this" {
  compartment_id             = var.compartment_id
  display_name               = "${var.label_prefix}-lb"
  shape                      = var.shape
  is_private                 = false
  network_security_group_ids = [var.nsg_lb_id]
  subnet_ids                 = [var.public_subnet_id]
  freeform_tags              = var.freeform_tags

  dynamic "shape_details" {
    for_each = var.shape == "flexible" ? [1] : []
    content {
      minimum_bandwidth_in_mbps = var.shape_min_mbps
      maximum_bandwidth_in_mbps = var.shape_max_mbps
    }
  }
}

# --- Backend Set (health-checks the frontend on its NodePort) ---

resource "oci_load_balancer_backend_set" "frontend" {
  load_balancer_id = oci_load_balancer_load_balancer.this.id
  name             = "frontend"
  policy           = "ROUND_ROBIN"

  health_checker {
    protocol          = "HTTP"
    port              = var.frontend_node_port
    url_path          = "/"
    return_code       = 200
    interval_ms       = 10000
    timeout_in_millis = 3000
    retries           = 3
  }
}

# --- One backend per active worker node ---

resource "oci_load_balancer_backend" "frontend" {
  for_each = {
    for node in data.oci_containerengine_node_pool.this.nodes :
    node.id => node
    if node.private_ip != null && node.private_ip != ""
  }

  load_balancer_id = oci_load_balancer_load_balancer.this.id
  backendset_name  = oci_load_balancer_backend_set.frontend.name
  ip_address       = each.value.private_ip
  port             = var.frontend_node_port
}

# --- Listener on port 80 ---

resource "oci_load_balancer_listener" "http" {
  load_balancer_id         = oci_load_balancer_load_balancer.this.id
  name                     = "http"
  default_backend_set_name = oci_load_balancer_backend_set.frontend.name
  port                     = 80
  protocol                 = "HTTP"
}
