# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# VCN
# -----------------------------------------------------------------------------

data "oci_core_services" "all" {
  filter {
    name   = "name"
    values = ["All .* Services In Oracle Services Network"]
    regex  = true
  }
}

resource "oci_core_vcn" "this" {
  compartment_id = var.compartment_id
  display_name   = "${var.label_prefix}-vcn"
  cidr_blocks    = [var.vcn_cidr]
  dns_label      = "${var.label_prefix}vcn"
  freeform_tags  = var.freeform_tags
}

# -----------------------------------------------------------------------------
# Gateways
# -----------------------------------------------------------------------------

resource "oci_core_internet_gateway" "this" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.label_prefix}-igw"
  enabled        = true
  freeform_tags  = var.freeform_tags
}

resource "oci_core_nat_gateway" "this" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.label_prefix}-natgw"
  freeform_tags  = var.freeform_tags
}

resource "oci_core_service_gateway" "this" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.label_prefix}-sgw"
  freeform_tags  = var.freeform_tags

  services {
    service_id = data.oci_core_services.all.services[0].id
  }
}

# -----------------------------------------------------------------------------
# Route Tables
# -----------------------------------------------------------------------------

resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.label_prefix}-rt-public"
  freeform_tags  = var.freeform_tags

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.this.id
  }
}

resource "oci_core_route_table" "private" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.label_prefix}-rt-private"
  freeform_tags  = var.freeform_tags

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_nat_gateway.this.id
  }

  route_rules {
    destination       = data.oci_core_services.all.services[0].cidr_block
    destination_type  = "SERVICE_CIDR_BLOCK"
    network_entity_id = oci_core_service_gateway.this.id
  }
}

# -----------------------------------------------------------------------------
# Network Security Groups
# -----------------------------------------------------------------------------

resource "oci_core_network_security_group" "lb" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.label_prefix}-nsg-lb"
  freeform_tags  = var.freeform_tags
}

resource "oci_core_network_security_group" "oke_workers" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.label_prefix}-nsg-oke-workers"
  freeform_tags  = var.freeform_tags
}

resource "oci_core_network_security_group" "oke_api" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.label_prefix}-nsg-oke-api"
  freeform_tags  = var.freeform_tags
}

# --- LB NSG Rules ---

resource "oci_core_network_security_group_security_rule" "lb_ingress_http" {
  network_security_group_id = oci_core_network_security_group.lb.id
  direction                 = "INGRESS"
  protocol                  = "6" # TCP
  source                    = "0.0.0.0/0"
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "Allow HTTP from internet"

  tcp_options {
    destination_port_range {
      min = 80
      max = 80
    }
  }
}

resource "oci_core_network_security_group_security_rule" "lb_ingress_https" {
  network_security_group_id = oci_core_network_security_group.lb.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = "0.0.0.0/0"
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "Allow HTTPS from internet"

  tcp_options {
    destination_port_range {
      min = 443
      max = 443
    }
  }
}

resource "oci_core_network_security_group_security_rule" "lb_to_oke_workers" {
  network_security_group_id = oci_core_network_security_group.lb.id
  direction                 = "EGRESS"
  protocol                  = "6"
  destination               = var.oke_subnet_cidr
  destination_type          = "CIDR_BLOCK"
  stateless                 = false
  description               = "LB to OKE NodePort range"

  tcp_options {
    destination_port_range {
      min = 30000
      max = 32767
    }
  }
}

# --- OKE Workers NSG Rules ---
# Per OCI docs: https://docs.oracle.com/en-us/iaas/Content/ContEng/Concepts/contengnetworkconfig.htm

resource "oci_core_network_security_group_security_rule" "oke_workers_ingress_from_lb" {
  network_security_group_id = oci_core_network_security_group.oke_workers.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = oci_core_network_security_group.lb.id
  source_type               = "NETWORK_SECURITY_GROUP"
  stateless                 = false
  description               = "Allow traffic from LB NSG"

  tcp_options {
    destination_port_range {
      min = 30000
      max = 32767
    }
  }
}

resource "oci_core_network_security_group_security_rule" "oke_workers_ingress_internal" {
  network_security_group_id = oci_core_network_security_group.oke_workers.id
  direction                 = "INGRESS"
  protocol                  = "all"
  source                    = var.oke_subnet_cidr
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "Allow all traffic between OKE workers (node-to-node + pod-to-pod)"
}

resource "oci_core_network_security_group_security_rule" "oke_workers_ingress_from_vcn" {
  network_security_group_id = oci_core_network_security_group.oke_workers.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = var.vcn_cidr
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "Allow all TCP from VCN (control plane, cross-subnet communication)"
}

resource "oci_core_network_security_group_security_rule" "oke_workers_ingress_api_kubelet" {
  network_security_group_id = oci_core_network_security_group.oke_workers.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = var.vcn_cidr
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "Allow K8s API server to worker kubelet"

  tcp_options {
    destination_port_range {
      min = 10250
      max = 10250
    }
  }
}

resource "oci_core_network_security_group_security_rule" "oke_workers_ingress_oke_mgmt" {
  network_security_group_id = oci_core_network_security_group.oke_workers.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = var.vcn_cidr
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "Allow OKE control plane node management (Enhanced cluster)"

  tcp_options {
    destination_port_range {
      min = 12250
      max = 12250
    }
  }
}

resource "oci_core_network_security_group_security_rule" "oke_workers_ingress_icmp_vcn" {
  network_security_group_id = oci_core_network_security_group.oke_workers.id
  direction                 = "INGRESS"
  protocol                  = "1" # ICMP
  source                    = var.vcn_cidr
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "ICMP Path MTU Discovery from VCN"

  icmp_options {
    type = 3
    code = 4
  }
}

resource "oci_core_network_security_group_security_rule" "oke_workers_ingress_icmp_all" {
  network_security_group_id = oci_core_network_security_group.oke_workers.id
  direction                 = "INGRESS"
  protocol                  = "1" # ICMP
  source                    = "0.0.0.0/0"
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "ICMP Path MTU Discovery from internet"

  icmp_options {
    type = 3
    code = 4
  }
}

resource "oci_core_network_security_group_security_rule" "oke_workers_egress_all" {
  network_security_group_id = oci_core_network_security_group.oke_workers.id
  direction                 = "EGRESS"
  protocol                  = "all"
  destination               = "0.0.0.0/0"
  destination_type          = "CIDR_BLOCK"
  stateless                 = false
  description               = "Allow all egress (NAT GW for NVIDIA API, Tavily, etc.)"
}

# --- K8s API Endpoint NSG Rules ---
# Per OCI docs: Enhanced clusters use the private endpoint for worker↔API communication

resource "oci_core_network_security_group_security_rule" "oke_api_ingress_6443_workers" {
  network_security_group_id = oci_core_network_security_group.oke_api.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = var.oke_subnet_cidr
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "K8s API access from worker nodes"

  tcp_options {
    destination_port_range {
      min = 6443
      max = 6443
    }
  }
}

resource "oci_core_network_security_group_security_rule" "oke_api_ingress_6443_public" {
  network_security_group_id = oci_core_network_security_group.oke_api.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = var.api_allowed_cidr
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "K8s API access from external (kubectl)"

  tcp_options {
    destination_port_range {
      min = 6443
      max = 6443
    }
  }
}

resource "oci_core_network_security_group_security_rule" "oke_api_ingress_12250_workers" {
  network_security_group_id = oci_core_network_security_group.oke_api.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = var.oke_subnet_cidr
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "OKE control plane from worker nodes"

  tcp_options {
    destination_port_range {
      min = 12250
      max = 12250
    }
  }
}

resource "oci_core_network_security_group_security_rule" "oke_api_ingress_icmp" {
  network_security_group_id = oci_core_network_security_group.oke_api.id
  direction                 = "INGRESS"
  protocol                  = "1"
  source                    = var.vcn_cidr
  source_type               = "CIDR_BLOCK"
  stateless                 = false
  description               = "ICMP Path MTU Discovery"

  icmp_options {
    type = 3
    code = 4
  }
}

resource "oci_core_network_security_group_security_rule" "oke_api_egress_to_workers" {
  network_security_group_id = oci_core_network_security_group.oke_api.id
  direction                 = "EGRESS"
  protocol                  = "6"
  destination               = var.oke_subnet_cidr
  destination_type          = "CIDR_BLOCK"
  stateless                 = false
  description               = "All TCP to worker/pod subnet"
}

resource "oci_core_network_security_group_security_rule" "oke_api_egress_to_services" {
  network_security_group_id = oci_core_network_security_group.oke_api.id
  direction                 = "EGRESS"
  protocol                  = "6"
  destination               = data.oci_core_services.all.services[0].cidr_block
  destination_type          = "SERVICE_CIDR_BLOCK"
  stateless                 = false
  description               = "HTTPS to OCI services"

  tcp_options {
    destination_port_range {
      min = 443
      max = 443
    }
  }
}

resource "oci_core_network_security_group_security_rule" "oke_api_egress_icmp" {
  network_security_group_id = oci_core_network_security_group.oke_api.id
  direction                 = "EGRESS"
  protocol                  = "1"
  destination               = "0.0.0.0/0"
  destination_type          = "CIDR_BLOCK"
  stateless                 = false
  description               = "ICMP Path MTU Discovery"

  icmp_options {
    type = 3
    code = 4
  }
}

# -----------------------------------------------------------------------------
# Subnets
# -----------------------------------------------------------------------------

resource "oci_core_subnet" "public" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.this.id
  display_name               = "${var.label_prefix}-subnet-public"
  cidr_block                 = var.public_subnet_cidr
  dns_label                  = "pub"
  prohibit_public_ip_on_vnic = false
  route_table_id             = oci_core_route_table.public.id
  freeform_tags              = var.freeform_tags
}

resource "oci_core_subnet" "oke_workers" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.this.id
  display_name               = "${var.label_prefix}-subnet-oke-workers"
  cidr_block                 = var.oke_subnet_cidr
  dns_label                  = "oke"
  prohibit_public_ip_on_vnic = true
  route_table_id             = oci_core_route_table.private.id
  freeform_tags              = var.freeform_tags
}

