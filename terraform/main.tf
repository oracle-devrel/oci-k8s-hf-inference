# HuggingFace OKE Terraform Module

provider "oci" {
region = var.region
}

variable "compartment_id" {}
variable "region" {}
variable "cluster_name" {}
variable "node_pool_name" {}
variable "boot_volume_size_in_gbs" {}
variable "image_id" {}
variable "kubernetes_version" {}

# Networking best practices: separate public/private subnets, internet gateway, route tables
resource "oci_core_vcn" "oke_vcn" {
  compartment_id = var.compartment_id
  display_name   = "oke-vcn"
  cidr_block     = "10.0.0.0/16"
}

resource "oci_core_internet_gateway" "oke_igw" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.oke_vcn.id
  display_name   = "oke-igw"
}

resource "oci_core_route_table" "oke_rt" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.oke_vcn.id
  display_name   = "oke-rt"
  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.oke_igw.id
  }
}

resource "oci_core_subnet" "oke_subnet" {
  compartment_id               = var.compartment_id
  vcn_id                       = oci_core_vcn.oke_vcn.id
  cidr_block                   = "10.0.1.0/24"
  display_name                 = "oke-subnet"
  prohibit_public_ip_on_vnic = false
  route_table_id               = oci_core_route_table.oke_rt.id
  security_list_ids            = [oci_core_vcn.oke_vcn.default_security_list_id]
}

resource "oci_core_subnet" "oke_node_subnet" {
  compartment_id               = var.compartment_id
  vcn_id                       = oci_core_vcn.oke_vcn.id
  cidr_block                   = "10.0.2.0/24"
  display_name                 = "oke-node-subnet"
  prohibit_public_ip_on_vnic = false
  route_table_id               = oci_core_route_table.oke_rt.id
  security_list_ids            = [oci_core_vcn.oke_vcn.default_security_list_id]
}

# OKE Cluster
resource "oci_containerengine_cluster" "oke" {
  compartment_id     = var.compartment_id
  name               = var.cluster_name
  vcn_id             = oci_core_vcn.oke_vcn.id
  kubernetes_version = var.kubernetes_version
  options {
    add_ons {
      is_kubernetes_dashboard_enabled = false
      is_tiller_enabled               = false
    }
    service_lb_subnet_ids = [oci_core_subnet.oke_subnet.id]
  }
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_id
}

data "oci_containerengine_node_pool_option" "oke_node_pool_option" {
  node_pool_option_id = oci_containerengine_cluster.oke.id
}

locals {
  oke_init_script = base64encode(<<-EOT
    #!/bin/bash
    curl --fail -H "Authorization: Bearer Oracle" -L0 http://169.254.169.254/opc/v2/instance/metadata/oke_init_script | base64 --decode >/var/run/oke-init.sh
    bash /var/run/oke-init.sh
    sudo /usr/libexec/oci-growfs -y
  EOT
  )
}

resource "oci_containerengine_node_pool" "oke_node_pool" {
  compartment_id     = var.compartment_id
  cluster_id         = oci_containerengine_cluster.oke.id
  name               = var.node_pool_name
  node_shape         = "VM.GPU.A10.2"
  kubernetes_version = var.kubernetes_version

  node_config_details {
    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.oke_node_subnet.id
    }
    size = 3
  }

  node_source_details {
    source_type             = "IMAGE"
    image_id                = var.image_id
    boot_volume_size_in_gbs = var.boot_volume_size_in_gbs
  }

  initial_node_labels {
    key   = "nvidia.com/gpu"
    value = "true"
  }

  node_metadata = {
    "user_data" = local.oke_init_script
  }
}

# Output kubeconfig for automation
output "kubeconfig" {
  value       = "oci ce cluster create-kubeconfig --cluster-id ${oci_containerengine_cluster.oke.id} --file $HOME/.kube/config --region ${var.region}"
  description = "Command to generate kubeconfig for the OKE cluster"
  sensitive   = true
}