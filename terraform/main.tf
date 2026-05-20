variable "cluster_name" {
  description = "The name of the GKE cluster to create."
  type        = string
  default     = "kubecon-india-environment2"
}

variable "project_id" {
  description = "The ID of the project in which to create the cluster."
  type        = string
}

locals {
  ssh_key_files = fileset("${path.module}/../ssh-keys", "*.pub")
  node_metadata = merge(
    {
      for f in local.ssh_key_files :
      "ssh-keys" => "hami_demo:${trimspace(file("${path.module}/../ssh-keys/${f}"))}"
      # "ssh-keys-${trimsuffix(f, ".pub")}" => "hami_demo:${trimspace(file("${path.module}/../ssh-keys/${f}"))}"
    },
    {
      "disable-legacy-endpoints" = true
    }
  )
}

# resource "google_storage_bucket" "terraform_state" {
#   name     = "demo-environments-hami"
# }

# terraform {
#   backend "gcs" {
#     bucket = "demo-environments-hami"
#     prefix = "kubecon-india"
#   }
#   required_providers {
#     kubernetes = {
#       source  = "hashicorp/kubernetes"
#       version = "~> 2.0"
#     }
#   }
# }

data "google_client_config" "default" {}

variable "region" {
  description = "The region in which to create the cluster."
  type        = string
  default     = "asia-east1"
}

variable "zone" {
  description = "The zone for the cluster nodes."
  type        = string
  default     = "asia-northeast3-b"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone

  initial_node_count = 3
  resource_labels    = {}

  deletion_protection = false
  node_config {
    machine_type = "a2-highgpu-2g"
    image_type   = "UBUNTU_CONTAINERD"
    labels = {
      gpu = "on"
    }
    guest_accelerator {
      type  = "nvidia-tesla-a100"
      count = 2
    }
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    service_account = "default"
    metadata        = local.node_metadata
  }
  # metadata_startup_script = "useradd -m -s /bin/bash 'admin'aaaaaaaa;echo 'admin:password@54w21@123@@' | chpasswd"
}

data "google_compute_instance_group" "gke_nodes" {
  # self_link = google_container_cluster.primary.node_pool[0].instance_group_urls[0]
  zone = var.zone
  name = "gke-kubecon-india-enviro-default-pool-7a3e2b4f-grp"
}
