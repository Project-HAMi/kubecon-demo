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
  instance_group_name = element(
    split("/", google_container_cluster.primary.node_pool[0].instance_group_urls[0]),
    length(split("/", google_container_cluster.primary.node_pool[0].instance_group_urls[0])) - 1
  )
  node_metadata = merge(
    {
      for f in local.ssh_key_files :
      "ssh-keys" => "hami_demo:${trimspace(file("${path.module}/../ssh-keys/${f}"))}"
      # "ssh-keys-${trimsuffix(f, ".pub")}" => "hami_demo:${trimspace(file("${path.module}/../ssh-keys/${f}"))}"
    },
    {
      "disable-legacy-endpoints" = true,
    }
  )
}

# resource "google_storage_bucket" "terraform_state" {
#   name     = "demo-environments-hami"
#   location = var.region
# }

terraform {
  backend "gcs" {
    bucket = "demo-environments-hami"
    prefix = "kubecon-india"
  }
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

data "google_client_config" "default" {}

variable "region" {
  description = "The region in which to create the cluster."
  type        = string
  default     = "asia-northeast3"
}

variable "zone" {
  description = "The zone for the cluster nodes."
  type        = string
  default     = "b"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = "${var.region}-${var.zone}"

  initial_node_count = 3
  resource_labels    = {}

  deletion_protection = false
  node_config {
    machine_type = "a2-highgpu-2g"
    image_type   = "UBUNTU_CONTAINERD"
    # image_type   = "COS_CONTAINERD" r/o rootfs does not work with HAMi
    labels = {
      gpu                                       = "on"
      "gke-no-default-nvidia-gpu-device-plugin" = "true"
    }
    guest_accelerator {
      type  = "nvidia-tesla-a100"
      count = 2
      gpu_driver_installation_config {
        gpu_driver_version = "INSTALLATION_DISABLED"
      }
    }
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    service_account = "default"
    metadata        = local.node_metadata
  }
}

