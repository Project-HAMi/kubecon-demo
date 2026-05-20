data "google_compute_instance_group" "gke_nodes" {
  zone       = "${var.region}-${var.zone}"
  name       = local.instance_group_name
  depends_on = [google_container_cluster.primary]
}

data "google_compute_instance" "gke_nodes" {
  count      = google_container_cluster.primary.initial_node_count
  self_link  = tolist(data.google_compute_instance_group.gke_nodes.instances)[count.index]
  depends_on = [data.google_compute_instance_group.gke_nodes]
}

output "node_ip_addresses" {
  description = "IP addresses of all GKE cluster nodes"
  value = [
    for inst in data.google_compute_instance.gke_nodes :
    try(inst.network_interface[0].access_config[0].nat_ip, null)
  ]
  depends_on = [google_container_cluster.primary]
}

output "credential_command" {
  value = "gcloud container clusters get-credentials --region ${var.region}-${var.zone} ${var.cluster_name}"
}

output "kubeconfig" {
  description = "kubeconfig for the cluster-admin user"
  value = yamlencode({
    apiVersion = "v1"
    kind       = "Config"
    clusters = [{
      name = google_container_cluster.primary.name
      cluster = {
        certificate-authority-data = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
        server                     = "https://${google_container_cluster.primary.endpoint}"
      }
    }]
    users = [{
      name = "admin-user"
      user = {
        token = kubernetes_secret_v1.admin_token.data["token"]
      }
    }]
    contexts = [{
      name = "admin"
      context = {
        cluster = google_container_cluster.primary.name
        user    = "admin-user"
      }
    }]
    current-context = "admin"
  })
  sensitive = true
}
