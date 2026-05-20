output "node_ip_addresses" {
  description = "IP addresses of all GKE cluster nodes"
  value       = data.google_compute_instance_group.gke_nodes.instances
}
