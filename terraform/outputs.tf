data "google_compute_instance" "gke_nodes" {
  count     = length(tolist(data.google_compute_instance_group.gke_nodes.instances))
  self_link = tolist(data.google_compute_instance_group.gke_nodes.instances)[count.index]
}

output "node_ip_addresses" {
  description = "IP addresses of all GKE cluster nodes"
  value = {
    for inst in data.google_compute_instance.gke_nodes :
    inst.name => {
      internal_ip = inst.network_interface[0].network_ip
      external_ip = try(inst.network_interface[0].access_config[0].nat_ip, null)
    }
  }
}
