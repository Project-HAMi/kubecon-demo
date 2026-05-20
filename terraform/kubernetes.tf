provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
}

resource "kubernetes_service_account_v1" "admin" {
  metadata {
    name      = "admin-user"
    namespace = "kube-system"
  }
  depends_on = [google_container_cluster.primary]
}

resource "kubernetes_cluster_role_binding_v1" "admin" {
  metadata {
    name = "admin-user"
  }
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = "editor"
  }
  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account_v1.admin.metadata[0].name
    namespace = "kube-system"
  }
  depends_on = [google_container_cluster.primary]
}

resource "kubernetes_secret_v1" "admin_token" {
  metadata {
    name      = "admin-user-token"
    namespace = "kube-system"
    annotations = {
      "kubernetes.io/service-account.name" = kubernetes_service_account_v1.admin.metadata[0].name
    }
  }
  type = "kubernetes.io/service-account-token"

  depends_on = [google_container_cluster.primary]
}

