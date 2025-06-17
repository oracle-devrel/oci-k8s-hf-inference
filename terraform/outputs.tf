# outputs.tf

output "oke_cluster_id" {
  value       = oci_containerengine_cluster.oke.id
  description = "The OCID of the OKE cluster."
}

output "kubeconfig_command" {
  value       = "oci ce cluster create-kubeconfig --cluster-id ${oci_containerengine_cluster.oke.id} --file $HOME/.kube/config --region ${var.region} --token-version 2.0.0 --overwrite"
  description = "The command to run to generate the kubeconfig file for cluster access."
}

output "node_pool_id" {
  value       = oci_containerengine_node_pool.oke_node_pool.id
  description = "The OCID of the GPU node pool."
}
