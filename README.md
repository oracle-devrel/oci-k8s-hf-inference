# Deploying LLMs using HuggingFace and Kubernetes on OCI Kubernetes Engine (OKE)

[![License: UPL](https://img.shields.io/badge/license-UPL-green)](https://img.shields.io/badge/license-UPL-green)<!--[![Quality gate](https://sonarcloud.io/api/project_badges/quality_gate?project=oracle-devrel_oci-k8s-hf-inference)](https://sonarcloud.io/dashboard?id=oracle-devrel_oci-k8s-hf-inference)-->

## Introduction

This guide provides step-by-step instructions on deploying Text Generation Inference (TGI) on Oracle Kubernetes Engine (OKE). TGI is an open-source toolkit for serving popular large language models (LLMs).

Check out the demo [here](https://www.youtube.com/watch?v=WQqlB19Dffg&t=1s)

### HuggingFace text generation inference

Text generation inference (TGI) is an open source toolkit available in containers for serving popular LLMs. The example fine-tuned model in this post is based on Llama 2, but you can use TGI to deploy other open source LLMs, including Mistral, Falcon, BLOOM, and GPT-NeoX. TGI enables high-performance text generation with various optimization features supported on multiple AI accelerators, including NVIDIA GPUs with `CUDA 12.2+`:

### GPU memory consideration

The GPU memory requirement is largely determined by the pretrained LLM's size.

For example, `Llama-2-7B` (7 billion parameters) loaded in *16-bit* precision requires 7 billion * 2 bytes (16 bits / 8 bits/byte) = 14 GB for the model weights.

**Quantization** is a technique used to reduce model size and improve inferencing performance by decreasing precision without significantly sacrificing accuracy. In this example, we use the quantization feature of TGI to load a fine-tuned model based on Llama 2 13B in 8-bit precision and fit it on `VM.GPU.A10.1` (single NVIDIA A10 Tensor Core GPU with 24-GB VRAM).

The following image depicts the real memory utilization after the inference container loads the quantized model. Alternatively, consider employing a smaller model, opting for a GPU instance with larger memory capacity, or selecting an instance with multiple GPUs, such as `VM.GPU.A10.2` (2x NVIDIA A10 GPUs), to prevent CUDA out-of-memory errors. By default, TGI shards across and uses all available GPUs to run the model:

![gpu spec](./img/gpu_specs.avif)

## 0. Prerequisites & Docs

### Prerequisites

- An OCI tenancy with available credits to spend, and access to NVIDIA A10 Tensor Core GPU(s).
- A registered and verified HuggingFace account with a valid Access Token.
- An OKE cluster with a node pool consisting of VM.GPU.A10.1 compute instances.

### Docs

For more information, see the following resources:

- [HuggingFace text generation inference](http://https://github.com/huggingface/text-generation-inference)
- [NVIDIA device plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin#deployment-via-helm)
- [HuggingFace model hub](https://huggingface.co/models)
- [OCI Kubernetes Engine (OKE)](https://www.oracle.com/cloud/cloud-native/container-engine-kubernetes/)
- [OCI Container Registry](https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm)
- [Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [NVIDIA GPU instances on OCI](https://www.oracle.com/cloud/compute/gpu/)

## Getting started

### Create an OKE Cluster using Terraform

First, create an OKE cluster using the provided Terraform configuration file (terraform/main.tf). This will create a VCN, subnets, an OKE cluster, and a node pool with GPU-enabled nodes.

To use the Terraform configuration, you'll need to provide the required variables:

- compartment_id: The ID of the compartment where you want to create the OKE cluster.
- region: The region where you want to create the OKE cluster.
- cluster_name: The name of the OKE cluster.
- node_pool_name: The name of the node pool.
- boot_volume_size_in_gbs: The size of the boot volume for the nodes in GB.
- image_id: The ID of the image to use for the nodes.
- kubernetes_version: The version of Kubernetes to use for the OKE cluster.
  
You can create a `terraform.tfvars` file to provide these variables.

  ```terraform
  compartment_id = ""
  region        = ""
  cluster_name  = "cluster"
  node_pool_name = ""
  kubernetes_version = "" # Replace with the desired Kubernetes version
  image_id = "" # Replace with your actual image OCID ie ocid1.image.oc1.iad.aaaaaaaaexample
  boot_volume_size_in_gbs = "50" # Default boot volume size is 50 in GBs, please adjust as needed
  ```

Then, initialize the Terraform working directory and apply the configuration:

  ```bash
  terraform init && terraform apply
  ```

### Generate Kubeconfig for the OKE Cluster

After creating the OKE cluster, Terraform will output a command to generate a kubeconfig file. You can reuse this command to generate the kubeconfig file:

```bash
$(terraform output -raw kubeconfig_command)
```

Run this command to generate the kubeconfig file:

```bash
oci ce cluster create-kubeconfig --cluster-id <cluster_id> --file $HOME/.kube/config --region <region> --token-version 2.0.0 --overwrite
```

### Verify Kubeconfig and Access the OKE Cluster

Verify that the kubeconfig file is correctly generated by checking the cluster nodes:

```bash
kubectl get nodes
```

The nodes should be in a Ready state. If they're not, you can check the node status and logs to troubleshoot.

Additionally, the terraform/main.tf file includes an init script that runs on node initialization. This script is used to configure the nodes.

### Add Toleration to the CoreDNS Pod

Before deploying TGI, you need to add a toleration to the CoreDNS pod to allow it to run on nodes with the NVIDIA GPU taint. You can do this by patching the CoreDNS deployment:

```bash
kubectl patch deployment coredns -n kube-system --patch '{"spec": {"template": {"spec": {"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]}}}}'
```

## Install NVIDIA Device Plugin

Install the NVIDIA device plugin to enable GPU support in Kubernetes:

```bash
# Apply the NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

Verify that the GPU is available in the cluster:

```bash
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

### Create a Kubernetes Secret for the Hugging Face Token

Create a Kubernetes Secret to store your Hugging Face token. You can do this using the following command:

```bash
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=your_hugging_face_token
```

### Create Persistent Volume Claims (PVCs)

Create PVCs for the TGI model cache and shared memory using the provided k8s/pvc.yaml file:

```bash
kubectl apply -f k8s/pvc.yaml
```

### Deploy TGI using Kubernetes

Deploy TGI using the provided k8s/deployment.yaml file:

```bash
kubectl apply -f k8s/deployment.yaml
```

This will create a Deployment for TGI with the specified configuration, including the model ID, number of shards, and quantization method.

### Expose the TGI Service

Now we need to create a Kubernetes deployment file. An example YAML file could look something like this:

```bash
kubectl apply -f k8s/service.yaml
```

This will create a Service of type LoadBalancer, which will expose the TGI service to the outside world.

## Test the TGI Service

Get the external IP address of the LoadBalancer:

```bash
kubectl get svc
```

Then, use curl to test the TGI service:

## Conclusion

Deploying a production-ready LLM becomes straightforward when using the HuggingFace TGI container and OKE. This approach allows you to harness the benefits of Kubernetes without the complexities of deploying and managing a Kubernetes cluster.

## Contributing

<!-- If your project has specific contribution requirements, update the
    CONTRIBUTING.md file to ensure those requirements are clearly explained. -->

This project welcomes contributions from the community. Before submitting a pull
request, please [review our contribution guide](./CONTRIBUTING.md).

## Security

Please consult the [security guide](./SECURITY.md) for our responsible security
vulnerability disclosure process.

## License

Copyright (c) 2024 Oracle and/or its affiliates.

Licensed under the Universal Permissive License (UPL), Version 1.0.

See [LICENSE](LICENSE.txt) for more details.

ORACLE AND ITS AFFILIATES DO NOT PROVIDE ANY WARRANTY WHATSOEVER, EXPRESS OR IMPLIED, FOR ANY SOFTWARE, MATERIAL OR CONTENT OF ANY KIND CONTAINED OR PRODUCED WITHIN THIS REPOSITORY, AND IN PARTICULAR SPECIFICALLY DISCLAIM ANY AND ALL IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.  FURTHERMORE, ORACLE AND ITS AFFILIATES DO NOT REPRESENT THAT ANY CUSTOMARY SECURITY REVIEW HAS BEEN PERFORMED WITH RESPECT TO ANY SOFTWARE, MATERIAL OR CONTENT CONTAINED OR PRODUCED WITHIN THIS REPOSITORY. IN ADDITION, AND WITHOUT LIMITING THE FOREGOING, THIRD PARTIES MAY HAVE POSTED SOFTWARE, MATERIAL OR CONTENT TO THIS REPOSITORY WITHOUT ANY REVIEW. USE AT YOUR OWN RISK.
