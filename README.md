# HAMi vLLM Deployment

Deployment of vLLM using HAMi heterogeneous GPU scheduling.

## Files

- `helmfile.yaml` - Main Helmfile configuration
- `helmfile.d/01-hami.yaml` - HAMi core
- `helmfile.d/02-workload.yaml` - vLLM deployment, and MIG test pod
- `helmfile.d/03-cv-deployment.yaml` - YOLOv8n deployment (20 replicas)

## Prerequisites

- Kubernetes cluster with GPU nodes
- Helm 3+
- HAMi installed in the cluster
- kubectl configured

## Installation

### 1. Install HAMi

```bash
helm repo add hami-charts https://project-hami.github.io/HAMi/
helmfile sync
```

### 2. Wait for vLLM to be ready

```bash
kubectl wait --for=condition=ready pod -l app-name=node-spread --timeout=300s
```

### 3. Port forward for testing

```bash
kubectl port-forward svc/node-spread 8000:8000
```

## Testing YOLOv8n

### Check deployment status

```bash
kubectl get pods -l app=yolov8n
kubectl logs -l app=yolov8n
```

### Run Kubernetes test script

```bash
bash test_yolov8_k8s.sh
```

### Test directly in container

```bash
kubectl exec -it <pod-name> -- python -c "from ultralytics import YOLO; print(YOLO('yolov8n.pt'))"
```

### Setup local environment

```bash
pip install -r requirements-cv.txt
python test_yolov8.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              HAMi Deployment Overview                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 vLLM Deployment (4 replicas)             │
│  ┌──────────────────────────────────────────────────┐  │
│  │  vllm/vllm-openai:latest                        │  │
│  │  - HAMi scheduler (spread policy)               │  │
│  │  - GPU: 1, GPU cores: 30, GPU mem: 15000m       │  │
│  └──────────────────────────────────────────────────┘  │
│                      │                                   │
│                      ▼                                   │
│           ┌────────────────────┐                         │
│           │    Service         │                         │
│           │    node-spread     │                         │
│           │    Port: 8000      │                         │
│           └────────────────────┘                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│            YOLOv8n Deployment (20 replicas)              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ultralytics/ultralytics:latest                 │  │
│  │  - HAMi scheduler (spread policy)               │  │
│  │  - GPU: 1, GPU mem: 1024                         │  │
│  └──────────────────────────────────────────────────┘  │
│  [20 instances across GPU nodes]                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              MIG Test Pod (1 instance)                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ubuntu:22.04                                    │  │
│  │  - MIG mode: mig                                 │  │
│  │  - GPU: 1, GPU mem: 1000                         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```
