# HAMi vLLM Deployment

Deployment of vLLM using HAMi heterogeneous GPU scheduling with support for multiple model sizes and resource policies.

## Files

- `helmfile.d/01-hami-core.yaml` - HAMi core scheduler
- `helmfile.d/02-workload.yaml` - vLLM 4B deployment, MIG test pod
- `helmfile.d/03-cv-deployment.yaml` - YOLOv8n deployment (20 replicas)
- `helmfile.d/04-qwen7b-deployment.yaml` - Qwen 7B vLLM deployment (NEW)
- `charts/spread/` - vLLM 4B deployment configuration
- `charts/qwen7b/` - Qwen 7B deployment configuration (NEW)
- `charts/mig/` - MIG test pod configuration
- `charts/yolo8/` - YOLOv8n deployment configuration
- `scripts/control.sh` - Testing and deployment management (NEW)
- `test_vllm.py` - vLLM API testing script (UPDATED)

## Prerequisites

- Kubernetes cluster with GPU nodes (NVIDIA GPUs recommended)
- Helm 3+ installed
- HAMi installed in the kube-system namespace: `helm repo add hami-charts https://project-hami.github.io/HAMi/`
- kubectl configured for cluster access
- Python 3.8+ with openai package: `pip install openai`

## Installation

### 1. Install HAMi Core

```bash
helm repo add hami-charts https://project-hami.github.io/HAMi/
helmfile sync
```

### 2. Deploy Workloads

#### Deploy All Workloads
```bash
helmfile sync
```

#### Deploy Individual Workloads
```bash
# Deploy vLLM 4B spread workload
helmfile -f helmfile.d/02-workload.yaml apply

# Deploy Qwen 7B workload
helmfile -f helmfile.d/04-qwen7b.yaml apply

# Deploy YOLOv8n workload
helmfile -f helmfile.d/03-cv-deployment.yaml apply
```

### 3. Verify Deployments

```bash
# Check all deployments
kubectl get pods -A | grep -E "(hami|qwen|yolo|spread)"

# Check Qwen 7B specifically
kubectl get pods -l app=qwen7b
kubectl get svc qwen7b-service
```

## Qwen 7B vLLM Deployment

The Qwen 7B deployment features:
- Model: Qwen/Qwen3-7B-Instruct
- Memory: 25GB GPU allocation
- Policy: binpack (efficient resource utilization)
- Service: qwen7b-service on port 8000

### Deployment

```bash
# Deploy Qwen 7B using helmfile
helmfile -f helmfile.d/04-qwen7b.yaml apply

# Wait for deployment to be ready
kubectl wait --for=condition=ready pod -l app=qwen7b --timeout=300s
```

### Testing

#### Option 1: Basic Testing (Service URL)
```bash
./scripts/control.sh --test-qwen7b
```
- Uses service URL directly
- Falls back to port-forward if service unavailable
- Tests nvidia-smi, chat API, and host GPU

#### Option 2: Port-Forward Testing
```bash
./scripts/control.sh --test-qwen7b --port-forward
```
- Uses port-forward for local access
- Automatic port-forward management
- Same testing workflow

#### Option 3: Manual Port-Forward
```bash
# Start port-forward manually
kubectl port-forward svc/qwen7b-service 8000:8000

# Test API locally
python3 test_vllm.py
```

#### Option 4: Direct API Testing
```bash
# Port-forward first
kubectl port-forward svc/qwen7b-service 8000:8000

# Test with OpenAI client
python3 -c "
import openai
client = openai.OpenAI(base_url='http://localhost:8000/v1', api_key='dummy')
response = client.chat.completions.create(
    model='Qwen/Qwen3-7B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}],
    max_tokens=50
)
print(response.choices[0].message.content)
"
```

## Testing Workflows

### Complete Qwen 7B Testing
```bash
# Full test with nvidia-smi verification and streaming chat
./scripts/control.sh --test-qwen7b

# Full test with port-forward
./scripts/control.sh --test-qwen7b --port-forward
```

### vLLM API Testing
```bash
# Test streaming chat response
python3 test_vllm.py

# Test with custom prompts
python3 -c "
import openai
client = openai.OpenAI(base_url='http://localhost:8000/v1', api_key='dummy')
response = client.chat.completions.create(
    model='Qwen/Qwen3-7B-Instruct',
    messages=[
        {'role': 'system', 'content': 'You are a helpful AI assistant.'},
        {'role': 'user', 'content': 'Explain quantum computing in simple terms.'}
    ],
    max_tokens=200,
    temperature=0.7,
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
print()
"
```

### GPU Resource Monitoring
```bash
# Check container GPU usage
kubectl exec -it <qwen7b-pod> -- nvidia-smi

# Check host GPU usage
kubectl get nodes -o name | xargs -I {} kubectl node-shell {} -- nvidia-smi

# Monitor GPU utilization over time
watch -n 2 kubectl exec -it <qwen7b-pod> -- nvidia-smi
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

### Current Deployment Overview

```
## Troubleshooting

### Common Issues

#### Port Forward Connection Issues
```bash
# Check if service is accessible
kubectl get svc qwen7b-service

# Check pod status
kubectl get pods -l app=qwen7b

# View pod logs
kubectl logs -l app=qwen7b

# Test service connectivity
kubectl exec -it <qwen7b-pod> -- curl http://localhost:8000/health
```

#### GPU Resource Allocation
```bash
# Check GPU resources in pod
kubectl exec -it <qwen7b-pod> -- nvidia-smi

# Check node GPU availability
kubectl describe nodes <node-name> | grep -A 10 nvidia.com

# Check HAMi scheduler events
kubectl get events --sort-by=.metadata.creationTimestamp | grep -i hami
```

#### API Connection Issues
```bash
# Test API endpoint directly
curl http://qwen7b-service.default.svc.cluster.local:8000/v1/models

# Test with Python
python3 -c "
import requests
try:
    response = requests.get('http://qwen7b-service.default.svc.cluster.local:8000/v1/models')
    print(response.json())
except Exception as e:
    print(f'Error: {e}')
"
```

### Debug Commands
```bash
# Debug port-forward
kubectl port-forward svc/qwen7b-service 8000:8000 --v=9

# Debug HAMi scheduler
kubectl logs -n kube-system deployment/hami-scheduler

# Debug vLLM container
kubectl exec -it <qwen7b-pod> -- /bin/bash
```

### Testing Workflow Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Complete Testing Workflow                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                Step 1: Deployment                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │   helmfile -f helmfile.d/04-qwen7b.yaml apply    │   │
│  │   └─► Wait for pod readiness (300s timeout)      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Step 2: Container nvidia-smi               │
│  ┌──────────────────────────────────────────────────┐   │
│  │   kubectl exec -it <pod> -- nvidia-smi           │   │
│  │   └─► Verify 25GB GPU allocation                 │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                Step 3: Chat API Test                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │   python3 test_vllm.py                           │   │
│  │   └─► Streaming response with port-forward fallback  │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                Step 4: Host nvidia-smi                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │   kubectl node-shell <node> -- nvidia-smi        │   │
│  │   └─► Verify host GPU utilization                │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```
