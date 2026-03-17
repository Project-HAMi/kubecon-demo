# HAMi vLLM Deployment

Deployment of vLLM using HAMi heterogeneous GPU scheduling with support for multiple model sizes and resource policies.

## Quick Start

```bash
# Initialize HAMi core and deploy all workloads
make init

# Test Qwen deployment
make test1

# Test Dynamic MIG deployment
make test2

# Clean up all workloads
make clean
```

## Prerequisites

- Kubernetes cluster with GPU nodes (NVIDIA GPUs recommended)
- Helm 3+ installed
- kubectl configured for cluster access
- Python 3.8+ with openai package: `pip install openai`

## Makefile Commands

### Core Commands
- `make init` - Install HAMi core and deploy all workloads
- `make test1` - Deploy Qwen and run test
- `make test2` - Deploy Dynamic MIG and run test
- `make clean` - Destroy all workloads and clean up

### Utility Commands
- `make status` - Show cluster and deployment status
- `make help` - Display all available commands

## Files

- `helmfile.d/` - Helmfile configurations for HAMi and workloads
- `charts/` - Kubernetes resource definitions
- `scripts/control.sh` - Testing and deployment management
- `Makefile` - Automated deployment workflow
- `test_vllm.py` - vLLM API testing script

## Architecture

```
HAMi Core → vLLM Workloads → Qwen 7B → YOLOv8n → Testing
     ↓              ↓           ↓          ↓        ↓
  Scheduler    vLLM 4B   25GB GPU    CV Workload   Control Script
           + MIG Test    (binpack)   20 replicas
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
