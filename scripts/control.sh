#!/usr/bin/env bash
# More safety, by turning some bugs into errors.
# Without `errexit` you don’t need ! and can replace
# PIPESTATUS with a simple $?, but I don’t do that.
set -o errexit -o pipefail -o noclobber -o nounset

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && (pwd -W 2> /dev/null || pwd))
cd "$SCRIPT_DIR/.."


test_qwen() {
    echo "=== Starting Qwen Test Workflow ==="

    echo "Step 1: Waiting for deployment readiness..."
    kubectl wait --for=condition=ready pod -l app=qwen --timeout=300s

    echo "Step 2: Checking GPU resources in container (should show 25GB allocation)..."
    POD_NAME=$(kubectl get pods -l app=qwen -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -it "$POD_NAME" -- nvidia-smi
    
    echo "Step 3: Checking GPU resources on host node..."
    NODE_NAME=$(kubectl get pods -l app=qwen -o jsonpath='{.items[0].spec.nodeName}')
    echo "Found qwen pod on node: $NODE_NAME"

    ssh "$NODE_NAME" nvidia-smi
    echo "Step 3: Checking GPU resources in container (should show 25GB allocation)..."
    python3 ./scripts/gpu_visualization.py

    echo "Step 4: Testing chat API with streaming output..."
    python3 ./scripts/test_vllm.py --app qwen --model "Qwen/Qwen3-1.7B"
    
    echo "Step 5: Checking GPU resources in container (should show 25GB allocation)..."
    POD_NAME=$(kubectl get pods -l app=qwen -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -it "$POD_NAME" -- nvidia-smi
    
    echo "Step 6: Checking GPU resources on host node..."
    NODE_NAME=$(kubectl get pods -l app=qwen -o jsonpath='{.items[0].spec.nodeName}')
    echo "Found qwen pod on node: $NODE_NAME"
    
    echo "=== Qwen Test Workflow Complete ==="
}

test_mig() {
    echo "=== Starting MIG Test Workflow ==="

    echo "Step 1: Waiting for deployment readiness..."
    kubectl wait --for=condition=ready pod -l app=mig --timeout=300s

    echo "Step 2: Checking GPU resources in container..."
    POD_NAME=$(kubectl get pods -l app=mig -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -it "$POD_NAME" -- nvidia-smi

    echo "Step 3: Checking GPU resources on host node..."
    NODE_NAME=$(kubectl get pods -l app=mig -o jsonpath='{.items[0].spec.nodeName}')
    echo "Found mig pod on node: $NODE_NAME"
    ssh "$NODE_NAME" nvidia-smi
    
    echo "Step 3: Checking GPU resources in container ..."
    python3 ./scripts/gpu_visualization.py

    echo "Step 4: Testing chat API with streaming output..."
    python3 ./scripts/test_vllm.py --app mig --model "Qwen/Qwen3-4B"

    echo "Step 5: Checking GPU resources in container ..."
    POD_NAME=$(kubectl get pods -l app=mig -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -it "$POD_NAME" -- nvidia-smi

    echo "Step 6: Checking GPU resources on host node..."
    NODE_NAME=$(kubectl get pods -l app=mig -o jsonpath='{.items[0].spec.nodeName}')
    echo "Found mig pod on node: $NODE_NAME"
    ssh "$NODE_NAME" nvidia-smi
    
    echo "=== MIG Test Workflow Complete ==="
}
visualize() {
    echo "=== Starting GPU Cluster Visualization ==="
    echo "Generating GPU cluster visualizations..."
    python3 scripts/gpu_visualization.py --output-dir ./output

    echo "=== Visualization Complete ==="
    echo "Output saved to ./output/"
    echo "Generated files:"
    echo "  - interactive_dashboard.html: Interactive Plotly dashboard"
    echo "  - visualization_report.txt: Text summary report"
}

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -q8|--test-qwen)
      TEST_QWEN=YES
      shift # past argument
      ;;
    --test-mig)
      TEST_MIG=YES
      shift # past argument
      ;;
    -pf|--port-forward)
      USE_PORT_FORWARD=YES
      shift # past argument
      ;;
    -v|--visualize)
      VISUALIZE=YES
      shift # past argument
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [[ -v TEST_QWEN ]]; then
  test_qwen
elif [[ -v TEST_MIG ]]; then
  test_mig
elif [[ -v VISUALIZE ]]; then
  visualize
else
    echo -n "\
Please specify the right commandline option:
-q8/--test-qwen : test qwen
-pf/--port-forward : use port-forward for vLLM API access
-v/--visualize    : generate GPU cluster visualizations

Example usage:
  ./control.sh -q8                            # Test Qwen without port-forward
  ./control.sh -q8 -pf                       # Test Qwen with port-forward
  ./control.sh -v                            # Generate visualizations
"
fi

