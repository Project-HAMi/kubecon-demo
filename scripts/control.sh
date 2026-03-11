#!/usr/bin/env bash
# More safety, by turning some bugs into errors.
# Without `errexit` you don’t need ! and can replace
# PIPESTATUS with a simple $?, but I don’t do that.
set -o errexit -o pipefail -o noclobber -o nounset

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && (pwd -W 2> /dev/null || pwd))
cd "$SCRIPT_DIR/.."


test_qwen7b() {
    echo "=== Starting Qwen 7B Test Workflow ==="
    
    # Check if user wants port-forward
    if [[ "${USE_PORT_FORWARD:-false}" == "true" ]]; then
        echo "Step 0: Starting kubectl port-forward..."
        kubectl port-forward svc/qwen7b-service 8000:8000 &
        PORT_FORWARD_PID=$!
        sleep 3  # Give port-forward time to establish
        echo "✓ Port-forward started (PID: $PORT_FORWARD_PID)"
    fi
    
    echo "Step 1: Waiting for deployment readiness..."
    kubectl wait --for=condition=ready pod -l app=qwen7b --timeout=300s
    
    echo "Step 2: Checking GPU resources in container (should show 25GB allocation)..."
    POD_NAME=$(kubectl get pods -l app=qwen7b -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -it "$POD_NAME" -- nvidia-smi
    
    echo "Step 3: Testing chat API with streaming output..."
    python3 test_vllm.py
    
    echo "Step 4: Checking GPU resources on host node..."
    kubectl get nodes -o name | xargs -I {} kubectl node-shell {} -- nvidia-smi
    
    if [[ "${USE_PORT_FORWARD:-false}" == "true" ]]; then
        echo "Step 5: Cleaning up port-forward..."
        kill $PORT_FORWARD_PID 2>/dev/null || true
        echo "✓ Port-forward cleaned up"
    fi
    
    echo "=== Qwen 7B Test Workflow Complete ==="
}

visualize() {
    echo "=== Starting GPU Cluster Visualization ==="
    
    # Check if python3 with required packages is available
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: python3 not found"
        exit 1
    fi
    
    
    # Generate visualizations
    echo "Generating GPU cluster visualizations..."
    python3 scripts/gpu_visualization.py --output-dir ./output
    
    echo "=== Visualization Complete ==="
    echo "Output saved to ./output/"
    echo "Generated files:"
    echo "  - cluster_heatmap.png: Cluster-wide VRAM usage heatmap"
    echo "  - pod_placement_map.png: Detailed pod placement on individual GPUs"
    echo "  - pod_type_distribution.png: Pod type distribution charts"
    echo "  - vram_utilization.png: VRAM utilization analysis"
    echo "  - interactive_dashboard.html: Interactive Plotly dashboard"
    echo "  - visualization_report.txt: Text summary report"
}

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -q7|--test-qwen7b)
      TEST_QWEN7B=YES
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

if [[ -v TEST_QWEN7B ]]; then
  test_qwen7b
elif [[ -v VISUALIZE ]]; then
  visualize
else
    echo -n "\
Please specify the right commandline option:
-q7/--test-qwen7b : test qwen7b
-pf/--port-forward : use port-forward for vLLM API access
-v/--visualize    : generate GPU cluster visualizations

Example usage:
  ./control.sh -q7                            # Test Qwen 7B without port-forward
  ./control.sh -q7 -pf                       # Test Qwen 7B with port-forward
  ./control.sh -v                            # Generate visualizations
"
fi

