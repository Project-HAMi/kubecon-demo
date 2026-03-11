#!/usr/bin/env bash
# More safety, by turning some bugs into errors.
# Without `errexit` you don’t need ! and can replace
# PIPESTATUS with a simple $?, but I don’t do that.
set -o errexit -o pipefail -o noclobber -o nounset

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && (pwd -W 2> /dev/null || pwd))
cd "$SCRIPT_DIR/.."


test_qwen8b() {
    echo "=== Starting Qwen 8B Test Workflow ==="

    echo "Step 1: Waiting for deployment readiness..."
    kubectl wait --for=condition=ready pod -l app=qwen8b --timeout=300s

    echo "Step 2: Checking GPU resources in container (should show 25GB allocation)..."
    POD_NAME=$(kubectl get pods -l app=qwen8b -o jsonpath='{.items[0].metadata.name}')
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
    
    echo "=== Qwen 8B Test Workflow Complete ==="
}

visualize() {
    echo "=== Starting GPU Cluster Visualization ==="
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
    -q8|--test-qwen8b)
      TEST_QWEN8B=YES
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

if [[ -v TEST_QWEN8B ]]; then
  test_qwen8b
elif [[ -v VISUALIZE ]]; then
  visualize
else
    echo -n "\
Please specify the right commandline option:
-q8/--test-qwen8b : test qwen8b
-pf/--port-forward : use port-forward for vLLM API access
-v/--visualize    : generate GPU cluster visualizations

Example usage:
  ./control.sh -q8                            # Test Qwen 8B without port-forward
  ./control.sh -q8 -pf                       # Test Qwen 8B with port-forward
  ./control.sh -v                            # Generate visualizations
"
fi

