.PHONY: init test1 status help clean check-deps

deploy-workloads:
	@echo "=== Deploying Basic Workloads ==="
	@helmfile -f helmfile.d/02-workload.yaml apply

deploy-cv-workload:
	@echo "=== Deploying Computer Vision Workload ==="
	@helmfile -f helmfile.d/03-cv-deployment.yaml apply

deploy-qwen:
	@echo "=== Deploying Qwen Workload ==="
	@echo "helmfile -f helmfile.d/04-qwen.yaml apply"
	@helmfile -f helmfile.d/04-qwen.yaml apply

deploy-mig:
	@echo "=== Deploying MIG Workload ==="
	@echo "helmfile -f helmfile.d/05-mig.yaml apply"
	@helmfile -f helmfile.d/05-mig.yaml apply

verify-deployment:
	@echo "=== Verifying Deployments ==="
	@kubectl get pods -A | grep -E "(hami|spread|mig|yolo|qwen)" || echo "No GPU workloads found"

init: deploy-workloads deploy-cv-workload verify-deployment
	@echo "=== HAMi Core and Workload Initialization Complete ==="
	@kubectl get pods -A | grep -E "(hami|spread|mig|yolo)" | head -10

test1: deploy-qwen
	@echo "=== Waiting for Qwen Deployment ==="
	@kubectl wait --for=condition=ready pod -l app=qwen --timeout=300s || { echo "Warning: Qwen pod not ready, continuing with test..."; }
	@echo "=== Running Qwen Test ==="
	@./scripts/control.sh -q8
	@echo "=== Qwen Test Complete ==="

test2: deploy-mig
	@echo "=== Waiting for MIG Deployment ==="
	@kubectl wait --for=condition=ready pod -l app=mig --timeout=300s || { echo "Warning: MIG pod not ready, continuing with test..."; }
	@./scripts/control.sh --test-mig

destroy-qwen:
	@echo "helmfile -f helmfile.d/04-qwen.yaml destroy"
	@helmfile -f helmfile.d/04-qwen.yaml destroy
	@sleep 2
	@./scripts/gpu_visualization.py

destroy-mig:
	@echo "helmfile -f helmfile.d/05-mig.yaml destroy"
	@helmfile -f helmfile.d/05-mig.yaml destroy
	@sleep 2
	@./scripts/gpu_visualization.py

destroy-cv-workload:
	@echo "=== Destroying Computer Vision Workload ==="
	@helmfile -f helmfile.d/03-cv-deployment.yaml destroy

destroy-workloads:
	@echo "=== Destroying Basic Workloads ==="
	@helmfile -f helmfile.d/02-workload.yaml destroy

destroy-hami-core:
	@echo "=== Destroying HAMi Core ==="
	@helmfile -f helmfile.d/01-hami-core.yaml destroy

status:
	@echo "=== Cluster Status ==="
	@kubectl get nodes -o wide | head -10
	@echo ""
	@echo "=== GPU Workload Status ==="
	@kubectl get pods -A | grep -E "(hami|spread|mig|yolo|qwen)" || echo "No GPU workloads found"
	@echo ""
	@echo "=== Resource Usage ==="
	@kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUs:.status.capacity.\"nvidia\.com/gpu\.count\",MEMORY:.status.capacity.memory 2>/dev/null || echo "Resource info not available"

# Display help information
help:
	@echo "Available targets:"
	@echo ""
	@echo "Core Commands:"
	@echo "  init          - Install HAMi core and deploy all workloads"
	@echo "  test1         - Deploy Qwen and run test"
	@echo "  clean         - Destroy all workloads and clean up"
	@echo ""
	@echo "Individual Commands:"
	@echo "  status        - Show cluster and deployment status"
	@echo ""
	@echo "Development Commands:"
	@echo "  add-hami-repo - Add HAMi charts repository"
	@echo "  install-hami-core    - Install HAMi core scheduler"
	@echo "  deploy-workloads     - Deploy vLLM 4B + MIG workloads"
	@echo "  deploy-cv-workload   - Deploy YOLOv8n CV workload"
	@echo "  deploy-qwen        - Deploy Qwen workload"
	@echo "  verify-deployment    - Check deployment status"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  destroy-qwen       - Destroy Qwen workload"
	@echo "  destroy-cv-workload  - Destroy CV workload"
	@echo "  destroy-workloads     - Destroy basic workloads"
	@echo "  destroy-hami-core     - Destroy HAMi core"
	@echo ""
	@echo "Examples:"
	@echo "  make init            # Full initialization"
	@echo "  make test1           # Deploy and test Qwen "
	@echo "  make test2           # Deploy dynamic MIG"
	@echo "  make clean           # Clean up everything"

clean: destroy-qwen destroy-mig
	@echo "✓ Cleanup complete"
