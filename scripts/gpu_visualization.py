#!/usr/bin/env python3
"""
GPU VRAM Visualization for HAMi vLLM Deployment

Uses nvidia-smi commands to get actual VRAM information from GPU pods.
Generates interactive HTML dashboard only.

Usage:
    python3 gpu_visualization.py
    python3 gpu_visualization.py --output-dir ./output
    python3 gpu_visualization.py --pod-status Running
"""

import argparse
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import plotly.graph_objects as go
import plotly.subplots as sps

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CLUSTER_NAME = "HAMi Cluster"
TOTAL_GPUS_PER_NODE = 1  # 1x A100 per node
GPU_MEMORY_GB = 80  # A100 80GB

POD_COLORS = {
    "qwen": "#1f77b4",  # Blue
    "yolo": "#ff7f0e",  # Orange
    "vllm": "#2ca02c",  # Green
    "mig": "#d62728",  # Red
    "spread": "#9467bd",  # Purple
    "other": "#8c564b",  # Brown
}


class SeabornGPUVisualizer:
    """Simplified GPU visualizer using seaborn and nvidia-smi for VRAM info."""

    def __init__(self, pod_status_filter="Scheduled,Running"):
        """Initialize visualizer."""
        self.pod_status_filter = pod_status_filter
        self.nodes = self._get_nodes()
        self.pods = self._get_pods()
        self.gpu_pods = self._filter_gpu_pods()

    def _run_command(self, command: str) -> str:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timeout: {command}")
            return ""
        except Exception as e:
            logger.error(f"Command failed: {command}, error: {e}")
            return ""

    def _get_nodes(self) -> List[Dict[str, str]]:
        """Get nodes from cluster."""
        nodes = []
        output = self._run_command("kubectl get nodes -o json")

        if not output:
            return []

        try:
            data = json.loads(output)
            for node in data["items"]:
                nodes.append(
                    {
                        "name": node["metadata"]["name"],
                        "status": "Ready"
                        if any(
                            condition["type"] == "Ready"
                            and condition["status"] == "True"
                            for condition in node["status"]["conditions"]
                        )
                        else "Not Ready",
                        "labels": node["metadata"]["labels"] or {},
                    }
                )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")

        return nodes

    def _get_pods(self) -> List[Dict[str, Any]]:
        """Get pods from cluster."""
        pods = []
        output = self._run_command("kubectl get pods -A -o json")

        if not output:
            return []

        try:
            data = json.loads(output)
            for pod in data["items"]:
                # Get detailed status including container states
                status = pod["status"]["phase"]
                container_statuses = pod.get("status", {}).get("containerStatuses", [])

                # Check if all containers are running
                all_running = True
                if container_statuses:
                    for container in container_statuses:
                        if container.get("state", {}).get("running") is None:
                            all_running = False
                            break
                    if all_running:
                        status = "Running"

                # Remove pod hash/version from name using regex
                pod_name = pod["metadata"]["name"]
                # Remove hash pattern (e.g., -7fdf758cf8, -769648db94, -5¢56989dd6)
                clean_name = re.sub(r"-[a-z0-9]{10}-[a-z0-9]{6}$", "", pod_name)
                # print(f"DEBUG: Converted {clean_name}")

                pods.append(
                    {
                        "name": pod_name,
                        "czlean_name": clean_name,
                        "namespace": pod["metadata"]["namespace"],
                        "node_name": pod["spec"]["nodeName"],
                        "status": status,
                        "pod_type": self._classify_pod_type(pod),
                        "gpu_memory_gb": self._get_gpu_memory_from_pod(pod),
                        "gpu_count": self._get_gpu_count_from_pod(pod),
                    }
                )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")

        return pods

    def _classify_pod_type(self, pod: Dict[str, Any]) -> str:
        """Classify pod type based on name, labels, and containers."""
        name = pod["metadata"]["name"].lower()
        namespace = pod["metadata"]["namespace"].lower()

        if "qwen" in name or namespace == "qwen":
            return "qwen"
        elif "yolo" in name or "cv" in namespace:
            return "yolo"
        elif "vllm" in name:
            return "vllm"
        elif "mig" in name:
            return "mig"
        elif "spread" in name:
            return "spread"

        if pod["spec"]["containers"]:
            for container in pod["spec"]["containers"]:
                image = container["image"].lower()
                if "qwen" in image:
                    return "qwen"
                elif "yolo" in image:
                    return "yolo"
                elif "vllm" in image:
                    return "vllm"

        return "other"

    def _get_gpu_count_from_pod(self, pod: Dict[str, Any]) -> int:
        """Get GPU count from pod resources limits."""
        resources = (
            pod.get("spec", {})
            .get("containers", [{}])[0]
            .get("resources", {})
            .get("limits", {})
        )
        gpu_count = resources.get("nvidia.com/gpu", "0")
        return int(gpu_count)

    def _get_gpu_memory_from_pod(self, pod: Dict[str, Any]) -> float:
        """Get GPU memory from pod resources limits."""
        resources = (
            pod.get("spec", {})
            .get("containers", [{}])[0]
            .get("resources", {})
            .get("limits", {})
        )
        gpumem = resources.get("nvidia.com/gpumem", "0")

        if gpumem.endswith("k"):  # since gpumem is always in MB 'k' is the same as 'G'
            result = float(gpumem[:-1])
        elif gpumem.endswith("G"):
            result = float(gpumem[:-1])
        elif gpumem.endswith("m"):
            result = float(gpumem[:-1]) / 1024
        elif gpumem.endswith("m"):
            result = float(gpumem[:-1]) / 1024
        else:
            # Most GPU pods in HAMi specify memory in bytes, but the values are
            # actually very large (e.g., 25600 bytes = 25GB), so this seems wrong.
            # Let's assume these values are actually in MB for realistic GPU allocations.
            result = float(gpumem) / 1024
            return result

        # print(f"DEBUG: Converted {gpumem} to {result}GB")
        return result

    def _filter_gpu_pods(self) -> List[Dict[str, Any]]:
        """Filter pods that use GPUs."""
        gpu_pods = []

        for pod in self.pods:
            # Filter by pod status if specified
            pod_status = pod.get("status", "")
            if hasattr(self, "pod_status_filter"):
                allowed_statuses = [
                    s.strip() for s in self.pod_status_filter.split(",")
                ]
                if pod_status not in allowed_statuses:
                    continue

            if pod["gpu_count"] > 0 or pod["pod_type"] in [
                "qwen",
                "yolo",
                "vllm",
                "mig",
                "spread",
            ]:
                gpu_pods.append(pod)

        return gpu_pods

    def create_interactive_dashboard(self, output_dir: Path) -> str:
        """Create interactive dashboard using Plotly."""
        logger.debug("Creating interactive dashboard...")

        node_data = []
        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
            total_vram_used = sum(p.get("gpu_memory_gb", 0) for p in node_pods)
            total_vram_capacity = TOTAL_GPUS_PER_NODE * GPU_MEMORY_GB

            node_data.append(
                {
                    "node": node["name"],
                    "vram_used_gb": total_vram_used,
                    "vram_capacity_gb": total_vram_capacity,
                    "vram_utilization": (total_vram_used / total_vram_capacity * 100)
                    if total_vram_capacity > 0
                    else 0,
                    "pod_count": len(node_pods),
                }
            )

        fig = sps.make_subplots(
            rows=1,
            cols=1,
            subplot_titles=("VRAM Utilization by Workload Type",),
            specs=[[{"secondary_y": False}]],
        )

        nodes = [data["node"] for data in node_data]
        node_pod_data = self._prepare_pod_data_for_chart()

        unused_capacity = []
        for node in nodes:
            node_pods = node_pod_data.get(node, [])
            total_used = sum(pod["memory_gb"] for pod in node_pods)
            unused_capacity.append(GPU_MEMORY_GB - total_used)

        fig.add_trace(
            go.Bar(
                x=nodes,
                y=unused_capacity,
                name="Unused Capacity",
                marker_color="lightgreen",
                text=[f"{uc:.1f}GB" for uc in unused_capacity],
                textposition="inside",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        all_pod_names = set()
        for node_pods in node_pod_data.values():
            for pod in node_pods:
                all_pod_names.add(pod["name"])

        sorted_pod_names = sorted(all_pod_names)

        for pod_name in sorted_pod_names:
            pod_data = []
            pod_type = None
            for node in nodes:
                node_pods = node_pod_data.get(node, [])
                found_pod = next((p for p in node_pods if p["name"] == pod_name), None)
                if found_pod:
                    pod_data.append(found_pod["memory_gb"])
                    pod_type = found_pod["pod_type"]
                else:
                    pod_data.append(0)

            if pod_type and any(pod_data):
                fig.add_trace(
                    go.Bar(
                        x=nodes,
                        y=pod_data,
                        name=f"{pod_name} ({pod_type.title()})",
                        marker_color=POD_COLORS.get(pod_type, "#8c564b"),
                        text=[f"{mem:.1f}GB" if mem > 0 else "" for mem in pod_data],
                        textposition="inside",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

        fig.update_layout(
            barmode="stack", yaxis_title="VRAM (GB)", xaxis_title="Node", height=600
        )

        fig.update_xaxes(title_text="Node", tickangle=45)

        # get full current file path and replace with QR code image path
        qr_code_path = Path(__file__).parent / "../github-qr.png"

        if qr_code_path.exists():
            # Add QR code image below legend on the right
            # Position: x=1.02 (right outside plot), y=0.3 (below legend)
            # Size: 0.15 of paper width/height (adjust as needed)
            fig.add_layout_image(
                dict(
                    source=f"file://{qr_code_path.resolve()}",
                    xref="paper",
                    yref="paper",
                    x=1.02,  # Right outside plot area (same as default legend x)
                    y=0.3,  # Position below legend (legend is around y=1.0)
                    sizex=0.15,  # Size relative to paper width
                    sizey=0.15,  # Size relative to paper height
                    xanchor="left",  # Anchor to left side of image
                    yanchor="middle",  # Anchor to middle of image
                    opacity=1.0,
                    layer="above",
                )
            )
        else:
            logger.warning(f"QR code not found at {qr_code_path}")

        output_path = output_dir / "interactive_dashboard.html"
        fig.write_html(str(output_path))

        logger.debug(f"Interactive dashboard saved to {output_path}")
        return str(output_path)

    def generate_all_visualizations(
        self, output_dir: Path, interactive_only: bool = False
    ) -> Dict[str, str]:
        """Generate interactive dashboard only."""
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        results["interactive_dashboard"] = self.create_interactive_dashboard(output_dir)

        self._generate_summary_report(output_dir)

        return results

    def _prepare_pod_data_for_chart(self) -> Dict[str, List[Dict[str, Any]]]:
        """Prepare pod data for individual stacked bar chart."""
        node_pod_data = {}

        for node in self.nodes:
            node_name = node["name"]
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node_name]

            pod_data = []
            for pod in node_pods:
                pod_data.append(
                    {
                        "name": pod["name"],
                        "pod_type": pod["pod_type"],
                        "memory_gb": pod["gpu_memory_gb"],
                        "color": POD_COLORS.get(pod["pod_type"], "#8c564b"),
                    }
                )

            node_pod_data[node_name] = pod_data

        return node_pod_data

    def _generate_summary_report(self, output_dir: Path) -> None:
        """Generate a text summary of the cluster state."""
        logger.debug("Generating summary report...")

        total_nodes = len(self.nodes)
        total_pods = len(self.gpu_pods)
        total_vram_used = sum(p.get("gpu_memory_gb", 0) for p in self.gpu_pods)
        total_vram_capacity = total_nodes * TOTAL_GPUS_PER_NODE * GPU_MEMORY_GB

        pod_type_counts = {}
        for pod in self.gpu_pods:
            pod_type = pod["pod_type"]
            pod_type_counts[pod_type] = pod_type_counts.get(pod_type, 0) + 1

        node_pod_counts = {}
        for pod in self.gpu_pods:
            node_name = pod["node_name"]
            node_pod_counts[node_name] = node_pod_counts.get(node_name, 0) + 1

        report_content = f"""
# HAMi Cluster GPU Visualization Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Cluster Overview
- **Total Nodes**: {total_nodes}
- **Total GPU Pods**: {total_pods}
- **Total VRAM Used**: {total_vram_used:.1f}GB
- **Total VRAM Capacity**: {total_vram_capacity}GB
- **Overall VRAM Utilization**: {(total_vram_used / total_vram_capacity * 100) if total_vram_capacity > 0 else 0:.1f}%

## Node Details
"""

        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
            node_vram_used = sum(p.get("gpu_memory_gb", 0) for p in node_pods)
            node_vram_capacity = TOTAL_GPUS_PER_NODE * GPU_MEMORY_GB

            report_content += f"""
### Node: {node["name"]}
- **Status**: {node["status"]}
- **GPUs**: {TOTAL_GPUS_PER_NODE}
- **GPU Memory**: {node_vram_capacity}GB total
- **VRAM Usage**: {node_vram_used:.1f}GB ({(node_vram_used / node_vram_capacity * 100) if node_vram_capacity > 0 else 0:.1f}%)
- **GPU Pods**: {len(node_pods)}
"""

        report_content += """

## Pod Type Distribution
"""

        for pod_type, count in pod_type_counts.items():
            report_content += f"- **{pod_type.title()}**: {count} pods\n"

        report_content += """

## Pod Distribution by Node
"""

        for node_name, count in node_pod_counts.items():
            report_content += f"- **{node_name}**: {count} pods\n"

        output_path = output_dir / "visualization_report.txt"
        with open(output_path, "w") as f:
            f.write(report_content)

        logger.debug(f"Summary report saved to {output_path}")


def main():
    """Main function to run the GPU visualization."""
    parser = argparse.ArgumentParser(description="Generate interactive GPU dashboard")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for dashboard",
    )
    parser.add_argument(
        "--pod-status",
        type=str,
        default="Scheduled,Running",
        help="Filter pods by status. Default: 'Scheduled,Running'. Use 'Running' for only running pods",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Starting GPU cluster visualization...")

    try:
        visualizer = SeabornGPUVisualizer(args.pod_status)
        results = visualizer.generate_all_visualizations(output_dir)

        logger.info("Generated dashboard:")
        for viz_type, path in results.items():
            if path:
                logger.info(f"{viz_type}: {path}")
            else:
                logger.info(f"{viz_type}: No data available")

        if not any(results.values()):
            logger.info("No dashboard generated - check cluster status and permissions")

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        raise


if __name__ == "__main__":
    main()
