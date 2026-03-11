#!/usr/bin/env python3
"""
Simplified GPU VRAM Visualization System for HAMi vLLM Deployment

Uses nvidia-smi commands to get actual VRAM information from GPU pods.
Generates interactive HTML dashboard only.

Usage:
    python3 gpu_visualization.py
    python3 gpu_visualization.py --output-dir ./output
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sps
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
CLUSTER_NAME = "HAMi Cluster"
TOTAL_GPUS_PER_NODE = 1  # 1x A100 per node
GPU_MEMORY_GB = 80  # A100 80GB

# Color scheme for pod types
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

    def __init__(self):
        """Initialize visualizer."""
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
                # Debug: Print pod info for troubleshooting
                print(
                    f"DEBUG: Pod {pod['metadata']['name']} on node {pod['spec']['nodeName']}"
                )

                pod_info = {
                    "name": pod["metadata"]["name"],
                    "namespace": pod["metadata"]["namespace"],
                    "node_name": pod["spec"][
                        "nodeName"
                    ],  # Use nodeName instead of node_name
                    "status": pod["status"]["phase"],
                    "pod_type": self._classify_pod_type(pod),
                    "gpu_memory_gb": self._get_gpu_memory_from_pod(pod),
                    "gpu_count": self._get_gpu_count_from_pod(pod),
                }

                # Debug: Print GPU info
                print(
                    f"DEBUG: GPU count: {pod_info['gpu_count']}, GPU memory: {pod_info['gpu_memory_gb']}GB"
                )

                pods.append(pod_info)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")

        return pods

    def _classify_pod_type(self, pod: Dict[str, Any]) -> str:
        """Classify pod type based on name, labels, and containers."""
        name = pod["metadata"]["name"].lower()
        namespace = pod["metadata"]["namespace"].lower()

        # Check for specific deployments
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

        # Check container images
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
        return float(gpumem) / 1024  # Convert MB to GB

    def _filter_gpu_pods(self) -> List[Dict[str, Any]]:
        """Filter pods that use GPUs."""
        gpu_pods = []

        for pod in self.pods:
            # Include if pod has GPU annotations or is classified as GPU type
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
        logger.info("Creating interactive dashboard...")

        # Prepare data
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

        # Create single subplot for stacked bar chart
        fig = sps.make_subplots(
            rows=1,
            cols=1,
            subplot_titles=("VRAM Utilization by Workload Type",),
            specs=[[{"secondary_y": False}]],
        )

        # Get nodes and aggregate VRAM data
        nodes = [data["node"] for data in node_data]
        node_workload_vram = self._aggregate_vram_by_workload()

        # Add unused capacity (green base)
        unused_capacity = []
        for node in nodes:
            total_used = sum(node_workload_vram[node].values())
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

        # Add workload segments
        for workload_type in POD_COLORS.keys():
            workload_data = []
            for node in nodes:
                workload_data.append(node_workload_vram[node].get(workload_type, 0.0))

            if any(workload_data):  # Only add if there's data
                fig.add_trace(
                    go.Bar(
                        x=nodes,
                        y=workload_data,
                        name=workload_type.title(),
                        marker_color=POD_COLORS[workload_type],
                        text=[f"{wd:.1f}GB" if wd > 0 else "" for wd in workload_data],
                        textposition="inside",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

        # Update layout to set barmode
        fig.update_layout(
            barmode="stack", yaxis_title="VRAM (GB)", xaxis_title="Node", height=600
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Node", tickangle=45)

        # Save as HTML
        output_path = output_dir / "interactive_dashboard.html"
        fig.write_html(str(output_path))

        logger.info(f"Interactive dashboard saved to {output_path}")
        return str(output_path)

    def generate_all_visualizations(
        self, output_dir: Path, interactive_only: bool = False
    ) -> Dict[str, str]:
        """Generate interactive dashboard only."""
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Only generate interactive dashboard
        results["interactive_dashboard"] = self.create_interactive_dashboard(output_dir)

        # Generate summary report
        self._generate_summary_report(output_dir)

        return results

    def _aggregate_vram_by_workload(self) -> Dict[str, Dict[str, float]]:
        """Aggregate VRAM by node and workload type."""
        node_workload_vram = {}

        for node in self.nodes:
            node_name = node["name"]
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node_name]

            # Initialize with all workload types
            workload_vram = {pod_type: 0.0 for pod_type in POD_COLORS.keys()}

            # Sum VRAM per workload type
            for pod in node_pods:
                workload_vram[pod["pod_type"]] += pod["gpu_memory_gb"]

            node_workload_vram[node_name] = workload_vram

        return node_workload_vram

    def create_stacked_bar_chart(self, node_workload_vram: Dict) -> go.Figure:
        """Create stacked bar chart showing VRAM utilization per node."""
        fig = go.Figure()

        nodes = list(node_workload_vram.keys())

        # Add unused capacity (green base)
        unused_capacity = []
        for node in nodes:
            total_used = sum(node_workload_vram[node].values())
            unused_capacity.append(GPU_MEMORY_GB - total_used)

        fig.add_trace(
            go.Bar(
                x=nodes,
                y=unused_capacity,
                name="Unused Capacity",
                marker_color="lightgreen",
                text=[f"{uc:.1f}GB" for uc in unused_capacity],
                textposition="inside",
            )
        )

        # Add workload segments
        for workload_type in POD_COLORS.keys():
            workload_data = []
            for node in nodes:
                workload_data.append(node_workload_vram[node].get(workload_type, 0.0))

            if any(workload_data):  # Only add if there's data
                fig.add_trace(
                    go.Bar(
                        x=nodes,
                        y=workload_data,
                        name=workload_type.title(),
                        marker_color=POD_COLORS[workload_type],
                        text=[f"{wd:.1f}GB" if wd > 0 else "" for wd in workload_data],
                        textposition="inside",
                    )
                )

        fig.update_layout(
            title="VRAM Utilization by Workload Type",
            barmode="stack",
            yaxis_title="VRAM (GB)",
            xaxis_title="Node",
            height=500,
        )

        return fig

    def _generate_summary_report(self, output_dir: Path) -> None:
        """Generate a text summary of the cluster state."""
        logger.info("Generating summary report...")

        # Calculate summary statistics
        total_nodes = len(self.nodes)
        total_pods = len(self.gpu_pods)
        total_vram_used = sum(p.get("gpu_memory_gb", 0) for p in self.gpu_pods)
        total_vram_capacity = total_nodes * TOTAL_GPUS_PER_NODE * GPU_MEMORY_GB

        # Pod type breakdown
        pod_type_counts = {}
        for pod in self.gpu_pods:
            pod_type = pod["pod_type"]
            pod_type_counts[pod_type] = pod_type_counts.get(pod_type, 0) + 1

        # Node breakdown
        node_pod_counts = {}
        for pod in self.gpu_pods:
            node_name = pod["node_name"]
            node_pod_counts[node_name] = node_pod_counts.get(node_name, 0) + 1

        # Generate report
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

        report_content += f"""

## Pod Distribution by Node
"""

        for node_name, count in node_pod_counts.items():
            report_content += f"- **{node_name}**: {count} pods\n"

        # Save report
        output_path = output_dir / "visualization_report.txt"
        with open(output_path, "w") as f:
            f.write(report_content)

        logger.info(f"Summary report saved to {output_path}")


def main():
    """Main function to run the GPU visualization."""
    parser = argparse.ArgumentParser(description="Generate interactive GPU dashboard")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for dashboard",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting GPU cluster visualization...")

    try:
        visualizer = SeabornGPUVisualizer()
        results = visualizer.generate_all_visualizations(output_dir)

        print("\nGenerated dashboard:")
        for viz_type, path in results.items():
            if path:
                print(f"  {viz_type}: {path}")
            else:
                print(f"  {viz_type}: No data available")

        if not any(results.values()):
            print(
                "Warning: No dashboard generated - check cluster status and permissions"
            )

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        raise


if __name__ == "__main__":
    main()
