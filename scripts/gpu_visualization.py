#!/usr/bin/env python3
"""
GPU VRAM Visualization System for HAMi vLLM Deployment

Generates 4 seaborn cluster maps showing pod placement across 3 nodes with 80GB A100 GPUs each.
Light theme styling, static PNG and interactive HTML output.

Usage:
    python3 gpu_visualization.py
    python3 gpu_visualization.py --interactive-only
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from kubernetes import client, config
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
CLUSTER_NAME = "HAMi Cluster"
TOTAL_GPUS_PER_NODE = 1  # 1x A100 per node
GPU_MEMORY_GB = 80  # A100 80GB
MAX_PODS_PER_GPU = 1  # Conservative for vLLM workloads

# Color scheme for pod types
POD_COLORS = {
    "qwen": "#1f77b4",  # Blue
    "yolo": "#ff7f0e",  # Orange
    "vllm": "#2ca02c",  # Green
    "mig": "#d62728",  # Red
    "spread": "#9467bd",  # Purple
    "other": "#8c564b",  # Brown
}


class GPUClusterVisualizer:
    """Visualizes GPU cluster state with pod placement and VRAM usage."""

    def __init__(self, kubeconfig: Optional[str] = None):
        """Initialize with Kubernetes configuration."""
        try:
            # Try to load in-cluster config first, then fall back to kubeconfig
            config.load_incluster_config()
            self.api_client = client.CoreV1Api()
        except config.ConfigException:
            if kubeconfig:
                config.load_kube_config(config_file=kubeconfig)
            else:
                config.load_kube_config()
            self.api_client = client.CoreV1Api()

        self.nodes = self._get_nodes()
        self.pods = self._get_pods()
        self.gpu_pods = self._filter_gpu_pods()

    def _get_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the cluster."""
        try:
            nodes = []
            api_response = self.api_client.list_node()

            for node in api_response.items:
                node_info = {
                    "name": node.metadata.name,
                    "status": "Ready"
                    if any(
                        condition.type == "Ready" and condition.status == "True"
                        for condition in node.status.conditions
                    )
                    else "Not Ready",
                    "gpu_count": self._get_gpu_count(node),
                    "memory_total_gb": self._get_memory_total_gb(node),
                    "memory_available_gb": self._get_memory_available_gb(node),
                    "labels": node.metadata.labels or {},
                }
                nodes.append(node_info)

            return nodes

        except Exception as e:
            logger.error(f"Error getting nodes: {e}")
            return []

    def _get_pods(self) -> List[Dict[str, Any]]:
        """Get all pods in the cluster."""
        try:
            pods = []
            api_response = self.api_client.list_pod_for_all_namespaces()

            for pod in api_response.items:
                pod_info = {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "node_name": pod.spec.node_name
                    if pod.spec.node_name
                    else "Unknown",
                    "status": pod.status.phase,
                    "gpu_memory_gb": 0,
                    "gpu_count": 0,
                    "pod_type": self._classify_pod_type(pod),
                    "container_resources": self._get_container_resources(pod),
                }

                # Get GPU information from annotations or labels
                if pod.metadata.annotations:
                    if "nvidia.com/gpu.memory" in pod.metadata.annotations:
                        pod_info["gpu_memory_gb"] = float(
                            pod.metadata.annotations["nvidia.com/gpu.memory"]
                        )

                    if "nvidia.com/gpu.count" in pod.metadata.annotations:
                        pod_info["gpu_count"] = int(
                            pod.metadata.annotations["nvidia.com/gpu.count"]
                        )

                pods.append(pod_info)

            return pods

        except Exception as e:
            logger.error(f"Error getting pods: {e}")
            return []

    def _get_gpu_count(self, node: Any) -> int:
        """Get GPU count for a node from labels."""
        labels = node.metadata.labels or {}
        return int(labels.get("nvidia.com/gpu.count", "0"))

    def _get_memory_total_gb(self, node: Any) -> int:
        """Get total memory for a node in GB."""
        if node.status.capacity:
            memory_bytes = int(
                node.status.capacity.get("memory", "0").replace("Ki", "")
            )
            return int(memory_bytes / (1024 * 1024 * 1024))
        return 0

    def _get_memory_available_gb(self, node: Any) -> int:
        """Get available memory for a node in GB."""
        if node.status.allocatable:
            memory_bytes = int(
                node.status.allocatable.get("memory", "0").replace("Ki", "")
            )
            return int(memory_bytes / (1024 * 1024 * 1024))
        return 0

    def _classify_pod_type(self, pod: Any) -> str:
        """Classify pod type based on name, labels, and containers."""
        name = pod.metadata.name.lower()
        namespace = pod.metadata.namespace.lower()

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
        if pod.spec.containers:
            for container in pod.spec.containers:
                image = container.image.lower()
                if "qwen" in image:
                    return "qwen"
                elif "yolo" in image:
                    return "yolo"
                elif "vllm" in image:
                    return "vllm"

        return "other"

    def _get_container_resources(self, pod: Any) -> Dict[str, Any]:
        """Extract container resource requirements."""
        resources = {}

        if pod.spec.containers:
            for container in pod.spec.containers:
                if container.resources and container.resources.requests:
                    resources[container.name] = {
                        "memory": container.resources.requests.get("memory"),
                        "cpu": container.resources.requests.get("cpu"),
                    }

        return resources

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

    def create_cluster_heatmap(self, output_dir: Path) -> str:
        """Create cluster-wide heatmap showing VRAM usage per node."""
        logger.info("Creating cluster heatmap...")

        # Prepare data
        node_data = []
        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
            total_vram_used = sum(p["gpu_memory_gb"] for p in node_pods)
            total_vram_available = node["gpu_count"] * GPU_MEMORY_GB

            node_data.append(
                {
                    "node": node["name"],
                    "vram_used_gb": total_vram_used,
                    "vram_available_gb": total_vram_available,
                    "vram_usage_percent": (total_vram_used / total_vram_available * 100)
                    if total_vram_available > 0
                    else 0,
                    "pod_count": len(node_pods),
                    "gpu_count": node["gpu_count"],
                }
            )

        df = pd.DataFrame(node_data)

        if df.empty:
            logger.warning("No node data found for heatmap")
            return ""

        # Create the plot
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")

        # VRAM usage heatmap
        plt.subplot(2, 2, 1)
        vram_data = df.set_index("node")["vram_usage_percent"].to_frame()
        sns.heatmap(
            vram_data.T,
            annot=True,
            cmap="RdYlBu_r",
            fmt=".1f",
            cbar_kws={"label": "VRAM Usage (%)"},
        )
        plt.title(f"{CLUSTER_NAME} - VRAM Usage per Node")
        plt.xlabel("")

        # Pod count bar chart
        plt.subplot(2, 2, 2)
        sns.barplot(data=df, x="node", y="pod_count", palette="viridis")
        plt.title("GPU Pods per Node")
        plt.xlabel("Node")
        plt.ylabel("Pod Count")
        plt.xticks(rotation=45)

        # VRAM comparison
        plt.subplot(2, 2, 3)
        df_sorted = df.sort_values("vram_available_gb")
        x = np.arange(len(df_sorted))
        width = 0.35

        plt.bar(
            x - width / 2, df_sorted["vram_used_gb"], width, label="Used", color="coral"
        )
        plt.bar(
            x + width / 2,
            df_sorted["vram_available_gb"],
            width,
            label="Available",
            color="lightblue",
        )
        plt.xlabel("Node")
        plt.ylabel("VRAM (GB)")
        plt.title("VRAM Used vs Available")
        plt.xticks(x, df_sorted["node"], rotation=45)
        plt.legend()

        # Node status overview
        plt.subplot(2, 2, 4)
        status_counts = df[["node", "gpu_count"]].set_index("node")["gpu_count"]
        status_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(df))),
        )
        plt.title("GPU Distribution Across Nodes")
        plt.ylabel("")

        plt.tight_layout()
        plt.suptitle(f"{CLUSTER_NAME} - GPU Cluster Overview", fontsize=16, y=0.98)

        # Save the plot
        output_path = output_dir / "cluster_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Cluster heatmap saved to {output_path}")
        return str(output_path)

    def create_pod_placement_map(self, output_dir: Path) -> str:
        """Create detailed pod placement map showing individual GPUs."""
        logger.info("Creating pod placement map...")

        # Create figure with subplots for each node
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{CLUSTER_NAME} - Pod Placement on GPUs", fontsize=16)

        for idx, node in enumerate(self.nodes):
            if node["gpu_count"] == 0:
                continue

            ax = axes[idx]
            ax.set_title(f"Node {node['name']}\n{node['gpu_count']} GPU(s)")

            # Create GPU representation
            for gpu_idx in range(node["gpu_count"]):
                # Draw GPU rectangle
                gpu_rect = patches.Rectangle(
                    (0.1, 0.1),
                    0.8,
                    0.8,
                    linewidth=2,
                    edgecolor="black",
                    facecolor="lightgray",
                    alpha=0.7,
                )
                ax.add_patch(gpu_rect)

                # Add GPU label
                ax.text(
                    0.5,
                    0.95,
                    f"GPU {gpu_idx + 1}",
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )

                # Get pods on this GPU
                node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
                gpu_pods = [
                    p
                    for p in node_pods
                    if p["gpu_count"] > 0 and p["gpu_memory_gb"] > 0
                ]

                # Calculate VRAM usage
                vram_used = sum(p["gpu_memory_gb"] for p in gpu_pods)
                vram_percent = vram_used / GPU_MEMORY_GB * 100

                # Update GPU color based on usage
                if vram_percent > 0:
                    color_intensity = min(vram_percent / 100, 1.0)
                    gpu_rect.set_facecolor(
                        (1.0, 1.0 - color_intensity, 1.0 - color_intensity)
                    )

                # Add VRAM info
                ax.text(
                    0.5,
                    0.5,
                    f"{vram_used:.1f}GB\n{vram_percent:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

                # List pods below GPU
                pod_names = [p["name"][:15] for p in gpu_pods[:3]]  # Show first 3 pods
                if len(gpu_pods) > 3:
                    pod_names.append(f"+{len(gpu_pods) - 3} more")

                pod_text = "\n".join(pod_names)
                ax.text(
                    0.5,
                    0.05,
                    pod_text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.axis("off")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save the plot
        output_path = output_dir / "pod_placement_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Pod placement map saved to {output_path}")
        return str(output_path)

    def create_pod_type_distribution(self, output_dir: Path) -> str:
        """Create visualization showing pod type distribution."""
        logger.info("Creating pod type distribution chart...")

        # Count pods by type
        pod_type_counts = {}
        for pod in self.gpu_pods:
            pod_type = pod["pod_type"]
            pod_type_counts[pod_type] = pod_type_counts.get(pod_type, 0) + 1

        if not pod_type_counts:
            logger.warning("No GPU pods found for distribution")
            return ""

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Pie chart
        plt.subplot(2, 2, 1)
        colors = [
            POD_COLORS.get(pod_type, "#gray") for pod_type in pod_type_counts.keys()
        ]
        plt.pie(
            pod_type_counts.values(),
            labels=pod_type_counts.keys(),
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
        )
        plt.title("Pod Type Distribution")

        # Bar chart
        plt.subplot(2, 2, 2)
        df_types = pd.DataFrame(
            list(pod_type_counts.items()), columns=["Type", "Count"]
        )
        sns.barplot(data=df_types, x="Count", y="Type", palette=colors)
        plt.title("Pod Count by Type")
        plt.xlabel("Count")

        # VRAM usage by type
        plt.subplot(2, 2, 3)
        vram_by_type = {}
        for pod in self.gpu_pods:
            pod_type = pod["pod_type"]
            vram_by_type[pod_type] = (
                vram_by_type.get(pod_type, 0) + pod["gpu_memory_gb"]
            )

        df_vram = pd.DataFrame(list(vram_by_type.items()), columns=["Type", "VRAM_GB"])
        sns.barplot(data=df_vram, x="VRAM_GB", y="Type", palette=colors)
        plt.title("VRAM Usage by Pod Type")
        plt.xlabel("VRAM (GB)")

        # Node breakdown
        plt.subplot(2, 2, 4)
        node_breakdown = {}
        for pod in self.gpu_pods:
            node_name = pod["node_name"]
            pod_type = pod["pod_type"]
            if node_name not in node_breakdown:
                node_breakdown[node_name] = {}
            node_breakdown[node_name][pod_type] = (
                node_breakdown[node_name].get(pod_type, 0) + 1
            )

        df_nodes = pd.DataFrame.from_dict(node_breakdown, orient="index").fillna(0)
        df_nodes.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Pod Breakdown by Node")
        plt.xlabel("Node")
        plt.ylabel("Pod Count")
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Save the plot
        output_path = output_dir / "pod_type_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Pod type distribution saved to {output_path}")
        return str(output_path)

    def create_vram_utilization_chart(self, output_dir: Path) -> str:
        """Create VRAM utilization chart over time (mock data for now)."""
        logger.info("Creating VRAM utilization chart...")

        # Calculate current VRAM usage
        vram_data = []
        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
            total_vram_used = sum(p["gpu_memory_gb"] for p in node_pods)
            total_vram_available = node["gpu_count"] * GPU_MEMORY_GB

            vram_data.append(
                {
                    "node": node["name"],
                    "used": total_vram_used,
                    "available": total_vram_available,
                    "usage_percent": (total_vram_used / total_vram_available * 100)
                    if total_vram_available > 0
                    else 0,
                }
            )

        # Create utilization chart
        plt.figure(figsize=(15, 10))

        # Overall utilization
        plt.subplot(2, 2, 1)
        nodes = [data["node"] for data in vram_data]
        usage = [data["usage_percent"] for data in vram_data]
        bars = plt.bar(nodes, usage, color="skyblue", alpha=0.7)

        # Add utilization percentage on bars
        for bar, percent in zip(bars, usage):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{percent:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.title("VRAM Utilization by Node")
        plt.xlabel("Node")
        plt.ylabel("Utilization (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=45)

        # VRAM capacity comparison
        plt.subplot(2, 2, 2)
        used_vram = [data["used"] for data in vram_data]
        available_vram = [data["available"] for data in vram_data]
        x = np.arange(len(nodes))
        width = 0.35

        plt.bar(x - width / 2, used_vram, width, label="Used", color="coral")
        plt.bar(
            x + width / 2, available_vram, width, label="Available", color="lightblue"
        )
        plt.xlabel("Node")
        plt.ylabel("VRAM (GB)")
        plt.title("VRAM Capacity vs Usage")
        plt.xticks(x, nodes, rotation=45)
        plt.legend()

        # Pod density per GPU
        plt.subplot(2, 2, 3)
        pod_counts = []
        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
            pod_count = len(node_pods) / max(node["gpu_count"], 1)  # Pods per GPU
            pod_counts.append(pod_count)

        plt.bar(nodes, pod_counts, color="lightgreen", alpha=0.7)
        plt.title("Pods per GPU (Density)")
        plt.xlabel("Node")
        plt.ylabel("Pods per GPU")
        plt.xticks(rotation=45)

        # Memory pressure indicators
        plt.subplot(2, 2, 4)
        memory_pressure = []
        for data in vram_data:
            if data["usage_percent"] > 80:
                pressure = "High"
            elif data["usage_percent"] > 60:
                pressure = "Medium"
            else:
                pressure = "Low"
            memory_pressure.append(pressure)

        pressure_counts = pd.Series(memory_pressure).value_counts()
        pressure_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            colors=["red", "orange", "green"],
        )
        plt.title("Memory Pressure Distribution")
        plt.ylabel("")

        plt.tight_layout()

        # Save the plot
        output_path = output_dir / "vram_utilization.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"VRAM utilization chart saved to {output_path}")
        return str(output_path)

    def create_interactive_dashboard(self, output_dir: Path) -> str:
        """Create interactive dashboard using Plotly."""
        logger.info("Creating interactive dashboard...")

        # Prepare data for interactive plots
        node_data = []
        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
            total_vram_used = sum(p["gpu_memory_gb"] for p in node_pods)
            total_vram_available = node["gpu_count"] * GPU_MEMORY_GB

            node_data.append(
                {
                    "node": node["name"],
                    "gpu_count": node["gpu_count"],
                    "vram_used_gb": total_vram_used,
                    "vram_available_gb": total_vram_available,
                    "vram_usage_percent": (total_vram_used / total_vram_available * 100)
                    if total_vram_available > 0
                    else 0,
                    "pod_count": len(node_pods),
                    "memory_total_gb": node["memory_total_gb"],
                    "memory_available_gb": node["memory_available_gb"],
                }
            )

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "VRAM Usage by Node",
                "Pods per Node",
                "Memory Overview",
                "Pod Types by Node",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # VRAM Usage
        nodes = [data["node"] for data in node_data]
        vram_usage = [data["vram_usage_percent"] for data in node_data]

        fig.add_trace(
            go.Bar(
                x=nodes,
                y=vram_usage,
                name="VRAM Usage %",
                marker_color="coral",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Pods per Node
        pod_counts = [data["pod_count"] for data in node_data]

        fig.add_trace(
            go.Bar(
                x=nodes,
                y=pod_counts,
                name="Pod Count",
                marker_color="lightblue",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Memory Overview
        memory_used = [
            data["memory_total_gb"] - data["memory_available_gb"] for data in node_data
        ]
        memory_total = [data["memory_total_gb"] for data in node_data]

        fig.add_trace(
            go.Bar(
                x=nodes,
                y=memory_used,
                name="Memory Used",
                marker_color="orange",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=nodes,
                y=memory_total,
                name="Memory Total",
                marker_color="lightgreen",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Pod Types by Node
        for node in nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node]
            for pod_type in ["qwen", "yolo", "vllm", "mig", "spread", "other"]:
                count = len([p for p in node_pods if p["pod_type"] == pod_type])
                if count > 0:
                    fig.add_trace(
                        go.Bar(
                            x=[node],
                            y=[count],
                            name=f"{pod_type}",
                            marker_color=POD_COLORS.get(pod_type, "gray"),
                            showlegend=False if pod_type != "qwen" else True,
                        ),
                        row=2,
                        col=2,
                    )

        # Update layout
        fig.update_layout(
            title_text=f"{CLUSTER_NAME} - Interactive GPU Dashboard",
            showlegend=True,
            height=800,
            hovermode="x unified",
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
        """Generate all visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        if not interactive_only:
            results["cluster_heatmap"] = self.create_cluster_heatmap(output_dir)
            results["pod_placement_map"] = self.create_pod_placement_map(output_dir)
            results["pod_type_distribution"] = self.create_pod_type_distribution(
                output_dir
            )
            results["vram_utilization"] = self.create_vram_utilization_chart(output_dir)

        results["interactive_dashboard"] = self.create_interactive_dashboard(output_dir)

        # Generate summary report
        self._generate_summary_report(output_dir)

        return results

    def _generate_summary_report(self, output_dir: Path) -> None:
        """Generate a text summary of the cluster state."""
        logger.info("Generating summary report...")

        # Calculate summary statistics
        total_nodes = len(self.nodes)
        total_pods = len(self.gpu_pods)
        total_vram_used = sum(p["gpu_memory_gb"] for p in self.gpu_pods)
        total_vram_capacity = sum(
            node["gpu_count"] * GPU_MEMORY_GB for node in self.nodes
        )

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
            node_vram_used = sum(p["gpu_memory_gb"] for p in node_pods)
            node_vram_capacity = node["gpu_count"] * GPU_MEMORY_GB

            report_content += f"""
### Node: {node["name"]}
- **Status**: {node["status"]}
- **GPUs**: {node["gpu_count"]}
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

        # Save report
        output_path = output_dir / "visualization_report.txt"
        with open(output_path, "w") as f:
            f.write(report_content)

        logger.info(f"Summary report saved to {output_path}")


def main():
    """Main function to run the GPU visualization."""
    parser = argparse.ArgumentParser(description="Generate GPU cluster visualizations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--kubeconfig", type=str, default=None, help="Path to kubeconfig file"
    )
    parser.add_argument(
        "--interactive-only",
        action="store_true",
        help="Generate only interactive dashboard",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting GPU cluster visualization...")

    try:
        visualizer = GPUClusterVisualizer(kubeconfig=args.kubeconfig)
        results = visualizer.generate_all_visualizations(
            output_dir, args.interactive_only
        )

        print("\nGenerated visualizations:")
        for viz_type, path in results.items():
            print(f"  {viz_type}: {path}")

        if not any(results.values()):
            print(
                "Warning: No visualizations generated - check cluster status and permissions"
            )

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise


if __name__ == "__main__":
    main()
