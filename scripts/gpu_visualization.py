#!/usr/bin/env python3
"""
Simplified GPU VRAM Visualization System for HAMi vLLM Deployment

Uses nvidia-smi commands to get actual VRAM information from GPU pods.
Generates seaborn cluster maps showing pod placement and VRAM usage.

Usage:
    python3 gpu_visualization.py
    python3 gpu_visualization.py --output-dir ./output
    python3 gpu_visualization.py --interactive-only
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
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

    def _get_pods(self) -> List[Dict[str, str]]:
        """Get pods from cluster."""
        pods = []
        output = self._run_command("kubectl get pods -A -o json")

        if not output:
            return []

        try:
            data = json.loads(output)
            for pod in data["items"]:
                pods.append(
                    {
                        "name": pod["metadata"]["name"],
                        "namespace": pod["metadata"]["namespace"],
                        "node_name": pod["spec"].get("node_name", "Unknown"),
                        "status": pod["status"]["phase"],
                        "pod_type": self._classify_pod_type(pod),
                        "gpu_memory_gb": self._get_gpu_memory_from_pod(pod),
                        "gpu_count": self._get_gpu_count_from_pod(pod),
                    }
                )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")

        return pods

    def _classify_pod_type(self, pod: Dict[str, str]) -> str:
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

    def _get_gpu_count_from_pod(self, pod: Dict[str, str]) -> int:
        """Get GPU count from pod annotations."""
        annotations = pod["metadata"].get("annotations", {})
        return int(annotations.get("nvidia.com/gpu.count", "0"))

    def _get_gpu_memory_from_pod(self, pod: Dict[str, str]) -> float:
        """Get GPU memory using nvidia-smi command."""
        if pod["spec"].get("node_name") and pod["metadata"].get("name"):
            try:
                # Run nvidia-smi in the pod to get detailed memory info
                command = f"kubectl exec -it {pod['metadata']['name']} -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
                output = self._run_command(command)

                if output:
                    # Parse memory usage in MB: format "used,total"
                    parts = output.split(",")
                    if len(parts) == 2:
                        memory_used_mb = float(parts[0].strip())
                        memory_total_mb = float(parts[1].strip())
                        return memory_used_mb / 1024  # Convert to GB

            except Exception as e:
                logger.warning(
                    f"Failed to get GPU memory for pod {pod['metadata']['name']}: {e}"
                )

        return 0.0

    def _filter_gpu_pods(self) -> List[Dict[str, str]]:
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

    def create_cluster_visualization(self, output_dir: Path) -> str:
        """Create cluster-wide visualization using seaborn."""
        logger.info("Creating cluster visualization...")

        # Prepare data
        cluster_data = []
        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
            total_vram_used = sum(p["gpu_memory_gb"] for p in node_pods)
            total_vram_capacity = TOTAL_GPUS_PER_NODE * GPU_MEMORY_GB

            cluster_data.append(
                {
                    "node": node["name"],
                    "vram_used_gb": total_vram_used,
                    "vram_available_gb": total_vram_capacity,
                    "vram_usage_percent": (total_vram_used / total_vram_capacity * 100)
                    if total_vram_capacity > 0
                    else 0,
                    "pod_count": len(node_pods),
                    "pod_type": ", ".join(set([p["pod_type"] for p in node_pods])),
                }
            )

        df = pd.DataFrame(cluster_data)

        if df.empty:
            logger.warning("No cluster data found")
            return ""

        # Create comprehensive visualization
        plt.figure(figsize=(20, 12))
        sns.set_style("whitegrid")

        # 1. VRAM Usage by Node
        plt.subplot(2, 3, 1)
        sns.barplot(data=df, x="node", y="vram_usage_percent", palette="viridis")
        plt.title(f"{CLUSTER_NAME} - VRAM Usage per Node")
        plt.ylabel("VRAM Usage (%)")
        plt.xlabel("Node")
        plt.xticks(rotation=45)
        for i, v in enumerate(df["vram_usage_percent"]):
            plt.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom")

        # 2. Pod Count by Node
        plt.subplot(2, 3, 2)
        sns.barplot(data=df, x="node", y="pod_count", palette="plasma")
        plt.title("GPU Pods per Node")
        plt.xlabel("Node")
        plt.ylabel("Pod Count")
        plt.xticks(rotation=45)

        # 3. VRAM Used vs Available
        plt.subplot(2, 3, 3)
        df_melted = df.melt(
            id_vars=["node"],
            value_vars=["vram_used_gb", "vram_available_gb"],
            var_name="VRAM Type",
            value_name="VRAM (GB)",
        )
        sns.barplot(
            data=df_melted,
            x="node",
            y="VRAM (GB)",
            hue="VRAM Type",
            palette=["coral", "lightblue"],
        )
        plt.title("VRAM Used vs Available")
        plt.xlabel("Node")
        plt.ylabel("VRAM (GB)")
        plt.xticks(rotation=45)

        # 4. VRAM Heatmap
        plt.subplot(2, 3, 4)
        heatmap_data = df.set_index("node")[["vram_used_gb", "vram_available_gb"]].T
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",
            cmap="RdYlBu_r",
            cbar_kws={"label": "VRAM (GB)"},
        )
        plt.title("VRAM Heatmap")
        plt.xlabel("Node")
        plt.ylabel("")

        # 5. Pod Type Distribution
        plt.subplot(2, 3, 5)
        pod_type_counts = {}
        for pod in self.gpu_pods:
            pod_type_counts[pod["pod_type"]] = (
                pod_type_counts.get(pod["pod_type"], 0) + 1
            )

        pod_df = pd.DataFrame(list(pod_type_counts.items()), columns=["Type", "Count"])
        sns.barplot(
            data=pod_df,
            x="Count",
            y="Type",
            palette=[POD_COLORS.get(t, "gray") for t in pod_df["Type"]],
        )
        plt.title("Pod Type Distribution")
        plt.xlabel("Count")
        plt.ylabel("Pod Type")

        # 6. Memory Pressure
        plt.subplot(2, 3, 6)
        pressure_data = []
        for _, row in df.iterrows():
            if row["vram_usage_percent"] > 80:
                pressure = "High"
            elif row["vram_usage_percent"] > 60:
                pressure = "Medium"
            else:
                pressure = "Low"
            pressure_data.append({"node": row["node"], "pressure": pressure})

        pressure_df = pd.DataFrame(pressure_data)
        pressure_counts = pressure_df["pressure"].value_counts()
        plt.pie(
            pressure_counts.values,
            labels=pressure_counts.index,
            autopct="%1.1f%%",
            colors=["red", "orange", "green"],
        )
        plt.title("Memory Pressure Distribution")

        plt.tight_layout()
        plt.suptitle(
            f"{CLUSTER_NAME} - Comprehensive GPU Overview", fontsize=16, y=0.98
        )

        # Save the plot
        output_path = output_dir / "cluster_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Cluster visualization saved to {output_path}")
        return str(output_path)

    def create_pod_placement_heatmap(self, output_dir: Path) -> str:
        """Create pod placement heatmap using seaborn."""
        logger.info("Creating pod placement heatmap...")

        # Create node-pod matrix
        placement_data = []
        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]

            for pod in node_pods:
                placement_data.append(
                    {
                        "node": node["name"],
                        "pod_name": pod["name"],
                        "pod_type": pod["pod_type"],
                        "gpu_memory_gb": pod["gpu_memory_gb"],
                        "gpu_usage_percent": (
                            pod["gpu_memory_gb"] / GPU_MEMORY_GB * 100
                        )
                        if GPU_MEMORY_GB > 0
                        else 0,
                    }
                )

        if not placement_data:
            logger.warning("No pod placement data found")
            return ""

        df = pd.DataFrame(placement_data)

        # Create pivot table for heatmap
        heatmap_df = df.pivot_table(
            index="node",
            columns="pod_type",
            values="gpu_usage_percent",
            aggfunc="sum",
            fill_value=0,
        )

        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "GPU Usage (%)"},
        )
        plt.title(f"{CLUSTER_NAME} - Pod Placement Heatmap")
        plt.xlabel("Pod Type")
        plt.ylabel("Node")
        plt.tight_layout()

        # Save the plot
        output_path = output_dir / "pod_placement_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Pod placement heatmap saved to {output_path}")
        return str(output_path)

    def create_resource_utilization_chart(self, output_dir: Path) -> str:
        """Create resource utilization chart using seaborn."""
        logger.info("Creating resource utilization chart...")

        # Prepare data
        utilization_data = []
        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
            total_vram_used = sum(p["gpu_memory_gb"] for p in node_pods)
            total_vram_capacity = TOTAL_GPUS_PER_NODE * GPU_MEMORY_GB

            utilization_data.append(
                {
                    "node": node["name"],
                    "vram_used_gb": total_vram_used,
                    "vram_capacity_gb": total_vram_capacity,
                    "vram_utilization": (total_vram_used / total_vram_capacity * 100)
                    if total_vram_capacity > 0
                    else 0,
                    "pod_count": len(node_pods),
                    "pod_density": len(node_pods) / TOTAL_GPUS_PER_NODE,
                }
            )

        df = pd.DataFrame(utilization_data)

        # Create the chart
        plt.figure(figsize=(15, 10))

        # 1. VRAM Utilization by Node
        plt.subplot(2, 2, 1)
        sns.barplot(data=df, x="node", y="vram_utilization", palette="coolwarm")
        plt.title("VRAM Utilization by Node")
        plt.xlabel("Node")
        plt.ylabel("Utilization (%)")
        plt.xticks(rotation=45)
        for i, v in enumerate(df["vram_utilization"]):
            plt.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom")

        # 2. VRAM Capacity vs Usage
        plt.subplot(2, 2, 2)
        df_melted = df.melt(
            id_vars=["node"],
            value_vars=["vram_used_gb", "vram_capacity_gb"],
            var_name="VRAM Type",
            value_name="VRAM (GB)",
        )
        sns.barplot(
            data=df_melted,
            x="node",
            y="VRAM (GB)",
            hue="VRAM Type",
            palette=["coral", "lightblue"],
        )
        plt.title("VRAM Capacity vs Usage")
        plt.xlabel("Node")
        plt.ylabel("VRAM (GB)")
        plt.xticks(rotation=45)

        # 3. Pod Density
        plt.subplot(2, 2, 3)
        sns.barplot(data=df, x="node", y="pod_density", palette="viridis")
        plt.title("Pod Density per GPU")
        plt.xlabel("Node")
        plt.ylabel("Pods per GPU")
        plt.xticks(rotation=45)

        # 4. Resource Overview
        plt.subplot(2, 2, 4)
        resource_data = []
        for _, row in df.iterrows():
            resource_data.extend(
                [
                    {
                        "node": row["node"],
                        "resource": "VRAM Used",
                        "value": row["vram_used_gb"],
                    },
                    {
                        "node": row["node"],
                        "resource": "VRAM Capacity",
                        "value": row["vram_capacity_gb"],
                    },
                    {
                        "node": row["node"],
                        "resource": "Pod Count",
                        "value": row["pod_count"],
                    },
                ]
            )

        resource_df = pd.DataFrame(resource_data)
        sns.barplot(
            data=resource_df, x="node", y="value", hue="resource", palette="Set2"
        )
        plt.title("Resource Overview")
        plt.xlabel("Node")
        plt.ylabel("Value")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.suptitle(
            f"{CLUSTER_NAME} - Resource Utilization Analysis", fontsize=16, y=0.98
        )

        # Save the plot
        output_path = output_dir / "resource_utilization.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Resource utilization chart saved to {output_path}")
        return str(output_path)

    def create_interactive_dashboard(self, output_dir: Path) -> str:
        """Create interactive dashboard using Plotly."""
        logger.info("Creating interactive dashboard...")

        # Prepare data
        node_data = []
        for node in self.nodes:
            node_pods = [p for p in self.gpu_pods if p["node_name"] == node["name"]]
            total_vram_used = sum(p["gpu_memory_gb"] for p in node_pods)
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

        # Create subplots
        fig = sps.make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "VRAM Utilization",
                "Pod Count",
                "VRAM Capacity",
                "Resource Overview",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # VRAM Utilization
        nodes = [data["node"] for data in node_data]
        vram_util = [data["vram_utilization"] for data in node_data]

        fig.add_trace(
            go.Bar(
                x=nodes, y=vram_util, name="VRAM Utilization %", marker_color="coral"
            ),
            row=1,
            col=1,
        )

        # Pod Count
        pod_counts = [data["pod_count"] for data in node_data]

        fig.add_trace(
            go.Bar(x=nodes, y=pod_counts, name="Pod Count", marker_color="lightblue"),
            row=1,
            col=2,
        )

        # VRAM Capacity
        vram_used = [data["vram_used_gb"] for data in node_data]
        vram_capacity = [data["vram_capacity_gb"] for data in node_data]

        fig.add_trace(
            go.Bar(x=nodes, y=vram_used, name="VRAM Used", marker_color="orange"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=nodes,
                y=vram_capacity,
                name="VRAM Capacity",
                marker_color="lightgreen",
            ),
            row=2,
            col=1,
        )

        # Resource Overview
        for node in nodes:
            node_data_local = [d for d in node_data if d["node"] == node][0]
            fig.add_trace(
                go.Bar(
                    x=[node],
                    y=[node_data_local["pod_count"]],
                    name="Pod Count",
                    marker_color="purple",
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
            results["cluster_visualization"] = self.create_cluster_visualization(
                output_dir
            )
            results["pod_placement_heatmap"] = self.create_pod_placement_heatmap(
                output_dir
            )
            results["resource_utilization"] = self.create_resource_utilization_chart(
                output_dir
            )

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
            node_vram_used = sum(p["gpu_memory_gb"] for p in node_pods)
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
    parser = argparse.ArgumentParser(description="Generate GPU cluster visualizations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for visualizations",
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
        visualizer = SeabornGPUVisualizer()
        results = visualizer.generate_all_visualizations(
            output_dir, args.interactive_only
        )

        print("\nGenerated visualizations:")
        for viz_type, path in results.items():
            if path:
                print(f"  {viz_type}: {path}")
            else:
                print(f"  {viz_type}: No data available")

        if not any(results.values()):
            print(
                "Warning: No visualizations generated - check cluster status and permissions"
            )

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise


if __name__ == "__main__":
    main()
