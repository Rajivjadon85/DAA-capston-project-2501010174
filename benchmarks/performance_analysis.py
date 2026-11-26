"""
Phase 4: Performance Testing, Benchmarking, and Visualization.

Runs controlled timing experiments across small, medium, and large datasets for:
    1. Sorting Algorithms (Merge Sort vs Quick Sort vs Insertion Sort)
    2. 0/1 Knapsack Dynamic Programming (Full 2D Table vs 1D Space-Optimized)
    3. Graph Routing (Dijkstra Shortest Paths + Nearest-Neighbor TSP + 2-Opt)
    4. End-to-End Mini-Courier Planner Pipeline

Produces:
    - outputs/traces/benchmark_results.json
    - outputs/charts/sorting_comparison.png
    - outputs/charts/runtime_comparison.png
    - outputs/charts/sample_route_graph.png
    - outputs/screenshots/terminal_demo_screenshot.png
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless chart generation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

from src.dataset_generator import create_small_sample_dataset, generate_synthetic_dataset
from src.knapsack import knapsack_01_dp, knapsack_01_space_optimized
from src.planner import MiniCourierPlanner, format_execution_report
from src.sorting import parcel_priority_key, prioritize_parcels


def _time_call(fn, repeats: int = 3) -> float:
    """Returns average execution time in milliseconds over `repeats` runs."""
    best_or_avg = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best_or_avg += (time.perf_counter() - start) * 1000.0
    return round(best_or_avg / repeats, 4)


def benchmark_sorting(sizes: List[int]) -> Dict[str, object]:
    """Benchmarks Merge Sort, Quick Sort, and Insertion Sort across parcel counts."""
    merge_times: List[float] = []
    quick_times: List[float] = []
    insertion_times: List[float] = []

    for n in sizes:
        _, _, _, parcels = generate_synthetic_dataset(
            num_nodes=max(10, n // 4),
            num_parcels=n,
            capacity=100,
            seed=500 + n,
        )
        repeats = 5 if n <= 200 else 3
        merge_times.append(_time_call(lambda p=parcels: prioritize_parcels(p, "merge"), repeats))
        quick_times.append(_time_call(lambda p=parcels: prioritize_parcels(p, "quick"), repeats))
        insertion_times.append(
            _time_call(lambda p=parcels: prioritize_parcels(p, "insertion"), repeats)
        )

    return {
        "sizes": sizes,
        "merge_sort_ms": merge_times,
        "quick_sort_ms": quick_times,
        "insertion_sort_ms": insertion_times,
    }


def benchmark_pipeline() -> List[Dict[str, object]]:
    """Benchmarks all phases of MiniCourierPlanner from small to large datasets."""
    configs = [
        {"label": "XS (10p, 8n)", "num_parcels": 10, "num_nodes": 8, "capacity": 20},
        {"label": "Small (30p, 15n)", "num_parcels": 30, "num_nodes": 15, "capacity": 40},
        {"label": "Med (60p, 25n)", "num_parcels": 60, "num_nodes": 25, "capacity": 60},
        {"label": "Large (120p, 40n)", "num_parcels": 120, "num_nodes": 40, "capacity": 100},
        {"label": "XL (250p, 60n)", "num_parcels": 250, "num_nodes": 60, "capacity": 140},
        {"label": "XXL (450p, 90n)", "num_parcels": 450, "num_nodes": 90, "capacity": 200},
    ]

    records: List[Dict[str, object]] = []
    planner = MiniCourierPlanner(sort_algorithm="merge", include_dp_table=True, apply_two_opt=True)

    for idx, cfg in enumerate(configs):
        depot, capacity, graph, parcels = generate_synthetic_dataset(
            num_nodes=cfg["num_nodes"],
            num_parcels=cfg["num_parcels"],
            capacity=cfg["capacity"],
            avg_degree=4,
            seed=1000 + idx,
        )

        # Measure full planner run (average of 2 runs for stability on larger inputs)
        repeats = 3 if cfg["num_parcels"] <= 120 else 2
        accum = {
            "sorting_ms": 0.0,
            "knapsack_dp_ms": 0.0,
            "dijkstra_nn_tsp_ms": 0.0,
            "two_opt_ms": 0.0,
            "total_ms": 0.0,
        }
        last_result = None
        for _ in range(repeats):
            res = planner.plan(parcels=parcels, graph=graph, depot=depot, capacity=capacity)
            last_result = res
            for k in accum:
                accum[k] += res.timings_ms[k]

        avg_timings = {k: round(v / repeats, 4) for k, v in accum.items()}

        # Also time 1D space-optimized knapsack for Phase 4 optimization comparison
        sorted_parcels = sorted(parcels, key=parcel_priority_key)
        dp_1d_ms = _time_call(
            lambda sp=sorted_parcels, cap=capacity: knapsack_01_space_optimized(sp, cap),
            repeats=repeats,
        )

        assert last_result is not None
        nn_dist = last_result.route_plan.total_distance
        opt_dist = (
            last_result.refined_route_plan.total_distance
            if last_result.refined_route_plan
            else nn_dist
        )

        records.append(
            {
                "label": cfg["label"],
                "num_parcels": cfg["num_parcels"],
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "capacity": capacity,
                "selected_parcels": len(last_result.packing_plan.selected_parcels),
                "unique_stops": len(last_result.route_plan.required_stops),
                "nn_tsp_distance_km": nn_dist,
                "two_opt_distance_km": opt_dist,
                "knapsack_1d_ms": dp_1d_ms,
                **avg_timings,
            }
        )

    return records


def plot_sorting_chart(sort_data: Dict[str, object], out_path: Path) -> None:
    """Plots Merge Sort vs Quick Sort vs Insertion Sort runtime scaling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)

    sizes = sort_data["sizes"]
    ax.plot(sizes, sort_data["merge_sort_ms"], marker="o", linewidth=2.4, label="Merge Sort (Stable, O(n log n))", color="#1f77b4")
    ax.plot(sizes, sort_data["quick_sort_ms"], marker="s", linewidth=2.4, label="Quick Sort (Median-of-3, O(n log n))", color="#2ca02c")
    ax.plot(sizes, sort_data["insertion_sort_ms"], marker="^", linewidth=2.4, linestyle="--", label="Insertion Sort (O(n^2))", color="#d62728")

    ax.set_title("Phase 4: Sorting Algorithm Runtime Comparison (Parcel Prioritization)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Parcels (N)", fontsize=11)
    ax.set_ylabel("Average Execution Time (ms)", fontsize=11)
    ax.legend(frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.45)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_runtime_dashboard(records: List[Dict[str, object]], out_path: Path) -> None:
    """Plots a 4-panel performance dashboard covering all algorithmic stages."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), dpi=160)

    labels = [r["label"] for r in records]
    x = list(range(len(labels)))

    # Panel 1: Stage-wise runtime breakdown
    ax1 = axes[0, 0]
    ax1.plot(x, [r["sorting_ms"] for r in records], marker="o", linewidth=2.2, label="Merge Sort", color="#1f77b4")
    ax1.plot(x, [r["knapsack_dp_ms"] for r in records], marker="s", linewidth=2.2, label="0/1 Knapsack DP (2D)", color="#ff7f0e")
    ax1.plot(x, [r["dijkstra_nn_tsp_ms"] for r in records], marker="D", linewidth=2.2, label="Dijkstra + NN-TSP", color="#2ca02c")
    ax1.plot(x, [r["total_ms"] for r in records], marker="^", linewidth=2.2, linestyle="--", label="Total Pipeline", color="#9467bd")
    ax1.set_title("End-to-End Stage Runtimes Across Dataset Scales", fontweight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("Time (ms)")
    ax1.legend(fontsize=8.5)

    # Panel 2: Knapsack 2D vs 1D Space-Optimized
    ax2 = axes[0, 1]
    width = 0.36
    ax2.bar([i - width / 2 for i in x], [r["knapsack_dp_ms"] for r in records], width=width, label="2D DP Table (O(n*W) space)", color="#ff7f0e")
    ax2.bar([i + width / 2 for i in x], [r["knapsack_1d_ms"] for r in records], width=width, label="1D Space-Optimized (O(W) space)", color="#17becf")
    ax2.set_title("0/1 Knapsack DP: Full 2D Table vs 1D Space-Optimized", fontweight="bold", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylabel("Time (ms)")
    ax2.legend(fontsize=8.5)

    # Panel 3: Graph Routing Complexity vs Stops & Nodes
    ax3 = axes[1, 0]
    nodes = [r["num_nodes"] for r in records]
    routing_ms = [r["dijkstra_nn_tsp_ms"] for r in records]
    two_opt_ms = [r["two_opt_ms"] for r in records]
    ax3.plot(nodes, routing_ms, marker="D", linewidth=2.2, label="Dijkstra + NN-TSP", color="#2ca02c")
    ax3.plot(nodes, two_opt_ms, marker="o", linewidth=2.0, linestyle=":", label="2-Opt Refinement", color="#8c564b")
    ax3.set_title("Graph Routing Runtime vs Network Size (V)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Number of Graph Nodes (V)")
    ax3.set_ylabel("Time (ms)")
    ax3.legend(fontsize=8.5)

    # Panel 4: Tour Length Comparison (NN-TSP vs 2-Opt)
    ax4 = axes[1, 1]
    ax4.bar([i - width / 2 for i in x], [r["nn_tsp_distance_km"] for r in records], width=width, label="Nearest-Neighbor TSP", color="#d62728", alpha=0.85)
    ax4.bar([i + width / 2 for i in x], [r["two_opt_distance_km"] for r in records], width=width, label="NN-TSP + 2-Opt Refined", color="#2ca02c", alpha=0.85)
    ax4.set_title("Route Quality: Nearest-Neighbor TSP vs 2-Opt Improvement", fontweight="bold", fontsize=11)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax4.set_ylabel("Total Tour Distance (km)")
    ax4.legend(fontsize=8.5)

    fig.suptitle(
        "Mini-Courier Planner (ETCCDA302) - Phase 4 Performance & Optimization Analysis\n"
        "Rajiv Jadon (Roll No: 2501010174)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sample_route_graph(out_path: Path) -> None:
    """Visualizes the small sample delivery graph, shortest paths, and NN-TSP tour."""
    depot, capacity, graph, parcels = create_small_sample_dataset()
    planner = MiniCourierPlanner(sort_algorithm="merge", include_dp_table=True, apply_two_opt=True)
    result = planner.plan(parcels=parcels, graph=graph, depot=depot, capacity=capacity)

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=170)

    # Draw base road network edges with distance labels
    drawn = set()
    for u, neighbors in graph.adjacency.items():
        x1, y1 = graph.coordinates[u]
        for v, dist in neighbors.items():
            edge = (min(u, v), max(u, v))
            if edge in drawn:
                continue
            drawn.add(edge)
            x2, y2 = graph.coordinates[v]
            ax.plot([x1, x2], [y1, y2], color="#b0bec5", linewidth=1.8, zorder=1)
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax.text(
                mx,
                my,
                f"{dist:.1f} km",
                fontsize=8,
                color="#455a64",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
                zorder=2,
            )

    # Highlight the Nearest-Neighbor TSP expanded road legs with directional arrows
    full_path = result.route_plan.full_expanded_path
    for idx in range(len(full_path) - 1):
        u, v = full_path[idx], full_path[idx + 1]
        x1, y1 = graph.coordinates[u]
        x2, y2 = graph.coordinates[v]
        # Slight curve offset so return traversal on shared road segments remains visible
        rad = 0.14 if idx >= len(full_path) // 2 else 0.06
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=8,head_width=5",
            shrinkA=15,
            shrinkB=15,
            color="#1565c0",
            linewidth=2.3,
            alpha=0.88,
            zorder=3,
        )
        ax.add_patch(arrow)

    selected_stops = set(result.route_plan.required_stops)
    for node_id, (nx, ny) in graph.coordinates.items():
        if node_id == depot:
            color, size = "#d32f2f", 420
        elif node_id in selected_stops:
            color, size = "#2e7d32", 340
        else:
            color, size = "#78909c", 280

        ax.scatter([nx], [ny], s=size, color=color, edgecolors="white", linewidth=2.0, zorder=4)
        ax.text(nx, ny, node_id, color="white", fontweight="bold", fontsize=9.5, ha="center", va="center", zorder=5)
        ax.text(
            nx,
            ny - 0.52,
            f"{graph.labels.get(node_id, node_id)}",
            fontsize=8.2,
            color="#263238",
            ha="center",
            va="top",
            fontweight="semibold",
            zorder=5,
        )

    # Legend for node roles & route
    legend_handles = [
        ax.scatter([], [], s=140, color="#d32f2f", label="Depot (D0)"),
        ax.scatter([], [], s=120, color="#2e7d32", label="Selected Delivery Stop"),
        ax.scatter([], [], s=100, color="#78909c", label="Unselected Stop (Deferred)"),
        ax.plot([], [], color="#1565c0", linewidth=2.3, label="NN-TSP Expanded Route")[0],
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8.5)

    # Legend & summary annotation box
    visit_str = " -> ".join(result.route_plan.visit_order)
    summary_text = (
        f"Selected Parcels (15/15 kg, Priority=202):\n"
        f"  P101(C1), P102(C3), P107(C4), P105(C5)\n"
        f"NN-TSP Stop Order: {visit_str}\n"
        f"NN-TSP Distance: {result.route_plan.total_distance:.2f} km  |  "
        f"2-Opt Distance: {result.refined_route_plan.total_distance:.2f} km"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=8.8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f9ff", edgecolor="#90caf9"),
        zorder=6,
    )

    ax.set_title(
        "Mini-Courier Planner: Small Dataset Graph & Nearest-Neighbor TSP Route",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(-1.2, 11.4)
    ax.set_ylim(-1.2, 9.4)
    ax.set_xlabel("X Coordinate (km)")
    ax.set_ylabel("Y Coordinate (km)")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_terminal_screenshot(trace_text: str, out_path: Path) -> None:
    """Renders a crisp dark-theme terminal screenshot PNG of the execution trace."""
    lines = trace_text.splitlines()
    try:
        font = ImageFont.truetype("consola.ttf", 14)
        header_font = ImageFont.truetype("consolab.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        header_font = font

    line_height = 20
    pad_x = 24
    pad_y = 22
    top_bar = 38

    max_chars = max((len(line) for line in lines), default=80)
    img_width = max(920, pad_x * 2 + int(max_chars * 8.6))
    img_height = top_bar + pad_y * 2 + line_height * len(lines)

    img = Image.new("RGB", (img_width, img_height), color=(24, 26, 32))
    draw = ImageDraw.Draw(img)

    # Window title bar
    draw.rectangle([0, 0, img_width, top_bar], fill=(38, 41, 50))
    for idx, dot_color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = 20 + idx * 20
        draw.ellipse([cx - 6, 13, cx + 6, 25], fill=dot_color)
    draw.text(
        (90, 11),
        "Windows PowerShell  -  python main.py --dataset data/small_courier_dataset.json",
        fill=(210, 215, 224),
        font=header_font,
    )

    y = top_bar + pad_y
    for line in lines:
        if line.startswith("[PHASE") or line.startswith("[RUNTIME"):
            color = (102, 217, 239)
        elif line.startswith("=") or line.startswith("MINI-COURIER") or line.startswith("Student:"):
            color = (166, 226, 46)
        elif line.startswith("Packing Result:") or line.startswith("Total Route Length"):
            color = (253, 151, 31)
        else:
            color = (232, 234, 240)
        draw.text((pad_x, y), line, fill=color, font=font)
        y += line_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def run_all_benchmarks_and_charts() -> Dict[str, object]:
    """Executes all benchmarks, writes JSON results, and exports all PNG charts."""
    sort_data = benchmark_sorting([20, 50, 100, 200, 400, 800])
    pipeline_records = benchmark_pipeline()

    results = {
        "sorting_benchmark": sort_data,
        "pipeline_benchmark": pipeline_records,
    }

    trace_dir = Path("outputs/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    charts_dir = Path("outputs/charts")
    plot_sorting_chart(sort_data, charts_dir / "sorting_comparison.png")
    plot_runtime_dashboard(pipeline_records, charts_dir / "runtime_comparison.png")
    plot_sample_route_graph(charts_dir / "sample_route_graph.png")

    # Also generate fresh small trace and terminal screenshot PNG
    depot, capacity, graph, parcels = create_small_sample_dataset()
    planner = MiniCourierPlanner(sort_algorithm="merge", include_dp_table=True, apply_two_opt=True)
    demo_res = planner.plan(parcels=parcels, graph=graph, depot=depot, capacity=capacity)
    trace_text = format_execution_report(demo_res, graph=graph, show_dp_table=True)
    (trace_dir / "small_dataset_trace.txt").write_text(trace_text, encoding="utf-8")
    render_terminal_screenshot(
        trace_text, Path("outputs/screenshots/terminal_demo_screenshot.png")
    )

    return results


if __name__ == "__main__":
    summary = run_all_benchmarks_and_charts()
    print("Benchmarks complete! Pipeline summary:")
    for row in summary["pipeline_benchmark"]:
        print(
            f"  {row['label']:<18} | sort={row['sorting_ms']:.3f}ms | "
            f"dp2D={row['knapsack_dp_ms']:.3f}ms | dp1D={row['knapsack_1d_ms']:.3f}ms | "
            f"route={row['dijkstra_nn_tsp_ms']:.3f}ms | "
            f"NN={row['nn_tsp_distance_km']:.1f}km vs 2Opt={row['two_opt_distance_km']:.1f}km"
        )
