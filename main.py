"""
CLI Entry Point for Mini-Courier Planner (DAA Capstone - ETCCDA302).
Student: Rajiv Jadon | Roll No: 2501010174 | Faculty: Dr. Ankita Gupta

Usage examples:
    python main.py
    python main.py --dataset data/small_courier_dataset.json
    python main.py --dataset data/medium_courier_dataset.json --sort quick
    python main.py --dataset path/to/unseen_dataset.json --save-trace outputs/traces/unseen.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset_generator import create_small_sample_dataset, load_dataset
from src.planner import MiniCourierPlanner, format_execution_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini-Courier Planner: Sorting + 0/1 Knapsack DP + Dijkstra + Nearest-Neighbor TSP"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to a JSON courier dataset. Defaults to built-in small sample dataset.",
    )
    parser.add_argument(
        "--sort",
        choices=["merge", "quick", "insertion"],
        default="merge",
        help="Sorting algorithm for Phase 2 parcel prioritization (default: merge).",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Override vehicle weight capacity W (kg).",
    )
    parser.add_argument(
        "--hide-dp-table",
        action="store_true",
        help="Hide the full 2D DP table in console output.",
    )
    parser.add_argument(
        "--no-two-opt",
        action="store_true",
        help="Skip optional 2-Opt route refinement comparison.",
    )
    parser.add_argument(
        "--save-trace",
        type=str,
        default="",
        help="Optional path to save the formatted execution report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset:
        depot, capacity, graph, parcels = load_dataset(args.dataset)
    else:
        default_small = Path("data/small_courier_dataset.json")
        if default_small.exists():
            depot, capacity, graph, parcels = load_dataset(default_small)
        else:
            depot, capacity, graph, parcels = create_small_sample_dataset()

    if args.capacity is not None:
        capacity = args.capacity

    planner = MiniCourierPlanner(
        sort_algorithm=args.sort,
        include_dp_table=not args.hide_dp_table,
        apply_two_opt=not args.no_two_opt,
    )
    result = planner.plan(parcels=parcels, graph=graph, depot=depot, capacity=capacity)
    report = format_execution_report(
        result,
        graph=graph,
        show_dp_table=not args.hide_dp_table,
    )
    print(report)

    if args.save_trace:
        out_path = Path(args.save_trace)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nSaved execution trace to: {out_path}")


if __name__ == "__main__":
    main()
