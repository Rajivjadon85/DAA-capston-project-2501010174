"""
One-Click Project Runner for Mini-Courier Planner (ETCCDA302).

Executes:
    1. Dataset generation (small, medium, large JSON datasets)
    2. Unit & integration test suite
    3. Performance benchmarks & matplotlib chart export
    4. Small dataset full execution trace & screenshot rendering
"""

from __future__ import annotations

import subprocess
import sys

from benchmarks.performance_analysis import run_all_benchmarks_and_charts
from src.dataset_generator import materialize_standard_datasets


def main() -> None:
    print("1/3 Materializing standard JSON datasets in data/ ...")
    materialize_standard_datasets("data")

    print("\n2/3 Running unit & integration tests ...")
    test_proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        check=False,
    )
    if test_proc.returncode != 0:
        raise SystemExit(test_proc.returncode)

    print("\n3/3 Running Phase 4 benchmarks, charts, trace, and screenshot ...")
    summary = run_all_benchmarks_and_charts()
    print("\nAll artifacts generated successfully:")
    print("  - outputs/traces/small_dataset_trace.txt")
    print("  - outputs/traces/benchmark_results.json")
    print("  - outputs/charts/sorting_comparison.png")
    print("  - outputs/charts/runtime_comparison.png")
    print("  - outputs/charts/sample_route_graph.png")
    print("  - outputs/screenshots/terminal_demo_screenshot.png")
    print(f"  - Benchmarked {len(summary['pipeline_benchmark'])} dataset scales.")


if __name__ == "__main__":
    main()
