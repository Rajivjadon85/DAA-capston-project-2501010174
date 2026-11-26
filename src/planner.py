"""
End-to-End Mini-Courier Planner Orchestrator.

Integrates:
    - Phase 2A: Parcel Prioritization (Merge Sort / Quick Sort / Insertion Sort)
    - Phase 2B: Trip Box Filling (0/1 Knapsack Dynamic Programming)
    - Phase 3:  Shortest Paths (Dijkstra) + Route Sequencing (Nearest-Neighbor TSP)
    - Phase 4:  Optional 2-Opt Refinement & Per-Stage Runtime Profiling
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.graph_routing import nearest_neighbor_tsp, two_opt_refine_tsp
from src.knapsack import format_dp_table, knapsack_01_dp
from src.models import DeliveryGraph, PackingPlan, Parcel, RoutePlan
from src.sorting import prioritize_parcels


@dataclass
class CourierPlanResult:
    """Complete result produced by MiniCourierPlanner."""

    depot: str
    capacity: int
    sort_algorithm: str
    input_parcels: List[Parcel]
    prioritized_parcels: List[Parcel]
    packing_plan: PackingPlan
    route_plan: RoutePlan
    refined_route_plan: Optional[RoutePlan] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)


class MiniCourierPlanner:
    """Coordinates sorting, knapsack packing, and shortest-path TSP routing."""

    def __init__(
        self,
        sort_algorithm: str = "merge",
        include_dp_table: bool = True,
        apply_two_opt: bool = True,
    ) -> None:
        self.sort_algorithm = sort_algorithm
        self.include_dp_table = include_dp_table
        self.apply_two_opt = apply_two_opt

    def plan(
        self,
        parcels: Sequence[Parcel],
        graph: DeliveryGraph,
        depot: str,
        capacity: int,
    ) -> CourierPlanResult:
        """Runs the complete 3-stage courier planning pipeline."""
        total_start = time.perf_counter()

        # Stage 1 (Phase 2A): Prioritize parcels by deadline & urgency
        t0 = time.perf_counter()
        prioritized = prioritize_parcels(parcels, algorithm=self.sort_algorithm)
        t_sort_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 2 (Phase 2B): 0/1 Knapsack DP for optimal one-trip parcel selection
        t1 = time.perf_counter()
        packing = knapsack_01_dp(
            prioritized,
            capacity=capacity,
            include_table=self.include_dp_table,
        )
        t_knapsack_ms = (time.perf_counter() - t1) * 1000.0

        # Stage 3 (Phase 3): Dijkstra shortest paths + Nearest-Neighbor TSP
        stops = [p.destination for p in packing.selected_parcels]
        t2 = time.perf_counter()
        route = nearest_neighbor_tsp(graph=graph, depot=depot, stops=stops)
        t_routing_ms = (time.perf_counter() - t2) * 1000.0

        # Stage 4 (Phase 4 Optimization): Optional 2-Opt refinement comparison
        refined_route: Optional[RoutePlan] = None
        t_two_opt_ms = 0.0
        if self.apply_two_opt:
            t3 = time.perf_counter()
            refined_route = two_opt_refine_tsp(graph=graph, initial_route=route)
            t_two_opt_ms = (time.perf_counter() - t3) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return CourierPlanResult(
            depot=depot,
            capacity=capacity,
            sort_algorithm=self.sort_algorithm,
            input_parcels=list(parcels),
            prioritized_parcels=prioritized,
            packing_plan=packing,
            route_plan=route,
            refined_route_plan=refined_route,
            timings_ms={
                "sorting_ms": round(t_sort_ms, 4),
                "knapsack_dp_ms": round(t_knapsack_ms, 4),
                "dijkstra_nn_tsp_ms": round(t_routing_ms, 4),
                "two_opt_ms": round(t_two_opt_ms, 4),
                "total_ms": round(total_ms, 4),
            },
        )


def format_execution_report(
    result: CourierPlanResult,
    graph: DeliveryGraph,
    show_dp_table: bool = True,
    max_table_rows: int = 25,
) -> str:
    """Formats a detailed, human-readable trace of all capstone phases."""
    lines: List[str] = []
    sep = "=" * 78

    lines.append(sep)
    lines.append("MINI-COURIER PLANNER  |  DAA CAPSTONE PROJECT (ETCCDA302)")
    lines.append("Student: Rajiv Jadon (Roll No: 2501010174) | Faculty: Dr. Ankita Gupta")
    lines.append(sep)
    lines.append(
        f"Dataset Summary: {len(result.input_parcels)} parcels | "
        f"{graph.num_nodes} nodes, {graph.num_edges} roads | "
        f"Depot: {result.depot} | Vehicle Capacity: {result.capacity} kg"
    )
    lines.append("")

    # Phase 2A: Sorting output
    lines.append(f"[PHASE 2A] PARCEL PRIORITIZATION ({result.sort_algorithm.upper()} SORT)")
    lines.append("-" * 78)
    lines.append(
        f"{'Rank':<5} {'ID':<7} {'Dest':<6} {'Wt(kg)':<7} {'Priority':<9} "
        f"{'Deadline(h)':<12} {'Description'}"
    )
    for idx, parcel in enumerate(result.prioritized_parcels[:max_table_rows], start=1):
        lines.append(
            f"{idx:<5} {parcel.parcel_id:<7} {parcel.destination:<6} "
            f"{parcel.weight:<7} {parcel.priority:<9} {parcel.deadline_hours:<12.1f} "
            f"{parcel.description}"
        )
    if len(result.prioritized_parcels) > max_table_rows:
        lines.append(f"... ({len(result.prioritized_parcels) - max_table_rows} more parcels)")
    lines.append("")

    # Phase 2B: 0/1 Knapsack DP
    pack = result.packing_plan
    lines.append("[PHASE 2B] OPTIMAL BOX FILLING (0/1 KNAPSACK DYNAMIC PROGRAMMING)")
    lines.append("-" * 78)
    if show_dp_table and pack.dp_table is not None and len(result.prioritized_parcels) <= 15:
        lines.append("DP Table (dp[i][w] = max priority with first i items & capacity w):")
        lines.append(format_dp_table(pack.dp_table, result.prioritized_parcels))
        lines.append("")

    lines.append(
        f"Packing Result: Selected {len(pack.selected_parcels)} / {len(result.prioritized_parcels)} parcels | "
        f"Weight Used: {pack.total_weight} / {pack.capacity} kg "
        f"({(pack.total_weight / pack.capacity * 100.0) if pack.capacity else 0.0:.1f}% utilization) | "
        f"Total Priority Score: {pack.total_priority}"
    )
    lines.append(
        "Selected Parcels : "
        + (", ".join(f"{p.parcel_id}->{p.destination}({p.weight}kg,val={p.priority})" for p in pack.selected_parcels) or "None")
    )
    lines.append(
        "Deferred Parcels : "
        + (", ".join(f"{p.parcel_id}({p.weight}kg,val={p.priority})" for p in pack.rejected_parcels[:10]) or "None")
        + (f" ... (+{len(pack.rejected_parcels) - 10} more)" if len(pack.rejected_parcels) > 10 else "")
    )
    lines.append("")

    # Phase 3: Graph, Dijkstra Shortest Paths & NN-TSP
    route = result.route_plan
    lines.append("[PHASE 3] SHORTEST PATHS (DIJKSTRA) & DELIVERY ROUTE (NEAREST-NEIGHBOR TSP)")
    lines.append("-" * 78)
    lines.append(f"Unique Delivery Stops Required: {route.required_stops or ['None']}")
    lines.append("Shortest Paths from Depot:")
    for stop in sorted(route.shortest_paths_from_depot):
        dist, path = route.shortest_paths_from_depot[stop]
        lines.append(f"  {result.depot} -> {stop:<5}: dist = {dist:>6.2f} km | path = {' -> '.join(path)}")

    lines.append("")
    lines.append(f"Nearest-Neighbor TSP Stop Sequence : {' -> '.join(route.visit_order)}")
    lines.append(f"Full Road-Network Traversal Path   : {' -> '.join(route.full_expanded_path)}")
    lines.append(f"Total Route Length (NN-TSP)        : {route.total_distance:.2f} km")
    lines.append("Leg-by-Leg Breakdown:")
    for leg in route.legs:
        lines.append(
            f"  {leg.from_stop} -> {leg.to_stop}: {leg.distance:.2f} km via [{' -> '.join(leg.path)}]"
        )

    if result.refined_route_plan is not None:
        opt = result.refined_route_plan
        improvement = route.total_distance - opt.total_distance
        lines.append("")
        lines.append("[PHASE 4 OPTIMIZATION] 2-OPT LOCAL SEARCH COMPARISON")
        lines.append("-" * 78)
        lines.append(f"2-Opt Refined Stop Sequence        : {' -> '.join(opt.visit_order)}")
        lines.append(f"Total Route Length (2-Opt)         : {opt.total_distance:.2f} km")
        lines.append(
            f"Distance Saved                     : {improvement:.2f} km "
            f"({(improvement / route.total_distance * 100.0) if route.total_distance > 0 else 0.0:.2f}%)"
        )

    lines.append("")
    lines.append("[RUNTIME PROFILE (ms)]")
    lines.append("-" * 78)
    for k, v in result.timings_ms.items():
        lines.append(f"  {k:<22}: {v:.4f} ms")
    lines.append(sep)
    return "\n".join(lines)
