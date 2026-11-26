"""
Phase 3: Route Maker and Graph Processing.

Combines:
    1. Dijkstra's Shortest-Path Algorithm (using a binary min-heap via heapq)
       to compute shortest road distances and paths from the Depot and between
       all selected delivery stops.
    2. Nearest-Neighbor Traveling Salesperson Problem (TSP) Approximation
       over the shortest-path metric closure to construct the delivery tour
       starting and ending at the Depot.
    3. Optional 2-Opt Local Search refinement (used in Phase 4 optimization study).

Complexities (V = vertices, E = edges, k = selected unique stops + depot):
    - Single-source Dijkstra: O((V + E) log V) time, O(V) space.
    - All-required-stops Dijkstra: O(k * (V + E) log V) time.
    - Nearest-Neighbor TSP: O(k^2) time over the k x k shortest-path matrix.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

from src.models import DeliveryGraph, RouteLeg, RoutePlan


def dijkstra(
    graph: DeliveryGraph,
    source: str,
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """Computes single-source shortest paths using Dijkstra's algorithm.

    Args:
        graph: DeliveryGraph with non-negative edge weights.
        source: Starting node ID (e.g., Depot 'D0').

    Returns:
        Tuple of:
            - distances: node -> shortest distance from source (inf if unreachable)
            - predecessors: node -> previous node on shortest path (None for source/unreachable)
    """
    if source not in graph.adjacency:
        raise ValueError(f"Source node '{source}' does not exist in the delivery graph.")

    distances: Dict[str, float] = {node: math.inf for node in graph.adjacency}
    predecessors: Dict[str, Optional[str]] = {node: None for node in graph.adjacency}
    distances[source] = 0.0

    # Priority queue stores (current_distance, node_id); lazy deletion skips stale entries.
    min_heap: List[Tuple[float, str]] = [(0.0, source)]

    while min_heap:
        curr_dist, u = heapq.heappop(min_heap)
        if curr_dist > distances[u]:
            continue

        for v, weight in graph.adjacency[u].items():
            candidate = curr_dist + weight
            if candidate < distances[v]:
                distances[v] = candidate
                predecessors[v] = u
                heapq.heappush(min_heap, (candidate, v))

    return distances, predecessors


def reconstruct_path(
    predecessors: Dict[str, Optional[str]],
    source: str,
    target: str,
) -> List[str]:
    """Reconstructs the shortest node sequence from source to target."""
    if source == target:
        return [source]
    if predecessors.get(target) is None:
        return []

    path: List[str] = []
    curr: Optional[str] = target
    while curr is not None:
        path.append(curr)
        if curr == source:
            break
        curr = predecessors[curr]
    path.reverse()

    if not path or path[0] != source:
        return []
    return path


def compute_stop_shortest_paths(
    graph: DeliveryGraph,
    depot: str,
    stops: Sequence[str],
) -> Tuple[Dict[Tuple[str, str], Tuple[float, List[str]]], Dict[str, Tuple[float, List[str]]]]:
    """Runs Dijkstra from the depot and every unique delivery stop.

    Returns:
        - pair_paths: (u, v) -> (shortest_distance, road_path)
        - from_depot: stop -> (shortest_distance_from_depot, road_path_from_depot)
    """
    unique_stops: List[str] = [depot]
    seen = {depot}
    for stop in stops:
        if stop not in graph.adjacency:
            raise ValueError(f"Delivery stop '{stop}' is missing from the graph.")
        if stop not in seen:
            seen.add(stop)
            unique_stops.append(stop)

    pair_paths: Dict[Tuple[str, str], Tuple[float, List[str]]] = {}
    from_depot: Dict[str, Tuple[float, List[str]]] = {}

    for src in unique_stops:
        dists, preds = dijkstra(graph, src)
        for dst in unique_stops:
            dist = dists[dst]
            if math.isinf(dist):
                raise ValueError(
                    f"Destination '{dst}' is unreachable from '{src}' in the delivery graph."
                )
            path = reconstruct_path(preds, src, dst)
            pair_paths[(src, dst)] = (dist, path)
            if src == depot and dst != depot:
                from_depot[dst] = (dist, path)

    return pair_paths, from_depot


def _build_route_plan_from_order(
    depot: str,
    required_stops: List[str],
    visit_order: List[str],
    pair_paths: Dict[Tuple[str, str], Tuple[float, List[str]]],
    from_depot: Dict[str, Tuple[float, List[str]]],
) -> RoutePlan:
    """Expands a TSP stop order into full road-network legs and total distance."""
    legs: List[RouteLeg] = []
    full_expanded_path: List[str] = []
    total_distance = 0.0

    for idx in range(len(visit_order) - 1):
        u = visit_order[idx]
        v = visit_order[idx + 1]
        dist, road_path = pair_paths[(u, v)]
        total_distance += dist
        legs.append(RouteLeg(from_stop=u, to_stop=v, distance=dist, path=road_path))

        if not full_expanded_path:
            full_expanded_path.extend(road_path)
        else:
            # Avoid duplicating the connecting stop between consecutive legs.
            full_expanded_path.extend(road_path[1:])

    if not visit_order:
        visit_order = [depot, depot]
        full_expanded_path = [depot]

    return RoutePlan(
        depot=depot,
        required_stops=required_stops,
        visit_order=visit_order,
        full_expanded_path=full_expanded_path,
        total_distance=round(total_distance, 4),
        legs=legs,
        shortest_paths_from_depot=from_depot,
    )


def nearest_neighbor_tsp(
    graph: DeliveryGraph,
    depot: str,
    stops: Sequence[str],
) -> RoutePlan:
    """Constructs an approximate TSP delivery tour using the Nearest-Neighbor heuristic.

    Algorithm:
        1. Compute shortest-path distances between the depot and all unique stops
           via Dijkstra's algorithm.
        2. Start at `depot`.
        3. Repeatedly move to the unvisited stop with the smallest shortest-path
           distance from the current stop (deterministic ID tie-breaking).
        4. Return to `depot` after all stops are visited.
    """
    unique_customers = sorted({s for s in stops if s != depot})
    pair_paths, from_depot = compute_stop_shortest_paths(graph, depot, unique_customers)

    if not unique_customers:
        return RoutePlan(
            depot=depot,
            required_stops=[],
            visit_order=[depot, depot],
            full_expanded_path=[depot],
            total_distance=0.0,
            legs=[],
            shortest_paths_from_depot={},
        )

    unvisited = set(unique_customers)
    visit_order: List[str] = [depot]
    current = depot

    while unvisited:
        # Choose minimum shortest-path distance; tie-break by node ID for determinism.
        next_stop = min(
            unvisited,
            key=lambda candidate: (pair_paths[(current, candidate)][0], candidate),
        )
        visit_order.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop

    visit_order.append(depot)
    return _build_route_plan_from_order(
        depot=depot,
        required_stops=unique_customers,
        visit_order=visit_order,
        pair_paths=pair_paths,
        from_depot=from_depot,
    )


def two_opt_refine_tsp(
    graph: DeliveryGraph,
    initial_route: RoutePlan,
) -> RoutePlan:
    """Applies 2-Opt local search to improve a Nearest-Neighbor TSP route (Phase 4).

    Repeatedly reverses route sub-segments (i..j) whenever doing so strictly
    reduces total tour length, until a local optimum is reached.
    """
    order = list(initial_route.visit_order)
    if len(order) <= 4:  # Depot -> 0, 1, or 2 stops -> Depot cannot improve via 2-opt
        return initial_route

    pair_paths, from_depot = compute_stop_shortest_paths(
        graph, initial_route.depot, initial_route.required_stops
    )

    def tour_length(seq: List[str]) -> float:
        return sum(pair_paths[(seq[k], seq[k + 1])][0] for k in range(len(seq) - 1))

    best_order = order
    best_dist = tour_length(best_order)
    improved = True

    while improved:
        improved = False
        # Keep first and last elements fixed as the Depot.
        for i in range(1, len(best_order) - 2):
            for j in range(i + 1, len(best_order) - 1):
                candidate = best_order[:i] + best_order[i : j + 1][::-1] + best_order[j + 1 :]
                cand_dist = tour_length(candidate)
                if cand_dist + 1e-9 < best_dist:
                    best_order = candidate
                    best_dist = cand_dist
                    improved = True

    return _build_route_plan_from_order(
        depot=initial_route.depot,
        required_stops=initial_route.required_stops,
        visit_order=best_order,
        pair_paths=pair_paths,
        from_depot=from_depot,
    )
