"""
Data models for the Mini-Courier Planner system (Phase 1: Problem Design).

Defines core entities for parcels, the weighted road network graph,
the 0/1 Knapsack packing plan, and the Dijkstra + Nearest-Neighbor TSP route plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Parcel:
    """Represents a single courier parcel awaiting dispatch.

    Attributes:
        parcel_id: Unique identifier (e.g., 'P101').
        destination: Target node ID in the delivery graph (e.g., 'C1').
        weight: Integer parcel weight in kg (must be positive for 0/1 Knapsack DP).
        priority: Business/urgency score used as value in 0/1 Knapsack (higher = more urgent).
        deadline_hours: Delivery deadline in hours from dispatch (lower = earlier deadline).
        description: Optional human-readable label for the parcel contents.
    """

    parcel_id: str
    destination: str
    weight: int
    priority: int
    deadline_hours: float
    description: str = ""

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"Parcel {self.parcel_id}: weight must be > 0, got {self.weight}")
        if self.priority < 0:
            raise ValueError(f"Parcel {self.parcel_id}: priority must be >= 0, got {self.priority}")
        if self.deadline_hours <= 0:
            raise ValueError(
                f"Parcel {self.parcel_id}: deadline_hours must be > 0, got {self.deadline_hours}"
            )


@dataclass
class DeliveryGraph:
    """Weighted graph representing delivery locations and road distances.

    Uses an adjacency-list mapping: node -> {neighbor: distance}.
    Optional 2D coordinates are stored for visualization and spatial analysis.
    """

    adjacency: Dict[str, Dict[str, float]] = field(default_factory=dict)
    coordinates: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)

    def add_node(
        self,
        node_id: str,
        label: str = "",
        coords: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Registers a delivery point or depot in the graph."""
        if node_id not in self.adjacency:
            self.adjacency[node_id] = {}
        if label:
            self.labels[node_id] = label
        if coords is not None:
            self.coordinates[node_id] = coords

    def add_edge(
        self,
        source: str,
        target: str,
        distance: float,
        bidirectional: bool = True,
    ) -> None:
        """Adds a non-negative weighted road segment between two nodes."""
        if distance < 0:
            raise ValueError(
                f"Edge ({source}, {target}) has negative weight {distance}; "
                "Dijkstra's algorithm requires non-negative weights."
            )
        self.add_node(source)
        self.add_node(target)
        # Keep the shortest direct road if duplicate edges exist.
        existing = self.adjacency[source].get(target)
        if existing is None or distance < existing:
            self.adjacency[source][target] = float(distance)
        if bidirectional:
            rev_existing = self.adjacency[target].get(source)
            if rev_existing is None or distance < rev_existing:
                self.adjacency[target][source] = float(distance)

    @property
    def num_nodes(self) -> int:
        return len(self.adjacency)

    @property
    def num_edges(self) -> int:
        """Counts unique undirected/directed edge entries in the adjacency list."""
        seen = set()
        for u, neighbors in self.adjacency.items():
            for v in neighbors:
                edge = (min(u, v), max(u, v))
                seen.add(edge)
        return len(seen)


@dataclass
class PackingPlan:
    """Output of Phase 2 (0/1 Knapsack optimization for one courier trip)."""

    capacity: int
    selected_parcels: List[Parcel]
    rejected_parcels: List[Parcel]
    total_weight: int
    total_priority: int
    dp_table: Optional[List[List[int]]] = None


@dataclass
class RouteLeg:
    """Represents one shortest-path transition between consecutive TSP stops."""

    from_stop: str
    to_stop: str
    distance: float
    path: List[str]


@dataclass
class RoutePlan:
    """Output of Phase 3 (Dijkstra shortest paths + Nearest-Neighbor TSP route)."""

    depot: str
    required_stops: List[str]
    visit_order: List[str]
    full_expanded_path: List[str]
    total_distance: float
    legs: List[RouteLeg]
    shortest_paths_from_depot: Dict[str, Tuple[float, List[str]]]
