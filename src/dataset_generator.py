"""
Dataset Loader and Reproducible Generator for Mini-Courier Planner.

Supports:
    - Hand-crafted small dataset (ideal for dry-run tracing and DP table display)
    - Seeded medium/large datasets for Phase 4 runtime scaling experiments
    - Arbitrary unseen JSON datasets for viva / evaluation testing
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.models import DeliveryGraph, Parcel


def load_dataset(path: str | Path) -> Tuple[str, int, DeliveryGraph, List[Parcel]]:
    """Loads a courier dataset from a JSON file.

    Returns:
        (depot_id, vehicle_capacity, DeliveryGraph, list_of_Parcels)
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    depot = str(raw["depot"])
    capacity = int(raw["capacity"])

    graph = DeliveryGraph()
    for node in raw["nodes"]:
        coords = tuple(node["coords"]) if "coords" in node and node["coords"] is not None else None
        graph.add_node(
            node_id=str(node["id"]),
            label=str(node.get("label", node["id"])),
            coords=(float(coords[0]), float(coords[1])) if coords else None,
        )

    for edge in raw["edges"]:
        graph.add_edge(
            source=str(edge["from"]),
            target=str(edge["to"]),
            distance=float(edge["distance"]),
            bidirectional=bool(edge.get("bidirectional", True)),
        )

    parcels = [
        Parcel(
            parcel_id=str(p["parcel_id"]),
            destination=str(p["destination"]),
            weight=int(p["weight"]),
            priority=int(p["priority"]),
            deadline_hours=float(p["deadline_hours"]),
            description=str(p.get("description", "")),
        )
        for p in raw["parcels"]
    ]
    return depot, capacity, graph, parcels


def save_dataset(
    path: str | Path,
    depot: str,
    capacity: int,
    graph: DeliveryGraph,
    parcels: List[Parcel],
    metadata: Dict[str, object] | None = None,
) -> None:
    """Serializes a courier problem instance to JSON."""
    nodes_payload = []
    for node_id in sorted(graph.adjacency):
        entry: Dict[str, object] = {
            "id": node_id,
            "label": graph.labels.get(node_id, node_id),
        }
        if node_id in graph.coordinates:
            entry["coords"] = list(graph.coordinates[node_id])
        nodes_payload.append(entry)

    seen_edges = set()
    edges_payload = []
    for u in sorted(graph.adjacency):
        for v in sorted(graph.adjacency[u]):
            pair = (min(u, v), max(u, v))
            if pair not in seen_edges:
                seen_edges.add(pair)
                edges_payload.append(
                    {
                        "from": pair[0],
                        "to": pair[1],
                        "distance": graph.adjacency[u][v],
                        "bidirectional": True,
                    }
                )

    parcels_payload = [
        {
            "parcel_id": p.parcel_id,
            "destination": p.destination,
            "weight": p.weight,
            "priority": p.priority,
            "deadline_hours": p.deadline_hours,
            "description": p.description,
        }
        for p in parcels
    ]

    payload: Dict[str, object] = {
        "metadata": metadata or {},
        "depot": depot,
        "capacity": capacity,
        "nodes": nodes_payload,
        "edges": edges_payload,
        "parcels": parcels_payload,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_small_sample_dataset() -> Tuple[str, int, DeliveryGraph, List[Parcel]]:
    """Creates a compact, hand-traceable dataset (8 parcels, 6 locations, W=15 kg).

    Designed so that:
        - Tied deadlines demonstrate stable Merge Sort behavior.
        - 0/1 Knapsack DP beats naive greedy selection.
        - Dijkstra chooses multi-hop shorter paths over direct longer edges.
    """
    depot = "D0"
    capacity = 15

    graph = DeliveryGraph()
    node_specs = [
        ("D0", "Central Hub (Depot)", (0.0, 0.0)),
        ("C1", "Sector 14 Market", (3.0, 4.0)),
        ("C2", "Cyber Park Hub", (7.0, 2.0)),
        ("C3", "University Campus", (5.0, 7.0)),
        ("C4", "South City Mall", (10.0, 6.0)),
        ("C5", "Industrial Estate", (2.0, 8.0)),
    ]
    for nid, label, coords in node_specs:
        graph.add_node(nid, label=label, coords=coords)

    edges = [
        ("D0", "C1", 4.0),
        ("D0", "C2", 8.0),
        ("D0", "C5", 9.0),
        ("C1", "C2", 3.0),   # D0->C1->C2 = 7.0 < direct D0->C2 (8.0)
        ("C1", "C3", 4.0),
        ("C1", "C5", 4.5),   # D0->C1->C5 = 8.5 < direct D0->C5 (9.0)
        ("C2", "C3", 5.0),
        ("C2", "C4", 4.0),
        ("C3", "C4", 4.5),
        ("C3", "C5", 3.5),
    ]
    for u, v, dist in edges:
        graph.add_edge(u, v, dist)

    parcels = [
        Parcel("P101", "C1", weight=3, priority=41, deadline_hours=2.0, description="Medical Kit"),
        Parcel("P102", "C3", weight=4, priority=50, deadline_hours=3.0, description="Lab Equipment"),
        Parcel("P103", "C2", weight=4, priority=49, deadline_hours=2.0, description="Legal Documents"),
        Parcel("P104", "C4", weight=5, priority=44, deadline_hours=4.0, description="Laptop Spare"),
        Parcel("P105", "C5", weight=5, priority=68, deadline_hours=5.0, description="Server PSU"),
        Parcel("P106", "C2", weight=2, priority=26, deadline_hours=1.5, description="Pharma Cold Pack"),
        Parcel("P107", "C4", weight=3, priority=43, deadline_hours=3.0, description="Optical Sensor"),
        Parcel("P108", "C1", weight=7, priority=58, deadline_hours=6.0, description="Printer Toner Box"),
    ]
    return depot, capacity, graph, parcels


def generate_synthetic_dataset(
    num_nodes: int,
    num_parcels: int,
    capacity: int,
    avg_degree: int = 4,
    seed: int = 42,
) -> Tuple[str, int, DeliveryGraph, List[Parcel]]:
    """Generates a connected road network graph and randomized parcel set."""
    rng = random.Random(seed)
    depot = "D0"
    graph = DeliveryGraph()

    # Place depot at center of a 100x100 km grid, customers around it.
    graph.add_node(depot, label="Central Depot", coords=(50.0, 50.0))
    customer_ids: List[str] = []
    for i in range(1, num_nodes):
        nid = f"C{i}"
        customer_ids.append(nid)
        x = round(rng.uniform(0.0, 100.0), 2)
        y = round(rng.uniform(0.0, 100.0), 2)
        graph.add_node(nid, label=f"Zone {i}", coords=(x, y))

    all_nodes = [depot] + customer_ids

    def road_distance(u: str, v: str) -> float:
        x1, y1 = graph.coordinates[u]
        x2, y2 = graph.coordinates[v]
        euclid = math.hypot(x2 - x1, y2 - y1)
        detour_factor = rng.uniform(1.05, 1.35)
        return round(max(1.0, euclid * detour_factor), 2)

    # Guarantee connectivity with a random spanning tree first.
    shuffled = list(all_nodes)
    rng.shuffle(shuffled)
    for idx in range(1, len(shuffled)):
        u = shuffled[idx]
        v = shuffled[rng.randint(0, idx - 1)]
        graph.add_edge(u, v, road_distance(u, v))

    # Connect each node to several nearest spatial neighbors for realistic road density.
    for u in all_nodes:
        xu, yu = graph.coordinates[u]
        neighbors_by_dist = sorted(
            (n for n in all_nodes if n != u),
            key=lambda n: math.hypot(graph.coordinates[n][0] - xu, graph.coordinates[n][1] - yu),
        )
        for v in neighbors_by_dist[:avg_degree]:
            if v not in graph.adjacency[u]:
                graph.add_edge(u, v, road_distance(u, v))

    categories = [
        "Electronics",
        "Medical Supplies",
        "Documents",
        "Apparel",
        "Auto Parts",
        "Books",
        "Groceries",
    ]
    parcels: List[Parcel] = []
    max_item_wt = max(5, min(20, capacity // 4))
    for i in range(1, num_parcels + 1):
        dest = rng.choice(customer_ids)
        wt = rng.randint(1, max_item_wt)
        # Correlate priority mildly with weight + urgency noise
        deadline = round(rng.choice([1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]), 1)
        urgency_bonus = int(30 / deadline)
        priority = rng.randint(10, 60) + wt * 3 + urgency_bonus
        parcels.append(
            Parcel(
                parcel_id=f"P{100 + i}",
                destination=dest,
                weight=wt,
                priority=priority,
                deadline_hours=deadline,
                description=rng.choice(categories),
            )
        )

    return depot, capacity, graph, parcels


def materialize_standard_datasets(data_dir: str | Path = "data") -> None:
    """Generates and saves small, medium, and large JSON datasets."""
    out = Path(data_dir)

    d_s, c_s, g_s, p_s = create_small_sample_dataset()
    save_dataset(
        out / "small_courier_dataset.json",
        d_s,
        c_s,
        g_s,
        p_s,
        metadata={"name": "Small Sample Dataset", "purpose": "Hand-traceable demo & DP table"},
    )

    d_m, c_m, g_m, p_m = generate_synthetic_dataset(
        num_nodes=20, num_parcels=40, capacity=50, avg_degree=4, seed=101
    )
    save_dataset(
        out / "medium_courier_dataset.json",
        d_m,
        c_m,
        g_m,
        p_m,
        metadata={"name": "Medium Courier Dataset", "purpose": "Mid-scale validation"},
    )

    d_l, c_l, g_l, p_l = generate_synthetic_dataset(
        num_nodes=60, num_parcels=200, capacity=120, avg_degree=5, seed=202
    )
    save_dataset(
        out / "large_courier_dataset.json",
        d_l,
        c_l,
        g_l,
        p_l,
        metadata={"name": "Large Courier Dataset", "purpose": "Large-scale stress & timing"},
    )
