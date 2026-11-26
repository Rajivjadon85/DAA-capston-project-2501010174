"""
Unit and Integration Test Suite for Mini-Courier Planner (ETCCDA302).
Covers Phase 1-4 components: Data validation, Sorting, 0/1 Knapsack DP,
Dijkstra Shortest Paths, Nearest-Neighbor TSP, 2-Opt, and End-to-End Pipeline.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.dataset_generator import (
    create_small_sample_dataset,
    generate_synthetic_dataset,
    load_dataset,
)
from src.graph_routing import dijkstra, nearest_neighbor_tsp, reconstruct_path, two_opt_refine_tsp
from src.knapsack import knapsack_01_dp, knapsack_01_space_optimized
from src.models import DeliveryGraph, Parcel
from src.planner import MiniCourierPlanner
from src.sorting import (
    insertion_sort,
    merge_sort,
    parcel_priority_key,
    prioritize_parcels,
    quick_sort,
)


class TestSortingAlgorithms(unittest.TestCase):
    def test_all_sorters_match_python_sorted_on_parcels(self) -> None:
        _, _, _, parcels = generate_synthetic_dataset(num_nodes=12, num_parcels=35, capacity=50, seed=77)
        expected = [p.parcel_id for p in sorted(parcels, key=parcel_priority_key)]

        for algo in ("merge", "quick", "insertion"):
            actual = [p.parcel_id for p in prioritize_parcels(parcels, algorithm=algo)]
            self.assertEqual(actual, expected, msg=f"{algo} sort mismatch")

    def test_merge_sort_stability_on_tied_keys(self) -> None:
        items = [("a", 2), ("b", 1), ("c", 2), ("d", 1), ("e", 2)]
        sorted_items = merge_sort(items, key=lambda x: x[1])
        self.assertEqual(sorted_items, [("b", 1), ("d", 1), ("a", 2), ("c", 2), ("e", 2)])

    def test_edge_cases_empty_single_and_reverse(self) -> None:
        self.assertEqual(merge_sort([]), [])
        self.assertEqual(quick_sort([]), [])
        self.assertEqual(insertion_sort([]), [])

        single = [Parcel("P1", "C1", 2, 10, 1.0)]
        self.assertEqual(prioritize_parcels(single, "merge"), single)
        self.assertEqual(prioritize_parcels(single, "quick"), single)

        nums = [9, 7, 5, 3, 1, 1, 0]
        self.assertEqual(quick_sort(nums), [0, 1, 1, 3, 5, 7, 9])

    def test_invalid_algorithm_raises(self) -> None:
        with self.assertRaises(ValueError):
            prioritize_parcels([], algorithm="bubble")


class TestKnapsackDP(unittest.TestCase):
    def test_dp_beats_greedy_ratio_on_small_dataset(self) -> None:
        _, capacity, _, parcels = create_small_sample_dataset()
        prioritized = prioritize_parcels(parcels, "merge")
        plan = knapsack_01_dp(prioritized, capacity=capacity, include_table=True)

        self.assertEqual(plan.total_weight, 15)
        self.assertEqual(plan.total_priority, 202)
        self.assertIsNotNone(plan.dp_table)
        self.assertEqual(len(plan.dp_table), len(parcels) + 1)
        self.assertEqual(len(plan.dp_table[0]), capacity + 1)
        selected_ids = {p.parcel_id for p in plan.selected_parcels}
        self.assertEqual(selected_ids, {"P101", "P102", "P105", "P107"})

    def test_2d_and_1d_knapsack_agree_on_random_cases(self) -> None:
        for seed in (11, 22, 33):
            _, capacity, _, parcels = generate_synthetic_dataset(15, 25, capacity=45, seed=seed)
            plan_2d = knapsack_01_dp(parcels, capacity=capacity, include_table=False)
            val_1d = knapsack_01_space_optimized(parcels, capacity=capacity)
            self.assertEqual(plan_2d.total_priority, val_1d)
            self.assertLessEqual(plan_2d.total_weight, capacity)

    def test_knapsack_edge_cases(self) -> None:
        p = [Parcel("P1", "C1", 10, 50, 2.0)]
        self.assertEqual(knapsack_01_dp([], 10).total_priority, 0)
        self.assertEqual(knapsack_01_dp(p, 0).total_priority, 0)
        self.assertEqual(knapsack_01_dp(p, 5).selected_parcels, [])
        with self.assertRaises(ValueError):
            knapsack_01_dp(p, -1)


class TestGraphAndRouting(unittest.TestCase):
    def test_dijkstra_prefers_multi_hop_shorter_path(self) -> None:
        depot, _, graph, _ = create_small_sample_dataset()
        dists, preds = dijkstra(graph, depot)

        # Direct D0->C2 is 8.0, but D0->C1->C2 is 4.0 + 3.0 = 7.0
        self.assertAlmostEqual(dists["C2"], 7.0)
        self.assertEqual(reconstruct_path(preds, depot, "C2"), ["D0", "C1", "C2"])
        self.assertAlmostEqual(dists["C4"], 11.0)
        self.assertEqual(reconstruct_path(preds, depot, "C4"), ["D0", "C1", "C2", "C4"])

    def test_negative_edge_and_disconnected_node_validation(self) -> None:
        g = DeliveryGraph()
        with self.assertRaises(ValueError):
            g.add_edge("D0", "C1", -2.5)

        g.add_edge("D0", "C1", 3.0)
        g.add_node("C99")  # Disconnected island
        with self.assertRaises(ValueError):
            nearest_neighbor_tsp(g, depot="D0", stops=["C1", "C99"])

    def test_nn_tsp_and_two_opt_on_small_dataset(self) -> None:
        depot, _, graph, _ = create_small_sample_dataset()
        route = nearest_neighbor_tsp(graph, depot=depot, stops=["C1", "C3", "C4", "C5"])
        self.assertEqual(route.visit_order, ["D0", "C1", "C3", "C5", "C4", "D0"])
        self.assertAlmostEqual(route.total_distance, 30.5)

        refined = two_opt_refine_tsp(graph, route)
        self.assertLessEqual(refined.total_distance, route.total_distance)
        self.assertAlmostEqual(refined.total_distance, 27.5)
        self.assertEqual(refined.visit_order[0], depot)
        self.assertEqual(refined.visit_order[-1], depot)
        self.assertEqual(set(refined.visit_order[1:-1]), {"C1", "C3", "C4", "C5"})


class TestEndToEndPlanner(unittest.TestCase):
    def test_pipeline_on_saved_datasets(self) -> None:
        planner = MiniCourierPlanner(sort_algorithm="merge", include_dp_table=False, apply_two_opt=True)
        for filename in (
            "small_courier_dataset.json",
            "medium_courier_dataset.json",
            "large_courier_dataset.json",
        ):
            path = Path("data") / filename
            depot, capacity, graph, parcels = load_dataset(path)
            result = planner.plan(parcels=parcels, graph=graph, depot=depot, capacity=capacity)

            self.assertLessEqual(result.packing_plan.total_weight, capacity)
            self.assertGreater(result.packing_plan.total_priority, 0)
            self.assertEqual(result.route_plan.visit_order[0], depot)
            self.assertEqual(result.route_plan.visit_order[-1], depot)
            self.assertIsNotNone(result.refined_route_plan)
            self.assertLessEqual(
                result.refined_route_plan.total_distance,
                result.route_plan.total_distance + 1e-6,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
