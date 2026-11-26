# Mini-Courier Planner — DAA Capstone Project

**Course:** Design & Analysis of Algorithms (`ETCCDA302`)  
**Programme:** B.Tech CSE & Specializations  
**University:** K.R. Mangalam University, Gurugram, Haryana  
**Student Name:** Rajiv Jadon  
**Roll No:** `2501010174`  
**Faculty:** Dr. Ankita Gupta  

---

## Problem Statement

Given a list of courier parcels (each with a destination, weight, priority score, and delivery deadline), a vehicle weight capacity $W$, and a weighted road network graph with a central depot `D0`, **Mini-Courier Planner** computes:
1. A prioritized parcel order (earliest deadline first, tie-broken stably by highest priority).
2. An optimal one-trip packing plan maximizing total priority within capacity $W$.
3. Shortest paths between the depot and selected customer stops, plus a complete round-trip delivery route.

---

## Algorithmic Pipeline (Phases 1–5)

| Stage | Module | Algorithm | Time Complexity | Space Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 2A: Prioritization** | `src/sorting.py` | **Merge Sort** *(plus Quick Sort & Insertion Sort for comparison)* | $\Theta(n \log n)$ | $O(n)$ |
| **Phase 2B: Box Filling** | `src/knapsack.py` | **0/1 Knapsack Dynamic Programming** *(2D DP + 1D $O(W)$ variant)* | $O(n \cdot W)$ | $O(n \cdot W)$ / $O(W)$ |
| **Phase 3A: Shortest Paths** | `src/graph_routing.py` | **Dijkstra's Algorithm** (`heapq` min-heap) | $O(k(V + E)\log V)$ | $O(V)$ |
| **Phase 3B: Route Sequence** | `src/graph_routing.py` | **Nearest-Neighbor TSP Approximation** *(+ 2-Opt refinement)* | $O(k^2)$ | $O(k^2)$ |

*(where $n$ = number of parcels, $W$ = vehicle capacity, $V$ = graph vertices, $E$ = graph edges, $k$ = unique selected delivery stops + depot)*

---

## Project Structure

```text
daa capston/
├── main.py                              # CLI entry point (supports custom/unseen JSON datasets)
├── run_all.py                           # One-click runner: datasets + unit tests + benchmarks + charts
├── requirements.txt                     # matplotlib, pillow (core algorithms use Python stdlib only)
├── src/
│   ├── models.py                        # Parcel, DeliveryGraph, PackingPlan, RouteLeg, RoutePlan
│   ├── sorting.py                       # Merge Sort, Quick Sort (median-of-3), Insertion Sort
│   ├── knapsack.py                      # 0/1 Knapsack 2D DP + backtracking, 1D DP, table formatter
│   ├── graph_routing.py                 # Dijkstra (min-heap), shortest-path matrix, NN-TSP, 2-Opt
│   ├── planner.py                       # MiniCourierPlanner orchestrator & report formatter
│   └── dataset_generator.py             # JSON loader & reproducible small/medium/large generator
├── benchmarks/
│   └── performance_analysis.py          # Phase 4 scaling benchmarks & matplotlib chart exporter
├── data/
│   ├── small_courier_dataset.json       # Hand-traceable demo (8 parcels, 6 nodes, W=15 kg)
│   ├── medium_courier_dataset.json      # Medium test set (40 parcels, 20 nodes, W=50 kg)
│   └── large_courier_dataset.json       # Large stress test (200 parcels, 60 nodes, W=120 kg)
├── tests/
│   └── test_mini_courier_planner.py     # 11 unit & integration tests (unittest)
├── outputs/
│   ├── charts/                          # sorting_comparison.png, runtime_comparison.png, sample_route_graph.png
│   ├── traces/                          # small_dataset_trace.txt, benchmark_results.json
│   └── screenshots/                     # terminal_demo_screenshot.png
└── docs/
    ├── DESIGN_AND_REPORT.md             # Complete Phase 1-5 Capstone Report
    └── VIVA_PREP.md                     # Viva-Voce Q&A, complexity proofs & dry-run trace
```

---

## Setup & Execution Steps

### 1. Install Dependencies (for chart generation only)
```powershell
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Recommended)
Generates datasets, runs all 11 unit tests, runs Phase 4 benchmarks, and exports all charts and screenshots:
```powershell
python run_all.py
```

### 3. Run on a Specific or Unseen Dataset
```powershell
# Default small sample dataset (displays full DP table, shortest paths & NN-TSP route)
python main.py --dataset data/small_courier_dataset.json

# Medium dataset using Quick Sort
python main.py --dataset data/medium_courier_dataset.json --sort quick --hide-dp-table

# Any unseen dataset during viva evaluation
python main.py --dataset path/to/unseen_dataset.json --capacity 30
```

### 4. Run Unit & Integration Tests Only
```powershell
python -m unittest discover -s tests -v
```

---

## Sample Output Highlights (Small Dataset)

- **Prioritized Order (Merge Sort):** `P106 -> P103 -> P101 -> P102 -> P107 -> P104 -> P105 -> P108`
- **0/1 Knapsack Selection ($W = 15\text{ kg}$):** `P101 (C1), P102 (C3), P107 (C4), P105 (C5)` -> **15 / 15 kg (100% capacity)**, **Priority Score = 202** (beats greedy-by-ratio score of 178)
- **Dijkstra Shortest Paths from `D0`:**
  - `D0 -> C1` = `4.00 km` (`D0 -> C1`)
  - `D0 -> C3` = `8.00 km` (`D0 -> C1 -> C3`)
  - `D0 -> C4` = `11.00 km` (`D0 -> C1 -> C2 -> C4`)
  - `D0 -> C5` = `8.50 km` (`D0 -> C1 -> C5`)
- **Nearest-Neighbor TSP Route:** `D0 -> C1 -> C3 -> C5 -> C4 -> D0` (**30.50 km**; expanded path: `D0 -> C1 -> C3 -> C5 -> C3 -> C4 -> C2 -> C1 -> D0`)
- **2-Opt Optimization Comparison:** `D0 -> C1 -> C5 -> C3 -> C4 -> D0` (**27.50 km**, saving **3.00 km / 9.84%**)

---

## Documentation & Evidence

- Full Phase 1–5 Academic Report: [docs/DESIGN_AND_REPORT.md](docs/DESIGN_AND_REPORT.md)
- Viva Preparation & Dry-Run Guide: [docs/VIVA_PREP.md](docs/VIVA_PREP.md)
- Execution Trace: [outputs/traces/small_dataset_trace.txt](outputs/traces/small_dataset_trace.txt)
- Charts: [outputs/charts/sample_route_graph.png](outputs/charts/sample_route_graph.png), [outputs/charts/sorting_comparison.png](outputs/charts/sorting_comparison.png), [outputs/charts/runtime_comparison.png](outputs/charts/runtime_comparison.png)
- Terminal Screenshot: [outputs/screenshots/terminal_demo_screenshot.png](outputs/screenshots/terminal_demo_screenshot.png)
