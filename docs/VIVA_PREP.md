# Viva-Voce & Demonstration Preparation Guide

**Project:** Mini-Courier Planner (`ETCCDA302`)  
**Student:** Rajiv Jadon (`2501010174`)  
**Faculty:** Dr. Ankita Gupta  

---

## 1. Quick Complexity Cheat Sheet

| Algorithm | Role in Project | Best Time | Average Time | Worst Time | Aux Space |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Merge Sort** | Primary parcel prioritization | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $O(n)$ |
| **Quick Sort (Median-of-3)** | Phase 4 sorting benchmark | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $O(n^2)$ | $O(\log n)$ |
| **Insertion Sort** | Phase 4 quadratic baseline | $\Theta(n)$ | $\Theta(n^2)$ | $\Theta(n^2)$ | $O(1)$ |
| **0/1 Knapsack DP (2D)** | Optimal parcel selection + backtrack | $\Theta(nW)$ | $\Theta(nW)$ | $\Theta(nW)$ | $O(nW)$ |
| **0/1 Knapsack DP (1D)** | Space-optimized value computation | $\Theta(nW)$ | $\Theta(nW)$ | $\Theta(nW)$ | $O(W)$ |
| **Dijkstra's Algorithm** | Shortest path from each stop ($k$ runs) | $O((V+E)\log V)$ | $O((V+E)\log V)$ | $O((V+E)\log V)$ | $O(V)$ |
| **Nearest-Neighbor TSP** | Delivery tour construction | $\Theta(k^2)$ | $\Theta(k^2)$ | $\Theta(k^2)$ | $O(k^2)$ |
| **2-Opt Local Search** | Tour refinement (Phase 4) | $O(k^2)$ | $O(k^2 \cdot \text{passes})$ | $O(k^3+)$ | $O(k)$ |

---

## 2. Likely Viva Questions & Crisp Answers

### Q1. Why did you choose Merge Sort over Quick Sort or Insertion Sort for parcel prioritization?
- **Answer:** Merge Sort has a guaranteed $\Theta(n \log n)$ worst-case runtime (unlike Insertion Sort's $O(n^2)$ or Quick Sort's $O(n^2)$ theoretical worst case) and is **stable** (`left[i] <= right[j]` in `_merge`), which preserves the original intake order when two parcels have identical priority keys. In Phase 4, Quick Sort with median-of-three pivot is slightly faster in practice (1.12 ms vs 2.01 ms at $N=800$) due to better cache locality, while Insertion Sort degrades quadratically (26.53 ms at $N=800$).

### Q2. Why can't we use Greedy Fractional Knapsack for box filling?
- **Answer:** Parcels are indivisible physical boxes (0/1 constraint: either load the whole parcel or defer it). Greedy selection by `priority/weight` can leave unused capacity that a different combination fills optimally. On `small_courier_dataset.json` ($W=15\text{ kg}$), greedy-by-ratio achieves only **178** (`13/15 kg`), whereas 0/1 Knapsack DP achieves **202** (`15/15 kg`).

### Q3. How do you reconstruct which parcels were packed from the DP table?
- **Answer:** Starting at `i = n, w = capacity`, if `dp[i][w] != dp[i - 1][w]`, parcel `i - 1` must have been included, so we record it and reduce `w -= parcel.weight`. Otherwise it was excluded and we move to `i - 1` with the same `w`.

### Q4. Why is 0/1 Knapsack $O(nW)$ called pseudo-polynomial?
- **Answer:** Because the runtime grows linearly with the **numeric value** of $W$, not the number of bits $\log_2 W$ needed to represent $W$. If $W$ is huge (e.g., $10^9$ grams instead of kg), $O(nW)$ becomes slow and memory-heavy; that's when 1D DP ($O(W)$ space) or Branch-and-Bound / scaling weights helps.

### Q5. Why does Dijkstra's algorithm fail with negative edge weights, and how do you handle stale heap entries?
- **Answer:** Dijkstra greedily finalizes the closest unvisited vertex assuming path distances never decrease when extended by another edge. Negative edges violate that invariant (Bellman-Ford handles negative weights in $O(VE)$). Since Python's `heapq` has no `decrease-key`, we push `(candidate, v)` whenever a shorter path is found and skip popped entries where `curr_dist > distances[u]`.

### Q6. Why use Nearest-Neighbor TSP instead of finding the exact optimal TSP tour?
- **Answer:** Exact TSP is NP-hard ($O(k!)$ brute force, $O(k^2 2^k)$ Held-Karp DP). For $k = 30$ delivery stops, exact Held-Karp is completely infeasible. Nearest-Neighbor runs in $O(k^2)$ over the shortest-path distance matrix and returns a valid tour immediately. In Phase 4 we also apply **2-Opt local search**, which reversed `C3 -> C5` on our small dataset to cut distance from `30.50 km` to `27.50 km` (9.84% saved).

### Q7. How do you run the program on an unseen dataset or modify inputs during evaluation?
- **Answer:** Pass any new JSON file via `--dataset`:
  ```powershell
  python main.py --dataset path/to/unseen.json --sort merge --capacity 25
  ```
  All sorting, DP, Dijkstra, and TSP functions are decoupled in `src/`, so changing capacity, adding parcels, or editing edges in JSON requires zero code changes.

---

## 3. Quick Dry-Run Trace Summary (Small Dataset)

1. **Input:** 8 parcels, Capacity $W = 15\text{ kg}$, 6 nodes (`D0`, `C1`..`C5`), 10 edges.
2. **Merge Sort Order:** `P106 (1.5h)`, `P103 (2.0h, p=49)`, `P101 (2.0h, p=41)`, `P102 (3.0h, p=50)`, `P107 (3.0h, p=43)`, `P104 (4.0h)`, `P105 (5.0h)`, `P108 (6.0h)`.
3. **0/1 Knapsack DP:** `dp[8][15] = 202` -> selects `P101(C1,3kg)`, `P102(C3,4kg)`, `P107(C4,3kg)`, `P105(C5,5kg)` (`15/15 kg`).
4. **Dijkstra from `D0`:** `C1=4.0`, `C3=8.0 (D0->C1->C3)`, `C5=8.5 (D0->C1->C5)`, `C4=11.0 (D0->C1->C2->C4)`.
5. **Nearest-Neighbor TSP:** `D0 -(4.0)-> C1 -(4.0)-> C3 -(3.5)-> C5 -(8.0)-> C4 -(11.0)-> D0` = **`30.50 km`**.
6. **2-Opt Comparison:** `D0 -(4.0)-> C1 -(4.5)-> C5 -(3.5)-> C3 -(4.5)-> C4 -(11.0)-> D0` = **`27.50 km`**.

---

## 4. Live Code Modification Scenarios (If Faculty Asks to Edit Code)

| Faculty Request | Exact File & Line to Change |
| :--- | :--- |
| **"Sort by highest priority first instead of earliest deadline"** | In [src/sorting.py](file:///c:/Users/ASUS/Desktop/daa%20capston/src/sorting.py) -> `parcel_priority_key()`: change `(parcel.deadline_hours, -parcel.priority, parcel.weight, parcel.parcel_id)` to `(-parcel.priority, parcel.deadline_hours, parcel.weight, parcel.parcel_id)`. |
| **"Switch from Merge Sort to Quick Sort by default"** | In [src/planner.py](file:///c:/Users/ASUS/Desktop/daa%20capston/src/planner.py) -> `MiniCourierPlanner.__init__`: change `sort_algorithm: str = "merge"` to `"quick"`, or run `python main.py --sort quick`. |
| **"Change vehicle capacity to 25 kg without editing JSON"** | Run `python main.py --capacity 25`. |
| **"Support one-way (directed) roads"** | In `data/*.json`, set `"bidirectional": false` on an edge (`DeliveryGraph.add_edge` already supports `bidirectional=False`). |

---

## 5. Error Identification & Debugging Traps (Viva Q7)

1. **Bug: Using `<` instead of `<=` in Merge Sort's `_merge()`**
   - *Effect:* Breaks stability when two parcels share identical deadlines and priorities.
   - *Fix:* Keep `if key(left[i]) <= key(right[j]):` in [src/sorting.py](file:///c:/Users/ASUS/Desktop/daa%20capston/src/sorting.py).
2. **Bug: Iterating `w` forwards (`0 -> W`) in 1D Space-Optimized Knapsack**
   - *Effect:* Turns 0/1 Knapsack into **Unbounded Knapsack** (same parcel gets packed multiple times!).
   - *Fix:* Iterate backwards `for w in range(capacity, wt - 1, -1):` in [src/knapsack.py](file:///c:/Users/ASUS/Desktop/daa%20capston/src/knapsack.py).
3. **Bug: Negative road weight in Dijkstra**
   - *Effect:* Greedy relaxation assumption fails; shortest distance becomes wrong.
   - *Fix:* `DeliveryGraph.add_edge()` raises `ValueError` if `distance < 0`; use Bellman-Ford if negative weights ever exist.
4. **Bug: Missing stale-entry check in Dijkstra (`if curr_dist > distances[u]: continue`)**
   - *Effect:* Re-expands outdated heap entries, wasting $O(E \log V)$ extra work.
   - *Fix:* Keep the `if curr_dist > distances[u]: continue` guard in [src/graph_routing.py](file:///c:/Users/ASUS/Desktop/daa%20capston/src/graph_routing.py).

