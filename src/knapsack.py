"""
Phase 2: Optimal Box Filling using 0/1 Knapsack Dynamic Programming.

Given prioritized parcels (each with integer weight w_i and priority value v_i)
and a courier vehicle/box capacity W, selects the subset of parcels that
maximizes total priority without exceeding W.

Recurrence:
    dp[i][w] = max(
        dp[i - 1][w],                             # Exclude parcel i
        dp[i - 1][w - weight_i] + priority_i      # Include parcel i (if weight_i <= w)
    )

Time Complexity:  O(n * W)
Space Complexity: O(n * W) for full table + backtracking (or O(W) value-only).
"""

from __future__ import annotations

from typing import List, Sequence

from src.models import PackingPlan, Parcel


def knapsack_01_dp(
    parcels: Sequence[Parcel],
    capacity: int,
    include_table: bool = True,
) -> PackingPlan:
    """Solves 0/1 Knapsack using bottom-up Dynamic Programming with backtracking.

    Args:
        parcels: Candidate parcels (typically already sorted by deadline/priority).
        capacity: Maximum integer weight capacity W of the courier vehicle/box.
        include_table: Whether to attach the 2D DP table in PackingPlan (disable
            for very large n * W benchmarks to save memory).

    Returns:
        PackingPlan containing selected parcels, rejected parcels, totals, and DP table.
    """
    if capacity < 0:
        raise ValueError(f"Vehicle capacity must be non-negative, got {capacity}")

    n = len(parcels)
    if n == 0 or capacity == 0:
        empty_table = [[0] * (capacity + 1) for _ in range(n + 1)] if include_table else None
        return PackingPlan(
            capacity=capacity,
            selected_parcels=[],
            rejected_parcels=list(parcels),
            total_weight=0,
            total_priority=0,
            dp_table=empty_table,
        )

    # dp[i][w] = max priority achievable using first i parcels with capacity w.
    dp: List[List[int]] = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        parcel = parcels[i - 1]
        wt = parcel.weight
        val = parcel.priority
        prev_row = dp[i - 1]
        curr_row = dp[i]

        for w in range(capacity + 1):
            best = prev_row[w]
            if wt <= w:
                take = prev_row[w - wt] + val
                if take > best:
                    best = take
            curr_row[w] = best

    # Backtrack through the DP table to identify which parcels were selected.
    selected_indices = set()
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_indices.add(i - 1)
            w -= parcels[i - 1].weight

    selected_parcels = [parcels[idx] for idx in range(n) if idx in selected_indices]
    rejected_parcels = [parcels[idx] for idx in range(n) if idx not in selected_indices]
    total_weight = sum(p.weight for p in selected_parcels)
    total_priority = dp[n][capacity]

    return PackingPlan(
        capacity=capacity,
        selected_parcels=selected_parcels,
        rejected_parcels=rejected_parcels,
        total_weight=total_weight,
        total_priority=total_priority,
        dp_table=dp if include_table else None,
    )


def knapsack_01_space_optimized(parcels: Sequence[Parcel], capacity: int) -> int:
    """Space-optimized 1D 0/1 Knapsack DP computing maximum achievable priority.

    Uses O(W) auxiliary space by iterating capacity backwards from W down to wt
    so each parcel is considered at most once per trip.
    """
    if capacity < 0:
        raise ValueError(f"Vehicle capacity must be non-negative, got {capacity}")
    if not parcels or capacity == 0:
        return 0

    dp = [0] * (capacity + 1)
    for parcel in parcels:
        wt = parcel.weight
        val = parcel.priority
        if wt > capacity:
            continue
        for w in range(capacity, wt - 1, -1):
            candidate = dp[w - wt] + val
            if candidate > dp[w]:
                dp[w] = candidate
    return dp[capacity]


def format_dp_table(
    dp_table: List[List[int]],
    parcels: Sequence[Parcel],
    max_cols: int = 21,
) -> str:
    """Formats the 2D Knapsack DP table as an aligned text grid for reports/traces."""
    if not dp_table:
        return "(DP table not retained)"

    num_cols = len(dp_table[0])
    if num_cols > max_cols:
        return (
            f"(DP table has {len(dp_table)} rows x {num_cols} cols; "
            f"omitted from console print because capacity > {max_cols - 1})"
        )

    header = f"{'Item \\ Cap':<12}| " + " ".join(f"{w:>3}" for w in range(num_cols))
    separator = "-" * len(header)
    lines = [header, separator]

    for i, row in enumerate(dp_table):
        label = "0 (none)" if i == 0 else f"{parcels[i - 1].parcel_id}(w={parcels[i - 1].weight})"
        row_str = " ".join(f"{val:>3}" for val in row)
        lines.append(f"{label:<12}| {row_str}")

    return "\n".join(lines)
