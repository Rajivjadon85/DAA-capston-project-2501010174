"""
Phase 2: Package Prioritization via Comparison Sorting Algorithms.

Primary algorithm for the Mini-Courier Planner:
    - Merge Sort: Deterministic O(n log n) worst-case time complexity and stable,
      preserving relative input order when two parcels share the same deadline/priority.

Additional algorithms included for Phase 4 empirical comparison:
    - Quick Sort (median-of-three pivot): Average O(n log n), in-place O(log n) stack.
    - Insertion Sort: O(n^2) worst-case, O(n) best-case on nearly sorted data.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, TypeVar

from src.models import Parcel

T = TypeVar("T")


def parcel_priority_key(parcel: Parcel) -> Tuple[float, int, int, str]:
    """Composite sort key for parcel dispatch prioritization.

    Ordering rules:
        1. Earlier delivery deadline first (ascending deadline_hours).
        2. Higher urgency/priority first on deadline ties (descending priority).
        3. Lighter weight first on equal deadline & priority (ascending weight).
        4. Lexicographical parcel_id for deterministic tie-breaking.
    """
    return (parcel.deadline_hours, -parcel.priority, parcel.weight, parcel.parcel_id)


def merge_sort(
    items: Sequence[T],
    key: Callable[[T], object] = lambda x: x,
) -> List[T]:
    """Sorts a sequence using bottom-up / top-down Merge Sort (stable, O(n log n)).

    Time Complexity:
        Best / Average / Worst: Theta(n log n)
    Space Complexity:
        O(n) auxiliary space for temporary merge buffers.
    """
    n = len(items)
    if n <= 1:
        return list(items)

    mid = n // 2
    left = merge_sort(items[:mid], key=key)
    right = merge_sort(items[mid:], key=key)
    return _merge(left, right, key)


def _merge(
    left: List[T],
    right: List[T],
    key: Callable[[T], object],
) -> List[T]:
    """Merges two sorted lists stably (<= keeps left element first on ties)."""
    merged: List[T] = []
    i = 0
    j = 0
    len_left = len(left)
    len_right = len(right)

    while i < len_left and j < len_right:
        if key(left[i]) <= key(right[j]):  # '<=' guarantees stability
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    if i < len_left:
        merged.extend(left[i:])
    if j < len_right:
        merged.extend(right[j:])
    return merged


def quick_sort(
    items: Sequence[T],
    key: Callable[[T], object] = lambda x: x,
) -> List[T]:
    """Sorts a sequence using Quick Sort with median-of-three pivot selection.

    Using median-of-three prevents O(n^2) degradation on already-sorted or
    reverse-sorted parcel lists.

    Time Complexity:
        Best / Average: O(n log n)
        Worst: O(n^2) (rare with median-of-three)
    Space Complexity:
        O(log n) average recursion stack (+ O(n) output copy to keep input immutable).
    """
    arr = list(items)
    if len(arr) <= 1:
        return arr

    # Precompute keys once per item to avoid repeated key() calls during partitioning.
    decorated = [(key(item), idx, item) for idx, item in enumerate(arr)]
    _quick_sort_inplace(decorated, 0, len(decorated) - 1)
    return [entry[2] for entry in decorated]


def _median_of_three_index(arr: List[Tuple[object, int, T]], low: int, high: int) -> int:
    """Selects the median of (low, mid, high) keys to avoid worst-case pivots."""
    mid = (low + high) // 2
    a, b, c = arr[low][0], arr[mid][0], arr[high][0]
    if (a <= b <= c) or (c <= b <= a):
        return mid
    if (b <= a <= c) or (c <= a <= b):
        return low
    return high


def _quick_sort_inplace(
    arr: List[Tuple[object, int, T]],
    low: int,
    high: int,
) -> None:
    """In-place Hoare/Lomuto partition recursion using tail-call loop optimization."""
    while low < high:
        pivot_idx = _median_of_three_index(arr, low, high)
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        pivot_key = arr[high][0]

        i = low
        for j in range(low, high):
            if arr[j][0] <= pivot_key:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[high] = arr[high], arr[i]

        # Recurse on the smaller partition first to bound stack depth to O(log n).
        if (i - low) < (high - i):
            _quick_sort_inplace(arr, low, i - 1)
            low = i + 1
        else:
            _quick_sort_inplace(arr, i + 1, high)
            high = i - 1


def insertion_sort(
    items: Sequence[T],
    key: Callable[[T], object] = lambda x: x,
) -> List[T]:
    """Sorts a sequence using Insertion Sort (stable, O(n^2) worst-case).

    Included to demonstrate quadratic growth vs O(n log n) Merge/Quick Sort
    in Phase 4 performance benchmarks.
    """
    arr = list(items)
    for i in range(1, len(arr)):
        current = arr[i]
        current_key = key(current)
        j = i - 1
        while j >= 0 and key(arr[j]) > current_key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = current
    return arr


def prioritize_parcels(
    parcels: Sequence[Parcel],
    algorithm: str = "merge",
) -> List[Parcel]:
    """Prioritizes parcels for delivery using the chosen sorting algorithm."""
    algorithms = {
        "merge": merge_sort,
        "quick": quick_sort,
        "insertion": insertion_sort,
    }
    sorter = algorithms.get(algorithm.lower())
    if sorter is None:
        raise ValueError(
            f"Unsupported sorting algorithm '{algorithm}'. Choose from {sorted(algorithms)}."
        )
    return sorter(parcels, key=parcel_priority_key)
