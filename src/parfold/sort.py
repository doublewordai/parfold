"""Parallel sorting algorithms built on fold/unfold primitives."""

from typing import TypeVar, Callable, Awaitable
import asyncio
from .primitives import unfold

T = TypeVar("T")

# Compare function: returns negative if a < b, positive if a > b, 0 if equal
CompareFunc = Callable[[T, T], Awaitable[int]]


async def quicksort(
    items: list[T],
    compare: CompareFunc,
) -> list[T]:
    """
    Parallel quicksort using unfold.

    Parallelism:
    - All comparisons to pivot within a partition are parallel
    - Left and right subtrees are sorted in parallel

    Args:
        items: List to sort
        compare: Async comparison function (negative = a wins, positive = b wins)

    Returns:
        Sorted list (ascending order based on compare)
    """
    if len(items) <= 1:
        return list(items)

    async def partition(items: list[T]) -> list[list[T]]:
        if len(items) <= 1:
            return []  # Leaf

        # Pick middle element as pivot
        pivot_idx = len(items) // 2
        pivot = items[pivot_idx]
        others = items[:pivot_idx] + items[pivot_idx + 1:]

        # Parallel: compare all elements to pivot
        async def cmp_to_pivot(x: T) -> int:
            return await compare(x, pivot)

        comparisons = await asyncio.gather(*[cmp_to_pivot(x) for x in others])

        left = [others[i] for i in range(len(others)) if comparisons[i] < 0]
        right = [others[i] for i in range(len(others)) if comparisons[i] >= 0]

        # Return [left, pivot, right] - unfold preserves order
        return [left, [pivot], right]

    # Unfold returns list of single-element lists, flatten them
    result = await unfold(items, partition)
    return [x for sublist in result for x in sublist]


async def mergesort(
    items: list[T],
    compare: CompareFunc,
) -> list[T]:
    """
    Parallel mergesort with pairwise comparisons.

    Parallelism:
    - Left and right halves are sorted in parallel
    - Merge step is sequential (each comparison depends on previous)

    Args:
        items: List to sort
        compare: Async comparison function (negative = a wins, positive = b wins)

    Returns:
        Sorted list (ascending order based on compare)
    """
    if len(items) <= 1:
        return list(items)

    async def merge(left: list[T], right: list[T]) -> list[T]:
        result = []
        i, j = 0, 0

        while i < len(left) and j < len(right):
            cmp = await compare(left[i], right[j])
            if cmp <= 0:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    async def sort(items: list[T]) -> list[T]:
        if len(items) <= 1:
            return items

        mid = len(items) // 2
        left, right = await asyncio.gather(
            sort(items[:mid]),
            sort(items[mid:])
        )
        return await merge(left, right)

    return await sort(items)
