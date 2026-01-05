"""Parallel primitives for async operations."""

from typing import TypeVar, Callable, Awaitable
import asyncio

T = TypeVar("T")
U = TypeVar("U")


async def map(
    items: list[T],
    f: Callable[[T], Awaitable[U]],
) -> list[U]:
    """
    Apply f to each item in parallel.

    Args:
        items: List of items to transform
        f: Async function to apply to each item

    Returns:
        List of transformed items (same order as input)
    """
    if not items:
        return []
    return list(await asyncio.gather(*[f(x) for x in items]))


async def filter(
    items: list[T],
    pred: Callable[[T], Awaitable[bool]],
) -> list[T]:
    """
    Keep items where pred returns True, evaluated in parallel.

    Args:
        items: List of items to filter
        pred: Async predicate function

    Returns:
        List of items where pred returned True (preserves order)
    """
    if not items:
        return []
    results = await asyncio.gather(*[pred(x) for x in items])
    return [items[i] for i in range(len(items)) if results[i]]


async def fold(
    items: list[T],
    combine: Callable[[T, T], Awaitable[T]],
) -> T:
    """
    Binary tree reduction with parallel execution.

    Achieves O(log n) depth by structuring the computation as a tree:
    pairs are combined in parallel at each level until a single result remains.

    Args:
        items: List of items to combine (must not be empty)
        combine: Async function to combine two items into one.
                 Should be approximately associative for consistent results.

    Returns:
        Single combined result

    Raises:
        ValueError: If items is empty
    """
    if len(items) == 0:
        raise ValueError("Cannot fold empty list")

    async def fold_tree(items: list[T]) -> T:
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return await combine(items[0], items[1])

        mid = len(items) // 2
        left_task = asyncio.create_task(fold_tree(items[:mid]))
        right_task = asyncio.create_task(fold_tree(items[mid:]))

        left_result, right_result = await asyncio.gather(left_task, right_task)
        return await combine(left_result, right_result)

    return await fold_tree(items)


async def unfold(
    seed: T,
    decompose: Callable[[T], Awaitable[list[T]]],
    max_depth: int = 100,
) -> list[T]:
    """
    Parallel tree expansion. Returns leaves in child order.

    Expands a seed value into a tree by repeatedly applying decompose,
    with all children at each level expanded in parallel.

    Args:
        seed: Initial value to decompose
        decompose: Async function returning children. Empty list = leaf node.
        max_depth: Safety limit for recursion depth

    Returns:
        List of leaf nodes in tree order (left-to-right)
    """
    async def unfold_node(item: T, depth: int) -> list[T]:
        if depth >= max_depth:
            return [item]

        children = await decompose(item)
        if not children:
            return [item]  # Leaf node

        child_tasks = [
            asyncio.create_task(unfold_node(child, depth + 1))
            for child in children
        ]
        results = await asyncio.gather(*child_tasks)
        return [leaf for leaves in results for leaf in leaves]

    return await unfold_node(seed, 0)
