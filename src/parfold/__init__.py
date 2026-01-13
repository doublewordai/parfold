"""
Parfold: Parallel async primitives for tree-based operations.

Provides fold, unfold, map, filter, sorting algorithms, and a BST
that execute async operations in parallel.

Usage:
    from parfold import fold, unfold, map, filter, quicksort, mergesort

    # Parallel tree reduction
    result = await fold(items, combine_fn)

    # Parallel tree expansion
    leaves = await unfold(seed, decompose_fn)

    # Parallel sorting with custom comparator
    sorted_items = await quicksort(items, compare_fn)

    # BST with LLM comparison
    from parfold import BST
    tree = BST(llm_compare)
    await asyncio.gather(*[tree.insert(x) for x in items])
"""

from .primitives import map, filter, fold, unfold
from .sort import quicksort, mergesort, CompareFunc
from .bst import BST, Node, CachedCompare

__version__ = "0.1.3"
__all__ = [
    "map",
    "filter",
    "fold",
    "unfold",
    "quicksort",
    "mergesort",
    "CompareFunc",
    "BST",
    "Node",
    "CachedCompare",
]
