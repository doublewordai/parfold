"""
Parfold: Parallel async primitives for tree-based operations.

Provides fold, unfold, map, filter, and sorting algorithms that execute
async operations in parallel using tree-structured computation.

Usage:
    from parfold import fold, unfold, map, filter, quicksort, mergesort

    # Parallel tree reduction
    result = await fold(items, combine_fn)

    # Parallel tree expansion
    leaves = await unfold(seed, decompose_fn)

    # Parallel sorting with custom comparator
    sorted_items = await quicksort(items, compare_fn)
"""

from .primitives import map, filter, fold, unfold
from .sort import quicksort, mergesort, CompareFunc

__version__ = "0.1.1"
__all__ = [
    # Core primitives
    "map",
    "filter",
    "fold",
    "unfold",
    # Sorting (built on primitives)
    "quicksort",
    "mergesort",
    "CompareFunc",
]
