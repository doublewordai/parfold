"""
Binary Search Tree with async comparison and parallel inserts.

Uses optimistic concurrency control for lock-free parallel operations,
designed for expensive comparison functions like LLM calls.
"""

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Awaitable
from dataclasses import dataclass
import asyncio

T = TypeVar("T")

CompareFunc = Callable[[T, T], Awaitable[int]]


@dataclass
class Node(Generic[T]):
    """BST node with version number for optimistic concurrency."""
    value: T
    left: Node[T] | None = None
    right: Node[T] | None = None
    version: int = 0


class BST(Generic[T]):
    """
    Lock-free Binary Search Tree with async comparison.

    Uses optimistic concurrency control:
    1. Read current state and version
    2. Perform expensive async comparison
    3. Attempt update, retry if version changed

    Designed for LLM-based comparisons where the comparison cost
    dominates and lock contention would be the bottleneck.

    Example:
        async def llm_compare(a: str, b: str) -> int:
            response = await llm.compare(a, b)
            return response  # -1, 0, or 1

        tree = BST(llm_compare)
        await asyncio.gather(*[tree.insert(item) for item in items])
        sorted_items = await tree.to_list()
    """

    def __init__(self, compare: CompareFunc[T], max_retries: int = 100):
        """
        Args:
            compare: Async function returning negative if a < b,
                     positive if a > b, zero if equal.
            max_retries: Maximum retry attempts on conflict.
        """
        self._compare = compare
        self._root: Node[T] | None = None
        self._max_retries = max_retries
        self._root_lock = asyncio.Lock()

    async def insert(self, value: T) -> None:
        """
        Insert value into tree. Safe to call concurrently.

        Uses optimistic concurrency: retries from conflict point
        if another insert modified the traversal path.
        """
        if self._root is None:
            async with self._root_lock:
                if self._root is None:
                    self._root = Node(value)
                    return

        retries = 0
        node = self._root
        parent: Node[T] | None = None
        go_left = False

        while retries < self._max_retries:
            if node is None:
                new_node = Node(value)
                if parent is None:
                    self._root = new_node
                    return

                expected_version = parent.version
                if go_left:
                    if parent.left is None and parent.version == expected_version:
                        parent.left = new_node
                        parent.version += 1
                        return
                else:
                    if parent.right is None and parent.version == expected_version:
                        parent.right = new_node
                        parent.version += 1
                        return

                retries += 1
                node = self._root
                parent = None
                continue

            saved_version = node.version
            saved_left = node.left
            saved_right = node.right

            cmp = await self._compare(value, node.value)

            if node.version != saved_version:
                retries += 1
                continue

            parent = node
            if cmp < 0:
                go_left = True
                node = saved_left
            else:
                go_left = False
                node = saved_right

        raise RuntimeError(f"Insert failed after {self._max_retries} retries")

    async def contains(self, value: T) -> bool:
        """Check if value exists in tree."""
        node = self._root
        while node is not None:
            cmp = await self._compare(value, node.value)
            if cmp == 0:
                return True
            node = node.left if cmp < 0 else node.right
        return False

    async def to_list(self) -> list[T]:
        """Return in-order traversal as list."""
        result: list[T] = []

        def inorder(node: Node[T] | None) -> None:
            if node is None:
                return
            inorder(node.left)
            result.append(node.value)
            inorder(node.right)

        inorder(self._root)
        return result

    def __len__(self) -> int:
        """Count nodes."""
        def count(node: Node[T] | None) -> int:
            if node is None:
                return 0
            return 1 + count(node.left) + count(node.right)
        return count(self._root)


class CachedCompare(Generic[T]):
    """
    Caches async comparison results.

    Useful when LLM comparisons are expensive and may be repeated.
    Handles both (a,b) and (b,a) lookups.

    Example:
        cached = CachedCompare(llm_compare)
        tree = BST(cached)
        # ... operations ...
        print(f"Cache: {cached.hits} hits, {cached.misses} misses")
    """

    def __init__(self, compare: CompareFunc[T]):
        self._compare = compare
        self._cache: dict[tuple[int, int], int] = {}
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def __call__(self, a: T, b: T) -> int:
        key = (id(a), id(b))
        rev_key = (id(b), id(a))

        async with self._lock:
            if key in self._cache:
                self.hits += 1
                return self._cache[key]
            if rev_key in self._cache:
                self.hits += 1
                return -self._cache[rev_key]

        result = await self._compare(a, b)

        async with self._lock:
            self._cache[key] = result
            self.misses += 1

        return result
