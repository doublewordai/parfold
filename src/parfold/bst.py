"""
Binary Search Tree with async comparison and parallel inserts.

Uses optimistic concurrency control for lock-free parallel operations,
designed for expensive comparison functions like LLM calls.
"""

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Awaitable, Iterator
from dataclasses import dataclass
import asyncio

T = TypeVar("T")

CompareFunc = Callable[[T, T], Awaitable[int]]
EvictFunc = Callable[[T], Awaitable[None]]


@dataclass
class Node(Generic[T]):
    """BST node with linked list threading for O(1) sorted traversal."""
    value: T
    left: Node[T] | None = None
    right: Node[T] | None = None
    prev: Node[T] | None = None  # in-order predecessor
    next: Node[T] | None = None  # in-order successor
    version: int = 0


class BST(Generic[T]):
    """
    Lock-free BST with async comparison and O(1) sorted access.

    Maintains a threaded linked list through nodes for cheap traversal.
    Uses optimistic concurrency for parallel inserts.

    Example:
        async def llm_compare(a: str, b: str) -> int:
            response = await llm.compare(a, b)
            return response  # -1, 0, or 1

        tree = BST(llm_compare)
        await asyncio.gather(*[tree.insert(item) for item in items])

        # O(1) access to sorted items
        for item in tree:
            print(item)
    """

    def __init__(
        self,
        compare: CompareFunc[T],
        max_retries: int = 100,
        max_size: int | None = None,
        on_evict: EvictFunc[T] | None = None,
    ):
        self._compare = compare
        self._root: Node[T] | None = None
        self._head: Node[T] | None = None  # smallest element
        self._tail: Node[T] | None = None  # largest element
        self._max_retries = max_retries
        self._max_size = max_size
        self._on_evict = on_evict
        self._link_lock = asyncio.Lock()  # Protects tree modification and linked list operations
        self._size = 0

    async def insert(self, value: T) -> None:
        """Insert value. Safe to call concurrently."""
        # Handle empty tree case
        if self._root is None:
            async with self._link_lock:
                if self._root is None:
                    node = Node(value)
                    self._root = node
                    self._head = node
                    self._tail = node
                    self._size = 1
                    await self._maybe_evict()
                    return

        retries = 0

        while retries < self._max_retries:
            # Phase 1: Traverse tree to find insertion point (parallel, no lock)
            node = self._root
            parent: Node[T] | None = None
            go_left = False

            while node is not None:
                saved_version = node.version
                saved_left = node.left
                saved_right = node.right

                cmp = await self._compare(value, node.value)

                # Check if tree changed during comparison
                if node.version != saved_version:
                    break  # Restart traversal

                parent = node
                if cmp < 0:
                    go_left = True
                    node = saved_left
                else:
                    go_left = False
                    node = saved_right
            else:
                # Phase 2: Link new node (serialized with lock)
                async with self._link_lock:
                    # Re-verify insertion point is still valid
                    if parent is None:
                        # Tree became empty, shouldn't happen but handle it
                        new_node = Node(value)
                        self._root = new_node
                        self._head = new_node
                        self._tail = new_node
                        self._size = 1
                        await self._maybe_evict()
                        return

                    if go_left:
                        if parent.left is not None:
                            # Slot taken, retry
                            retries += 1
                            continue
                        new_node = Node(value)
                        parent.left = new_node
                        parent.version += 1
                        # Link: new_node goes before parent in sorted order
                        new_node.next = parent
                        new_node.prev = parent.prev
                        if parent.prev:
                            parent.prev.next = new_node
                        else:
                            self._head = new_node
                        parent.prev = new_node
                    else:
                        if parent.right is not None:
                            # Slot taken, retry
                            retries += 1
                            continue
                        new_node = Node(value)
                        parent.right = new_node
                        parent.version += 1
                        # Link: new_node goes after parent in sorted order
                        new_node.prev = parent
                        new_node.next = parent.next
                        if parent.next:
                            parent.next.prev = new_node
                        else:
                            self._tail = new_node
                        parent.next = new_node

                    self._size += 1
                    await self._maybe_evict()
                    return

            retries += 1

        raise RuntimeError(f"Insert failed after {self._max_retries} retries")

    async def _maybe_evict(self) -> None:
        """Evict lowest-ranked items if over max_size."""
        if self._max_size is None or self._size <= self._max_size:
            return

        while self._size > self._max_size and self._head:
            evicted = self._head.value
            # Unlink from list (leave orphaned in tree - harmless for correctness)
            old_head = self._head
            self._head = old_head.next
            if self._head:
                self._head.prev = None
            else:
                self._tail = None  # Tree is now empty
            old_head.next = None  # Help GC
            self._size -= 1

            if self._on_evict:
                await self._on_evict(evicted)

    async def contains(self, value: T) -> bool:
        """Check if value exists in tree."""
        node = self._root
        while node is not None:
            cmp = await self._compare(value, node.value)
            if cmp == 0:
                return True
            node = node.left if cmp < 0 else node.right
        return False

    def __iter__(self) -> Iterator[T]:
        """Iterate in sorted order via linked list. O(1) to start."""
        node = self._head
        while node is not None:
            yield node.value
            node = node.next

    def __reversed__(self) -> Iterator[T]:
        """Iterate in reverse sorted order."""
        node = self._tail
        while node is not None:
            yield node.value
            node = node.prev

    def to_list(self) -> list[T]:
        """Return sorted list. O(n) but no comparisons needed."""
        return list(self)

    def __len__(self) -> int:
        return self._size

    @property
    def min(self) -> T | None:
        """Smallest element. O(1)."""
        return self._head.value if self._head else None

    @property
    def max(self) -> T | None:
        """Largest element. O(1)."""
        return self._tail.value if self._tail else None


class CachedCompare(Generic[T]):
    """
    Caches async comparison results.

    Handles both (a,b) and (b,a) lookups.

    Example:
        cached = CachedCompare(llm_compare)
        tree = BST(cached)
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
