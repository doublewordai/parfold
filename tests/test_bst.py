"""Tests for BST with async comparison."""

import asyncio
import random
import pytest
from parfold import BST, CachedCompare


async def int_compare(a: int, b: int) -> int:
    return a - b


class TestBST:
    @pytest.mark.asyncio
    async def test_empty_tree(self):
        tree = BST(int_compare)
        assert len(tree) == 0
        assert tree.to_list() == []
        assert tree.min is None
        assert tree.max is None

    @pytest.mark.asyncio
    async def test_single_insert(self):
        tree = BST(int_compare)
        await tree.insert(5)
        assert len(tree) == 1
        assert tree.to_list() == [5]
        assert tree.min == 5
        assert tree.max == 5

    @pytest.mark.asyncio
    async def test_sequential_inserts(self):
        tree = BST(int_compare)
        for x in [5, 3, 7, 1, 9]:
            await tree.insert(x)
        assert len(tree) == 5
        assert tree.to_list() == [1, 3, 5, 7, 9]
        assert tree.min == 1
        assert tree.max == 9

    @pytest.mark.asyncio
    async def test_parallel_inserts(self):
        tree = BST(int_compare)
        items = [5, 3, 7, 1, 9, 4, 6, 2, 8]
        await asyncio.gather(*[tree.insert(x) for x in items])
        assert len(tree) == 9
        assert tree.to_list() == sorted(items)

    @pytest.mark.asyncio
    async def test_parallel_inserts_large(self):
        tree = BST(int_compare)
        items = list(range(100))
        random.shuffle(items)
        await asyncio.gather(*[tree.insert(x) for x in items])
        assert len(tree) == 100
        assert tree.to_list() == sorted(items)

    @pytest.mark.asyncio
    async def test_iterator(self):
        tree = BST(int_compare)
        items = [5, 3, 7, 1, 9]
        for x in items:
            await tree.insert(x)
        assert list(tree) == [1, 3, 5, 7, 9]

    @pytest.mark.asyncio
    async def test_reversed_iterator(self):
        tree = BST(int_compare)
        items = [5, 3, 7, 1, 9]
        for x in items:
            await tree.insert(x)
        assert list(reversed(tree)) == [9, 7, 5, 3, 1]

    @pytest.mark.asyncio
    async def test_contains_found(self):
        tree = BST(int_compare)
        items = [5, 3, 7, 1, 9]
        for x in items:
            await tree.insert(x)
        for x in items:
            assert await tree.contains(x)

    @pytest.mark.asyncio
    async def test_contains_not_found(self):
        tree = BST(int_compare)
        for x in [5, 3, 7]:
            await tree.insert(x)
        assert not await tree.contains(1)
        assert not await tree.contains(10)

    @pytest.mark.asyncio
    async def test_contains_empty(self):
        tree = BST(int_compare)
        assert not await tree.contains(5)

    @pytest.mark.asyncio
    async def test_duplicates(self):
        tree = BST(int_compare)
        await asyncio.gather(*[tree.insert(5) for _ in range(5)])
        assert len(tree) == 5
        assert tree.to_list() == [5, 5, 5, 5, 5]

    @pytest.mark.asyncio
    async def test_sorted_input(self):
        tree = BST(int_compare)
        items = list(range(20))
        await asyncio.gather(*[tree.insert(x) for x in items])
        assert tree.to_list() == items

    @pytest.mark.asyncio
    async def test_reverse_sorted_input(self):
        tree = BST(int_compare)
        items = list(range(20, 0, -1))
        await asyncio.gather(*[tree.insert(x) for x in items])
        assert tree.to_list() == sorted(items)

    @pytest.mark.asyncio
    async def test_linked_list_integrity(self):
        """Verify linked list is properly threaded after parallel inserts."""
        tree = BST(int_compare)
        items = list(range(50))
        random.shuffle(items)
        await asyncio.gather(*[tree.insert(x) for x in items])

        # Forward traversal
        forward = list(tree)
        assert forward == sorted(items)

        # Backward traversal
        backward = list(reversed(tree))
        assert backward == sorted(items, reverse=True)

        # Verify consistency
        assert forward == list(reversed(backward))


class TestCachedCompare:
    @pytest.mark.asyncio
    async def test_caches_results(self):
        call_count = 0

        async def counting_compare(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            return a - b

        cached = CachedCompare(counting_compare)

        result1 = await cached(3, 5)
        assert result1 < 0
        assert call_count == 1
        assert cached.misses == 1
        assert cached.hits == 0

        result2 = await cached(3, 5)
        assert result2 == result1
        assert call_count == 1
        assert cached.hits == 1

    @pytest.mark.asyncio
    async def test_reverse_lookup(self):
        call_count = 0

        async def counting_compare(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            return a - b

        cached = CachedCompare(counting_compare)

        result1 = await cached(3, 5)
        assert result1 == -2
        assert call_count == 1

        result2 = await cached(5, 3)
        assert result2 == 2
        assert call_count == 1
        assert cached.hits == 1

    @pytest.mark.asyncio
    async def test_with_bst(self):
        call_count = 0

        async def counting_compare(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            return a - b

        cached = CachedCompare(counting_compare)
        tree = BST(cached)

        items = [5, 3, 7, 1, 9]
        for x in items:
            await tree.insert(x)

        assert tree.to_list() == sorted(items)
        assert cached.misses > 0
