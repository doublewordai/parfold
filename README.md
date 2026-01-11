# parfold

Parallel async primitives for tree-based operations: `fold`, `unfold`, `map`, `filter`, and sorting.

## Why?

When you need to process many items with async functions (like LLM calls), sequential execution is slow. `parfold` provides tree-structured parallelism that achieves O(log n) depth instead of O(n).

```python
# Sequential: O(n) round trips
result = items[0]
for item in items[1:]:
    result = await combine(result, item)

# Parallel fold: O(log n) round trips
from parfold import fold
result = await fold(items, combine)
```

## Installation

```bash
pip install parfold
```

## Primitives

### `fold` — parallel tree reduction

Combines a list into a single result using parallel binary tree reduction.

```python
from parfold import fold

async def combine(a: str, b: str) -> str:
    # Your async combining logic (e.g., LLM call)
    return await summarize_together(a, b)

chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
summary = await fold(chunks, combine)
```

**How it works:** Instead of combining left-to-right, `fold` pairs items and combines each pair in parallel, then pairs the results, and so on. For 8 items, that's 3 levels of parallelism instead of 7 sequential operations.

### `unfold` — parallel tree expansion

Expands a seed into leaves by recursively decomposing, with all children expanded in parallel.

```python
from parfold import unfold

async def decompose(query: str) -> list[str]:
    if is_specific_enough(query):
        return []  # Leaf node
    return await generate_subqueries(query)

specific_queries = await unfold("broad research question", decompose)
```

### `map` — parallel transform

Applies an async function to each item in parallel.

```python
from parfold import map

results = await map(items, async_transform)
```

### `filter` — parallel predicate

Keeps items where the async predicate returns True.

```python
from parfold import filter

relevant = await filter(items, async_is_relevant)
```

## Sorting

Sorting algorithms using async comparison functions:

```python
from parfold import quicksort, mergesort

async def compare(a, b) -> int:
    # Return negative if a < b, positive if a > b, 0 if equal
    return await llm_compare(a, b)

sorted_items = await quicksort(items, compare)
# or
sorted_items = await mergesort(items, compare)
```

**Quicksort** parallelizes comparisons to the pivot within each partition.
**Mergesort** parallelizes the recursive sorting of left/right halves.

## BST — Binary Search Tree

Lock-free BST with parallel inserts and O(1) sorted traversal.

```python
from parfold import BST

async def compare(a: str, b: str) -> int:
    return await llm_compare(a, b)

tree = BST(compare)

# Parallel inserts
await asyncio.gather(*[tree.insert(x) for x in items])

# O(1) access to sorted order (no comparisons needed)
for item in tree:
    print(item)

# O(1) min/max
print(tree.min, tree.max)
```

Uses optimistic concurrency control for parallel inserts. Maintains a threaded linked list through nodes for cheap traversal.

| Operation | Comparisons | Time |
|-----------|-------------|------|
| `insert()` | O(log n) | O(1) pointer ops |
| `min`/`max` | 0 | O(1) |
| `for x in tree` | 0 | O(n) |
| `contains()` | O(log n) | — |

### CachedCompare

Wrap your comparison function to cache results:

```python
from parfold import BST, CachedCompare

cached = CachedCompare(llm_compare)
tree = BST(cached)

# After operations:
print(f"Cache: {cached.hits} hits, {cached.misses} misses")
```

## Use Cases

- **Summarization**: Fold document chunks into a single summary
- **Search**: Fold chunks while filtering by relevance to a query
- **Research expansion**: Unfold a broad question into specific searches
- **Ranking**: Sort items using LLM-based comparison
- **Clustering**: Fold items into groups using LLM-based merging

## Requirements

- Python 3.10+
- No dependencies (just `asyncio`)

## License

MIT
