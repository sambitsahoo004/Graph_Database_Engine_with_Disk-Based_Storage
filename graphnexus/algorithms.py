"""Graph algorithms.

Every traversal here is iterative. The original used recursive DFS with the
interpreter recursion limit raised to 10,000,000, which does not raise the
underlying C stack and so segfaults on deep graphs rather than raising.

Each function takes a ``neighbors(node) -> list[(neighbor, weight)]`` callable
so the algorithms are independent of where the adjacency actually lives.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Callable, Iterable

NeighborFn = Callable[[int], list[tuple[int, int]]]


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


def tarjan_scc(num_nodes: int, neighbors: NeighborFn) -> list[int]:
    """Strongly connected components, iterative Tarjan.

    Returns a list mapping each node to its component id. Component ids are
    assigned in reverse topological order of the condensation.
    """
    UNVISITED = -1
    index = 0
    ids = [UNVISITED] * num_nodes
    low = [0] * num_nodes
    on_stack = [False] * num_nodes
    component = [UNVISITED] * num_nodes
    stack: list[int] = []
    num_components = 0

    for root in range(num_nodes):
        if ids[root] != UNVISITED:
            continue
        # Each frame is (node, iterator over its neighbours).
        work: list[tuple[int, Iterable[int]]] = [
            (root, iter([n for n, _ in neighbors(root)]))
        ]
        ids[root] = low[root] = index
        index += 1
        stack.append(root)
        on_stack[root] = True

        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt < 0 or nxt >= num_nodes:
                    continue
                if ids[nxt] == UNVISITED:
                    ids[nxt] = low[nxt] = index
                    index += 1
                    stack.append(nxt)
                    on_stack[nxt] = True
                    work.append((nxt, iter([n for n, _ in neighbors(nxt)])))
                    advanced = True
                    break
                if on_stack[nxt]:
                    low[node] = min(low[node], ids[nxt])
            if advanced:
                continue

            work.pop()
            if low[node] == ids[node]:
                while True:
                    top = stack.pop()
                    on_stack[top] = False
                    component[top] = num_components
                    if top == node:
                        break
                num_components += 1
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])

    return component


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def weakly_connected_components(
    num_nodes: int, neighbors: NeighborFn
) -> list[int]:
    """Weakly connected components via union-find.

    Union-find is used instead of a DFS over the undirected view: it needs one
    pass over the outgoing adjacency only, no stack proportional to graph
    depth, and no separate incoming adjacency scan.
    """
    uf = _UnionFind(num_nodes)
    for node in range(num_nodes):
        for nxt, _ in neighbors(node):
            if 0 <= nxt < num_nodes:
                uf.union(node, nxt)

    labels = [-1] * num_nodes
    next_label = 0
    for node in range(num_nodes):
        root = uf.find(node)
        if labels[root] == -1:
            labels[root] = next_label
            next_label += 1
        labels[node] = labels[root]
    return labels


def component_sizes(labels: list[int]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for label in labels:
        sizes[label] = sizes.get(label, 0) + 1
    return sizes


# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------


def pagerank(
    num_nodes: int,
    neighbors: NeighborFn,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> tuple[list[float], int]:
    """PageRank with dangling-node mass redistribution.

    A node with no outgoing edges is a rank sink. Without redistributing its
    mass the vector does not sum to 1 -- on the Gnutella sample the original
    implementation converged to a total of roughly 0.25. Here the mass held by
    dangling nodes each iteration is spread uniformly over all nodes, which is
    the standard formulation and keeps the total at 1.0.

    Returns ``(scores, iterations_run)``.
    """
    if num_nodes == 0:
        return [], 0

    initial = 1.0 / num_nodes
    scores = [initial] * num_nodes

    out_lists: list[list[int]] = [
        [n for n, _ in neighbors(node) if 0 <= n < num_nodes]
        for node in range(num_nodes)
    ]
    dangling = [node for node in range(num_nodes) if not out_lists[node]]
    teleport = (1.0 - damping) / num_nodes

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        dangling_mass = sum(scores[node] for node in dangling)
        base = teleport + damping * dangling_mass / num_nodes
        updated = [base] * num_nodes

        for node in range(num_nodes):
            targets = out_lists[node]
            if not targets:
                continue
            share = damping * scores[node] / len(targets)
            for target in targets:
                updated[target] += share

        delta = sum(abs(updated[i] - scores[i]) for i in range(num_nodes))
        scores = updated
        if delta < tolerance:
            break

    # Guard against float drift so the vector sums to exactly 1.
    total = sum(scores)
    if total > 0:
        scores = [s / total for s in scores]
    return scores, iterations


def rank_order(scores: list[float]) -> list[int]:
    """1-based rank per node, highest PageRank first. Ties break by node id."""
    order = sorted(range(len(scores)), key=lambda n: (-scores[n], n))
    ranks = [0] * len(scores)
    for position, node in enumerate(order, start=1):
        ranks[node] = position
    return ranks, order


# ---------------------------------------------------------------------------
# Cycle statistics
# ---------------------------------------------------------------------------


def cycle_statistics(
    num_nodes: int, neighbors: NeighborFn, scc_labels: list[int]
) -> dict[str, int | bool]:
    """Well-defined cycle measures.

    The original code advertised a count of *simple cycles*. Enumerating those
    is #P-hard and the number is astronomically large on any real graph -- what
    that code actually returned was a partial back-edge count, because its
    visited set was shared across the outer loop with no guard.

    This reports three quantities that are cheap, exact, and unambiguous:

    ``back_edges``        edges closing a cycle in a full DFS forest
    ``nodes_on_cycles``   nodes belonging to an SCC of size > 1, plus self-loops
    ``is_dag``            True when the graph has no directed cycle at all
    """
    sizes = component_sizes(scc_labels)
    nodes_on_cycles = sum(
        1 for node in range(num_nodes) if sizes[scc_labels[node]] > 1
    )

    self_loops = 0
    for node in range(num_nodes):
        for nxt, _ in neighbors(node):
            if nxt == node:
                self_loops += 1
    nodes_on_cycles += self_loops

    WHITE, GREY, BLACK = 0, 1, 2
    colour = [WHITE] * num_nodes
    back_edges = 0

    for root in range(num_nodes):
        if colour[root] != WHITE:
            continue
        colour[root] = GREY
        work: list[tuple[int, Iterable[int]]] = [
            (root, iter([n for n, _ in neighbors(root)]))
        ]
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt < 0 or nxt >= num_nodes:
                    continue
                if colour[nxt] == GREY:
                    back_edges += 1
                elif colour[nxt] == WHITE:
                    colour[nxt] = GREY
                    work.append((nxt, iter([n for n, _ in neighbors(nxt)])))
                    advanced = True
                    break
            if not advanced:
                colour[node] = BLACK
                work.pop()

    return {
        "back_edges": back_edges,
        "nodes_on_cycles": nodes_on_cycles,
        "is_dag": back_edges == 0,
    }


# ---------------------------------------------------------------------------
# Path queries
# ---------------------------------------------------------------------------


def bfs_distance(source: int, target: int, neighbors: NeighborFn) -> int:
    """Unweighted shortest distance, or -1 if unreachable.

    Nodes are marked visited when they are *enqueued*. The original marked on
    dequeue, so a node reachable by many paths entered the queue many times.
    """
    if source == target:
        return 0
    visited = {source}
    queue = deque([(source, 0)])
    while queue:
        node, distance = queue.popleft()
        for nxt, _ in neighbors(node):
            if nxt < 0 or nxt in visited:
                continue
            if nxt == target:
                return distance + 1
            visited.add(nxt)
            queue.append((nxt, distance + 1))
    return -1


def dijkstra_distance(source: int, target: int, neighbors: NeighborFn) -> int | float:
    """Weighted shortest distance, stopping as soon as ``target`` settles."""
    if source == target:
        return 0
    best: dict[int, int] = {source: 0}
    queue: list[tuple[int, int]] = [(0, source)]
    settled: set[int] = set()
    while queue:
        distance, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        if node == target:
            return distance
        for nxt, weight in neighbors(node):
            if nxt < 0 or nxt in settled:
                continue
            candidate = distance + weight
            if candidate < best.get(nxt, float("inf")):
                best[nxt] = candidate
                heapq.heappush(queue, (candidate, nxt))
    return float("inf")


def k_nearest(source: int, k: int, neighbors: NeighborFn) -> list[tuple[int, int]]:
    """The ``k`` nearest reachable nodes as ``(node, distance)``, closest first.

    The search stops once ``k`` nodes have settled instead of computing
    distances to every node in the graph and then sorting.
    """
    if k <= 0:
        return []
    best: dict[int, int] = {source: 0}
    queue: list[tuple[int, int]] = [(0, source)]
    found: list[tuple[int, int]] = []
    settled: set[int] = set()
    while queue and len(found) < k:
        distance, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        if node != source:
            found.append((node, distance))
            if len(found) == k:
                break
        for nxt, weight in neighbors(node):
            if nxt < 0 or nxt in settled:
                continue
            candidate = distance + weight
            if candidate < best.get(nxt, float("inf")):
                best[nxt] = candidate
                heapq.heappush(queue, (candidate, nxt))
    return found
