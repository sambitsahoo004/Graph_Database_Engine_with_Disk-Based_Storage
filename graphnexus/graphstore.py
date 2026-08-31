"""Build and query a block-paged graph.

On-disk layout for one graph, all paths relative to the graph directory:

    source.txt   the uploaded edge list, kept verbatim
    meta.json    graph-level statistics
    nodes.dat    node table, 64-byte records, 8 per block
    adj_out.dat  outgoing adjacency, compressed-sparse-row style
    adj_in.dat   incoming adjacency, same layout
    rank.dat     node ids ordered by descending PageRank
    edge.hash    static hash index over edges, linear probing
    rank.btree   B+-tree ordered index over (pagerank, node_id)

Adjacency is stored as one contiguous run per node, so a node's neighbour list
is a sequential read from ``out_start`` rather than a pointer chase. ``rank.dat``
is a sorted array addressed positionally, which makes a rank-range query a
sequential scan of ``(r2 - r1) / 128`` blocks.

Two indexes sit alongside those heap files and are described in ``indexes.py``:
a static hash index that resolves an edge without scanning an adjacency run,
and a B+-tree that resolves a PageRank *score* range, which the positional
rank list cannot do. Both are read through the same buffer pool, so their cost
appears in the same block counters.

The build phase works in memory and writes each heap file sequentially, the
same way a bulk loader does. Query paths read through the buffer pool, which is
where the reported block-access figures come from.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field

from . import algorithms
from .edgelist import ParsedGraph, parse_edge_list
from .indexes import (
    EdgeHashIndex,
    PageRankBTree,
    build_edge_hash_index,
    build_pagerank_btree,
)
from .storage import (
    BLOCK_SIZE,
    INTS_PER_BLOCK,
    BlockFile,
    BufferPool,
    IOStats,
    NodeRecord,
    NodeTable,
    StorageError,
)

META_FILENAME = "meta.json"
SOURCE_FILENAME = "source.txt"
NODES_FILE = "nodes.dat"
ADJ_OUT_FILE = "adj_out.dat"
ADJ_IN_FILE = "adj_in.dat"
RANK_FILE = "rank.dat"
EDGE_HASH_FILE = "edge.hash"
RANK_BTREE_FILE = "rank.btree"

SAFE_NAME_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)


class GraphError(Exception):
    """Raised for invalid graph names, missing graphs, or bad queries."""


def sanitize_graph_name(name: str) -> str:
    """Reduce a user-supplied name to a single safe path segment."""
    base = os.path.basename(str(name)).strip()
    if base.lower().endswith((".txt", ".csv", ".edges", ".tsv")):
        base = base.rsplit(".", 1)[0]
    cleaned = "".join(c for c in base if c in SAFE_NAME_CHARS).strip("._-")
    if not cleaned:
        raise GraphError("graph name must contain letters, digits, - or _")
    return cleaned[:64]


@dataclass
class GraphMeta:
    name: str
    nodes: int
    edges: int
    weighted: bool
    num_scc: int = 0
    largest_scc: int = 0
    num_wcc: int = 0
    largest_wcc: int = 0
    back_edges: int = 0
    nodes_on_cycles: int = 0
    is_dag: bool = False
    pagerank_iterations: int = 0
    pagerank_total: float = 0.0
    hash_capacity: int = 0
    hash_longest_probe: int = 0
    btree_height: int = 0
    btree_blocks: int = 0
    build_ms: int = 0
    on_disk_bytes: int = 0
    total_blocks: int = 0
    had_header: bool = False
    self_loops_removed: int = 0
    duplicates_removed: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResult:
    """A query answer paired with what it cost."""

    value: object
    elapsed_ms: float
    stats: IOStats

    @property
    def blocks_read(self) -> int:
        return self.stats.physical_reads

    @property
    def hit_ratio(self) -> float:
        return self.stats.hit_ratio


class GraphStore:
    """Read access to one built graph."""

    def __init__(self, directory: str, buffer_blocks: int = 256):
        self.directory = os.path.abspath(directory)
        if not os.path.isdir(self.directory):
            raise GraphError(f"graph directory not found: {self.directory}")
        meta_path = os.path.join(self.directory, META_FILENAME)
        if not os.path.exists(meta_path):
            raise GraphError(
                f"{os.path.basename(self.directory)} has not been built yet"
            )
        with open(meta_path, encoding="utf-8") as handle:
            self.meta = GraphMeta(**json.load(handle))

        self.pool = BufferPool(capacity=buffer_blocks)
        self.nodes_file = BlockFile(os.path.join(self.directory, NODES_FILE), self.pool)
        self.adj_out = BlockFile(os.path.join(self.directory, ADJ_OUT_FILE), self.pool)
        self.adj_in = BlockFile(os.path.join(self.directory, ADJ_IN_FILE), self.pool)
        self.rank_file = BlockFile(os.path.join(self.directory, RANK_FILE), self.pool)
        self.node_table = NodeTable(self.nodes_file)
        self.stride = 2 if self.meta.weighted else 1

        # Graphs built before the indexes existed are still readable; only the
        # two index-backed queries are unavailable, and they say so.
        self.edge_hash: EdgeHashIndex | None = None
        self.rank_btree: PageRankBTree | None = None
        self._hash_file: BlockFile | None = None
        self._btree_file: BlockFile | None = None

        hash_path = os.path.join(self.directory, EDGE_HASH_FILE)
        if os.path.exists(hash_path):
            self._hash_file = BlockFile(hash_path, self.pool)
            self.edge_hash = EdgeHashIndex(self._hash_file)

        btree_path = os.path.join(self.directory, RANK_BTREE_FILE)
        if os.path.exists(btree_path):
            self._btree_file = BlockFile(btree_path, self.pool)
            self.rank_btree = PageRankBTree(self._btree_file)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        handles = [self.nodes_file, self.adj_out, self.adj_in, self.rank_file]
        handles.extend(h for h in (self._hash_file, self._btree_file) if h)
        for handle in handles:
            handle.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- validation --------------------------------------------------------

    def check_node(self, node: int) -> int:
        """Validate a node id. Out-of-range ids raise instead of touching disk.

        The original built a filename from any integer and opened it with mode
        ``x``, so querying a node that did not exist created an empty block file
        as a side effect.
        """
        try:
            node = int(node)
        except (TypeError, ValueError):
            raise GraphError("node id must be an integer") from None
        if node < 0 or node >= self.meta.nodes:
            raise GraphError(
                f"node {node} is out of range: this graph has ids 0 to "
                f"{self.meta.nodes - 1}"
            )
        return node

    # -- adjacency ---------------------------------------------------------

    def out_neighbors(self, node: int) -> list[tuple[int, int]]:
        record = self.node_table.read(node)
        raw = self.adj_out.read_ints(record.out_start, record.out_deg * self.stride)
        if self.stride == 1:
            return [(n, 1) for n in raw]
        return [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]

    def in_neighbors(self, node: int) -> list[tuple[int, int]]:
        record = self.node_table.read(node)
        raw = self.adj_in.read_ints(record.in_start, record.in_deg * self.stride)
        if self.stride == 1:
            return [(n, 1) for n in raw]
        return [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]

    # -- measured queries --------------------------------------------------

    def _measure(self, fn, *args, **kwargs) -> QueryResult:
        self.pool.reset_stats()
        start = time.perf_counter()
        value = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0
        return QueryResult(value=value, elapsed_ms=elapsed, stats=self.pool.reset_stats())

    def in_degree(self, node: int) -> QueryResult:
        node = self.check_node(node)
        return self._measure(lambda: self.node_table.read(node).in_deg)

    def out_degree(self, node: int) -> QueryResult:
        node = self.check_node(node)
        return self._measure(lambda: self.node_table.read(node).out_deg)

    def pagerank_of(self, node: int) -> QueryResult:
        node = self.check_node(node)

        def run():
            record = self.node_table.read(node)
            return {"pagerank": record.pagerank, "rank": record.rank}

        return self._measure(run)

    def same_components(self, a: int, b: int) -> QueryResult:
        a, b = self.check_node(a), self.check_node(b)

        def run():
            ra, rb = self.node_table.read(a), self.node_table.read(b)
            return {
                "same_scc": ra.scc_id == rb.scc_id,
                "same_wcc": ra.wcc_id == rb.wcc_id,
                "scc_a": ra.scc_id,
                "scc_b": rb.scc_id,
                "wcc_a": ra.wcc_id,
                "wcc_b": rb.wcc_id,
            }

        return self._measure(run)

    def shortest_distance(self, source: int, target: int) -> QueryResult:
        source, target = self.check_node(source), self.check_node(target)
        if self.meta.weighted:
            fn = algorithms.dijkstra_distance
        else:
            fn = algorithms.bfs_distance

        def run():
            value = fn(source, target, self.out_neighbors)
            return -1 if value == float("inf") else value

        return self._measure(run)

    def knn(self, source: int, k: int) -> QueryResult:
        source = self.check_node(source)
        try:
            k = int(k)
        except (TypeError, ValueError):
            raise GraphError("k must be an integer") from None
        if k < 1:
            raise GraphError("k must be at least 1")
        if k > self.meta.nodes:
            k = self.meta.nodes
        return self._measure(algorithms.k_nearest, source, k, self.out_neighbors)

    def rank_list(self, first: int, last: int) -> QueryResult:
        try:
            first, last = int(first), int(last)
        except (TypeError, ValueError):
            raise GraphError("rank bounds must be integers") from None
        if first < 1:
            raise GraphError("ranks start at 1")
        if last < first:
            raise GraphError("the end of the range must not precede the start")
        if first > self.meta.nodes:
            raise GraphError(
                f"rank {first} is beyond the last rank ({self.meta.nodes})"
            )
        last = min(last, self.meta.nodes)
        if last - first + 1 > 10_000:
            raise GraphError("rank ranges are limited to 10,000 entries per query")

        def run():
            ids = self.rank_file.read_ints(first - 1, last - first + 1)
            rows = []
            for offset, node_id in enumerate(ids):
                record = self.node_table.read(node_id)
                rows.append(
                    {
                        "rank": first + offset,
                        "node": node_id,
                        "pagerank": record.pagerank,
                    }
                )
            return rows

        return self._measure(run)

    # -- index-backed queries ---------------------------------------------

    def has_edge(self, source: int, target: int) -> QueryResult:
        """Edge existence through the hash index.

        One bucket-block read plus whatever the probe sequence crosses, with no
        dependence on the degree of ``source``.
        """
        source, target = self.check_node(source), self.check_node(target)
        if self.edge_hash is None:
            raise GraphError(
                "this graph was built before the edge hash index existed; "
                "rebuild it to use this query"
            )

        def run():
            found, weight, probes = self.edge_hash.lookup(source, target)
            return {"exists": found, "weight": weight, "probes": probes}

        return self._measure(run)

    def has_edge_by_scan(self, source: int, target: int) -> QueryResult:
        """The same answer without the index, for comparison.

        Reads the node record and then scans the whole outgoing run, so its
        cost grows with the out-degree of ``source``.
        """
        source, target = self.check_node(source), self.check_node(target)

        def run():
            for neighbor, weight in self.out_neighbors(source):
                if neighbor == target:
                    return {"exists": True, "weight": weight}
            return {"exists": False, "weight": 0}

        return self._measure(run)

    def nodes_by_score(
        self, low: float, high: float, limit: int = 1000
    ) -> QueryResult:
        """Nodes whose PageRank falls in ``[low, high]``, via the B+-tree.

        ``rank.dat`` cannot answer this: it is ordered by rank position and
        holds node ids, not scores, so searching it by score would mean a node
        record read per probe.
        """
        if self.rank_btree is None:
            raise GraphError(
                "this graph was built before the PageRank B+-tree existed; "
                "rebuild it to use this query"
            )
        try:
            low, high = float(low), float(high)
        except (TypeError, ValueError):
            raise GraphError("score bounds must be numbers") from None
        if high < low:
            raise GraphError("the upper bound must not be below the lower bound")
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise GraphError("limit must be an integer") from None
        limit = max(1, min(limit, 10_000))

        def run():
            matches = self.rank_btree.range_search(low, high, limit=limit)
            return [
                {"node": node_id, "pagerank": score}
                for node_id, score in reversed(matches)
            ]

        return self._measure(run)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _write_int_heap(path: str, values: list[int], pool: BufferPool) -> None:
    with BlockFile(path, pool, create=True) as handle:
        handle.append_ints(values if values else [0] * INTS_PER_BLOCK)


def build_graph(
    source_path: str,
    graph_dir: str,
    weighted: bool = False,
    damping: float = 0.85,
    progress=None,
) -> GraphMeta:
    """Parse an edge list and write the full on-disk representation."""
    started = time.perf_counter()
    name = os.path.basename(os.path.normpath(graph_dir))

    def report(message: str) -> None:
        if progress:
            progress(message)

    report("parsing edge list")
    parsed: ParsedGraph = parse_edge_list(source_path, weighted=weighted)
    num_nodes = parsed.num_nodes
    stride = 2 if weighted else 1

    warnings: list[str] = []
    if not parsed.had_header:
        warnings.append(
            "No <nodes> <edges> header line was found, so the node count was "
            "inferred from the largest node id."
        )
    if parsed.self_loops_removed:
        warnings.append(f"Removed {parsed.self_loops_removed:,} self-loops.")
    if parsed.duplicates_removed:
        warnings.append(f"Removed {parsed.duplicates_removed:,} duplicate edges.")

    # --- degrees and CSR offsets -----------------------------------------
    report("computing degrees")
    out_deg = [0] * num_nodes
    in_deg = [0] * num_nodes
    for u, v, _ in parsed.edges:
        out_deg[u] += 1
        in_deg[v] += 1

    out_start = [0] * num_nodes
    in_start = [0] * num_nodes
    running_out = running_in = 0
    for node in range(num_nodes):
        out_start[node] = running_out
        in_start[node] = running_in
        running_out += out_deg[node] * stride
        running_in += in_deg[node] * stride

    report("laying out adjacency")
    adj_out = [0] * running_out
    adj_in = [0] * running_in
    cursor_out = list(out_start)
    cursor_in = list(in_start)
    for u, v, w in parsed.edges:
        pos = cursor_out[u]
        adj_out[pos] = v
        if stride == 2:
            adj_out[pos + 1] = w
        cursor_out[u] += stride

        pos = cursor_in[v]
        adj_in[pos] = u
        if stride == 2:
            adj_in[pos + 1] = w
        cursor_in[v] += stride

    # In-memory neighbour view, used only during the build.
    def neighbors(node: int) -> list[tuple[int, int]]:
        start = out_start[node]
        end = start + out_deg[node] * stride
        if stride == 1:
            return [(adj_out[i], 1) for i in range(start, end)]
        return [(adj_out[i], adj_out[i + 1]) for i in range(start, end, 2)]

    # --- analytics --------------------------------------------------------
    report("finding strongly connected components")
    scc_labels = algorithms.tarjan_scc(num_nodes, neighbors)
    scc_sizes = algorithms.component_sizes(scc_labels)

    report("finding weakly connected components")
    wcc_labels = algorithms.weakly_connected_components(num_nodes, neighbors)
    wcc_sizes = algorithms.component_sizes(wcc_labels)

    report("computing PageRank")
    scores, iterations = algorithms.pagerank(num_nodes, neighbors, damping=damping)
    ranks, order = algorithms.rank_order(scores)

    report("measuring cycles")
    cycles = algorithms.cycle_statistics(num_nodes, neighbors, scc_labels)

    # --- write heap files -------------------------------------------------
    report("writing heap files")
    os.makedirs(graph_dir, exist_ok=True)
    for filename in (
        NODES_FILE,
        ADJ_OUT_FILE,
        ADJ_IN_FILE,
        RANK_FILE,
        EDGE_HASH_FILE,
        RANK_BTREE_FILE,
    ):
        target = os.path.join(graph_dir, filename)
        if os.path.exists(target):
            os.remove(target)

    pool = BufferPool(capacity=1024)
    _write_int_heap(os.path.join(graph_dir, ADJ_OUT_FILE), adj_out, pool)
    _write_int_heap(os.path.join(graph_dir, ADJ_IN_FILE), adj_in, pool)
    _write_int_heap(os.path.join(graph_dir, RANK_FILE), order, pool)

    with BlockFile(os.path.join(graph_dir, NODES_FILE), pool, create=True) as handle:
        table = NodeTable(handle)
        # Pre-size the file so every record's block exists before writing.
        blocks = (num_nodes + 7) // 8
        handle.append_ints([0] * (blocks * INTS_PER_BLOCK))
        for node in range(num_nodes):
            table.write(
                NodeRecord(
                    node_id=node,
                    out_start=out_start[node],
                    in_start=in_start[node],
                    in_deg=in_deg[node],
                    out_deg=out_deg[node],
                    scc_id=scc_labels[node],
                    wcc_id=wcc_labels[node],
                    rank=ranks[node],
                    pagerank=scores[node],
                )
            )
        handle.flush()

    report("building edge hash index")
    hash_capacity, longest_probe = build_edge_hash_index(
        os.path.join(graph_dir, EDGE_HASH_FILE), parsed.edges
    )

    report("bulk-loading PageRank B+-tree")
    btree_height, btree_blocks = build_pagerank_btree(
        os.path.join(graph_dir, RANK_BTREE_FILE), scores
    )

    on_disk = sum(
        os.path.getsize(os.path.join(graph_dir, f))
        for f in (
            NODES_FILE,
            ADJ_OUT_FILE,
            ADJ_IN_FILE,
            RANK_FILE,
            EDGE_HASH_FILE,
            RANK_BTREE_FILE,
        )
    )

    meta = GraphMeta(
        name=name,
        nodes=num_nodes,
        edges=parsed.num_edges,
        weighted=weighted,
        num_scc=len(scc_sizes),
        largest_scc=max(scc_sizes.values()) if scc_sizes else 0,
        num_wcc=len(wcc_sizes),
        largest_wcc=max(wcc_sizes.values()) if wcc_sizes else 0,
        back_edges=int(cycles["back_edges"]),
        nodes_on_cycles=int(cycles["nodes_on_cycles"]),
        is_dag=bool(cycles["is_dag"]),
        pagerank_iterations=iterations,
        pagerank_total=sum(scores),
        hash_capacity=hash_capacity,
        hash_longest_probe=longest_probe,
        btree_height=btree_height,
        btree_blocks=btree_blocks,
        build_ms=int((time.perf_counter() - started) * 1000),
        on_disk_bytes=on_disk,
        total_blocks=on_disk // BLOCK_SIZE,
        had_header=parsed.had_header,
        self_loops_removed=parsed.self_loops_removed,
        duplicates_removed=parsed.duplicates_removed,
        warnings=warnings,
    )

    with open(os.path.join(graph_dir, META_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(meta.to_dict(), handle, indent=2)

    report("done")
    return meta


def list_graphs(data_dir: str) -> list[str]:
    """Names of every built graph under ``data_dir``."""
    if not os.path.isdir(data_dir):
        return []
    names = []
    for entry in sorted(os.listdir(data_dir)):
        if os.path.exists(os.path.join(data_dir, entry, META_FILENAME)):
            names.append(entry)
    return names


def delete_graph(data_dir: str, name: str) -> None:
    target = os.path.join(data_dir, sanitize_graph_name(name))
    if os.path.isdir(target):
        shutil.rmtree(target)
