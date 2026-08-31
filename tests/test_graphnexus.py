"""Test suite.

Includes a regression test for each defect found in the original version, so
none of them can come back silently.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphnexus import algorithms  # noqa: E402
from graphnexus.edgelist import EdgeListError, parse_edge_list  # noqa: E402
from graphnexus.indexes import (  # noqa: E402
    EdgeHashIndex,
    build_edge_hash_index,
    edge_hash,
    hash_capacity_for,
    murmurhash3_x86_32,
)
from graphnexus.graphstore import (  # noqa: E402
    GraphError,
    GraphStore,
    build_graph,
    sanitize_graph_name,
)
from graphnexus.storage import (  # noqa: E402
    BLOCK_SIZE,
    NODE_RECORD_SIZE,
    RECORDS_PER_BLOCK,
    BlockFile,
    BufferPool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_graph(tmp_path, text, name="g", weighted=False):
    graph_dir = tmp_path / name
    graph_dir.mkdir(parents=True, exist_ok=True)
    source = graph_dir / "source.txt"
    source.write_text(text)
    meta = build_graph(str(source), str(graph_dir), weighted=weighted)
    return meta, str(graph_dir)


TRIANGLE_PLUS_PAIR = "5 5\n0 1\n1 2\n2 0\n3 4\n4 3\n"
CHAIN_WITH_SINK = "3 2\n0 1\n1 2\n"


# ---------------------------------------------------------------------------
# Storage layer
# ---------------------------------------------------------------------------


def test_block_geometry_matches_the_documented_layout():
    assert BLOCK_SIZE == 512
    assert NODE_RECORD_SIZE == 64
    assert RECORDS_PER_BLOCK == 8


def test_read_ints_spans_block_boundaries(tmp_path):
    pool = BufferPool(capacity=8)
    path = str(tmp_path / "ints.dat")
    with BlockFile(path, pool, create=True) as handle:
        handle.append_ints(list(range(500)))
        assert handle.read_ints(120, 20) == list(range(120, 140))
        assert handle.read_ints(0, 500) == list(range(500))


def test_reading_past_the_end_returns_minus_one_without_growing_the_file(tmp_path):
    pool = BufferPool(capacity=4)
    path = str(tmp_path / "short.dat")
    with BlockFile(path, pool, create=True) as handle:
        handle.append_ints([1, 2, 3])
        size_before = os.path.getsize(path)
        assert handle.read_int(10_000) == -1
        assert os.path.getsize(path) == size_before


def test_buffer_pool_counts_hits_and_misses(tmp_path):
    pool = BufferPool(capacity=4)
    path = str(tmp_path / "pool.dat")
    with BlockFile(path, pool, create=True) as handle:
        handle.append_ints(list(range(1000)))
        pool.reset_stats()
        handle.read_int(0)
        handle.read_int(1)  # same block, should be a hit
        stats = pool.reset_stats()
        assert stats.logical_reads == 2
        assert stats.physical_reads == 1


def test_a_smaller_buffer_pool_causes_more_physical_reads(tmp_path):
    _, graph_dir = make_graph(tmp_path, TRIANGLE_PLUS_PAIR)
    reads = {}
    for capacity in (1, 64):
        with GraphStore(graph_dir, buffer_blocks=capacity) as store:
            for node in range(store.meta.nodes):
                store.out_neighbors(node)
            reads[capacity] = store.pool.reset_stats().physical_reads
    assert reads[1] >= reads[64]


# ---------------------------------------------------------------------------
# Edge list parsing
# ---------------------------------------------------------------------------


def test_header_line_is_used_when_it_is_consistent(tmp_path):
    path = tmp_path / "with_header.txt"
    path.write_text("3 3\n0 1\n1 2\n2 0\n")
    parsed = parse_edge_list(str(path))
    assert parsed.had_header is True
    assert parsed.num_nodes == 3
    assert parsed.num_edges == 3


def test_headerless_file_is_detected_and_no_edge_is_swallowed(tmp_path):
    """Regression: the first edge used to be consumed as a header.

    A raw SNAP download has no header, so ``0 1`` was read as
    ``nodes=0, edges=1`` and every subsequent loop over range(0) did nothing.
    """
    path = tmp_path / "headerless.txt"
    path.write_text("0 1\n1 2\n2 0\n")
    parsed = parse_edge_list(str(path))
    assert parsed.had_header is False
    assert parsed.num_nodes == 3
    assert parsed.num_edges == 3


def test_comment_lines_are_skipped(tmp_path):
    path = tmp_path / "commented.txt"
    path.write_text("# Directed graph\n# FromNodeId\tToNodeId\n0 1\n1 2\n")
    parsed = parse_edge_list(str(path))
    assert parsed.num_edges == 2


def test_self_loops_and_duplicates_are_removed(tmp_path):
    path = tmp_path / "messy.txt"
    path.write_text("0 1\n0 1\n2 2\n1 2\n")
    parsed = parse_edge_list(str(path))
    assert parsed.num_edges == 2
    assert parsed.duplicates_removed == 1
    assert parsed.self_loops_removed == 1


def test_malformed_rows_raise_rather_than_being_swallowed(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 1\nnot a number\n")
    with pytest.raises(EdgeListError):
        parse_edge_list(str(path))


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------


def test_strongly_connected_components(tmp_path):
    meta, _ = make_graph(tmp_path, TRIANGLE_PLUS_PAIR)
    assert meta.num_scc == 2
    assert meta.largest_scc == 3


def test_weakly_connected_components(tmp_path):
    meta, _ = make_graph(tmp_path, TRIANGLE_PLUS_PAIR)
    assert meta.num_wcc == 2
    assert meta.largest_wcc == 3


def test_weak_connectivity_ignores_edge_direction(tmp_path):
    # 0 -> 1 <- 2 is one weak component but three strong ones.
    meta, _ = make_graph(tmp_path, "3 2\n0 1\n2 1\n")
    assert meta.num_wcc == 1
    assert meta.num_scc == 3


def test_pagerank_sums_to_one_with_a_dangling_node(tmp_path):
    """Regression: rank mass used to leak out through nodes with no out-edges.

    On the Gnutella sample the original converged to a total of about 0.25.
    """
    meta, _ = make_graph(tmp_path, CHAIN_WITH_SINK)
    assert meta.pagerank_total == pytest.approx(1.0, abs=1e-9)


def test_pagerank_sums_to_one_when_every_node_dangles(tmp_path):
    meta, _ = make_graph(tmp_path, "2 1\n0 1\n")
    assert meta.pagerank_total == pytest.approx(1.0, abs=1e-9)


def test_pagerank_ranks_a_hub_above_a_leaf(tmp_path):
    # Everything points at node 0.
    meta, graph_dir = make_graph(tmp_path, "4 3\n1 0\n2 0\n3 0\n")
    with GraphStore(graph_dir) as store:
        assert store.node_table.read(0).rank == 1


def test_cycle_statistics_are_exact_not_a_partial_back_edge_count(tmp_path):
    """Regression: the old counter shared one visited set across the outer loop.

    On this graph -- a triangle plus the chord 0 -> 2, which has two simple
    cycles -- it reported one. The replacement reports well-defined quantities.
    """
    meta, _ = make_graph(tmp_path, "3 4\n0 1\n1 2\n2 0\n0 2\n")
    assert meta.is_dag is False
    assert meta.nodes_on_cycles == 3
    assert meta.back_edges >= 1


def test_a_dag_is_reported_as_acyclic(tmp_path):
    meta, _ = make_graph(tmp_path, "4 3\n0 1\n1 2\n2 3\n")
    assert meta.is_dag is True
    assert meta.back_edges == 0
    assert meta.nodes_on_cycles == 0


def test_traversals_survive_a_graph_deeper_than_the_recursion_limit():
    """Regression: recursive DFS with sys.setrecursionlimit(10_000_000).

    Raising the limit does not raise the C stack, so a deep graph crashed the
    interpreter instead of raising RecursionError. A 50,000-node path is far
    past Python's default limit of 1,000.
    """
    depth = 50_000
    adjacency = {i: [(i + 1, 1)] for i in range(depth - 1)}
    adjacency[depth - 1] = []

    def neighbors(node):
        return adjacency[node]

    labels = algorithms.tarjan_scc(depth, neighbors)
    assert len(set(labels)) == depth

    weak = algorithms.weakly_connected_components(depth, neighbors)
    assert len(set(weak)) == 1

    assert algorithms.bfs_distance(0, depth - 1, neighbors) == depth - 1


def test_bfs_marks_visited_on_enqueue(tmp_path):
    # A node reachable by many paths must not be enqueued many times.
    edges = "\n".join(f"0 {i}" for i in range(1, 200))
    edges += "\n" + "\n".join(f"{i} 200" for i in range(1, 200))
    meta, graph_dir = make_graph(tmp_path, edges + "\n")
    with GraphStore(graph_dir) as store:
        assert store.shortest_distance(0, 200).value == 2


def test_unreachable_target_reports_no_path(tmp_path):
    _, graph_dir = make_graph(tmp_path, "4 2\n0 1\n2 3\n")
    with GraphStore(graph_dir) as store:
        assert store.shortest_distance(0, 3).value == -1


def test_dijkstra_respects_weights(tmp_path):
    # 0 -> 1 -> 2 costs 3; the direct edge 0 -> 2 costs 10.
    _, graph_dir = make_graph(
        tmp_path, "3 3\n0 1 1\n1 2 2\n0 2 10\n", weighted=True
    )
    with GraphStore(graph_dir) as store:
        assert store.shortest_distance(0, 2).value == 3


def test_knn_returns_k_nodes_in_distance_order(tmp_path):
    _, graph_dir = make_graph(tmp_path, "5 4\n0 1\n1 2\n2 3\n3 4\n")
    with GraphStore(graph_dir) as store:
        result = store.knn(0, 3).value
    assert [node for node, _ in result] == [1, 2, 3]
    assert [distance for _, distance in result] == [1, 2, 3]


def test_rank_list_returns_a_contiguous_range(tmp_path):
    _, graph_dir = make_graph(tmp_path, "4 3\n1 0\n2 0\n3 0\n")
    with GraphStore(graph_dir) as store:
        rows = store.rank_list(1, 4).value
    assert [row["rank"] for row in rows] == [1, 2, 3, 4]
    scores = [row["pagerank"] for row in rows]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Hash index: MurmurHash3 and linear probing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, seed, expected",
    [
        (b"", 0x00000000, 0x00000000),
        (b"", 0x00000001, 0x514E28B7),
        (b"", 0xFFFFFFFF, 0x81F16F39),
        (b"test", 0x00000000, 0xBA6BD213),
        (b"Hello, world!", 0x00000000, 0xC0363E43),
        (b"aaaa", 0x9747B28C, 0x5A97808A),
        (b"abcd", 0x9747B28C, 0xF0478627),
    ],
)
def test_murmurhash3_matches_the_reference_test_vectors(data, seed, expected):
    """The hash is worth nothing if it is not the hash it claims to be."""
    assert murmurhash3_x86_32(data, seed) == expected


def test_hash_capacity_is_a_power_of_two_under_the_load_factor():
    for entries in (0, 1, 100, 39_994, 1_000_000):
        capacity = hash_capacity_for(entries)
        assert capacity & (capacity - 1) == 0
        assert entries / capacity <= 0.75


def test_hash_index_finds_every_edge_and_invents_none(tmp_path):
    text = "6 7\n0 1\n0 2\n1 2\n2 3\n3 4\n4 5\n5 0\n"
    present = {(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)}
    _, graph_dir = make_graph(tmp_path, text)
    with GraphStore(graph_dir) as store:
        for u in range(6):
            for v in range(6):
                found = store.has_edge(u, v).value["exists"]
                assert found == ((u, v) in present), (u, v)


def test_hash_index_is_direction_sensitive(tmp_path):
    _, graph_dir = make_graph(tmp_path, "3 2\n0 1\n1 2\n")
    with GraphStore(graph_dir) as store:
        assert store.has_edge(0, 1).value["exists"]
        assert not store.has_edge(1, 0).value["exists"]


def test_hash_index_agrees_with_the_adjacency_scan(tmp_path):
    """The index and the scan are two paths to one answer; they must not part."""
    text = "8 9\n0 1\n0 3\n1 4\n2 5\n3 6\n4 7\n5 0\n6 2\n7 3\n"
    _, graph_dir = make_graph(tmp_path, text)
    with GraphStore(graph_dir) as store:
        for u in range(8):
            for v in range(8):
                assert (
                    store.has_edge(u, v).value["exists"]
                    == store.has_edge_by_scan(u, v).value["exists"]
                )


def test_hash_index_returns_the_weight_it_stored(tmp_path):
    _, graph_dir = make_graph(tmp_path, "0 1 7\n1 2 3\n", weighted=True)
    with GraphStore(graph_dir) as store:
        assert store.has_edge(0, 1).value["weight"] == 7
        assert store.has_edge(1, 2).value["weight"] == 3


def test_linear_probing_resolves_forced_collisions(tmp_path):
    """Drive many keys into one bucket and check every one is still found.

    A probe sequence that terminated early, or one that failed to wrap at the
    end of the table, would show up here and nowhere else.
    """
    capacity = 64
    mask = capacity - 1
    target_slot = 0
    colliding = []
    u = 0
    while len(colliding) < 40:
        for v in range(2000):
            if edge_hash(u, v) & mask == target_slot:
                colliding.append((u, v, 1))
                break
        u += 1
    path = str(tmp_path / "forced.hash")
    built_capacity, longest = build_edge_hash_index(path, colliding)
    assert built_capacity >= capacity
    pool = BufferPool(capacity=8)
    with BlockFile(path, pool) as handle:
        index = EdgeHashIndex(handle)
        for a, b, _ in colliding:
            found, _, _ = index.lookup(a, b)
            assert found
        assert not index.lookup(999_999, 999_999)[0]
    assert longest >= 2  # the collisions really did collide


def test_hash_lookup_cost_does_not_grow_with_out_degree(tmp_path):
    """The point of the index: a hub costs the same as a leaf.

    The scan reads the whole neighbour run, so its logical block reads climb
    with degree; the index reads one bucket block plus any probe overflow.
    """
    hub_edges = "".join(f"0 {v}\n" for v in range(1, 900))
    _, graph_dir = make_graph(tmp_path, f"901 900\n{hub_edges}900 1\n")
    with GraphStore(graph_dir, buffer_blocks=256) as store:
        indexed = store.has_edge(0, 899)
        scanned = store.has_edge_by_scan(0, 899)
        assert indexed.value["exists"] and scanned.value["exists"]
        assert indexed.stats.logical_reads < scanned.stats.logical_reads


# ---------------------------------------------------------------------------
# B+-tree ordered index
# ---------------------------------------------------------------------------


def test_btree_holds_every_node_and_stays_balanced(tmp_path):
    text = "".join(f"{i} {(i * 7) % 300}\n" for i in range(300))
    meta, graph_dir = make_graph(tmp_path, text)
    with GraphStore(graph_dir) as store:
        assert store.rank_btree.num_entries == meta.nodes
        assert meta.btree_height >= 2  # 300 entries cannot fit one 41-slot leaf
        for node in range(meta.nodes):
            score = store.node_table.read(node).pagerank
            assert store.rank_btree.lookup(score, node)


def test_btree_range_matches_a_full_scan(tmp_path):
    text = "".join(f"{i} {(i * 13) % 200}\n" for i in range(200))
    _, graph_dir = make_graph(tmp_path, text)
    with GraphStore(graph_dir) as store:
        scores = {i: store.node_table.read(i).pagerank for i in range(200)}
        ordered = sorted(scores.values())
        for low, high in ((ordered[0], ordered[-1]),
                          (ordered[10], ordered[150]),
                          (ordered[75], ordered[75])):
            expected = sorted(n for n, s in scores.items() if low <= s <= high)
            got = sorted(
                row["node"] for row in store.nodes_by_score(low, high, 10_000).value
            )
            assert got == expected


def test_btree_returns_every_node_sharing_one_score(tmp_path):
    """Regression: ties are why the key is composite.

    With a bare float key, separators could not keep the strict "child i is
    below separator i" invariant once several nodes shared a score, and a
    descent would land past some of them.
    """
    text = "".join(f"{i} {(i + 1) % 60}\n" for i in range(60))
    _, graph_dir = make_graph(tmp_path, text)
    with GraphStore(graph_dir) as store:
        scores = [store.node_table.read(i).pagerank for i in range(60)]
        shared = max(set(scores), key=scores.count)
        assert scores.count(shared) > 1
        rows = store.nodes_by_score(shared, shared, 10_000).value
        assert len(rows) == scores.count(shared)


def test_btree_results_run_from_strongest_score_down(tmp_path):
    _, graph_dir = make_graph(tmp_path, "4 3\n1 0\n2 0\n3 0\n")
    with GraphStore(graph_dir) as store:
        rows = store.nodes_by_score(0.0, 1.0, 10_000).value
        returned = [row["pagerank"] for row in rows]
    assert returned == sorted(returned, reverse=True)


def test_btree_range_honours_its_row_limit(tmp_path):
    text = "".join(f"{i} {(i * 3) % 150}\n" for i in range(150))
    _, graph_dir = make_graph(tmp_path, text)
    with GraphStore(graph_dir) as store:
        assert len(store.nodes_by_score(0.0, 1.0, 5).value) == 5


def test_btree_descent_reads_one_block_per_level(tmp_path):
    """A descent is log_f n block reads, not a scan of the leaves."""
    text = "".join(f"{i} {(i * 11) % 1200}\n" for i in range(1200))
    meta, graph_dir = make_graph(tmp_path, text)
    with GraphStore(graph_dir, buffer_blocks=512) as store:
        leaf_blocks = (meta.nodes + 40) // 41
        result = store.nodes_by_score(0.5, 1.0, 10)  # above every real score
        assert result.stats.logical_reads <= meta.btree_height + 1
        assert result.stats.logical_reads < leaf_blocks


def test_empty_score_range_returns_nothing(tmp_path):
    _, graph_dir = make_graph(tmp_path, CHAIN_WITH_SINK)
    with GraphStore(graph_dir) as store:
        assert store.nodes_by_score(0.9, 1.0).value == []
        with pytest.raises(GraphError):
            store.nodes_by_score(0.5, 0.1)


def test_missing_index_files_are_reported_not_crashed(tmp_path):
    """Graphs built before the indexes existed stay readable."""
    _, graph_dir = make_graph(tmp_path, CHAIN_WITH_SINK)
    os.remove(os.path.join(graph_dir, "edge.hash"))
    os.remove(os.path.join(graph_dir, "rank.btree"))
    with GraphStore(graph_dir) as store:
        assert store.in_degree(1).value == 1  # heap queries still work
        with pytest.raises(GraphError):
            store.has_edge(0, 1)
        with pytest.raises(GraphError):
            store.nodes_by_score(0.0, 1.0)


def test_index_blocks_are_whole_blocks(tmp_path):
    _, graph_dir = make_graph(tmp_path, TRIANGLE_PLUS_PAIR)
    for name in ("edge.hash", "rank.btree"):
        assert os.path.getsize(os.path.join(graph_dir, name)) % BLOCK_SIZE == 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_out_of_range_node_raises_and_creates_nothing(tmp_path):
    """Regression: any node id used to be turned into a filename and opened
    with mode 'x', so querying a node that did not exist created a block file.
    """
    _, graph_dir = make_graph(tmp_path, CHAIN_WITH_SINK)
    before = sorted(os.listdir(graph_dir))
    with GraphStore(graph_dir) as store:
        with pytest.raises(GraphError):
            store.in_degree(99_999)
        with pytest.raises(GraphError):
            store.in_degree(-1)
    assert sorted(os.listdir(graph_dir)) == before


def test_invalid_rank_ranges_are_rejected(tmp_path):
    _, graph_dir = make_graph(tmp_path, CHAIN_WITH_SINK)
    with GraphStore(graph_dir) as store:
        with pytest.raises(GraphError):
            store.rank_list(0, 5)
        with pytest.raises(GraphError):
            store.rank_list(5, 1)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("graph.txt", "graph"),
        ("../../etc/passwd", "passwd"),
        ("my graph!.csv", "mygraph"),
        ("/tmp/nested/name.tsv", "name"),
    ],
)
def test_graph_names_are_reduced_to_one_safe_segment(raw, expected):
    assert sanitize_graph_name(raw) == expected


def test_empty_graph_name_is_rejected():
    with pytest.raises(GraphError):
        sanitize_graph_name("///")


# ---------------------------------------------------------------------------
# Published reference figures
# ---------------------------------------------------------------------------

SNAP_SAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "samples",
    "p2p-Gnutella04.txt",
)


@pytest.mark.skipif(
    not os.path.exists(SNAP_SAMPLE), reason="Gnutella sample not present"
)
def test_matches_published_snap_statistics(tmp_path):
    """SNAP publishes 10,876 nodes in the largest WCC and 4,317 in the largest SCC."""
    graph_dir = tmp_path / "gnutella"
    graph_dir.mkdir()
    source = graph_dir / "source.txt"
    source.write_text(open(SNAP_SAMPLE, encoding="utf-8").read())
    meta = build_graph(str(source), str(graph_dir))

    assert meta.edges == 39_994
    assert meta.largest_scc == 4_317
    assert meta.largest_wcc == 10_876
    assert meta.pagerank_total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Web layer
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    from config import TestConfig
    from graphnexus import create_app

    class Cfg(TestConfig):
        DATA_DIR = str(tmp_path / "data")

    app = create_app(Cfg)
    with app.test_client() as test_client:
        yield test_client


def test_pages_load_without_a_graph_loaded(client):
    assert client.get("/").status_code == 200
    assert client.get("/graphs").status_code == 200


def test_query_pages_redirect_when_no_graph_is_active(client):
    response = client.get("/indegree")
    assert response.status_code == 302
    assert "/graphs" in response.headers["Location"]


def test_upload_builds_a_graph_and_makes_it_active(client):
    import io

    payload = {
        "graph_file": (io.BytesIO(b"0 1\n1 2\n2 0\n"), "triangle.txt"),
        "graph_type": "unweighted",
    }
    response = client.post(
        "/graphs", data=payload, content_type="multipart/form-data", follow_redirects=True
    )
    assert response.status_code == 200
    assert b"triangle" in response.data
    assert client.get("/indegree").status_code == 200


def test_unknown_page_returns_404(client):
    assert client.get("/no-such-page").status_code == 404
