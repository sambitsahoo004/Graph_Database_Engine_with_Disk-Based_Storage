"""Command-line interface.

Everything the web UI does, without a browser. Useful for scripting, for
benchmarking the buffer pool, and for checking a graph built correctly before
serving it.

    python cli.py build samples/p2p-Gnutella04.txt
    python cli.py info p2p-Gnutella04
    python cli.py query p2p-Gnutella04 distance 1 500
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config  # noqa: E402
from graphnexus.graphstore import (  # noqa: E402
    SOURCE_FILENAME,
    GraphError,
    GraphStore,
    build_graph,
    delete_graph,
    list_graphs,
    sanitize_graph_name,
)


def _data_dir(args) -> str:
    return args.data_dir or Config.DATA_DIR


def _open(args, name: str) -> GraphStore:
    return GraphStore(
        os.path.join(_data_dir(args), sanitize_graph_name(name)),
        buffer_blocks=args.buffer_blocks,
    )


def _report_cost(result) -> None:
    print(
        f"    {result.elapsed_ms:.3f} ms   "
        f"{result.stats.physical_reads:,} blocks read / "
        f"{result.stats.logical_reads:,} requested   "
        f"({result.hit_ratio:.1%} hit ratio)"
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_build(args) -> int:
    if not os.path.exists(args.source):
        print(f"No such file: {args.source}", file=sys.stderr)
        return 1

    name = sanitize_graph_name(args.name or os.path.basename(args.source))
    graph_dir = os.path.join(_data_dir(args), name)
    os.makedirs(graph_dir, exist_ok=True)

    destination = os.path.join(graph_dir, SOURCE_FILENAME)
    if os.path.abspath(args.source) != os.path.abspath(destination):
        with open(args.source, "rb") as src, open(destination, "wb") as dst:
            dst.write(src.read())

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"  {message}...")

    try:
        meta = build_graph(
            destination, graph_dir, weighted=args.weighted, progress=progress
        )
    except Exception as exc:  # noqa: BLE001
        delete_graph(_data_dir(args), name)
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nBuilt {name} in {meta.build_ms:,} ms")
    print(f"  {meta.nodes:,} nodes, {meta.edges:,} edges")
    print(f"  {meta.total_blocks:,} blocks of 512 bytes on disk")
    for warning in meta.warnings:
        print(f"  note: {warning}")
    return 0


def cmd_list(args) -> int:
    names = list_graphs(_data_dir(args))
    if not names:
        print(f"No graphs built yet in {_data_dir(args)}")
        return 0
    for name in names:
        print(name)
    return 0


def cmd_info(args) -> int:
    with _open(args, args.name) as store:
        m = store.meta
        rows = [
            ("Nodes", f"{m.nodes:,}"),
            ("Edges", f"{m.edges:,}"),
            ("Weighted", "yes" if m.weighted else "no"),
            ("Strongly connected components", f"{m.num_scc:,}"),
            ("Largest SCC", f"{m.largest_scc:,}"),
            ("Weakly connected components", f"{m.num_wcc:,}"),
            ("Largest WCC", f"{m.largest_wcc:,}"),
            ("Acyclic", "yes" if m.is_dag else "no"),
            ("Back edges", f"{m.back_edges:,}"),
            ("Nodes on a cycle", f"{m.nodes_on_cycles:,}"),
            ("PageRank iterations", f"{m.pagerank_iterations}"),
            ("PageRank total mass", f"{m.pagerank_total:.10f}"),
            ("Build time", f"{m.build_ms:,} ms"),
            ("On disk", f"{m.on_disk_bytes:,} bytes ({m.total_blocks:,} blocks)"),
            (
                "Edge hash index",
                f"{m.hash_capacity:,} buckets, load "
                f"{(m.edges / m.hash_capacity if m.hash_capacity else 0):.2f}, "
                f"longest probe {m.hash_longest_probe}",
            ),
            (
                "PageRank B+-tree",
                f"height {m.btree_height}, {m.btree_blocks:,} blocks",
            ),
        ]
        width = max(len(label) for label, _ in rows)
        print(m.name)
        for label, value in rows:
            print(f"  {label:<{width}}  {value}")
        for warning in m.warnings:
            print(f"  note: {warning}")
    return 0


def cmd_query(args) -> int:
    with _open(args, args.name) as store:
        kind = args.kind
        try:
            if kind == "indegree":
                result = store.in_degree(args.values[0])
                print(f"in-degree of {args.values[0]}: {result.value:,}")
            elif kind == "outdegree":
                result = store.out_degree(args.values[0])
                print(f"out-degree of {args.values[0]}: {result.value:,}")
            elif kind == "pagerank":
                result = store.pagerank_of(args.values[0])
                print(f"PageRank of {args.values[0]}: {result.value['pagerank']:.12f}")
                print(f"rank: {result.value['rank']:,} of {store.meta.nodes:,}")
            elif kind == "distance":
                result = store.shortest_distance(args.values[0], args.values[1])
                if result.value < 0:
                    print(f"no path from {args.values[0]} to {args.values[1]}")
                else:
                    print(
                        f"distance {args.values[0]} -> {args.values[1]}: "
                        f"{result.value:,}"
                    )
            elif kind == "knn":
                result = store.knn(args.values[0], args.values[1])
                if not result.value:
                    print(f"node {args.values[0]} cannot reach any others")
                for node, distance in result.value:
                    print(f"  {node:<10} distance {distance:,}")
            elif kind == "components":
                result = store.same_components(args.values[0], args.values[1])
                print(f"same SCC: {'yes' if result.value['same_scc'] else 'no'}")
                print(f"same WCC: {'yes' if result.value['same_wcc'] else 'no'}")
            elif kind == "ranklist":
                result = store.rank_list(args.values[0], args.values[1])
                for row in result.value:
                    print(
                        f"  {row['rank']:<8} node {row['node']:<10} "
                        f"{row['pagerank']:.12f}"
                    )
            elif kind == "edge":
                u, v = args.values[0], args.values[1]
                result = store.has_edge(u, v)
                if result.value["exists"]:
                    detail = (
                        f" (weight {result.value['weight']:,})"
                        if store.meta.weighted
                        else ""
                    )
                    print(f"edge {u} -> {v}: present{detail}")
                else:
                    print(f"edge {u} -> {v}: absent")
                print(f"probe sequence length: {result.value['probes']}")
            elif kind == "scorerange":
                low, high = args.values[0], args.values[1]
                limit = args.values[2] if len(args.values) > 2 else 100
                result = store.nodes_by_score(low, high, limit)
                if not result.value:
                    print(f"no node scores between {low} and {high}")
                for row in result.value:
                    print(f"  node {row['node']:<10} {row['pagerank']:.12f}")
            else:  # pragma: no cover - argparse restricts the choices
                raise GraphError(f"unknown query {kind}")
        except GraphError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        _report_cost(result)
    return 0


def cmd_delete(args) -> int:
    delete_graph(_data_dir(args), args.name)
    print(f"Deleted {sanitize_graph_name(args.name)}")
    return 0


def cmd_bench(args) -> int:
    """Show how buffer pool size changes the number of physical block reads."""
    print(f"{'pool blocks':>12}  {'blocks read':>12}  {'hit ratio':>10}")
    for capacity in (1, 4, 16, 64, 256, 1024):
        store = GraphStore(
            os.path.join(_data_dir(args), sanitize_graph_name(args.name)),
            buffer_blocks=capacity,
        )
        with store:
            result = store.shortest_distance(args.source, args.target)
        print(
            f"{capacity:>12}  {result.stats.physical_reads:>12,}  "
            f"{result.hit_ratio:>9.1%}"
        )
    return 0


def cmd_benchindex(args) -> int:
    """Time the two index-backed queries against their unindexed equivalents."""
    import random
    import statistics

    random.seed(args.seed)
    with _open(args, args.name) as store:
        n = store.meta.nodes
        if store.edge_hash is None or store.rank_btree is None:
            print("Rebuild this graph to add its indexes.", file=sys.stderr)
            return 1

        # Sample real edges so the lookups mostly hit.
        sample = []
        while len(sample) < args.samples:
            u = random.randrange(n)
            neighbors = store.out_neighbors(u)
            if neighbors:
                sample.append((u, random.choice(neighbors)[0]))

        def timed(fn):
            times, blocks = [], []
            for u, v in sample:
                result = fn(u, v)
                times.append(result.elapsed_ms)
                blocks.append(result.stats.logical_reads)
            times.sort()
            return (
                statistics.mean(times),
                times[int(len(times) * 0.95)],
                statistics.mean(blocks),
            )

        print(f"{args.samples:,} edge lookups on {store.meta.name}")
        print(f"{'method':<26}{'mean ms':>10}{'p95 ms':>10}{'blocks':>9}")
        for label, fn in (
            ("hash index", store.has_edge),
            ("adjacency scan", store.has_edge_by_scan),
        ):
            mean, p95, blocks = timed(fn)
            print(f"{label:<26}{mean:>10.4f}{p95:>10.4f}{blocks:>9.2f}")

        probes = [store.has_edge(u, v).value["probes"] for u, v in sample]
        probes.sort()
        print(
            f"  probe sequence: mean {statistics.mean(probes):.2f}, "
            f"p95 {probes[int(len(probes) * 0.95)]}, max {max(probes)}, "
            f"load factor {store.edge_hash.load_factor:.2f}"
        )

        scores = sorted(store.node_table.read(i).pagerank for i in range(n))
        times, blocks = [], []
        for _ in range(args.samples):
            j = random.randrange(0, max(1, n - 50))
            result = store.nodes_by_score(scores[j], scores[min(j + 49, n - 1)], 10_000)
            times.append(result.elapsed_ms)
            blocks.append(result.stats.logical_reads)
        times.sort()
        print(
            f"\nB+-tree score range (50 nodes), height {store.meta.btree_height}: "
            f"mean {statistics.mean(times):.4f} ms, "
            f"p95 {times[int(len(times) * 0.95)]:.4f} ms, "
            f"{statistics.mean(blocks):.2f} blocks"
        )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="Build and query block-paged graphs."
    )
    parser.add_argument("--data-dir", help="Override where graphs are stored")
    parser.add_argument(
        "--buffer-blocks",
        type=int,
        default=Config.BUFFER_POOL_BLOCKS,
        help="Buffer pool size in 512-byte blocks (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="Build a graph from an edge list")
    p.add_argument("source", help="Path to the edge list")
    p.add_argument("--name", help="Name to store it under")
    p.add_argument("--weighted", action="store_true", help="Third column is a weight")
    p.add_argument("--quiet", action="store_true", help="Hide progress output")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("list", help="List built graphs")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="Show a graph's statistics")
    p.add_argument("name")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("query", help="Run a query")
    p.add_argument("name")
    p.add_argument(
        "kind",
        choices=[
            "indegree",
            "outdegree",
            "pagerank",
            "distance",
            "knn",
            "components",
            "ranklist",
            "edge",
            "scorerange",
        ],
    )
    p.add_argument("values", nargs="+", help="Query arguments")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("delete", help="Delete a built graph")
    p.add_argument("name")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("bench", help="Compare block reads across buffer pool sizes")
    p.add_argument("name")
    p.add_argument("source", type=int)
    p.add_argument("target", type=int)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser(
        "bench-index", help="Time index lookups against their unindexed equivalents"
    )
    p.add_argument("name")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_benchindex)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GraphError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
