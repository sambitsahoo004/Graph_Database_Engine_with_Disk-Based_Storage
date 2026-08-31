# GraphNexus

A directed-graph store that keeps its data in fixed 512-byte blocks and answers
structural queries against it through a buffer pool. Every query reports how
long it took and how many blocks it actually read from disk.

The point of the project is the storage layer, not the graph algorithms: the
algorithms exist to generate realistic access patterns against a block-addressed
file format.

## Running it

```bash
unzip graphnexus.zip
cd graphnexus

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check the install first:

```bash
pytest -q                        # 59 passed
```

### Web interface

```bash
python run.py                    # http://127.0.0.1:5000
```

Open `/graphs`, upload an edge list from `samples/`, and it becomes the active
graph. Every query page then shows the answer plus what it cost in block reads.

### Command line

The same operations without a browser:

```bash
python cli.py build samples/p2p-Gnutella04.txt
python cli.py list
python cli.py info p2p-Gnutella04

python cli.py query p2p-Gnutella04 indegree 1
python cli.py query p2p-Gnutella04 outdegree 1
python cli.py query p2p-Gnutella04 pagerank 1
python cli.py query p2p-Gnutella04 distance 1 500
python cli.py query p2p-Gnutella04 knn 1 5
python cli.py query p2p-Gnutella04 components 1 2
python cli.py query p2p-Gnutella04 ranklist 1 20
python cli.py query p2p-Gnutella04 edge 0 1
python cli.py query p2p-Gnutella04 scorerange 0.0004 0.0006 20

python cli.py build samples/tiny-weighted.txt --weighted
python cli.py query tiny-weighted distance 0 4

python cli.py delete tiny-weighted
```

Every query prints its cost:

```
$ python cli.py query p2p-Gnutella04 distance 1 500
distance 1 -> 500: 5
    6.237 ms   240 blocks read / 533 requested   (55.0% hit ratio)
```

`bench` sweeps the buffer pool to show the storage layer working:

```
$ python cli.py bench p2p-Gnutella04 1 500
 pool blocks   blocks read   hit ratio
           1           439      17.6%
           4           299      43.9%
          16           289      45.8%
          64           266      50.1%
         256           240      55.0%
        1024           240      55.0%
```

The same query costs 439 block reads with a single-block pool and 240 with a
256-block one, and stops improving past that because the working set fits.

### Included samples

| File | What it is |
| --- | --- |
| `samples/p2p-Gnutella04.txt` | SNAP Gnutella peer-to-peer network, 10,879 nodes, 39,994 edges. No header line, so it also exercises header detection. |
| `samples/tiny-dag.txt` | Seven-node acyclic graph with a sink, for checking dangling-node PageRank. |
| `samples/tiny-weighted.txt` | Five-node weighted graph where the cheapest route is not the direct edge. |


## Input format

One edge per line:

```
0 1
0 2
1 2
```

For weighted graphs, a third column:

```
0 1 5
1 2 3
```

A leading `<nodes> <edges>` header line is optional; it is used when present
and consistent with the rest of the file, and ignored otherwise. Lines starting
with `#` or `%` are skipped, so raw SNAP downloads load without editing. Self-
loops and duplicate edges are dropped, and both counts are reported after the
build.

## Configuration

Everything is read from the environment, with working defaults.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GRAPHNEXUS_DATA_DIR` | `./data` | Where built graphs are written |
| `GRAPHNEXUS_SECRET_KEY` | random per process | Flask session signing key |
| `GRAPHNEXUS_BUFFER_BLOCKS` | `256` | Buffer pool size, in 512-byte blocks |
| `GRAPHNEXUS_MAX_UPLOAD_MB` | `64` | Upload size limit |
| `GRAPHNEXUS_DEBUG` | unset | Set to `1` for Flask debug mode |

Set a real `GRAPHNEXUS_SECRET_KEY` and serve with a WSGI server for anything
beyond local development:

```bash
gunicorn "graphnexus:create_app()"
```

## Storage design

Each graph is a directory of heap files, all addressed in 512-byte blocks.

```
data/<graph>/
    source.txt    the uploaded edge list, kept verbatim
    meta.json     graph-level statistics
    nodes.dat     node table, 64-byte records, 8 per block
    adj_out.dat   outgoing adjacency, compressed-sparse-row style
    adj_in.dat    incoming adjacency, same layout
    rank.dat      node ids ordered by descending PageRank
    edge.hash     static hash index over edges, linear probing
    rank.btree    B+-tree over (pagerank, node_id)
```

**Node record, 64 bytes.** Eight records fit exactly in one block, so a node
lookup is one block read.

| Offset | Field |
| --- | --- |
| 0 | `node_id` |
| 4 | `out_start` — entry index into `adj_out.dat` |
| 8 | `in_start` — entry index into `adj_in.dat` |
| 12 | `in_deg` |
| 16 | `out_deg` |
| 20 | `scc_id` |
| 24 | `wcc_id` |
| 28 | `rank` |
| 32 | `pagerank`, float64 |
| 40–63 | reserved |

**Adjacency.** Each node's neighbours occupy one contiguous run starting at
`out_start`, so reading a neighbour list is a sequential scan rather than a
pointer chase. Unweighted graphs store one int32 per neighbour; weighted graphs
store an interleaved `(neighbour, weight)` pair.

**Rank list.** `rank.dat` is a sorted array addressed positionally: rank *r*
lives at int32 index *r − 1*. A rank range query therefore touches
`ceil((r2 − r1 + 1) / 128)` blocks and nothing else.

## Indexes

Two indexes sit beside the heap files. Both are addressed in the same 512-byte
blocks and read through the same buffer pool, so their cost lands in the same
counters every other query reports. Both are bulk-loaded at build time and
never mutated afterwards.

**`edge.hash` — static hash index, open addressing with linear probing.**
Answers *is there an edge u → v*. The key is the edge itself, the 8-byte pair
`(u, v)`, hashed with MurmurHash3 x86_32; the bucket is 16 bytes, so 32 fit in
a block and a probe sequence stays inside one block for its first 31 steps.
Capacity is a power of two sized to keep the load factor under 0.75. Because
the table is built once and never deleted from, there are no tombstones.

The alternative is to read `u`'s node record and scan its whole outgoing run,
which costs `ceil(out_deg / 128)` blocks and grows with the degree of the node.
The index does not. On a synthetic node with out-degree 899 the scan reads 9
blocks and the index reads 2.

**`rank.btree` — bulk-loaded B+-tree, ordered index.** Answers *which nodes
score between lo and hi*. `rank.dat` cannot: it is ordered by rank position and
stores node ids, not scores, so binary searching it by score means reading a
node record — a random block elsewhere in the file — at every probe. The
B+-tree stores the score inline, so a descent is one block per level and the
matching range is then a sequential walk along linked leaf blocks. On 1,200
nodes a descent reads 3 blocks where the leaf level alone is 30.

Keys are the composite `(pagerank, node_id)`. That composite is what makes keys
unique, which is what lets the tree hold the strict separator invariant — every
key in child *i* is below separator *i* — when many nodes share a score. On the
Gnutella sample 26 of them share the most common one. A bare float key would
break the invariant and a descent would land past some of the ties.

Loading bottom-up from sorted input fills every leaf to its 41-entry capacity,
rather than leaving the half-full blocks repeated insertion produces.

Measured on `p2p-Gnutella04`, 2,000 random edge lookups, 256-block pool:

```
$ python cli.py bench-index p2p-Gnutella04
2,000 edge lookups on p2p-Gnutella04
method                       mean ms    p95 ms   blocks
hash index                    0.0081    0.0135     1.79
adjacency scan                0.0123    0.0193     2.05
  probe sequence: mean 1.79, p95 5, max 26, load factor 0.61

B+-tree score range (50 nodes), height 3: mean 0.0253 ms, p95 0.0353 ms, 5.26 blocks
```

The gap between the two edge-lookup rows is small here because Gnutella's mean
out-degree is under four, so the scan it replaces is short. The index earns its
place on the tail, not the mean: the scan's cost is a function of degree and
the index's is not.

**Buffer pool.** All reads go through a shared LRU pool. The block figures shown
in the UI are counted there:

- *Block requests* — every block the query asked for.
- *Blocks read* — the requests that missed the pool and reached the file.

Shrinking `GRAPHNEXUS_BUFFER_BLOCKS` raises the second number, which is the
easiest way to see the pool doing its job.

The build phase works in memory and writes each heap file sequentially, the way
a bulk loader does. Block accounting covers the query path.

## Queries

| Query | Method |
| --- | --- |
| In-degree, out-degree | Single node record read |
| PageRank score and rank | Single node record read |
| Same SCC / same WCC | Two node record reads |
| Shortest distance | BFS, or Dijkstra when weighted |
| K nearest neighbours | Dijkstra, stopping once *k* nodes settle |
| Rank range | Sequential scan of `rank.dat` |
| Edge exists | Hash index probe, no adjacency read |
| Score range | B+-tree descent plus a leaf walk |

Component ids and PageRank are computed once at build time and stored in the
node record, so those queries are point lookups rather than traversals.

## What the build reports

Node and edge counts, SCC and WCC counts with their largest members, PageRank
convergence, cycle measures, build time, and on-disk size.

**PageRank.** Mass held by nodes with no outgoing edges is redistributed
uniformly each iteration, so the vector sums to 1. Scores are stored as float64
in the node record.

**Cycles.** Counting every simple cycle is #P-hard, and on a graph like the
Gnutella sample the true number is astronomically large, so it is not reported.
Three exact quantities are reported instead:

- `is_dag` — whether the graph contains any directed cycle
- `back_edges` — edges closing a cycle in a DFS forest
- `nodes_on_cycles` — nodes in an SCC of size greater than 1, plus self-loops

## Verification

The test suite checks the pipeline against SNAP's published figures for
`p2p-Gnutella04` (included in `samples/`): 39,994 edges, a largest strongly
connected component of 4,317 nodes, and a largest weakly connected component of
10,876 nodes. It also carries a regression test for each defect fixed in the
rewrite, so none of them can return silently.

```
$ pytest -q
59 passed
```

## Notes on the rewrite

This version corrects several things the earlier one got wrong. They are
recorded here because the earlier README described some of them as features.

- **PageRank leaked rank mass.** Without dangling-node handling the vector
  converged to a total of roughly 0.25 on the Gnutella sample.
- **The cycle counter was not counting simple cycles.** Its visited set was
  shared across the outer loop with no guard, so it returned a partial
  back-edge count. On a triangle with one chord — two simple cycles — it
  returned one.
- **There was no index.** The old `File_Index` stored the value `16 × i` at
  position `i`, an identity map that cost a block access to compute arithmetic.
  It was removed, and the two indexes now in the tree are keyed on things that
  are not their own position: the edge pair, and the PageRank score.
- **There was no static hashing.** There is now, but over edges, not over the
  rank list. Hashing the rank list would have been the wrong structure — a hash
  index cannot answer a range query — which is why the ordered index over
  scores is a B+-tree instead.
- **Block accounting was a proxy.** The old counter incremented only when a
  path differed from the immediately preceding one, and its state was never
  reset. Reads now go through a real buffer pool with hit and miss counters.
- **Traversals were recursive**, with the interpreter recursion limit raised to
  10,000,000. That does not grow the C stack, so a deep graph segfaulted rather
  than raising. Every traversal is now iterative.
- **Reads created files.** Node lookups opened a computed filename with mode
  `x`, so querying a node that did not exist left an empty block file behind.
  Node ids are now range-checked before any I/O.
- **The active graph was a module global**, so concurrent users overwrote each
  other's selection. It now lives in the signed session cookie.
- **The data directory was hardcoded** to one developer's home directory.

## Layout

```
graphnexus/
    __init__.py       application factory
    storage.py        blocks, buffer pool, node table
    edgelist.py       edge list parsing and header detection
    algorithms.py     SCC, WCC, PageRank, BFS, Dijkstra, KNN
    indexes.py        MurmurHash3, hash index, B+-tree
    graphstore.py     build and query
    forms.py          form definitions and validation
    routes.py         HTTP routes
    static/css/       one stylesheet
    templates/        base template plus one page per query
config.py             environment-driven configuration
run.py                web entry point
cli.py                command-line interface
tests/                test suite
samples/              example edge lists, including p2p-Gnutella04 from SNAP
```

## Licence

MIT.
