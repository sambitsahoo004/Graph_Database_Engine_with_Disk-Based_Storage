"""On-disk index structures.

Two indexes are built alongside the heap files, both addressed in the same
512-byte blocks and read through the same buffer pool, so their cost shows up
in the block counters the rest of the project reports.

``EdgeHashIndex`` -- static hash index, open addressing with linear probing
--------------------------------------------------------------------------
Answers "is there an edge u -> v, and what does it weigh" without touching the
adjacency heap. The alternative is to read the node record for ``u`` and then
scan its whole neighbour run, which costs ``ceil(out_deg / 128)`` blocks and
grows with the degree of the node. The hash index answers the same question in
one bucket-block read plus however many blocks the probe sequence crosses.

The key is the edge itself -- the 8-byte little-endian pair ``(u, v)`` -- hashed
with MurmurHash3 x86_32. Collisions are resolved by linear probing, which keeps
a probe sequence inside one block for its first 31 steps because 32 buckets fit
in a block. Capacity is a power of two, sized so the table never exceeds a 0.75
load factor, and the table is built once at load time and never mutated, so no
tombstones are needed.

``PageRankBTree`` -- bulk-loaded B+-tree, ordered index
------------------------------------------------------
Answers "which nodes have a PageRank score in [lo, hi]". ``rank.dat`` cannot:
it stores node ids ordered by rank, not the scores themselves, so binary
searching it by score means reading a node record -- a random block, elsewhere
in the file -- at every probe. The B+-tree stores the score inline in its
separators and leaves, so a descent costs one block per level and the matching
range is then a sequential walk along linked leaf blocks.

Keys are the composite ``(pagerank, node_id)``. The composite is what makes the
keys unique, which is what lets the tree keep the strict separator invariant
"every key in child i is less than separator i" even when many nodes share a
score -- and on a real graph many of them do.

Both files start with a header block so they are self-describing; nothing about
their geometry is inferred from the file length.
"""

from __future__ import annotations

import struct
from bisect import bisect_left, bisect_right

from .storage import BLOCK_SIZE, BlockFile, StorageError

# ---------------------------------------------------------------------------
# MurmurHash3 x86_32
# ---------------------------------------------------------------------------

_C1 = 0xCC9E2D51
_C2 = 0x1B873593
_M32 = 0xFFFFFFFF


def murmurhash3_x86_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3, x86 32-bit variant.

    Matches the reference implementation's published test vectors; see
    ``tests/test_graphnexus.py``.
    """
    length = len(data)
    h1 = seed & _M32
    tail_start = length & ~0x03

    for i in range(0, tail_start, 4):
        k1 = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24)
        k1 = (k1 * _C1) & _M32
        k1 = ((k1 << 15) | (k1 >> 17)) & _M32
        k1 = (k1 * _C2) & _M32
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & _M32
        h1 = (h1 * 5 + 0xE6546B64) & _M32

    remainder = length & 0x03
    if remainder:
        k1 = 0
        if remainder == 3:
            k1 ^= data[tail_start + 2] << 16
        if remainder >= 2:
            k1 ^= data[tail_start + 1] << 8
        k1 ^= data[tail_start]
        k1 = (k1 * _C1) & _M32
        k1 = ((k1 << 15) | (k1 >> 17)) & _M32
        k1 = (k1 * _C2) & _M32
        h1 ^= k1

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & _M32
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & _M32
    h1 ^= h1 >> 16
    return h1


# ---------------------------------------------------------------------------
# Static hash index over edges
# ---------------------------------------------------------------------------

HASH_MAGIC = 0x47_4E_48_31  # "GNH1"
HASH_SEED = 0x9747B28C

BUCKET_SIZE = 16  # u:int32, v:int32, weight:int32, state:int32
BUCKETS_PER_BLOCK = BLOCK_SIZE // BUCKET_SIZE  # 32
BUCKET_FORMAT = "<4i"

STATE_EMPTY = 0
STATE_OCCUPIED = 1

MAX_LOAD_FACTOR = 0.75
MIN_HASH_CAPACITY = BUCKETS_PER_BLOCK


def _next_power_of_two(value: int) -> int:
    capacity = 1
    while capacity < value:
        capacity <<= 1
    return capacity


def hash_capacity_for(num_entries: int) -> int:
    """Smallest power-of-two bucket count that stays under the load factor."""
    needed = int(num_entries / MAX_LOAD_FACTOR) + 1
    return max(MIN_HASH_CAPACITY, _next_power_of_two(needed))


def edge_hash(u: int, v: int) -> int:
    """Hash one edge key. The key is the 8 bytes the bucket itself stores."""
    return murmurhash3_x86_32(struct.pack("<II", u & _M32, v & _M32), HASH_SEED)


class EdgeHashIndex:
    """Read access to a built edge hash index."""

    def __init__(self, block_file: BlockFile):
        self.file = block_file
        header = self.file.read_block(0)
        magic, capacity, entries, max_probe = struct.unpack_from("<4i", header, 0)
        if magic != HASH_MAGIC:
            raise StorageError(f"{self.file.path} is not an edge hash index")
        if capacity <= 0 or capacity & (capacity - 1):
            raise StorageError(f"{self.file.path} has a non power-of-two capacity")
        self.capacity = capacity
        self.num_entries = entries
        self.max_probe = max_probe

    @property
    def load_factor(self) -> float:
        return self.num_entries / self.capacity if self.capacity else 0.0

    def _read_bucket(self, slot: int) -> tuple[int, int, int, int]:
        block_no, index = divmod(slot, BUCKETS_PER_BLOCK)
        frame = self.file.read_block(block_no + 1)  # block 0 is the header
        return struct.unpack_from(BUCKET_FORMAT, frame, index * BUCKET_SIZE)

    def lookup(self, u: int, v: int) -> tuple[bool, int, int]:
        """Return ``(found, weight, probes)`` for the edge ``u -> v``.

        The probe count is returned rather than hidden so the caller can show
        what the collision behaviour actually cost on this key.
        """
        mask = self.capacity - 1
        slot = edge_hash(u, v) & mask
        for probe in range(1, self.capacity + 1):
            bu, bv, weight, state = self._read_bucket(slot)
            if state == STATE_EMPTY:
                return False, 0, probe
            if bu == u and bv == v:
                return True, weight, probe
            slot = (slot + 1) & mask
        return False, 0, self.capacity


def build_edge_hash_index(
    path: str, edges: list[tuple[int, int, int]]
) -> tuple[int, int]:
    """Bulk-load the hash index. Returns ``(capacity, longest probe sequence)``.

    The table is filled in memory and written out as whole blocks, the way a
    bulk loader does; the probe logic here is the same linear probe the reader
    replays.
    """
    capacity = hash_capacity_for(len(edges))
    mask = capacity - 1
    slots = [None] * capacity  # type: list[tuple[int, int, int] | None]
    longest_probe = 0

    for u, v, weight in edges:
        slot = edge_hash(u, v) & mask
        probe = 1
        while slots[slot] is not None:
            existing = slots[slot]
            if existing[0] == u and existing[1] == v:
                break  # duplicate edge; the parser already drops these
            slot = (slot + 1) & mask
            probe += 1
            if probe > capacity:
                raise StorageError("edge hash index is full")
        slots[slot] = (u, v, weight)
        longest_probe = max(longest_probe, probe)

    header = bytearray(BLOCK_SIZE)
    struct.pack_into(
        "<4i", header, 0, HASH_MAGIC, capacity, len(edges), longest_probe
    )

    payload = bytearray(header)
    for base in range(0, capacity, BUCKETS_PER_BLOCK):
        block = bytearray(BLOCK_SIZE)
        for offset in range(BUCKETS_PER_BLOCK):
            entry = slots[base + offset]
            if entry is None:
                continue
            struct.pack_into(
                BUCKET_FORMAT,
                block,
                offset * BUCKET_SIZE,
                entry[0],
                entry[1],
                entry[2],
                STATE_OCCUPIED,
            )
        payload.extend(block)

    with open(path, "wb") as handle:
        handle.write(payload)
    return capacity, longest_probe


# ---------------------------------------------------------------------------
# Bulk-loaded B+-tree over (pagerank, node_id)
# ---------------------------------------------------------------------------

BTREE_MAGIC = 0x47_4E_42_31  # "GNB1"

NODE_LEAF = 1
NODE_INTERNAL = 2

# Leaf block: header(16) + keys(41 * 8) + ids(41 * 4) = 508 bytes.
LEAF_CAPACITY = 41
LEAF_KEYS_OFFSET = 16
LEAF_IDS_OFFSET = LEAF_KEYS_OFFSET + LEAF_CAPACITY * 8

# Internal block: header(16) + children(31 * 4) + sep keys(30 * 8)
#                 + sep ids(30 * 4) = 500 bytes.
INTERNAL_FANOUT = 31
CHILDREN_OFFSET = 16
SEP_KEYS_OFFSET = CHILDREN_OFFSET + INTERNAL_FANOUT * 4
SEP_IDS_OFFSET = SEP_KEYS_OFFSET + (INTERNAL_FANOUT - 1) * 8

assert LEAF_IDS_OFFSET + LEAF_CAPACITY * 4 <= BLOCK_SIZE
assert SEP_IDS_OFFSET + (INTERNAL_FANOUT - 1) * 4 <= BLOCK_SIZE


class PageRankBTree:
    """Read access to a bulk-loaded B+-tree keyed on ``(pagerank, node_id)``."""

    def __init__(self, block_file: BlockFile):
        self.file = block_file
        header = self.file.read_block(0)
        magic, root, height, entries = struct.unpack_from("<4i", header, 0)
        if magic != BTREE_MAGIC:
            raise StorageError(f"{self.file.path} is not a PageRank B+-tree")
        self.root_block = root
        self.height = height
        self.num_entries = entries

    # -- block decoding ----------------------------------------------------

    def _leaf_entries(self, frame) -> tuple[list[float], list[int], int]:
        _, count, next_leaf, _ = struct.unpack_from("<4i", frame, 0)
        keys = list(struct.unpack_from(f"<{count}d", frame, LEAF_KEYS_OFFSET))
        ids = list(struct.unpack_from(f"<{count}i", frame, LEAF_IDS_OFFSET))
        return keys, ids, next_leaf

    def _descend(self, probe: tuple[float, int]) -> int:
        """Walk from the root to the leaf that would hold ``probe``.

        One block read per level, which is what makes this ``log_f n`` rather
        than a scan.
        """
        block_no = self.root_block
        while True:
            frame = self.file.read_block(block_no)
            node_type, count, _, _ = struct.unpack_from("<4i", frame, 0)
            if node_type == NODE_LEAF:
                return block_no
            separators = count - 1
            keys = struct.unpack_from(f"<{separators}d", frame, SEP_KEYS_OFFSET)
            ids = struct.unpack_from(f"<{separators}i", frame, SEP_IDS_OFFSET)
            composite = list(zip(keys, ids))
            child = bisect_right(composite, probe)
            block_no = struct.unpack_from(
                "<i", frame, CHILDREN_OFFSET + child * 4
            )[0]

    # -- queries -----------------------------------------------------------

    def lookup(self, score: float, node_id: int) -> bool:
        """Exact point lookup on the composite key."""
        frame = self.file.read_block(self._descend((score, node_id)))
        keys, ids, _ = self._leaf_entries(frame)
        position = bisect_left(list(zip(keys, ids)), (score, node_id))
        return position < len(keys) and (keys[position], ids[position]) == (
            score,
            node_id,
        )

    def range_search(
        self, low: float, high: float, limit: int = 1000
    ) -> list[tuple[int, float]]:
        """Every ``(node_id, score)`` with ``low <= score <= high``.

        Descends once, then walks the linked leaves. Ordered ascending by
        score; callers wanting the strongest nodes reverse the result.
        """
        if high < low:
            return []
        # ``-1`` sorts below every real node id, so this lands on the first
        # entry whose score is >= low rather than skipping ties.
        block_no = self._descend((low, -1))
        results: list[tuple[int, float]] = []

        while block_no >= 0 and len(results) < limit:
            frame = self.file.read_block(block_no)
            keys, ids, next_leaf = self._leaf_entries(frame)
            start = bisect_left(list(zip(keys, ids)), (low, -1))
            for position in range(start, len(keys)):
                if keys[position] > high:
                    return results
                results.append((ids[position], keys[position]))
                if len(results) >= limit:
                    return results
            block_no = next_leaf
        return results


def build_pagerank_btree(path: str, scores: list[float]) -> tuple[int, int]:
    """Bulk-load the tree from every node's score. Returns ``(height, blocks)``.

    Loading bottom-up from sorted input fills every leaf to capacity instead of
    leaving the half-full blocks that repeated insertion produces.
    """
    entries = sorted((score, node) for node, score in enumerate(scores))

    blocks: list[bytearray] = [bytearray(BLOCK_SIZE)]  # slot 0 is the header

    # --- leaf level -------------------------------------------------------
    leaf_blocks: list[int] = []
    leaf_first_key: list[tuple[float, int]] = []
    for start in range(0, len(entries), LEAF_CAPACITY):
        chunk = entries[start:start + LEAF_CAPACITY]
        frame = bytearray(BLOCK_SIZE)
        struct.pack_into("<4i", frame, 0, NODE_LEAF, len(chunk), -1, 0)
        struct.pack_into(
            f"<{len(chunk)}d", frame, LEAF_KEYS_OFFSET, *[k for k, _ in chunk]
        )
        struct.pack_into(
            f"<{len(chunk)}i", frame, LEAF_IDS_OFFSET, *[n for _, n in chunk]
        )
        leaf_blocks.append(len(blocks))
        leaf_first_key.append(chunk[0])
        blocks.append(frame)

    if not leaf_blocks:  # empty graph: one empty leaf so the tree is still valid
        frame = bytearray(BLOCK_SIZE)
        struct.pack_into("<4i", frame, 0, NODE_LEAF, 0, -1, 0)
        leaf_blocks.append(len(blocks))
        leaf_first_key.append((0.0, 0))
        blocks.append(frame)

    # Link the leaves so a range walk never returns to the root.
    for position, block_no in enumerate(leaf_blocks[:-1]):
        struct.pack_into("<i", blocks[block_no], 8, leaf_blocks[position + 1])

    # --- internal levels --------------------------------------------------
    height = 1
    level_blocks = leaf_blocks
    level_keys = leaf_first_key

    while len(level_blocks) > 1:
        height += 1
        parent_blocks: list[int] = []
        parent_keys: list[tuple[float, int]] = []
        for start in range(0, len(level_blocks), INTERNAL_FANOUT):
            children = level_blocks[start:start + INTERNAL_FANOUT]
            child_keys = level_keys[start:start + INTERNAL_FANOUT]
            frame = bytearray(BLOCK_SIZE)
            struct.pack_into("<4i", frame, 0, NODE_INTERNAL, len(children), -1, 0)
            struct.pack_into(
                f"<{len(children)}i", frame, CHILDREN_OFFSET, *children
            )
            # Separator i is the first key of child i + 1, so every key in
            # child i is strictly below it.
            separators = child_keys[1:]
            if separators:
                struct.pack_into(
                    f"<{len(separators)}d",
                    frame,
                    SEP_KEYS_OFFSET,
                    *[k for k, _ in separators],
                )
                struct.pack_into(
                    f"<{len(separators)}i",
                    frame,
                    SEP_IDS_OFFSET,
                    *[n for _, n in separators],
                )
            parent_blocks.append(len(blocks))
            parent_keys.append(child_keys[0])
            blocks.append(frame)
        level_blocks = parent_blocks
        level_keys = parent_keys

    root_block = level_blocks[0]
    struct.pack_into(
        "<4i", blocks[0], 0, BTREE_MAGIC, root_block, height, len(entries)
    )

    with open(path, "wb") as handle:
        for frame in blocks:
            handle.write(frame)
    return height, len(blocks)
