"""Block-addressed storage layer.

Each relation lives in one heap file that is addressed in fixed-size blocks.
All reads go through a buffer pool with an LRU replacement policy, so the
"blocks accessed" figure reported by the UI is the number of *physical* block
reads that actually reached the file -- not a proxy for it.

Layout
------
BLOCK_SIZE   512 bytes
INTS_PER_BLOCK   128 int32 values per block
NODE_RECORD_SIZE 64 bytes, so 8 node records fit in one block

Node record (64 bytes, struct format ``<8id6i``)

    slot 0   node_id
    slot 1   out_start    entry index into the outgoing adjacency heap
    slot 2   in_start     entry index into the incoming adjacency heap
    slot 3   in_deg
    slot 4   out_deg
    slot 5   scc_id
    slot 6   wcc_id
    slot 7   rank         1-based rank by descending PageRank
    slot 8   pagerank     float64, occupies 8 bytes
    slots 9-14  reserved

PageRank is stored as a native float64 rather than a scaled integer, so the
value the UI shows is the value the algorithm computed.
"""

from __future__ import annotations

import os
import struct
from collections import OrderedDict
from dataclasses import dataclass

BLOCK_SIZE = 512
INT_SIZE = 4
INTS_PER_BLOCK = BLOCK_SIZE // INT_SIZE  # 128

NODE_RECORD_SIZE = 64
NODE_RECORD_FORMAT = "<8id6i"
RECORDS_PER_BLOCK = BLOCK_SIZE // NODE_RECORD_SIZE  # 8

assert struct.calcsize(NODE_RECORD_FORMAT) == NODE_RECORD_SIZE

# Slot indices within a node record's tuple representation.
F_NODE_ID = 0
F_OUT_START = 1
F_IN_START = 2
F_IN_DEG = 3
F_OUT_DEG = 4
F_SCC_ID = 5
F_WCC_ID = 6
F_RANK = 7
F_PAGERANK = 8

_EMPTY_BLOCK = b"\x00" * BLOCK_SIZE


class StorageError(Exception):
    """Raised when a block store is used incorrectly or is corrupt."""


@dataclass
class IOStats:
    """Counters for one measured operation."""

    physical_reads: int = 0
    logical_reads: int = 0
    physical_writes: int = 0

    @property
    def hit_ratio(self) -> float:
        if self.logical_reads == 0:
            return 0.0
        hits = self.logical_reads - self.physical_reads
        return hits / self.logical_reads


class BufferPool:
    """LRU buffer pool shared by every heap file in a graph.

    ``capacity`` is measured in blocks. A capacity of 1 reproduces the
    behaviour of a single-block cache; the default of 256 blocks (128 KB) is
    what the web UI uses.
    """

    def __init__(self, capacity: int = 256):
        if capacity < 1:
            raise ValueError("buffer pool capacity must be at least 1 block")
        self.capacity = capacity
        self._frames: OrderedDict[tuple[str, int], bytearray] = OrderedDict()
        self._dirty: set[tuple[str, int]] = set()
        self.stats = IOStats()

    def reset_stats(self) -> IOStats:
        """Return the counters accumulated so far and zero them."""
        stats, self.stats = self.stats, IOStats()
        return stats

    def _evict_if_needed(self, owner: "BlockFile") -> None:
        while len(self._frames) > self.capacity:
            key, frame = self._frames.popitem(last=False)
            if key in self._dirty:
                self._dirty.discard(key)
                owner._flush_frame(key[1], frame)

    def fetch(self, owner: "BlockFile", block_no: int) -> bytearray:
        key = (owner.path, block_no)
        self.stats.logical_reads += 1
        frame = self._frames.get(key)
        if frame is not None:
            self._frames.move_to_end(key)
            return frame
        self.stats.physical_reads += 1
        frame = bytearray(owner._read_from_disk(block_no))
        self._frames[key] = frame
        self._evict_if_needed(owner)
        return frame

    def mark_dirty(self, owner: "BlockFile", block_no: int) -> None:
        self._dirty.add((owner.path, block_no))

    def flush(self, owner: "BlockFile") -> None:
        for key in list(self._dirty):
            if key[0] != owner.path:
                continue
            self._dirty.discard(key)
            frame = self._frames.get(key)
            if frame is not None:
                owner._flush_frame(key[1], frame)

    def drop(self, owner: "BlockFile") -> None:
        """Forget every cached frame belonging to ``owner`` without writing."""
        for key in [k for k in self._frames if k[0] == owner.path]:
            del self._frames[key]
            self._dirty.discard(key)


class BlockFile:
    """One heap file addressed in fixed-size blocks."""

    def __init__(self, path: str, pool: BufferPool, create: bool = False):
        self.path = os.path.abspath(path)
        self.pool = pool
        if create:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            if not os.path.exists(self.path):
                with open(self.path, "wb"):
                    pass
        elif not os.path.exists(self.path):
            raise StorageError(f"heap file not found: {self.path}")
        self._fh = open(self.path, "r+b")

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if not self._fh.closed:
            self.pool.flush(self)
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "BlockFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- raw disk access ---------------------------------------------------

    @property
    def num_blocks(self) -> int:
        size = os.path.getsize(self.path)
        return (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    def _read_from_disk(self, block_no: int) -> bytes:
        if block_no < 0:
            raise StorageError(f"negative block number {block_no}")
        self._fh.seek(block_no * BLOCK_SIZE)
        raw = self._fh.read(BLOCK_SIZE)
        if len(raw) < BLOCK_SIZE:
            raw = raw + _EMPTY_BLOCK[len(raw):]
        return raw

    def _flush_frame(self, block_no: int, frame: bytes) -> None:
        self._fh.seek(block_no * BLOCK_SIZE)
        self._fh.write(frame)
        self.pool.stats.physical_writes += 1

    # -- block access ------------------------------------------------------

    def read_block(self, block_no: int) -> bytearray:
        return self.pool.fetch(self, block_no)

    def write_block(self, block_no: int, data: bytes) -> None:
        if len(data) != BLOCK_SIZE:
            raise StorageError(f"block must be exactly {BLOCK_SIZE} bytes")
        frame = self.pool.fetch(self, block_no)
        frame[:] = data
        self.pool.mark_dirty(self, block_no)

    def flush(self) -> None:
        self.pool.flush(self)
        self._fh.flush()

    # -- integer-level helpers --------------------------------------------

    def read_int(self, index: int) -> int:
        """Read the ``index``-th int32 in the file, or -1 past the end."""
        if index < 0:
            return -1
        block_no, offset = divmod(index, INTS_PER_BLOCK)
        if block_no >= self.num_blocks:
            return -1
        frame = self.read_block(block_no)
        start = offset * INT_SIZE
        return int.from_bytes(frame[start:start + INT_SIZE], "little", signed=True)

    def write_int(self, index: int, value: int) -> None:
        if index < 0:
            raise StorageError(f"negative int index {index}")
        block_no, offset = divmod(index, INTS_PER_BLOCK)
        frame = self.read_block(block_no)
        start = offset * INT_SIZE
        frame[start:start + INT_SIZE] = int(value).to_bytes(
            INT_SIZE, "little", signed=True
        )
        self.pool.mark_dirty(self, block_no)

    def read_ints(self, start_index: int, count: int) -> list[int]:
        """Read ``count`` consecutive int32 values.

        Blocks are fetched whole, so a run of values inside one block costs a
        single block access rather than one per value.
        """
        if count <= 0:
            return []
        values: list[int] = []
        index = start_index
        remaining = count
        total_blocks = self.num_blocks
        while remaining > 0:
            block_no, offset = divmod(index, INTS_PER_BLOCK)
            if block_no >= total_blocks:
                values.extend([-1] * remaining)
                break
            take = min(remaining, INTS_PER_BLOCK - offset)
            frame = self.read_block(block_no)
            start = offset * INT_SIZE
            values.extend(
                struct.unpack_from(f"<{take}i", frame, start)
            )
            index += take
            remaining -= take
        return values

    def append_ints(self, values: list[int]) -> None:
        """Append int32 values, padding the final block with zeros."""
        self.pool.flush(self)
        self._fh.seek(0, os.SEEK_END)
        payload = struct.pack(f"<{len(values)}i", *values)
        self._fh.write(payload)
        pad = (-len(payload)) % BLOCK_SIZE
        if pad:
            self._fh.write(b"\x00" * pad)
        self._fh.flush()
        self.pool.drop(self)


class NodeTable:
    """Fixed-size node records packed 8 to a block."""

    def __init__(self, block_file: BlockFile):
        self.file = block_file

    @staticmethod
    def _locate(node_id: int) -> tuple[int, int]:
        block_no, slot = divmod(node_id, RECORDS_PER_BLOCK)
        return block_no, slot * NODE_RECORD_SIZE

    @property
    def num_nodes(self) -> int:
        return os.path.getsize(self.file.path) // NODE_RECORD_SIZE

    def read(self, node_id: int) -> "NodeRecord":
        if node_id < 0 or node_id >= self.num_nodes:
            raise StorageError(f"node {node_id} is outside this graph")
        block_no, offset = self._locate(node_id)
        frame = self.file.read_block(block_no)
        fields = struct.unpack_from(NODE_RECORD_FORMAT, frame, offset)
        return NodeRecord(*fields[:9])

    def read_field(self, node_id: int, field: int) -> int | float:
        return getattr(self.read(node_id), _FIELD_NAMES[field])

    def write(self, record: "NodeRecord") -> None:
        block_no, offset = self._locate(record.node_id)
        frame = self.file.read_block(block_no)
        struct.pack_into(
            NODE_RECORD_FORMAT,
            frame,
            offset,
            record.node_id,
            record.out_start,
            record.in_start,
            record.in_deg,
            record.out_deg,
            record.scc_id,
            record.wcc_id,
            record.rank,
            record.pagerank,
            0, 0, 0, 0, 0, 0,
        )
        self.file.pool.mark_dirty(self.file, block_no)

    def update(self, node_id: int, **changes) -> None:
        record = self.read(node_id)
        for key, value in changes.items():
            setattr(record, key, value)
        self.write(record)


@dataclass
class NodeRecord:
    node_id: int = -1
    out_start: int = 0
    in_start: int = 0
    in_deg: int = 0
    out_deg: int = 0
    scc_id: int = -1
    wcc_id: int = -1
    rank: int = 0
    pagerank: float = 0.0


_FIELD_NAMES = {
    F_NODE_ID: "node_id",
    F_OUT_START: "out_start",
    F_IN_START: "in_start",
    F_IN_DEG: "in_deg",
    F_OUT_DEG: "out_deg",
    F_SCC_ID: "scc_id",
    F_WCC_ID: "wcc_id",
    F_RANK: "rank",
    F_PAGERANK: "pagerank",
}
