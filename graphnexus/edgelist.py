"""Edge list parsing.

The original loader assumed every file opened with a ``<nodes> <edges>``
header line. Raw SNAP downloads do not have one, so the first edge was silently
consumed as the header -- which is how the checked-in Gnutella sample ended up
recording ``Nodes : 0``. This parser detects the header instead of assuming it,
and skips ``#`` comment lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class EdgeListError(Exception):
    """Raised when an edge list cannot be parsed."""


@dataclass
class ParsedGraph:
    num_nodes: int
    edges: list[tuple[int, int, int]] = field(default_factory=list)
    weighted: bool = False
    had_header: bool = False
    self_loops_removed: int = 0
    duplicates_removed: int = 0

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def _tokenize(path: str) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            rows.append(line.split())
    if not rows:
        raise EdgeListError("edge list is empty")
    return rows


def _looks_like_header(rows: list[list[str]], weighted: bool) -> bool:
    """Decide whether row 0 is a ``<nodes> <edges>`` header or a real edge.

    A header must have exactly two integers, and both must be consistent with
    the rest of the file: the declared edge count has to match the number of
    remaining rows, and every node id has to be below the declared node count.
    """
    head = rows[0]
    if len(head) != 2:
        return False
    try:
        declared_nodes, declared_edges = int(head[0]), int(head[1])
    except ValueError:
        return False
    if declared_nodes <= 0 or declared_edges < 0:
        return False
    if declared_edges != len(rows) - 1:
        return False
    width = 3 if weighted else 2
    max_id = 0
    for row in rows[1:]:
        if len(row) < width:
            return False
        try:
            max_id = max(max_id, int(row[0]), int(row[1]))
        except ValueError:
            return False
    return max_id < declared_nodes


def parse_edge_list(
    path: str,
    weighted: bool = False,
    drop_self_loops: bool = True,
    drop_duplicates: bool = True,
) -> ParsedGraph:
    """Parse an edge list into node count plus a list of ``(u, v, w)`` triples."""
    rows = _tokenize(path)
    width = 3 if weighted else 2

    had_header = _looks_like_header(rows, weighted)
    body = rows[1:] if had_header else rows
    declared_nodes = int(rows[0][0]) if had_header else 0

    edges: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    max_id = -1
    self_loops = 0
    duplicates = 0

    for line_no, row in enumerate(body, start=2 if had_header else 1):
        if len(row) < width:
            raise EdgeListError(
                f"line {line_no}: expected {width} values, found {len(row)}"
            )
        try:
            u, v = int(row[0]), int(row[1])
            w = int(row[2]) if weighted else 1
        except ValueError as exc:
            raise EdgeListError(f"line {line_no}: {exc}") from exc
        if u < 0 or v < 0:
            raise EdgeListError(f"line {line_no}: node ids must be non-negative")
        if weighted and w < 0:
            raise EdgeListError(
                f"line {line_no}: negative weights are not supported"
            )
        if drop_self_loops and u == v:
            self_loops += 1
            continue
        if drop_duplicates:
            if (u, v) in seen:
                duplicates += 1
                continue
            seen.add((u, v))
        max_id = max(max_id, u, v)
        edges.append((u, v, w))

    num_nodes = max(declared_nodes, max_id + 1)
    if num_nodes <= 0:
        raise EdgeListError("edge list contains no usable edges")

    return ParsedGraph(
        num_nodes=num_nodes,
        edges=edges,
        weighted=weighted,
        had_header=had_header,
        self_loops_removed=self_loops,
        duplicates_removed=duplicates,
    )
