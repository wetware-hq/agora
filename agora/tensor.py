"""The board as a `tmap1` tensor map of kind `board` (wetware-hq/web contract §14).

The multigraph `Hub.edges()` becomes one adjacency tensor: `adjacency[k, i, j]` counts the
live edges of kind `KINDS[k]` from `fid[i]` to `fid[j]`. Names are ascending, dtypes are
TensorFlow `DataType` numbers, content is row-major little-endian, and `sha256` of the
bytes is the board's tensor root. The map is a view: it holds the graph and the hub root,
never a cast's text.
"""

from __future__ import annotations

import hashlib
import struct

from agora.hub import Hub

KINDS = ("follow", "like", "mention", "recast", "reply")
UINT8, UINT32, UINT64 = 4, 22, 23
FMT = {UINT8: "B", UINT32: "I", UINT64: "Q"}
MAX_BYTES = 1 << 30  # web §14: a tensor map is at most 2^30 bytes


def _record(name: str, code: int, shape: tuple[int, ...], values: list[int]) -> bytes:
    head = bytes([len(name)]) + name.encode() + bytes([code, len(shape)])
    return (
        head
        + struct.pack(f"<{len(shape)}I", *shape)
        + struct.pack(f"<{len(values)}{FMT[code]}", *values)
    )


def board_map(hub: Hub) -> bytes:
    """Canonical tmap1 bytes of the board: adjacency, casts, fid, messages, root."""
    edges = hub.edges()
    fids = sorted(set(hub.names) | {e.src for e in edges} | {e.dst for e in edges})
    at = {fid: i for i, fid in enumerate(fids)}
    n = len(fids)
    if 4 * len(KINDS) * n * n > MAX_BYTES:
        raise ValueError(
            f"too_large: a board of {n} fids is over {MAX_BYTES} bytes as a tensor map"
        )
    adjacency = [0] * (len(KINDS) * n * n)
    for e in edges:
        adjacency[(KINDS.index(e.kind) * n + at[e.src]) * n + at[e.dst]] += 1
    records = [
        _record("adjacency", UINT32, (len(KINDS), n, n), adjacency),
        _record("casts", UINT64, (), [len(hub.casts)]),
        _record("fid", UINT64, (n,), fids),
        _record("messages", UINT64, (), [len(hub.log)]),
        _record("root", UINT8, (32,), list(bytes.fromhex(hub.root))),
    ]
    return b"tmap1" + bytes([5]) + b"board" + struct.pack("<H", len(records)) + b"".join(records)


def tensor_root(hub: Hub) -> str:
    """sha256 of the board's tensor map, cited `tmap1:<root>`."""
    return hashlib.sha256(board_map(hub)).hexdigest()
