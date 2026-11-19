"""Snap hardpoint coordinates to a fabrication grid (absolute vehicle-frame coordinates).

Pure numpy. A coordinate is on the grid when it is an integer multiple of the resolution
``res`` (metres). Axis-aligned domains (box, offset) snap each axis to the nearest node
inside the bounds; a hexahedron snaps to the nearest node inside the region (the rounded
point first, then the neighbouring nodes in growing cubes).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin.optimize.regions import hex_contains

F64 = NDArray[np.float64]
_EPS = 1e-9  # tolerance in grid steps for bounds that lie on a node
_DECIMALS = 12  # k * res is rounded to this many decimals (clean YAML values)
_CHUNK = 400_000  # candidate points tested per hexahedron call
_ENUM_MAX = 3_000_000  # largest node count enumerated when looking for any node in a region

__all__ = ["has_grid_node", "node_range", "snap_hexahedron", "snap_interval"]


def node_range(lo: ArrayLike, hi: ArrayLike, res: float) -> tuple[F64, F64]:
    """Integer index range ``(kmin, kmax)`` of the grid nodes inside ``[lo, hi]`` (floats)."""
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return np.ceil(lo_a / res - _EPS), np.floor(hi_a / res + _EPS)


def snap_interval(p: ArrayLike, lo: ArrayLike, hi: ArrayLike, res: float) -> F64:
    """Nearest grid node of ``p`` (``(..., k)``) inside ``[lo, hi]`` per axis.

    Raises
    ------
    ValueError
        If an axis has no grid node inside its bounds.
    """
    kmin, kmax = node_range(lo, hi, res)
    if np.any(kmin > kmax):
        raise ValueError("no grid node inside the bounds")
    k = np.clip(np.rint(np.asarray(p, dtype=np.float64) / res), kmin, kmax)
    out: F64 = np.round(k * res, _DECIMALS)
    return out


@lru_cache(maxsize=16)
def _offsets(n: int) -> F64:
    """All integer offsets of the cube ``[-n, n]^3`` sorted by distance, ``(M, 3)``."""
    r = np.arange(-n, n + 1, dtype=np.float64)
    g = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    out: F64 = g[np.argsort(np.sum(g * g, axis=1), kind="stable")]
    out.setflags(write=False)
    return out


def snap_hexahedron(p: ArrayLike, vertices: ArrayLike, res: float) -> F64:
    """Nearest grid node inside the hexahedron for each point ``p`` (``(..., 3)``).

    The rounded point is kept when it lies inside; otherwise the closest (to ``p``) inside
    node among the neighbours is taken, in cubes of growing radius.

    Raises
    ------
    ValueError
        If the hexahedron contains no grid node near ``p``.
    """
    pa = np.asarray(p, dtype=np.float64)
    pts = pa.reshape(-1, 3)
    v = np.asarray(vertices, dtype=np.float64)
    k0 = np.rint(pts / res)
    out = np.round(k0 * res, _DECIMALS)
    idx = np.flatnonzero(~hex_contains(v, out))
    if idx.size:
        extent = float(np.max(v.max(axis=0) - v.min(axis=0)))
        todo = idx
        for n in range(1, int(np.ceil(extent / res)) + 3):
            offs = _offsets(n)
            step = max(1, _CHUNK // len(offs))
            still: list[np.ndarray[tuple[int], np.dtype[np.intp]]] = []
            for s in range(0, todo.size, step):
                sub = todo[s : s + step]
                cand = np.round((k0[sub, None, :] + offs[None]) * res, _DECIMALS)
                ins = hex_contains(v, cand)
                d = np.linalg.norm(cand - pts[sub, None, :], axis=-1)
                d[~ins] = np.inf
                j = np.argmin(d, axis=1)
                rows = np.arange(sub.size)
                found = np.isfinite(d[rows, j])
                out[sub[found]] = cand[rows[found], j[found]]
                still.append(sub[~found])
            todo = np.concatenate(still) if still else todo[:0]
            if todo.size == 0:
                break
        else:
            raise ValueError("no grid node inside the hexahedron")
    res_out: F64 = out.reshape(pa.shape)
    return res_out


def has_grid_node(vertices: ArrayLike, res: float) -> bool:
    """True if the hexahedron contains at least one grid node (huge regions count as yes)."""
    v = np.asarray(vertices, dtype=np.float64)
    k0 = np.rint(v.mean(axis=0) / res)
    for n in (2, 6):
        if hex_contains(v, (k0 + _offsets(n)) * res).any():
            return True
    kmin, kmax = node_range(v.min(axis=0), v.max(axis=0), res)
    if np.any(kmin > kmax):
        return False
    counts = kmax - kmin + 1
    if float(np.prod(counts)) > _ENUM_MAX:
        return True
    axes = [np.arange(a, b + 1) * res for a, b in zip(kmin, kmax, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return any(
        bool(hex_contains(v, grid[s : s + _CHUNK]).any()) for s in range(0, len(grid), _CHUNK)
    )
