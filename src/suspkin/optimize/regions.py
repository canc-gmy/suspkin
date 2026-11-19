"""Regions in the vehicle frame: trilinear hexahedron map and signed distances.

Pure numpy/scipy. Hexahedron vertex order: bottom face 0-1-2-3 counter-clockwise seen from
+z, then top face 4-5-6-7 above 0-1-2-3 (FEM convention). Vertex ``i`` sits at the unit-cube
corner ``_CORNERS[i]``. Signed distances are negative inside, positive outside (exact
Euclidean outside for all four shapes).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull, QhullError

F64 = NDArray[np.float64]

_CORNERS: F64 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float64,
)
_TWISTED_MSG = "hexahedron is degenerate or twisted, check vertex order"


def _vertices(v: ArrayLike) -> F64:
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (8, 3):
        raise ValueError(f"hexahedron needs 8 vertices of 3 coordinates, got shape {arr.shape}")
    return arr


def _shape_derivs(uvw: F64) -> tuple[F64, F64]:
    """Shape functions ``(..., 8)`` and their derivatives ``(..., 8, 3)`` at ``uvw``."""
    s = 2.0 * _CORNERS - 1.0  # +-1 per corner and axis
    f = np.where(s > 0, uvw[..., None, :], 1.0 - uvw[..., None, :])  # (..., 8, 3)
    n = f[..., 0] * f[..., 1] * f[..., 2]
    d = np.empty(f.shape, dtype=np.float64)
    d[..., 0] = s[:, 0] * f[..., 1] * f[..., 2]
    d[..., 1] = s[:, 1] * f[..., 0] * f[..., 2]
    d[..., 2] = s[:, 2] * f[..., 0] * f[..., 1]
    return n, d


def hex_map(vertices: ArrayLike, uvw: ArrayLike) -> F64:
    """Trilinear map from the unit cube ``uvw`` (``(..., 3)``) to the hexahedron."""
    v = _vertices(vertices)
    n, _ = _shape_derivs(np.asarray(uvw, dtype=np.float64))
    out: F64 = n @ v
    return out


def hex_jacobian(vertices: ArrayLike, uvw: ArrayLike) -> F64:
    """Jacobian ``dp/d(u,v,w)`` of :func:`hex_map`, shape ``(..., 3, 3)`` (columns = u, v, w)."""
    v = _vertices(vertices)
    _, d = _shape_derivs(np.asarray(uvw, dtype=np.float64))
    out: F64 = np.einsum("...ik,ij->...jk", d, v)
    return out


def check_hexahedron(vertices: ArrayLike) -> None:
    """Raise ``ValueError`` unless the Jacobian determinant is > 0 at the 8 corners and centre."""
    v = _vertices(vertices)
    pts = np.vstack([_CORNERS, [[0.5, 0.5, 0.5]]])
    det = np.linalg.det(hex_jacobian(v, pts))
    if not np.all(det > 0.0):
        raise ValueError(_TWISTED_MSG)


def hex_inverse(
    vertices: ArrayLike, points: ArrayLike, tol: float = 1e-10, max_iter: int = 50
) -> F64:
    """Invert :func:`hex_map` by Newton iteration; ``points`` is ``(..., 3)``.

    Raises
    ------
    ValueError
        If Newton does not converge or the point lies outside the hexahedron.
    """
    v = _vertices(vertices)
    p = np.asarray(points, dtype=np.float64)
    uvw = np.full(p.shape, 0.5)
    for _ in range(max_iter):
        r = hex_map(v, uvw) - p
        jac = hex_jacobian(v, uvw)
        try:
            step = np.linalg.solve(jac, r[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError("hex_inverse: singular Jacobian") from exc
        uvw = uvw - step
        if np.max(np.abs(step), initial=0.0) < tol:
            break
    else:
        raise ValueError("hex_inverse: Newton did not converge")
    if np.any(uvw < -tol) or np.any(uvw > 1.0 + tol):
        raise ValueError("point lies outside the hexahedron")
    out: F64 = np.clip(uvw, 0.0, 1.0)
    return out


def hex_locate(
    vertices: ArrayLike, points: ArrayLike, tol: float = 1e-9, max_iter: int = 50
) -> tuple[F64, NDArray[np.bool_]]:
    """Unit-cube coordinates of ``points`` (``(..., 3)``) in the hexahedron, never raising.

    Returns ``(uvw, inside)``: ``uvw`` is clipped to ``[0, 1]``; ``inside`` is True where
    Newton converged to a point within ``tol`` (relative to the cube) of the unit cube.
    """
    v = _vertices(vertices)
    p = np.asarray(points, dtype=np.float64)
    uvw = np.full(p.shape, 0.5)
    with np.errstate(all="ignore"):
        for _ in range(max_iter):
            r = hex_map(v, uvw) - p
            jac = hex_jacobian(v, uvw)
            det = np.linalg.det(jac)
            bad = ~np.isfinite(det) | (np.abs(det) < 1e-300)
            jac = np.where(bad[..., None, None], np.eye(3), jac)
            step = np.linalg.solve(jac, r[..., None])[..., 0]
            step = np.where(np.isfinite(step), step, 0.0)
            uvw = np.clip(uvw - step, -2.0, 3.0)
            if np.max(np.abs(step), initial=0.0) < 1e-13:
                break
        res = np.max(np.abs(hex_map(v, uvw) - p), axis=-1)
    scale = max(float(np.max(np.abs(v - v.mean(axis=0)))), 1e-300)
    ok = res <= 1e-9 * scale
    inside = ok & np.all((uvw >= -tol) & (uvw <= 1.0 + tol), axis=-1)
    return np.clip(uvw, 0.0, 1.0), inside


def hex_contains(vertices: ArrayLike, points: ArrayLike, tol: float = 1e-9) -> NDArray[np.bool_]:
    """Mask of ``points`` (``(..., 3)``) inside or on the boundary of the hexahedron."""
    return hex_locate(vertices, points, tol)[1]


# --------------------------------------------------------------------------------------
# signed distances
# --------------------------------------------------------------------------------------


def sd_box(points: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> F64:
    """Signed distance to the axis-aligned box ``[lo, hi]``."""
    p = np.asarray(points, dtype=np.float64)
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    c, h = 0.5 * (lo_a + hi_a), 0.5 * (hi_a - lo_a)
    q = np.abs(p - c) - h
    out: F64 = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)
    return out


def sd_sphere(points: ArrayLike, center: ArrayLike, radius: float) -> F64:
    """Signed distance to a sphere."""
    p = np.asarray(points, dtype=np.float64)
    out: F64 = np.linalg.norm(p - np.asarray(center, dtype=np.float64), axis=-1) - radius
    return out


def sd_cylinder(points: ArrayLike, p0: ArrayLike, p1: ArrayLike, radius: float) -> F64:
    """Signed distance to the finite (capped) cylinder with axis endpoints ``p0``, ``p1``."""
    p = np.asarray(points, dtype=np.float64)
    a = np.asarray(p0, dtype=np.float64)
    axis = np.asarray(p1, dtype=np.float64) - a
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("cylinder axis endpoints coincide")
    u = axis / length
    rel = p - a
    along = rel @ u
    radial = np.linalg.norm(rel - along[..., None] * u, axis=-1)
    d = np.stack([radial - radius, np.abs(along - 0.5 * length) - 0.5 * length], axis=-1)
    out: F64 = np.minimum(np.max(d, axis=-1), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    return out


_HULL_CACHE: dict[bytes, tuple[ConvexHull, F64]] = {}
_HULL_CACHE_MAX = 256


def _hull_tri(vertices: ArrayLike) -> tuple[ConvexHull, F64]:
    """Convex hull and its boundary triangles, cached per vertex set (failures not cached)."""
    v = _vertices(vertices)
    key = np.ascontiguousarray(v).tobytes()
    hit = _HULL_CACHE.get(key)
    if hit is None:
        hull = _build_hull(v)
        hit = (hull, v[hull.simplices])
        if len(_HULL_CACHE) >= _HULL_CACHE_MAX:
            _HULL_CACHE.clear()
        _HULL_CACHE[key] = hit
    return hit


def _build_hull(v: F64) -> ConvexHull:
    try:
        return ConvexHull(v)
    except QhullError as exc:
        raise ValueError(_TWISTED_MSG) from exc


def _hull(vertices: ArrayLike) -> ConvexHull:
    return _hull_tri(vertices)[0]


def _hex_volume(v: F64) -> float:
    g = 0.5 / np.sqrt(3.0)
    pts = np.array(
        [
            [0.5 + sx * g, 0.5 + sy * g, 0.5 + sz * g]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    return float(np.sum(np.linalg.det(hex_jacobian(v, pts))) / 8.0)


def is_convex_hexahedron(vertices: ArrayLike, rtol: float = 1e-6) -> bool:
    """True if the hexahedron equals its convex hull (all vertices extreme, planar faces)."""
    v = _vertices(vertices)
    try:
        hull = ConvexHull(v)
    except QhullError:
        return False
    if len(hull.vertices) != 8:
        return False
    vol = _hex_volume(v)
    return bool(vol > 0.0 and abs(hull.volume - vol) <= rtol * max(vol, hull.volume))


def _dist_to_triangles(p: F64, tri: F64) -> F64:
    """Distance from points ``p`` (N, 3) to the nearest of triangles ``tri`` (T, 3, 3)."""
    best = np.full(p.shape[0], np.inf)
    for a, b, c in tri:
        # in-plane projection
        n = np.cross(b - a, c - a)
        nn = float(np.linalg.norm(n))
        cand = np.full(p.shape[0], np.inf)
        if nn > 0.0:
            n = n / nn
            h = (p - a) @ n
            proj = p - h[:, None] * n
            m = np.stack([b - a, c - a], axis=0)  # (2, 3)
            gram = m @ m.T
            rhs = (proj - a) @ m.T  # (N, 2)
            st = np.linalg.solve(gram, rhs.T).T
            inside = (st[:, 0] >= 0) & (st[:, 1] >= 0) & (st.sum(axis=1) <= 1.0)
            cand = np.where(inside, np.abs(h), np.inf)
        for e0, e1 in ((a, b), (b, c), (c, a)):
            ab = e1 - e0
            t = np.clip(((p - e0) @ ab) / float(ab @ ab), 0.0, 1.0)
            seg = np.linalg.norm(p - (e0 + t[:, None] * ab), axis=-1)
            cand = np.minimum(cand, seg)
        best = np.minimum(best, cand)
    return best


def sd_hexahedron(points: ArrayLike, vertices: ArrayLike) -> F64:
    """Signed distance to a convex hexahedron (see :func:`is_convex_hexahedron`).

    Inside: minus the distance to the nearest face plane. Outside: exact distance to the
    boundary triangles of the convex hull.
    """
    p = np.asarray(points, dtype=np.float64)
    hull, tri = _hull_tri(vertices)
    eq = hull.equations  # (F, 4): outward unit normal, offset
    plane = p @ eq[:, :3].T + eq[:, 3]  # (..., F)
    d = np.max(plane, axis=-1)
    flat_d = np.atleast_1d(d).reshape(-1)
    out_mask = flat_d > 0.0
    if np.any(out_mask):
        flat_p = p.reshape(-1, 3)
        flat_d = flat_d.copy()
        flat_d[out_mask] = _dist_to_triangles(flat_p[out_mask], tri)
        d = flat_d.reshape(np.shape(d))
    out: F64 = np.asarray(d, dtype=np.float64)
    return out
