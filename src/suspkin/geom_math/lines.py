"""2D/3D line and plane helpers used by the instant-centre metrics.

A 2D line is stored as ``(normal, offset)`` with unit ``normal`` and the line being
``normal . q = offset``. Parallel lines never raise: the intersection is ``None``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec = NDArray[np.float64]

PARALLEL_TOL = 1e-9


def plane_from_points(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> tuple[Vec, Vec]:
    """Plane through three points as ``(point, unit_normal)``.

    Raises
    ------
    ValueError
        If the points are collinear.
    """
    pa = np.asarray(a, dtype=np.float64)
    n = np.cross(np.asarray(b, dtype=np.float64) - pa, np.asarray(c, dtype=np.float64) - pa)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        raise ValueError("collinear points do not define a plane")
    return pa, n / norm


def plane_section_2d(
    point: ArrayLike, normal: ArrayLike, axis: int, value: float
) -> tuple[Vec, float]:
    """Section of a plane with the axis-aligned plane ``coord[axis] == value``.

    The section is a 2D line in the two remaining coordinates (in increasing index
    order, e.g. ``axis=0`` gives the (y, z) plane), returned as ``(unit_normal2, offset)``.

    Raises
    ------
    ValueError
        If the plane is parallel to the cutting plane (no unique section line).
    """
    p = np.asarray(point, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    keep = [i for i in range(3) if i != axis]
    n2 = n[keep]
    norm = float(np.linalg.norm(n2))
    if norm < 1e-12:
        raise ValueError("plane is parallel to the cutting plane")
    rhs = float(n @ p) - n[axis] * value
    return n2 / norm, rhs / norm


def line_through_2d(p: ArrayLike, q: ArrayLike) -> tuple[Vec, float]:
    """2D line through two distinct points as ``(unit_normal, offset)``."""
    a = np.asarray(p, dtype=np.float64)
    d = np.asarray(q, dtype=np.float64) - a
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        raise ValueError("points coincide")
    normal = np.array([-d[1], d[0]]) / norm
    return normal, float(normal @ a)


def line_direction_2d(normal: ArrayLike) -> Vec:
    """Unit direction of a 2D line given its unit normal (normal rotated by -90 deg)."""
    n = np.asarray(normal, dtype=np.float64)
    return np.array([n[1], -n[0]])


def intersect_lines_2d(
    l1: tuple[ArrayLike, float], l2: tuple[ArrayLike, float], tol: float = PARALLEL_TOL
) -> Vec | None:
    """Intersection of two 2D lines, or ``None`` if parallel (|cross of normals| < tol)."""
    n1 = np.asarray(l1[0], dtype=np.float64)
    n2 = np.asarray(l2[0], dtype=np.float64)
    det = float(n1[0] * n2[1] - n1[1] * n2[0])
    if abs(det) < tol * float(np.linalg.norm(n1) * np.linalg.norm(n2)):
        return None
    return np.linalg.solve(np.array([n1, n2]), np.array([l1[1], l2[1]]))
