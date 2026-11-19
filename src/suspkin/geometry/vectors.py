"""Vector, rotation and line/plane helpers (ISO 8855 frame, SI units).

A 2D line is stored as ``(normal, offset)`` with unit ``normal`` and the line being
``normal . q = offset``. Parallel lines never raise: the intersection is ``None``.

The private ``_*3`` helpers are pure-Python tuple versions used by the scalar solver
hot path; they must stay free of numpy.
"""

from __future__ import annotations

from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec = NDArray[np.float64]

_EYE = np.eye(3)
PARALLEL_TOL = 1e-9
_V3 = tuple[float, float, float]


def mirror_y(p: ArrayLike) -> NDArray[np.float64]:
    """Mirror point(s) across the x-z plane (y -> -y).

    Parameters
    ----------
    p : array-like
        Shape (3,) or (N, 3).

    Returns
    -------
    numpy.ndarray
        New array of the same shape with the y component negated.
    """
    out = np.array(p, dtype=np.float64, copy=True)
    out[..., 1] = -out[..., 1]
    return out


def unit(v: ArrayLike) -> NDArray[np.float64]:
    """Return ``v`` scaled to unit length; raise ``ValueError`` if its norm is ~0."""
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        raise ValueError("cannot normalize a near-zero vector")
    out: NDArray[np.float64] = arr / n
    return out


def distance(a: ArrayLike, b: ArrayLike) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def wheel_spin_axis(
    side: Literal["left", "right"], camber: float, toe: float
) -> NDArray[np.float64]:
    """Unit wheel spin axis in the ISO 8855 frame, pointing outboard.

    Zero camber and toe give ``(0, 1, 0)`` for the left wheel and ``(0, -1, 0)``
    for the right wheel. Conventions (DESIGN_PLAN section 2): negative camber
    means the top of the wheel leans toward the car centre; positive toe is
    toe-in (front of the wheel points toward the car centre). The right-side
    axis is the y-mirror of the left one.

    Parameters
    ----------
    side : {"left", "right"}
    camber, toe : float
        Static camber and toe [rad].
    """
    n = np.array([0.0, np.cos(camber), -np.sin(camber)])
    c, s = np.cos(-toe), np.sin(-toe)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    n = rz @ n
    if side == "right":
        n = n * np.array([1.0, -1.0, 1.0])
    return np.asarray(n / np.linalg.norm(n), dtype=np.float64)


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


def axis_angle_matrix(axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Rotation matrix for a right-hand rotation by ``angle`` [rad] about ``axis``.

    ``axis`` is normalized internally; a near-zero axis raises ``ValueError``.
    """
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.sqrt(a @ a))
    if n < 1e-12:
        raise ValueError("rotation axis must be non-zero")
    x, y, z = a / n
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    out: NDArray[np.float64] = _EYE + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)
    return out


def rotation_between(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Minimal rotation matrix taking the direction of ``a`` onto the direction of ``b``."""
    u = np.asarray(a, dtype=np.float64)
    v = np.asarray(b, dtype=np.float64)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        raise ValueError("vectors must be non-zero")
    u, v = u / nu, v / nv
    c = float(u @ v)
    if c < -1.0 + 1e-12:
        # Antiparallel: rotate by pi about any axis perpendicular to u.
        helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return axis_angle_matrix(np.cross(u, helper), np.pi)
    w = np.cross(u, v)
    k = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    out: NDArray[np.float64] = _EYE + k + (k @ k) / (1.0 + c)
    return out


def body_rotation(roll: float, pitch: float) -> Vec:
    """Body-to-world rotation ``R_y(pitch) R_x(roll)``."""
    r: Vec = axis_angle_matrix([0.0, 1.0, 0.0], pitch) @ axis_angle_matrix([1.0, 0.0, 0.0], roll)
    return r


def body_rotation_batch(roll: Vec, pitch: Vec) -> Vec:
    """Batched ``body_rotation``: ``R_y(pitch) R_x(roll)``, shape ``(N, 3, 3)``."""

    def axis_rot(k: Vec, ang: Vec) -> Vec:
        out: Vec = (
            np.eye(3)
            + np.sin(ang)[:, None, None] * k
            + (1.0 - np.cos(ang))[:, None, None] * (k @ k)
        )
        return out

    kx = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    ky = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    out: Vec = axis_rot(ky, pitch) @ axis_rot(kx, roll)
    return out


def _as3(v: ArrayLike) -> _V3:
    a = np.asarray(v, dtype=np.float64)
    return (float(a[0]), float(a[1]), float(a[2]))


def _sub3(a: _V3, b: _V3) -> _V3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _unit3(a: _V3) -> _V3:
    n = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    return (a[0] / n, a[1] / n, a[2] / n)


def _cross3(a: _V3, b: _V3) -> _V3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot3(a: _V3, b: _V3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
