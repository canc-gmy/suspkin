"""Rotation helpers: axis-angle matrices, rotation about a line, aligning rotation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_EYE = np.eye(3)


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


def rotate_about_line(
    p: ArrayLike, origin: ArrayLike, axis: ArrayLike, angle: float
) -> NDArray[np.float64]:
    """Rotate point ``p`` by ``angle`` about the line through ``origin`` along ``axis``."""
    o = np.asarray(origin, dtype=np.float64)
    r = axis_angle_matrix(axis, angle)
    out: NDArray[np.float64] = o + r @ (np.asarray(p, dtype=np.float64) - o)
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
