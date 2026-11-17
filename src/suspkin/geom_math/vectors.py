"""Small vector helpers (ISO 8855 frame, SI units)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
