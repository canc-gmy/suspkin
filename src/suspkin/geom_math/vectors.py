"""Small vector helpers (ISO 8855 frame, SI units)."""

from __future__ import annotations

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
