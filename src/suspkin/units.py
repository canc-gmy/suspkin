"""Display-only unit conversions. Internals use SI; mm and deg are for display."""

import numpy as np
import numpy.typing as npt

ArrayLike = float | npt.NDArray[np.float64]


def m_to_mm(x: ArrayLike) -> ArrayLike:
    """Convert metres to millimetres."""
    return np.multiply(x, 1000.0)


def mm_to_m(x: ArrayLike) -> ArrayLike:
    """Convert millimetres to metres."""
    return np.multiply(x, 0.001)


def rad_to_deg(x: ArrayLike) -> ArrayLike:
    """Convert radians to degrees."""
    return np.degrees(x)


def deg_to_rad(x: ArrayLike) -> ArrayLike:
    """Convert degrees to radians."""
    return np.radians(x)
