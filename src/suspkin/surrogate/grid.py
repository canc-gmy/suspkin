"""Fixed output grid of the surrogate per axle (wheel travel x rack)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from suspkin.optimize.spec import GridSpec


@dataclass(frozen=True)
class AxleGrid:
    """Grid of one axle. ``travel`` (T,) and ``rack`` (R,) in SI; the rear has ``rack == [0]``."""

    axle: Literal["front", "rear"]
    travel: NDArray[np.float64]
    rack: NDArray[np.float64]

    @property
    def shape(self) -> tuple[int, int]:
        """``(R, T)``."""
        return (self.rack.size, self.travel.size)

    def grid_hash(self) -> str:
        """Stable sha256 (hex) of the axle name and the values rounded to 1e-9."""
        h = hashlib.sha256(self.axle.encode())
        for arr in (self.travel, self.rack):
            r = np.round(np.asarray(arr, dtype=np.float64), 9) + 0.0  # +0.0 folds -0.0
            h.update(np.int64(r.size).tobytes())
            h.update(r.astype("<f8").tobytes())
        return h.hexdigest()


def axle_grids(spec: GridSpec) -> dict[str, AxleGrid]:
    """Front and rear grids from the study ``surrogate.grid`` block.

    The rack axis applies to the front axle only; without ``rack`` the front uses ``[0]``.
    """
    t = spec.travel
    travel = np.linspace(t.min, t.max, t.points)
    if spec.rack is None:
        front_rack = np.array([0.0])
    else:
        front_rack = np.linspace(spec.rack.min, spec.rack.max, spec.rack.points)
    return {
        "front": AxleGrid("front", travel, front_rack),
        "rear": AxleGrid("rear", travel.copy(), np.array([0.0])),
    }
