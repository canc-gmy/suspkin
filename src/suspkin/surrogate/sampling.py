"""Design-space sampling of one axle: sub design space and Latin hypercube.

Experimental (see ``docs/guides/stability.md``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

from suspkin.optimize.design import DesignSpace
from suspkin.optimize.spec import StudyConfigError

F64 = NDArray[np.float64]


class NoAxleVariablesError(StudyConfigError):
    """The study has no design variable on the requested axle."""


class AxleSpace:
    """The sub design space of one axle: the study slots whose point ref is on that axle.

    Only symmetric variables are supported (the surrogate models the left corner and mirrors
    it). The other axle's parameters stay at the baseline.

    Parameters
    ----------
    space
        Full study design space.
    axle
        ``"front"`` or ``"rear"``.
    """

    def __init__(self, space: DesignSpace, axle: Literal["front", "rear"]) -> None:
        if axle not in ("front", "rear"):
            raise ValueError(f"axle must be 'front' or 'rear', got {axle!r}")
        self.space = space
        self.axle = axle
        idx: list[int] = []
        names: list[str] = []
        for s in space._slots:
            if s.axle != axle:
                continue
            if s.mode != "sym":
                raise StudyConfigError(
                    f"variable {s.var.point!r} is asymmetric; the surrogate supports "
                    f"symmetric variables only"
                )
            idx.extend(range(s.offset, s.offset + s.n))
            names.extend(s.names)
        self.idx: NDArray[np.intp] = np.array(idx, dtype=np.intp)
        self.names: tuple[str, ...] = tuple(names)
        self._base_full: F64 = space.from_vehicle(clip=True)

    @property
    def n(self) -> int:
        """Number of design parameters on this axle."""
        return int(self.idx.size)

    def baseline(self) -> F64:
        """Axle parameters of the baseline vehicle (clipped to the unit cube)."""
        out: F64 = self._base_full[self.idx].copy()
        return out

    def to_full(self, x_axle: ArrayLike) -> F64:
        """Full design vector: ``x_axle`` on this axle, the other axle at the baseline."""
        x = np.asarray(x_axle, dtype=np.float64)
        if x.shape != (self.n,):
            raise ValueError(f"x_axle must have shape ({self.n},), got {x.shape}")
        full = self._base_full.copy()
        full[self.idx] = x
        return full


#: ``auto`` uses ``random-cd`` up to this many samples and plain LHS above.
AUTO_RANDOM_CD_MAX = 5000


def resolve_lhs_optimization(setting: str, n: int) -> Literal["random-cd", "none"]:
    """Resolve ``auto`` / ``random-cd`` / ``none`` for ``n`` samples."""
    if setting == "auto":
        return "random-cd" if n <= AUTO_RANDOM_CD_MAX else "none"
    if setting in ("random-cd", "none"):
        return setting  # type: ignore[return-value]
    raise ValueError(f"lhs_optimization must be auto, random-cd or none, got {setting!r}")


def lhs(n: int, d: int, seed: int, optimization: str = "random-cd") -> F64:
    """Latin hypercube in the unit cube, ``(n, d)``.

    ``optimization`` is ``"random-cd"`` or ``"none"``. Deterministic: depends only on
    ``(n, d, seed, optimization)``. The caller replaces row 0 by the baseline.
    """
    if n < 1 or d < 1:
        raise ValueError("n and d must be >= 1")
    opt = resolve_lhs_optimization(optimization, 0 if optimization == "auto" else n)
    sampler = qmc.LatinHypercube(d=d, optimization=None if opt == "none" else opt, seed=seed)
    out: F64 = np.asarray(sampler.random(n), dtype=np.float64)
    return out
