"""Axle data contract: in-memory dataset, design-space sampling and the fixed output grid.

Experimental (see ``docs/guides/stability.md``). Shapes use ``N`` samples, ``d`` design
parameters of the axle, ``C`` channels, ``R`` rack points (1 on the rear axle) and ``T``
travel points.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

from suspkin.optimize.design import DesignSpace
from suspkin.optimize.spec import StudyConfigError
from suspkin.optimize.spec.surrogate import GridSpec

F64 = NDArray[np.float64]

# --------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------

Axle = Literal["front", "rear"]

#: Values of :attr:`AxleDataset.split`.
SPLIT_TRAIN: int = 0
SPLIT_VAL: int = 1
SPLIT_TEST: int = 2


@dataclass(frozen=True)
class AxleDataset:
    """Samples of one axle.

    Attributes:
        axle: ``"front"`` or ``"rear"``.
        X: design parameters in the unit cube, ``(N, d)`` float64.
        Y: channel curves, ``(N, C, R, T)`` float32, nan where ``mask`` is False.
        mask: kinematic validity, ``(N, R, T)`` bool.
        feasible: ``mask.all()`` per sample, ``(N,)`` bool.
        split: :data:`SPLIT_TRAIN`, :data:`SPLIT_VAL` or :data:`SPLIT_TEST`, ``(N,)`` uint8.
        source: ``"lhs"`` or ``"al_<k>"`` per sample, ``(N,)`` str.
        weight: training weight per sample, ``(N,)`` float64 (1.0 for LHS samples).
        channels: channel names, length ``C`` (order of the channel axis of ``Y``).
        param_names: design parameter names, length ``d`` (order of the columns of ``X``).
        travel: travel grid ``(T,)`` [m].
        rack: rack grid ``(R,)`` [m]; ``array([0.0])`` on the rear axle.
        meta: free metadata (hashes, package version, seed, ...).
    """

    axle: Axle
    X: NDArray[np.float64]
    Y: NDArray[np.float32]
    mask: NDArray[np.bool_]
    feasible: NDArray[np.bool_]
    split: NDArray[np.uint8]
    source: NDArray[np.str_]
    weight: NDArray[np.float64]
    channels: tuple[str, ...]
    param_names: tuple[str, ...]
    travel: NDArray[np.float64]
    rack: NDArray[np.float64]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n, d = self.X.shape
        c, r, t = len(self.channels), len(self.rack), len(self.travel)
        if self.Y.shape != (n, c, r, t):
            raise ValueError(f"Y must have shape {(n, c, r, t)}, got {self.Y.shape}")
        if self.mask.shape != (n, r, t):
            raise ValueError(f"mask must have shape {(n, r, t)}, got {self.mask.shape}")
        for name in ("feasible", "split", "source", "weight"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if len(self.param_names) != d:
            raise ValueError(f"param_names must have length {d}, got {len(self.param_names)}")

    @property
    def n(self) -> int:
        """Number of samples."""
        return int(self.X.shape[0])

    def subset(self, which: int | NDArray[np.bool_]) -> AxleDataset:
        """Samples of one split (an ``int`` split code) or of a boolean row mask."""
        rows = self.split == which if isinstance(which, int) else np.asarray(which, dtype=bool)
        return AxleDataset(
            axle=self.axle,
            X=self.X[rows],
            Y=self.Y[rows],
            mask=self.mask[rows],
            feasible=self.feasible[rows],
            split=self.split[rows],
            source=self.source[rows],
            weight=self.weight[rows],
            channels=self.channels,
            param_names=self.param_names,
            travel=self.travel,
            rack=self.rack,
            meta=dict(self.meta),
        )


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------


class NoAxleVariablesError(StudyConfigError):
    """The study has no design variable on the requested axle."""


class AxleSpace:
    """The sub design space of one axle: the study slots whose point ref is on that axle.

    Only symmetric variables are supported (the surrogate models the left corner and mirrors
    it). The other axle's parameters stay at the baseline.

    Args:
        space: Full study design space.
        axle: ``"front"`` or ``"rear"``.
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
    """Resolve ``auto`` / ``random-cd`` / ``none`` for ``n`` samples.

    Args:
        setting: ``"auto"``, ``"random-cd"`` or ``"none"``.
        n: Number of samples.

    Returns:
        The concrete optimization name.
    """
    if setting == "auto":
        return "random-cd" if n <= AUTO_RANDOM_CD_MAX else "none"
    if setting in ("random-cd", "none"):
        return setting  # type: ignore[return-value]
    raise ValueError(f"lhs_optimization must be auto, random-cd or none, got {setting!r}")


def lhs(n: int, d: int, seed: int, optimization: str = "random-cd") -> F64:
    """Latin hypercube in the unit cube, ``(n, d)``.

    ``optimization`` is ``"random-cd"`` or ``"none"``. Deterministic: depends only on
    ``(n, d, seed, optimization)``. The caller replaces row 0 by the baseline.

    Args:
        n: Number of samples.
        d: Dimension.
        seed: Random seed.
        optimization: ``"random-cd"`` or ``"none"``.

    Returns:
        Samples ``(n, d)`` in [0, 1].
    """
    if n < 1 or d < 1:
        raise ValueError("n and d must be >= 1")
    opt = resolve_lhs_optimization(optimization, 0 if optimization == "auto" else n)
    sampler = qmc.LatinHypercube(d=d, optimization=None if opt == "none" else opt, seed=seed)
    out: F64 = np.asarray(sampler.random(n), dtype=np.float64)
    return out


# --------------------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------------------


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
