"""In-memory dataset of one axle: the contract between sampling (S3) and training (S4).

Experimental (see ``docs/guides/stability.md``). Shapes use ``N`` samples, ``d`` design
parameters of the axle, ``C`` channels, ``R`` rack points (1 on the rear axle) and ``T``
travel points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

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
