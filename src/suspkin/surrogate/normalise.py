"""Standardisation and POD compression of channel curves (numpy only, experimental).

Curves are ``(N, C, R, T)`` with a validity mask ``(N, R, T)``. Masked entries are set to
0 after scaling, which is the channel mean, so they carry no signal into the POD.
"""

# ruff: noqa: N803  (X, Y, Z follow the dataset contract)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-12


@dataclass(frozen=True)
class ChannelScaler:
    """Per-channel standardisation fitted on valid points only.

    Attributes:
        mean: channel means ``(C,)``.
        std: channel standard deviations ``(C,)`` (1 for constant channels).
    """

    mean: NDArray[np.float64]
    std: NDArray[np.float64]

    @classmethod
    def fit(cls, Y: NDArray[Any], mask: NDArray[np.bool_]) -> ChannelScaler:
        """Fit on ``Y (N, C, R, T)`` over the points where ``mask (N, R, T)`` is True."""
        m = np.broadcast_to(mask[:, None, :, :], Y.shape)
        cnt = np.maximum(m.sum(axis=(0, 2, 3)), 1).astype(np.float64)
        y = np.where(m, Y, 0.0).astype(np.float64)
        mean = y.sum(axis=(0, 2, 3)) / cnt
        dev = np.where(m, y - mean[None, :, None, None], 0.0)
        std = np.sqrt((dev**2).sum(axis=(0, 2, 3)) / cnt)
        std = np.where(std < _EPS, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(
        self, Y: NDArray[Any], mask: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float32]:
        """Standardise ``Y``; masked (or non-finite) entries become 0."""
        z = (np.nan_to_num(Y.astype(np.float64), nan=0.0) - self.mean[None, :, None, None]) / (
            self.std[None, :, None, None]
        )
        if mask is not None:
            z = np.where(mask[:, None, :, :], z, 0.0)
        z = np.where(np.isfinite(Y), z, 0.0)
        return z.astype(np.float32)

    def inverse(self, Z: NDArray[Any]) -> NDArray[np.float64]:
        """Map standardised curves ``(..., C, R, T)`` back to physical units."""
        z = np.asarray(Z, dtype=np.float64)
        shape = (1,) * (z.ndim - 3) + (-1, 1, 1)
        return z * self.std.reshape(shape) + self.mean.reshape(shape)

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Plain arrays for npz storage."""
        return {"scaler_mean": self.mean, "scaler_std": self.std}

    @classmethod
    def from_dict(cls, d: Any) -> ChannelScaler:
        """Inverse of :meth:`to_dict`."""
        return cls(mean=np.asarray(d["scaler_mean"]), std=np.asarray(d["scaler_std"]))


@dataclass(frozen=True)
class PodBasis:
    """Orthonormal POD basis of flattened standardised curves.

    Attributes:
        components: ``(k, D)`` orthonormal rows, ``D = C*R*T``.
        explained: explained variance ratio per component ``(k,)``.
    """

    components: NDArray[np.float64]
    explained: NDArray[np.float64]

    @property
    def k(self) -> int:
        """Number of retained modes."""
        return int(self.components.shape[0])

    @property
    def explained_total(self) -> float:
        """Total explained variance ratio of the retained modes."""
        return float(self.explained.sum())

    @classmethod
    def fit(cls, Z: NDArray[Any], k: int, *, max_rows: int = 4000, seed: int = 0) -> PodBasis:
        """SVD of standardised curves ``Z`` (N, ...) flattened; ``k`` is clipped to the rank.

        At most ``max_rows`` rows (random subset) enter the SVD to bound memory.
        """
        a = np.asarray(Z, dtype=np.float32).reshape(len(Z), -1)
        if len(a) > max_rows:
            rows = np.random.default_rng(seed).choice(len(a), size=max_rows, replace=False)
            a = a[np.sort(rows)]
        _, s, vt = np.linalg.svd(a, full_matrices=False)
        var = s.astype(np.float64) ** 2
        total = max(var.sum(), _EPS)
        k = int(max(1, min(k, len(s))))
        return cls(components=vt[:k].astype(np.float64), explained=var[:k] / total)

    def encode(self, Z: NDArray[Any]) -> NDArray[np.float64]:
        """Coefficients ``(N, k)`` of curves ``Z (N, ...)``."""
        z = np.asarray(Z, dtype=np.float64).reshape(len(Z), -1)
        return z @ self.components.T

    def decode(self, coef: NDArray[Any]) -> NDArray[np.float64]:
        """Flattened curves ``(N, D)`` from coefficients ``(N, k)``."""
        return np.asarray(coef, dtype=np.float64) @ self.components

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Plain arrays for npz storage."""
        return {"pod_components": self.components, "pod_explained": self.explained}

    @classmethod
    def from_dict(cls, d: Any) -> PodBasis:
        """Inverse of :meth:`to_dict`."""
        return cls(
            components=np.asarray(d["pod_components"]), explained=np.asarray(d["pod_explained"])
        )
