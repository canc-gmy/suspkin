"""Per-axle MLP (regression and validity heads) plus the curve standardisation and POD basis.

Experimental, needs torch. Curves are ``(N, C, R, T)`` with a validity mask ``(N, R, T)``.
Masked entries are set to 0 after scaling, which is the channel mean, so they carry no
signal into the POD.
"""

# ruff: noqa: N803  (X, Y, Z follow the dataset contract)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

_EPS = 1e-12

# --------------------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------------------

Activation = Literal["silu", "gelu", "relu", "tanh"]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


@dataclass(frozen=True)
class MLPConfig:
    """Architecture of :class:`SurrogateMLP`."""

    n_layers: int = 4
    width: int = 256
    activation: Activation = "silu"
    residual: bool = True
    layernorm: bool = False
    dropout: float = 0.0
    output: Literal["raw", "pod"] = "raw"
    pod_k: int = 32

    def __post_init__(self) -> None:
        if self.n_layers < 1 or self.width < 1:
            raise ValueError("n_layers and width must be >= 1")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.output not in ("raw", "pod"):
            raise ValueError(f"output must be 'raw' or 'pod', got {self.output!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _Block(nn.Module):
    def __init__(self, n_in: int, n_out: int, cfg: MLPConfig) -> None:
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.act = _ACTIVATIONS[cfg.activation]()
        self.norm: nn.Module = nn.LayerNorm(n_out) if cfg.layernorm else nn.Identity()
        self.drop: nn.Module = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.skip = cfg.residual and n_in == n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.norm(self.act(self.lin(x))))
        out: torch.Tensor = x + h if self.skip else h
        return out


class SurrogateMLP(nn.Module):
    """MLP ``(B, d) -> (regression (B, n_out), validity logits (B, n_valid))``."""

    def __init__(self, d: int, n_out: int, n_valid: int, cfg: MLPConfig) -> None:
        super().__init__()
        widths = [d] + [cfg.width] * cfg.n_layers
        self.blocks = nn.ModuleList(
            _Block(widths[i], widths[i + 1], cfg) for i in range(cfg.n_layers)
        )
        self.head_reg = nn.Linear(cfg.width, n_out)
        self.head_valid = nn.Linear(cfg.width, n_valid)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the regression output and the validity logits."""
        h = x
        for blk in self.blocks:
            h = blk(h)
        return self.head_reg(h), self.head_valid(h)


# --------------------------------------------------------------------------------------
# standardisation and POD
# --------------------------------------------------------------------------------------


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
