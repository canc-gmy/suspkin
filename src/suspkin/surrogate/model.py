"""Per-axle MLP with a regression head and a validity head (experimental, needs torch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

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
