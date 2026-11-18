"""Batched inference with a saved axle model (experimental, needs torch)."""

# ruff: noqa: N803  (X, Y, Z follow the dataset contract)
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from suspkin.surrogate.train import TrainedModel, predict_physical, resolve_device


class SurrogatePredictor:
    """Evaluate a trained axle network on design vectors in the unit cube."""

    def __init__(self, model: TrainedModel, device: str = "auto", batch_size: int = 8192) -> None:
        self.model = model
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._net = model.build_network(self.device)

    @classmethod
    def load(
        cls, model_dir: str | Path, device: str = "auto", batch_size: int = 8192
    ) -> SurrogatePredictor:
        """Load ``model.pt`` / ``model.json`` / ``norm.npz`` from ``model_dir``."""
        return cls(
            TrainedModel.load(model_dir, device=device), device=device, batch_size=batch_size
        )

    @property
    def param_names(self) -> tuple[str, ...]:
        """Expected design parameter order."""
        return self.model.param_names

    def predict(
        self, X: NDArray[np.float64], *, param_names: Sequence[str] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(Y (B, C, R, T) physical units float64, p_valid (B, R, T))``.

        Raises ``ValueError`` if ``X`` is not ``(B, d)`` or ``param_names`` differ.
        """
        x = np.asarray(X, dtype=np.float64)
        d = len(self.model.param_names)
        if x.ndim != 2 or x.shape[1] != d:
            raise ValueError(f"X must have shape (B, {d}), got {x.shape}")
        if param_names is not None and tuple(param_names) != self.model.param_names:
            raise ValueError("param_names do not match the model")
        with torch.no_grad():
            return predict_physical(
                self.model, x, net=self._net, device=self.device, batch_size=self.batch_size
            )
