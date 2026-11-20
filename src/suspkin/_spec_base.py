"""Shared building blocks of the study spec models (leaf module, no suspkin imports)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from suspkin.units import Dim, parse_quantity


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _scalar(v: Any, dim: Dim, factor: float = 1.0) -> float:
    """Number or unit string -> SI; ``factor`` scales bare numbers (``units: mm``)."""
    if isinstance(v, str):
        return parse_quantity(v, dim)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"expected a number or a string with unit, got {v!r}")
    return float(v) * factor
