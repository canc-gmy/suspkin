"""Surrogate model support: output grid, channel registry and corner-curve extraction.

The evaluators and the problem are imported lazily (PEP 562): ``suspkin.optimize.spec``
imports :mod:`suspkin.surrogate.spec`, so importing this package must not pull
``suspkin.optimize`` in eagerly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from suspkin.surrogate.channels import CHANNELS, Channel, channels_for
from suspkin.surrogate.extract import CornerSample, extract_axles, extract_corner
from suspkin.surrogate.grid import AxleGrid, axle_grids
from suspkin.surrogate.vehicle_layer import (
    AxleBatch,
    AxleStatics,
    LayerResult,
    UnsupportedColumnError,
    VehicleStatics,
    layer_error,
    stack_samples,
    summary_fields,
    supported_sweep_columns,
    vehicle_layer,
    vehicle_statics,
)

if TYPE_CHECKING:
    from suspkin.surrogate.evaluate import (
        AxlePredictor,
        BaselineAxle,
        NetworkPredictor,
        OraclePredictor,
        SurrogateEvaluator,
    )
    from suspkin.surrogate.problem import SurrogateProblem

_LAZY = {
    "AxlePredictor": "suspkin.surrogate.evaluate",
    "BaselineAxle": "suspkin.surrogate.evaluate",
    "NetworkPredictor": "suspkin.surrogate.evaluate",
    "OraclePredictor": "suspkin.surrogate.evaluate",
    "SurrogateEvaluator": "suspkin.surrogate.evaluate",
    "SurrogateProblem": "suspkin.surrogate.problem",
}

__all__ = [
    "CHANNELS",
    "AxleBatch",
    "AxleGrid",
    "AxlePredictor",
    "AxleStatics",
    "BaselineAxle",
    "Channel",
    "CornerSample",
    "LayerResult",
    "NetworkPredictor",
    "OraclePredictor",
    "SurrogateEvaluator",
    "SurrogateProblem",
    "UnsupportedColumnError",
    "VehicleStatics",
    "axle_grids",
    "channels_for",
    "extract_axles",
    "extract_corner",
    "layer_error",
    "stack_samples",
    "summary_fields",
    "supported_sweep_columns",
    "vehicle_layer",
    "vehicle_statics",
]


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value
    return value
