"""Surrogate model support: output grid, channel registry and corner-curve extraction."""

from suspkin.surrogate.channels import CHANNELS, Channel, channels_for
from suspkin.surrogate.evaluate import (
    AxlePredictor,
    BaselineAxle,
    NetworkPredictor,
    OraclePredictor,
    SurrogateEvaluator,
)
from suspkin.surrogate.extract import CornerSample, extract_axles, extract_corner
from suspkin.surrogate.grid import AxleGrid, axle_grids
from suspkin.surrogate.problem import SurrogateProblem
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
