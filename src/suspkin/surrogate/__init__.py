"""Surrogate model support: output grid, channel registry and corner-curve extraction."""

from suspkin.surrogate.channels import CHANNELS, Channel, channels_for
from suspkin.surrogate.extract import CornerSample, extract_axles, extract_corner
from suspkin.surrogate.grid import AxleGrid, axle_grids

__all__ = [
    "CHANNELS",
    "AxleGrid",
    "Channel",
    "CornerSample",
    "axle_grids",
    "channels_for",
    "extract_axles",
    "extract_corner",
]
