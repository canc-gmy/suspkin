"""Hardpoint models, YAML I/O and left/right mirroring.

Hardpoints are entered for the left side only (ISO 8855 frame, SI units, static ride-height
pose) and mirrored to the right by :func:`build_corners`. See ``docs/concepts/vehicle-model.md``
and ``docs/concepts/conventions.md``.
"""

from suspkin.geometry.corners import Corner, CornerId, arb_points, build_corners
from suspkin.geometry.io import VehicleConfigError, load_vehicle, save_vehicle, vehicle_yaml
from suspkin.geometry.model import (
    HARDPOINT_NAMES,
    Axle,
    BeamARB,
    BladeARB,
    Coilover,
    CornerHardpoints,
    Steering,
    TorsionalARB,
    Vec3,
    Vehicle,
    VehicleParams,
    Wheel,
)

__all__ = [
    "HARDPOINT_NAMES",
    "Axle",
    "BeamARB",
    "BladeARB",
    "Coilover",
    "Corner",
    "CornerHardpoints",
    "CornerId",
    "Steering",
    "TorsionalARB",
    "Vec3",
    "Vehicle",
    "VehicleConfigError",
    "VehicleParams",
    "Wheel",
    "arb_points",
    "build_corners",
    "load_vehicle",
    "save_vehicle",
    "vehicle_yaml",
]
