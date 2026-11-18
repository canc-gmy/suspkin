"""suspkin: Formula Student double-wishbone suspension kinematics toolkit.

The names in ``__all__`` form the stable public API. They are imported lazily
(PEP 562) so that ``import suspkin`` stays cheap.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

_LAZY: dict[str, str] = {
    "Vehicle": "suspkin.geometry.models",
    "VehicleConfigError": "suspkin.geometry.io",
    "load_vehicle": "suspkin.geometry.io",
    "save_vehicle": "suspkin.geometry.io",
    "build_corners": "suspkin.geometry.corners",
    "solve_corner": "suspkin.kinematics.corner",
    "sweep_corner": "suspkin.kinematics.corner",
    "sweep_rack": "suspkin.kinematics.corner",
    "solve_vehicle": "suspkin.kinematics.vehicle",
    "KinematicsError": "suspkin.kinematics.rocker",
    "sweep_vehicle": "suspkin.sweeps.vehicle_sweeps",
    "vehicle_frame": "suspkin.sweeps.vehicle_sweeps",
    "states_to_frame": "suspkin.sweeps.corner_sweeps",
    "metrics_frame": "suspkin.metrics.table",
    "add_gradients": "suspkin.metrics.table",
    "vehicle_summary": "suspkin.metrics.vehicle_summary",
    "build_report": "suspkin.report.builder",
    "run_standard_studies": "suspkin.report.studies",
    "plot_vehicle_3d": "suspkin.plotting.geometry3d",
}

__all__ = ["__version__", *_LAZY]

if TYPE_CHECKING:
    from suspkin.geometry.corners import build_corners as build_corners
    from suspkin.geometry.io import VehicleConfigError as VehicleConfigError
    from suspkin.geometry.io import load_vehicle as load_vehicle
    from suspkin.geometry.io import save_vehicle as save_vehicle
    from suspkin.geometry.models import Vehicle as Vehicle
    from suspkin.kinematics.corner import solve_corner as solve_corner
    from suspkin.kinematics.corner import sweep_corner as sweep_corner
    from suspkin.kinematics.corner import sweep_rack as sweep_rack
    from suspkin.kinematics.rocker import KinematicsError as KinematicsError
    from suspkin.kinematics.vehicle import solve_vehicle as solve_vehicle
    from suspkin.metrics.table import add_gradients as add_gradients
    from suspkin.metrics.table import metrics_frame as metrics_frame
    from suspkin.metrics.vehicle_summary import vehicle_summary as vehicle_summary
    from suspkin.plotting.geometry3d import plot_vehicle_3d as plot_vehicle_3d
    from suspkin.report.builder import build_report as build_report
    from suspkin.report.studies import run_standard_studies as run_standard_studies
    from suspkin.sweeps.corner_sweeps import states_to_frame as states_to_frame
    from suspkin.sweeps.vehicle_sweeps import sweep_vehicle as sweep_vehicle
    from suspkin.sweeps.vehicle_sweeps import vehicle_frame as vehicle_frame


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'suspkin' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
