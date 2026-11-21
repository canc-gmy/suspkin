"""suspkin: Formula Student double-wishbone suspension kinematics toolkit.

The names in ``__all__`` form the stable public API (vehicle loading, corner and vehicle
solvers, sweep tables, metrics, report and 3D plot). They are imported lazily (PEP 562
module ``__getattr__``) so that ``import suspkin`` stays cheap and does not load the
solver, plotting or report dependencies. Frame, unit and sign rules are in
``docs/concepts/conventions.md``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "2.0.0"

_LAZY: dict[str, str] = {
    "Vehicle": "suspkin.geometry.model",
    "VehicleConfigError": "suspkin.geometry.io",
    "load_vehicle": "suspkin.geometry.io",
    "save_vehicle": "suspkin.geometry.io",
    "build_corners": "suspkin.geometry.corners",
    "solve_corner": "suspkin.kinematics.corner",
    "sweep_corner": "suspkin.kinematics.corner",
    "sweep_rack": "suspkin.kinematics.corner",
    "solve_vehicle": "suspkin.kinematics.vehicle",
    "KinematicsError": "suspkin.kinematics.rocker",
    "sweep_vehicle": "suspkin.sweeps",
    "vehicle_frame": "suspkin.sweeps",
    "states_to_frame": "suspkin.sweeps",
    "metrics_frame": "suspkin.metrics.table",
    "add_gradients": "suspkin.metrics.table",
    "vehicle_summary": "suspkin.metrics.vehicle",
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
    from suspkin.geometry.model import Vehicle as Vehicle
    from suspkin.kinematics.corner import solve_corner as solve_corner
    from suspkin.kinematics.corner import sweep_corner as sweep_corner
    from suspkin.kinematics.corner import sweep_rack as sweep_rack
    from suspkin.kinematics.rocker import KinematicsError as KinematicsError
    from suspkin.kinematics.vehicle import solve_vehicle as solve_vehicle
    from suspkin.metrics.table import add_gradients as add_gradients
    from suspkin.metrics.table import metrics_frame as metrics_frame
    from suspkin.metrics.vehicle import vehicle_summary as vehicle_summary
    from suspkin.plotting.geometry3d import plot_vehicle_3d as plot_vehicle_3d
    from suspkin.report.builder import build_report as build_report
    from suspkin.report.studies import run_standard_studies as run_standard_studies
    from suspkin.sweeps import states_to_frame as states_to_frame
    from suspkin.sweeps import sweep_vehicle as sweep_vehicle
    from suspkin.sweeps import vehicle_frame as vehicle_frame


def __getattr__(name: str) -> Any:
    """Import a public name on first access (PEP 562) and cache it in the module globals.

    Args:
        name: Attribute requested from the ``suspkin`` module.

    Returns:
        The object registered under ``name`` in ``_LAZY``.

    Raises:
        AttributeError: If ``name`` is not a public lazy name.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'suspkin' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Module attributes plus the lazy public names (for completion)."""
    return sorted([*globals(), *_LAZY])
