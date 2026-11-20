"""Corner, rocker, ARB and vehicle-level kinematic solvers.

Rigid-body, ideal-joint solvers in the ISO 8855 body frame (SI units, angles in rad). The
single-corner solver is :func:`solve_corner`; :func:`solve_vehicle` poses all four corners
for a heave/roll/pitch/rack input. See ``docs/concepts/corner-kinematics.md`` and
``docs/concepts/vehicle-kinematics.md``; sign rules in ``docs/concepts/conventions.md``.
"""

from suspkin.kinematics.arb import ArbState, arb_energy, arb_stiffness, solve_arb
from suspkin.kinematics.corner import (
    CornerState,
    solve_corner,
    sweep_corner,
    sweep_rack,
)
from suspkin.kinematics.rocker import KinematicsError, RockerResult, solve_rocker
from suspkin.kinematics.vehicle import VehicleState, cg_pivot, contact_point, solve_vehicle

__all__ = [
    "ArbState",
    "CornerState",
    "KinematicsError",
    "RockerResult",
    "VehicleState",
    "arb_energy",
    "arb_stiffness",
    "cg_pivot",
    "contact_point",
    "solve_arb",
    "solve_corner",
    "solve_rocker",
    "solve_vehicle",
    "sweep_corner",
    "sweep_rack",
]
