"""Vehicle-level kinematics: prescribed body motion, wheel travel from ground contact.

Body motion convention (body pose relative to the fixed ground, ISO 8855 axes):

* ``heave`` h [m]: positive = body UP (wheels droop, travel < 0).
* ``roll`` phi [rad]: right-hand rotation about +x; positive lifts the +y (left) side.
* ``pitch`` theta [rad]: right-hand rotation about +y; positive = nose DOWN.
* ``rack`` [m]: rack displacement, applied to the front corners only.

A body point ``x_b`` maps to the world as ``x_w = R (x_b - pivot) + pivot + h e_z``
with ``R = R_y(theta) R_x(phi)``; ``pivot`` defaults to the vehicle-frame origin. The
corner solver works in the body frame, so the ground is expressed there: the height
of a body point is ``n . (x_b - pivot) + pivot_z + h`` with ``n = R^T e_z`` (the
ground normal in the body frame).

Each wheel's contact point is the point of the tyre circle lowest along ``-n``. The
ground height under wheel k is its *static* contact height (so the static pose has
travel 0 exactly, even when the loaded radius or camber puts the static contact point
slightly off z = 0). Wheel travel is found per corner by a 1-D secant iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin.geom_math.rotations import axis_angle_matrix
from suspkin.geom_math.vectors import wheel_spin_axis
from suspkin.geometry.corners import Corner, CornerId, build_corners
from suspkin.geometry.models import Vehicle
from suspkin.kinematics.arb import ArbState, solve_arb
from suspkin.kinematics.corner import CornerState, solve_corner
from suspkin.kinematics.rocker import KinematicsError

Vec = NDArray[np.float64]
_E_Z = np.array([0.0, 0.0, 1.0])
_ORDER: tuple[CornerId, ...] = ("FL", "FR", "RL", "RR")
_G_TOL = 1e-11
_MAX_ITER = 30
_MAX_TRAVEL = 0.25


@dataclass(frozen=True)
class VehicleState:
    """Solved vehicle pose. Lengths in m, angles in rad, body frame."""

    heave: float
    roll: float
    pitch: float
    rack: float
    corners: dict[CornerId, CornerState]
    contact_points: dict[CornerId, Vec]
    ground_normal: Vec
    arb: dict[str, ArbState | None]
    max_residual: float


def contact_point(state: CornerState, ground_normal: ArrayLike, tyre_radius: float) -> Vec:
    """Tyre point lowest along ``-ground_normal`` (body frame)."""
    n = np.asarray(ground_normal, dtype=np.float64)
    s = np.asarray(state.spin_axis, dtype=np.float64)
    up = n - (n @ s) * s
    up = up / np.linalg.norm(up)
    out: Vec = np.asarray(state.points["wheel_center"], dtype=np.float64) - tyre_radius * up
    return out


def body_rotation(roll: float, pitch: float) -> Vec:
    """Body-to-world rotation ``R_y(pitch) R_x(roll)``."""
    r: Vec = axis_angle_matrix([0.0, 1.0, 0.0], pitch) @ axis_angle_matrix([1.0, 0.0, 0.0], roll)
    return r


def _static_contact(corner: Corner) -> Vec:
    cfg = corner.axle_cfg
    s = wheel_spin_axis(corner.side, cfg.wheel.static_camber, cfg.wheel.static_toe)
    up = _E_Z - s[2] * s
    up = up / np.linalg.norm(up)
    out: Vec = np.asarray(corner.points["wheel_center"]) - cfg.wheel.tyre_radius * up
    return out


def _solve_corner_ground(
    corner: Corner,
    n: Vec,
    offset: float,
    z_static: float,
    rack: float,
    guess: CornerState | None,
) -> tuple[CornerState, Vec, float]:
    """Secant iteration on travel so that the contact height equals ``z_static``."""
    radius = corner.axle_cfg.wheel.tyre_radius
    nz = float(n[2])

    def height(st: CornerState) -> tuple[Vec, float]:
        cp = contact_point(st, n, radius)
        return cp, float(n @ cp) + offset - z_static

    if guess is not None:
        t = guess.travel
    else:
        g0 = float(n @ _static_contact(corner)) + offset - z_static
        t = -g0 / nz
    if abs(t) > _MAX_TRAVEL:
        raise KinematicsError(f"{corner.id}: wheel cannot reach the ground (travel {t:.3f} m)")
    st = solve_corner(corner, t, rack, guess=guess)
    cp, g = height(st)
    t_prev, g_prev = t, g
    t = t - g / nz
    for _ in range(_MAX_ITER):
        if abs(g) < _G_TOL:
            return st, cp, g
        if abs(t) > _MAX_TRAVEL:
            raise KinematicsError(f"{corner.id}: wheel cannot reach the ground (travel {t:.3f} m)")
        st = solve_corner(corner, t, rack, guess=st)
        cp, g = height(st)
        if abs(g) < _G_TOL:
            return st, cp, g
        dg = g - g_prev
        slope = dg / (t - t_prev) if abs(t - t_prev) > 1e-15 and abs(dg) > 1e-18 else nz
        t_prev, g_prev = t, g
        t = t - g / slope
    raise KinematicsError(f"{corner.id}: ground contact did not converge (residual {g:.2e} m)")


def solve_vehicle(
    vehicle: Vehicle,
    corners: dict[CornerId, Corner] | None = None,
    *,
    heave: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    rack: float = 0.0,
    pivot: ArrayLike = (0.0, 0.0, 0.0),
    guess: VehicleState | None = None,
) -> VehicleState:
    """Solve all corners (and ARBs) for a prescribed body pose.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle description (used for the ARBs and, if ``corners`` is None, the corners).
    corners : dict, optional
        Result of ``build_corners(vehicle)``; built on demand when omitted.
    heave, roll, pitch, rack : float
        Body heave [m, up positive], roll [rad, left up], pitch [rad, nose down], rack [m].
    pivot : array-like
        Centre of the body rotations (default: vehicle-frame origin).
    guess : VehicleState, optional
        Neighbouring solution for continuation.

    Raises
    ------
    KinematicsError
        If a corner cannot reach the ground, or a corner/ARB solve fails.
    """
    if corners is None:
        corners = build_corners(vehicle)
    pv = np.asarray(pivot, dtype=np.float64)
    rot = body_rotation(roll, pitch)
    n: Vec = rot.T @ _E_Z
    offset = float(pv[2]) - float(n @ pv) + heave
    states: dict[CornerId, CornerState] = {}
    cps: dict[CornerId, Vec] = {}
    worst = 0.0
    for cid in _ORDER:
        c = corners[cid]
        z_static = float(_static_contact(c)[2])
        st, cp, g = _solve_corner_ground(
            c,
            n,
            offset,
            z_static,
            rack if c.axle == "front" else 0.0,
            None if guess is None else guess.corners[cid],
        )
        cp = cp.copy()
        cp.setflags(write=False)
        states[cid] = st
        cps[cid] = cp
        worst = max(worst, st.max_residual, abs(g))
    arbs: dict[str, ArbState | None] = {}
    for axle, left, right in (("front", "FL", "FR"), ("rear", "RL", "RR")):
        cfg = vehicle.front if axle == "front" else vehicle.rear
        if cfg.arb is None:
            arbs[axle] = None
            continue
        prev = None if guess is None else guess.arb.get(axle)
        a = solve_arb(cfg, states[left], states[right], guess=prev)  # type: ignore[index]
        arbs[axle] = a
        worst = max(worst, a.max_residual)
    n.setflags(write=False)
    return VehicleState(
        heave=float(heave),
        roll=float(roll),
        pitch=float(pitch),
        rack=float(rack),
        corners=states,
        contact_points=cps,
        ground_normal=n,
        arb=arbs,
        max_residual=worst,
    )
