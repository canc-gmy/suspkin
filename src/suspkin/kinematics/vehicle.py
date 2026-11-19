"""Vehicle-level kinematics: prescribed body motion, wheel travel from ground contact.

Body motion convention (body pose relative to the fixed ground, ISO 8855 axes):

* ``heave`` h [m]: positive = body UP (wheels droop, travel < 0).
* ``roll`` phi [rad]: right-hand rotation about +x; positive lifts the +y (left) side.
* ``pitch`` theta [rad]: right-hand rotation about +y; positive = nose DOWN.
* ``rack`` [m]: rack displacement, applied to the front corners only.

A body point ``x_b`` maps to the world as ``x_w = R (x_b - pivot) + pivot + h e_z``
with ``R = R_y(theta) R_x(phi)``. ``pivot`` defaults to the vehicle CG
(``cg_pivot``: x = -a, y = 0, z = cg_height, with a = wheelbase (1 - front_mass_ratio)
measured rearward from the front axle); it applies to roll and pitch. The
corner solver works in the body frame, so the ground is expressed there: the height
of a body point is ``n . (x_b - pivot) + pivot_z + h`` with ``n = R^T e_z`` (the
ground normal in the body frame).

Each wheel's contact point is the point of the tyre circle lowest along ``-n``. The
ground height under wheel k is its *static* contact height (so the static pose has
travel 0 exactly, even when the loaded radius or camber puts the static contact point
slightly off z = 0). Wheel travel is found per corner by a 1-D secant iteration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin.geometry.corners import Corner, CornerId, build_corners
from suspkin.geometry.model import Vehicle
from suspkin.geometry.vectors import _as3, body_rotation, wheel_spin_axis
from suspkin.kinematics import _numba
from suspkin.kinematics.arb import ArbState, solve_arb
from suspkin.kinematics.corner import (
    _NEWTON_F_TOL,
    CornerState,
    _Ctx,
    _finish,
    _Pose,
    _pose,
    _solve_light,
    _state,
    _tie_inner,
)
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
    pivot: Vec
    # Converged ground-contact secant slopes per corner (warm start of the next pose).
    _slopes: dict[CornerId, float] = field(default_factory=dict, repr=False, compare=False)


def contact_point(state: CornerState, ground_normal: ArrayLike, tyre_radius: float) -> Vec:
    """Tyre point lowest along ``-ground_normal`` (body frame)."""
    return _contact(
        np.asarray(state.points["wheel_center"], dtype=np.float64),
        np.asarray(state.spin_axis, dtype=np.float64),
        np.asarray(ground_normal, dtype=np.float64),
        tyre_radius,
    )


def _contact(
    wc: Sequence[float] | Vec, s: Sequence[float] | Vec, n: Vec, tyre_radius: float
) -> Vec:
    n0, n1, n2 = float(n[0]), float(n[1]), float(n[2])
    s0, s1, s2 = float(s[0]), float(s[1]), float(s[2])
    k = n0 * s0 + n1 * s1 + n2 * s2
    u0, u1, u2 = n0 - k * s0, n1 - k * s1, n2 - k * s2
    nu = math.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
    return np.array(
        [
            float(wc[0]) - tyre_radius * (u0 / nu),
            float(wc[1]) - tyre_radius * (u1 / nu),
            float(wc[2]) - tyre_radius * (u2 / nu),
        ]
    )


def cg_pivot(vehicle: Vehicle) -> Vec:
    """Default rotation pivot: the vehicle CG ``(-a, 0, cg_height)`` in the vehicle frame."""
    a = vehicle.wheelbase * (1.0 - vehicle.vehicle.front_mass_ratio)
    return np.array([-a, 0.0, vehicle.vehicle.cg_height])


def _static_contact(corner: Corner, spin: Vec | None = None) -> Vec:
    cfg = corner.axle_cfg
    s = (
        wheel_spin_axis(corner.side, cfg.wheel.static_camber, cfg.wheel.static_toe)
        if spin is None
        else spin
    )
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
    ctx: _Ctx | None = None,
    slope0: float | None = None,
) -> tuple[CornerState, Vec, float, float]:
    """Secant iteration on travel so that the contact height equals ``z_static``.

    The iterates are solved without their point tables (``_solve_light``); only the
    converged one is turned into a ``CornerState``. ``slope0`` is the converged secant
    slope of the neighbouring pose, used for the first step (default ``n_z``). Returns
    the state, its contact point, the height residual and the last secant slope.
    """
    radius = corner.axle_cfg.wheel.tyre_radius
    nz = float(n[2])
    c = _Ctx(corner) if ctx is None else ctx

    def height(st: CornerState | _Pose) -> tuple[Vec, float]:
        if isinstance(st, CornerState):
            cp = contact_point(st, n, radius)
        else:
            cp = _contact(st.wheel_center, st.spin_axis, n, radius)
        return cp, float(n @ cp) + offset - z_static

    if guess is not None and _numba.available() and (c.rack_axis is not None or rack == 0.0):
        tie_inner = _tie_inner(c, rack)
        res = _numba.ground(c, n, offset, z_static, _as3(tie_inner), guess, slope0)
        if res is not None:
            t_k, x_k, f_k, phi0, slope_k = res
            try:
                pose = _pose(c, x_k, f_k, t_k, rack, tie_inner, phi0)
            except KinematicsError:
                pose = None
            if pose is not None:
                cp, g = height(pose)
                return _finish(c, pose), cp, g, slope_k
        # Any failure: run the pure-Python path below from the same inputs.

    if guess is not None:
        t = guess.travel
    else:
        g0 = (
            float(n @ _static_contact(corner, None if ctx is None else ctx.spin0))
            + offset
            - z_static
        )
        t = -g0 / nz
    if abs(t) > _MAX_TRAVEL:
        raise KinematicsError(f"{corner.id}: wheel cannot reach the ground (travel {t:.3f} m)")
    st: CornerState | _Pose
    if guess is not None and guess.rack == rack and guess.max_residual < _NEWTON_F_TOL:
        # Re-solving the guess at its own travel and rack returns the guess.
        st = guess
    else:
        st = _solve_light(c, t, rack, guess)
    cp, g = height(st)
    slope = nz
    if slope0 is not None and 0.5 < slope0 / nz < 2.0:
        slope = slope0
    t_prev, g_prev = t, g
    t = t - g / slope
    for _ in range(_MAX_ITER):
        if abs(g) < _G_TOL:
            return _state(c, st), cp, g, slope
        if abs(t) > _MAX_TRAVEL:
            raise KinematicsError(f"{corner.id}: wheel cannot reach the ground (travel {t:.3f} m)")
        st = _solve_light(c, t, rack, st)
        cp, g = height(st)
        if abs(g) < _G_TOL:
            return _state(c, st), cp, g, slope
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
    pivot: ArrayLike | None = None,
    guess: VehicleState | None = None,
    _ctxs: Mapping[CornerId, _Ctx] | None = None,
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
        Centre of the body rotations (default: the vehicle CG, see ``cg_pivot``).
    guess : VehicleState, optional
        Neighbouring solution for continuation.

    Raises
    ------
    KinematicsError
        If a corner cannot reach the ground, or a corner/ARB solve fails.
    """
    if corners is None:
        corners = build_corners(vehicle)
    pv = cg_pivot(vehicle) if pivot is None else np.asarray(pivot, dtype=np.float64)
    rot = body_rotation(roll, pitch)
    n: Vec = rot.T @ _E_Z
    offset = float(pv[2]) - float(n @ pv) + heave
    states: dict[CornerId, CornerState] = {}
    cps: dict[CornerId, Vec] = {}
    slopes: dict[CornerId, float] = {}
    worst = 0.0
    for cid in _ORDER:
        c = corners[cid]
        ctx = _Ctx(c) if _ctxs is None else _ctxs[cid]
        z_static = float(_static_contact(c, ctx.spin0)[2])
        st, cp, g, slopes[cid] = _solve_corner_ground(
            c,
            n,
            offset,
            z_static,
            rack if c.axle == "front" else 0.0,
            None if guess is None else guess.corners[cid],
            ctx,
            None if guess is None else guess._slopes.get(cid),
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
        pivot=np.array(pv, dtype=np.float64),
        _slopes=slopes,
    )
