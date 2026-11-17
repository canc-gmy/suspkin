"""Actuation metrics, motion ratios and wheel/ride rates (DESIGN_PLAN section 5.5).

All functions are pure and return SI floats (m, rad, N, N/m). Travel ``z`` is positive
in bump. The spring force follows ``F = preload + k * (free_length - L)``: ``preload``
is an extra collar preload on top of the compression implied by the free length, and
nothing is clamped (a negative force means the spring would be unloaded).

Virtual work links wheel and spring: with spring motion ratio ``MR = -dL/dz`` the wheel
load of the spring is ``F_w = F_s * MR`` and the wheel rate is
``dF_w/dz = k * MR**2 + F_s * dMR/dz``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.kinematics.corner import CornerState, solve_corner

Vec = NDArray[np.float64]


def _rod(state: CornerState) -> Vec:
    """Pushrod vector, outer -> inner."""
    d: Vec = np.asarray(
        state.points["pushrod_inner"] - state.points["pushrod_outer"], dtype=np.float64
    )
    return d


def pushrod_angle(state: CornerState, corner: Corner) -> float:
    """Angle [rad] of the pushrod (outer -> inner) above the horizontal plane."""
    d = _rod(state)
    return math.atan2(float(d[2]), math.hypot(float(d[0]), float(d[1])))


def pushrod_angle_fv(state: CornerState, corner: Corner) -> float:
    """Front-view (y-z projection) pushrod angle [rad] above horizontal (same on both sides)."""
    d = _rod(state)
    return math.atan2(float(d[2]), abs(float(d[1])))


def rocker_angle(state: CornerState, static: CornerState) -> float:
    """Rocker rotation [rad] relative to the static pose (side-signed like the state)."""
    return state.rocker_angle - static.rocker_angle


def coilover_length(state: CornerState) -> float:
    """Coilover length [m]."""
    return state.coilover_length


def coilover_compression(state: CornerState, static: CornerState) -> float:
    """Compression [m] relative to static, ``L_static - L`` (positive in bump)."""
    return static.coilover_length - state.coilover_length


def bump_stroke_remaining(state: CornerState, corner: Corner) -> float:
    """Remaining compression stroke [m], ``L - length_min``."""
    return state.coilover_length - corner.axle_cfg.coilover.length_min


def droop_stroke_remaining(state: CornerState, corner: Corner) -> float:
    """Remaining extension stroke [m], ``length_max - L``."""
    return corner.axle_cfg.coilover.length_max - state.coilover_length


def spring_force(state: CornerState, corner: Corner) -> float:
    """Spring force [N] ``preload + k * (free_length - L)`` (positive = compressive)."""
    c = corner.axle_cfg.coilover
    return c.preload + c.spring_rate * (c.free_length - state.coilover_length)


def actuation_metrics(state: CornerState, corner: Corner, static: CornerState) -> dict[str, float]:
    """Per-state actuation metrics keyed by snake_case name (m, rad, N).

    ``coilover_length`` and the absolute ``rocker_angle`` are already columns of the
    sweep frame, hence the relative ``rocker_rotation`` and ``coilover_compression``.
    """
    return {
        "pushrod_angle": pushrod_angle(state, corner),
        "pushrod_angle_fv": pushrod_angle_fv(state, corner),
        "rocker_rotation": rocker_angle(state, static),
        "coilover_compression": coilover_compression(state, static),
        "bump_stroke_remaining": bump_stroke_remaining(state, corner),
        "droop_stroke_remaining": droop_stroke_remaining(state, corner),
        "spring_force": spring_force(state, corner),
    }


@dataclass(frozen=True)
class MotionRatios:
    """Motion ratios at one travel (dimensionless); ``progression`` in 1/m.

    ``spring = -dL/dz`` is positive when bump compresses the spring. ``damper`` equals
    ``spring`` (a single coilover is both spring and damper). ``pushrod`` is the rate at
    which the pushrod-inner point moves along the pushrod axis (outer -> inner) per unit
    wheel travel (positive = rod pushes the rocker in bump); it is the inner-point velocity
    component along the rod axis, equal to the outer-point component since the rod is rigid.
    ``rocker = spring / pushrod``.
    ``progression = d(spring)/dz`` (second difference of the coilover length).
    """

    spring: float
    damper: float
    pushrod: float
    rocker: float
    progression: float


def motion_ratios(
    corner: Corner,
    travel: float,
    rack: float = 0.0,
    h: float = 1e-5,
    guess: CornerState | None = None,
) -> MotionRatios:
    """Motion ratios by central finite difference of the solver at ``travel`` +- ``h``.

    Independent of the metric tables. ``guess`` (optional) seeds the solve at ``travel``;
    the neighbouring solves use that state as their guess. The progression uses a
    second difference with the wider step ``10 * h`` to limit round-off.
    """
    state = solve_corner(corner, travel, rack, guess=guess)
    hi = solve_corner(corner, travel + h, rack, guess=state)
    lo = solve_corner(corner, travel - h, rack, guess=state)
    spring = -(hi.coilover_length - lo.coilover_length) / (2.0 * h)
    u = _rod(state)
    u = u / float(np.linalg.norm(u))
    dp = (hi.points["pushrod_inner"] - lo.points["pushrod_inner"]) / (2.0 * h)
    pushrod = float(dp @ u)
    hp = 10.0 * h
    hi2 = solve_corner(corner, travel + hp, rack, guess=state)
    lo2 = solve_corner(corner, travel - hp, rack, guess=state)
    progression = -(hi2.coilover_length - 2.0 * state.coilover_length + lo2.coilover_length) / (
        hp * hp
    )
    return MotionRatios(
        spring=spring,
        damper=spring,
        pushrod=pushrod,
        rocker=spring / pushrod,
        progression=progression,
    )


def wheel_load_spring(force: float, mr_spring: float) -> float:
    """Vertical wheel load [N] of the spring, ``F_s * MR`` (virtual work)."""
    return force * mr_spring


def wheel_rate_linear(spring_rate: float, mr_spring: float) -> float:
    """Linearised wheel rate [N/m], ``k * MR**2``."""
    return spring_rate * mr_spring**2


def wheel_rate(spring_rate: float, mr_spring: float, force: float, progression: float) -> float:
    """Wheel rate [N/m] from virtual work, ``k * MR**2 + F_s * dMR/dz``."""
    return spring_rate * mr_spring**2 + force * progression


def ride_rate(wheel_rate_value: float, tyre_stiffness: float) -> float:
    """Ride rate [N/m]: wheel rate in series with the vertical tyre stiffness."""
    return wheel_rate_value * tyre_stiffness / (wheel_rate_value + tyre_stiffness)
