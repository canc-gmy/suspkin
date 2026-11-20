"""Scalar per-pose metric functions, kept as test-only oracles for ``metrics.arrays``.

These were the per-state reference implementations in ``suspkin.metrics.corner``; the
package computes the same columns with the vectorised ``metrics.arrays`` and
``kinematics.batch``. Sign conventions: see docs/concepts/conventions.md.
All return SI floats (m, rad, N, N/m); metrics relative to a static pose take it explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.kinematics.corner import CornerState, solve_corner
from suspkin.metrics.corner import (
    _NAN,
    _angle,
    _out,
    _roll_line,
    front_view_ic,
    side_view_ic,
    sv_swing_angle,
    sv_wc_angle,
)
from suspkin.metrics.corner import (
    steer_angle as steer_angle,
)

Vec = NDArray[np.float64]


def static_state(corner: Corner) -> CornerState:
    """Solve the static pose (travel 0, rack 0) of a corner."""
    return solve_corner(corner, 0.0)


def _axis(state: CornerState) -> Vec:
    """Steering axis direction, LBJ -> UBJ (not normalised)."""
    d: Vec = np.asarray(state.points["uca_outer"] - state.points["lca_outer"], dtype=np.float64)
    return d


def _ground_intercept(state: CornerState, z: float) -> Vec:
    """Point where the steering axis crosses the horizontal plane at height ``z``."""
    lbj = state.points["lca_outer"]
    u = _axis(state)
    t = (z - lbj[2]) / u[2]
    out: Vec = np.asarray(lbj + t * u, dtype=np.float64)
    return out


def camber(state: CornerState, corner: Corner) -> float:
    """Camber angle [rad] to the body; negative = top leaning toward the centreline."""
    return -math.asin(float(np.clip(state.spin_axis[2], -1.0, 1.0)))


def toe(state: CornerState, corner: Corner) -> float:
    """Toe angle [rad]; positive = toe-in on both sides."""
    n = state.spin_axis
    return math.atan2(float(n[0]), _out(corner) * float(n[1]))


def caster(state: CornerState, corner: Corner) -> float:
    """Caster angle [rad]; positive = top of steering axis rearward."""
    u = _axis(state)
    return math.atan2(-float(u[0]), float(u[2]))


def kpi(state: CornerState, corner: Corner) -> float:
    """Kingpin inclination [rad]; positive = top of steering axis inboard."""
    u = _axis(state)
    return math.atan2(-_out(corner) * float(u[1]), float(u[2]))


def scrub_radius(state: CornerState, corner: Corner) -> float:
    """Scrub radius [m]; positive when the ground intercept is inboard of the patch."""
    cp = state.points["contact_patch"]
    g = _ground_intercept(state, float(cp[2]))
    return _out(corner) * float(cp[1] - g[1])


def mechanical_trail(state: CornerState, corner: Corner) -> float:
    """Mechanical trail [m]; positive when the ground intercept is ahead (+x) of the patch."""
    cp = state.points["contact_patch"]
    g = _ground_intercept(state, float(cp[2]))
    return float(g[0] - cp[0])


def kingpin_offset_wc(state: CornerState, corner: Corner) -> float:
    """Kingpin offset at wheel centre [m].

    Lateral (y) distance at wheel-centre height between the steering axis and the
    wheel centre; positive when the axis is inboard of the wheel centre.
    """
    wc = state.points["wheel_center"]
    a = _ground_intercept(state, float(wc[2]))
    return _out(corner) * float(wc[1] - a[1])


def spindle_length(state: CornerState, corner: Corner) -> float:
    """Shortest 3D distance [m] from the wheel centre to the steering axis line."""
    u = _axis(state)
    u = u / np.linalg.norm(u)
    r = state.points["wheel_center"] - state.points["lca_outer"]
    return float(np.linalg.norm(r - (r @ u) * u))


def half_track(state: CornerState, corner: Corner) -> float:
    """Half-track [m]: ``|y|`` of the contact patch."""
    return abs(float(state.points["contact_patch"][1]))


def half_track_change(state: CornerState, static: CornerState, corner: Corner) -> float:
    """Half-track change [m] relative to ``static``; positive = wheel moves outboard."""
    return half_track(state, corner) - half_track(static, corner)


def contact_patch_dx(state: CornerState, static: CornerState, corner: Corner) -> float:
    """Longitudinal contact-patch shift [m] relative to ``static`` (positive = forward)."""
    return float(state.points["contact_patch"][0] - static.points["contact_patch"][0])


def contact_patch_dy(state: CornerState, static: CornerState, corner: Corner) -> float:
    """Lateral contact-patch shift [m] relative to ``static`` (positive = outboard)."""
    d = float(state.points["contact_patch"][1] - static.points["contact_patch"][1])
    return _out(corner) * d


def wheel_center_recession(state: CornerState, static: CornerState, corner: Corner) -> float:
    """Wheel-centre x shift [m] relative to ``static`` (positive = forward)."""
    return float(state.points["wheel_center"][0] - static.points["wheel_center"][0])


def wheelbase_change_contribution(state: CornerState, static: CornerState, corner: Corner) -> float:
    """Contribution of this corner to wheelbase change [m].

    Wheelbase = x_front - x_rear, so a front corner contributes ``+dx`` and a rear
    corner ``-dx`` of its contact-patch x shift relative to ``static``.
    """
    dx = contact_patch_dx(state, static, corner)
    return dx if corner.axle == "front" else -dx


def alignment_metrics(state: CornerState, corner: Corner, static: CornerState) -> dict[str, float]:
    """All alignment and wheel-position metrics of a state, keyed by snake_case name."""
    return {
        "camber": camber(state, corner),
        "toe": toe(state, corner),
        "steer_angle": steer_angle(state, corner),
        "caster": caster(state, corner),
        "kpi": kpi(state, corner),
        "scrub_radius": scrub_radius(state, corner),
        "mechanical_trail": mechanical_trail(state, corner),
        "kingpin_offset_wc": kingpin_offset_wc(state, corner),
        "spindle_length": spindle_length(state, corner),
        "half_track": half_track(state, corner),
        "half_track_change": half_track_change(state, static, corner),
        "contact_patch_dx": contact_patch_dx(state, static, corner),
        "contact_patch_dy": contact_patch_dy(state, static, corner),
        "wheel_center_recession": wheel_center_recession(state, static, corner),
        "wheelbase_change_contribution": wheelbase_change_contribution(state, static, corner),
    }


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


def fvsa_length(state: CornerState, corner: Corner) -> float:
    """Front-view swing-arm length |y_ic - y_cp| [m]; ``inf`` if the IC is at infinity."""
    ic = front_view_ic(state, corner)
    if not ic.is_finite:
        return math.inf
    return abs(ic.h - float(state.points["contact_patch"][1]))


def svsa_length(state: CornerState, corner: Corner) -> float:
    """Side-view swing-arm length |x_ic - x_cp| [m]; ``inf`` if the IC is at infinity."""
    ic = side_view_ic(state, corner)
    if not ic.is_finite:
        return math.inf
    return abs(ic.h - float(state.points["contact_patch"][0]))


def _inboard(corner: Corner) -> float:
    return -1.0 if corner.side == "left" else 1.0


def fv_swing_angle(state: CornerState, corner: Corner) -> float:
    """Angle [rad] of the line contact patch -> FVIC.

    Positive when the FVIC is above the patch; equals ``atan2(dz, inboard * dy)``
    (inboard positive), range (-pi, pi]. An inboard IC gives |angle| < pi/2.
    """
    cp = state.points["contact_patch"]
    return _angle(front_view_ic(state, corner), float(cp[1]), float(cp[2]), _inboard(corner))


def roll_centre_symmetric(state: CornerState, corner: Corner) -> float:
    """Roll-centre height z [m]: line contact patch -> FVIC cut with the centreline y = 0.

    Returns nan if that line is parallel to y = 0 (vertical in the front view).
    """
    p0, d = _roll_line(state)
    if abs(d[0]) < 1e-12:
        return _NAN
    return float(p0[1] + d[1] / d[0] * (0.0 - p0[0]))


def centre_metrics(state: CornerState, corner: Corner) -> dict[str, float]:
    """All corner-level centre metrics (lengths in m, angles in rad)."""
    fv = front_view_ic(state, corner)
    sv = side_view_ic(state, corner)
    return {
        "fvic_y": fv.h,
        "fvic_z": fv.z,
        "fvsa": fvsa_length(state, corner),
        "fv_swing_angle": fv_swing_angle(state, corner),
        "svic_x": sv.h,
        "svic_z": sv.z,
        "svsa": svsa_length(state, corner),
        "sv_swing_angle": sv_swing_angle(state, corner),
        "sv_wc_angle": sv_wc_angle(state, corner),
        "rc_height": roll_centre_symmetric(state, corner),
    }
