"""Wheel alignment and wheel position metrics (DESIGN_PLAN sections 5.1 and 5.2).

All functions are pure and return SI floats (m, rad). They work on the
body-fixed corner frame (ISO 8855: x forward, y left, z up) where the ground is
the plane ``z = contact_patch z``. Sign conventions are identical for left and
right corners:

* camber: negative = top of wheel leans toward the car centreline.
* toe: positive = toe-in (wheel front points toward the centreline).
* steer angle: road-wheel heading, ISO yaw sign (positive = wheel points left).
  Hence ``steer = toe`` on right corners and ``steer = -toe`` on left corners.
* caster: positive = top of the steering axis rearward.
* KPI: positive = top of the steering axis inboard.
* scrub radius: positive = ground intercept of the steering axis inboard of the
  contact patch.
* mechanical trail: positive = ground intercept ahead of the contact patch.
* kingpin offset at wheel centre: positive = steering axis inboard of the wheel
  centre, measured along y at wheel-centre height.
* ``contact_patch_dy`` positive = outboard; ``wheel_center_recession`` positive =
  forward (+x) relative to static.

The steering axis is the line from ``lca_outer`` to ``uca_outer``. Metrics that
refer to the static pose take the static state explicitly (``solve_corner(corner,
0.0)``, see :func:`static_state`).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.kinematics.corner import CornerState, solve_corner

Vec = NDArray[np.float64]


def static_state(corner: Corner) -> CornerState:
    """Solve the static pose (travel 0, rack 0) of a corner."""
    return solve_corner(corner, 0.0)


def _out(corner: Corner) -> float:
    """+1 for left (outboard = +y), -1 for right."""
    return 1.0 if corner.side == "left" else -1.0


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


def steer_angle(state: CornerState, corner: Corner) -> float:
    """Road-wheel heading [rad], ISO yaw sign (positive = wheel points left)."""
    n = state.spin_axis
    o = _out(corner)
    return math.atan2(-o * float(n[0]), o * float(n[1]))


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
    """Contribution of this corner to wheelbase change [m]: contact-patch x shift."""
    return contact_patch_dx(state, static, corner)


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
