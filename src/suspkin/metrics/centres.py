"""Instant centres, swing arms and kinematic roll centre (corner level).

Plane method: each wishbone plane passes through its two chassis pivots and its
current outer ball joint. Front view: both planes are cut with the transverse plane
``x = contact_patch.x`` and the two section lines (in y-z) are intersected -> FVIC.
Side view: cut with ``y = contact_patch.y`` (lines in x-z) -> SVIC.

Parallel section lines mean an instant centre at infinity: ``is_finite=False`` with
nan coordinates (never an exception); the common line direction is kept in
``InstantCentre.direction`` where a direction is needed (swing angles, roll centre).

Angle sign: angles are signed, range (-pi, pi]. Front view: positive = FVIC above the
contact patch (inboard side is the positive horizontal direction). Side view: positive =
anti-geometry (front axle: SVIC rearward of and above the reference point; rear axle:
forward of and above). For an IC at infinity the line direction is oriented to the same
reference side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.geometry.vectors import (
    intersect_lines_2d,
    line_direction_2d,
    line_through_2d,
    plane_from_points,
    plane_section_2d,
)
from suspkin.kinematics.corner import CornerState

Vec = NDArray[np.float64]
_NAN = float("nan")


@dataclass(frozen=True)
class InstantCentre:
    """Instant centre in a view plane.

    ``h`` is the horizontal coordinate (y in front view, x in side view), ``z`` the
    height [m]; both nan if ``is_finite`` is False. ``direction`` is the unit
    direction (h, z) of the section lines (the common one when parallel).
    """

    h: float
    z: float
    is_finite: bool
    direction: Vec


def _ic(state: CornerState, axis: int) -> InstantCentre:
    p = state.points
    cut = float(p["contact_patch"][axis])
    lines = []
    try:
        for w in ("uca", "lca"):
            pt, n = plane_from_points(p[f"{w}_front"], p[f"{w}_rear"], p[f"{w}_outer"])
            lines.append(plane_section_2d(pt, n, axis, cut))
    except ValueError:  # degenerate plane (collinear points / plane parallel to the cut)
        return InstantCentre(_NAN, _NAN, False, np.array([_NAN, _NAN]))
    d = line_direction_2d(lines[0][0])
    hit = intersect_lines_2d(lines[0], lines[1])
    if hit is None:
        return InstantCentre(_NAN, _NAN, False, d)
    return InstantCentre(float(hit[0]), float(hit[1]), True, d)


def front_view_ic(state: CornerState, corner: Corner) -> InstantCentre:
    """Front-view instant centre (h = y, z) of the corner."""
    return _ic(state, 0)


def side_view_ic(state: CornerState, corner: Corner) -> InstantCentre:
    """Side-view instant centre (h = x, z) of the corner."""
    return _ic(state, 1)


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


def _angle(ic: InstantCentre, ref_h: float, ref_z: float, sign: float) -> float:
    """Signed angle of the line ref -> IC: ``atan2(dz, sign * (ic_h - ref_h))`` [rad].

    ``sign`` picks the positive horizontal direction. Positive angle = IC above ``ref``;
    the range is (-pi, pi] (beyond +-pi/2 the IC lies on the opposite side). For an IC
    at infinity the common line direction is oriented so that its horizontal component
    is on the ``sign`` side, giving the same convention. Degenerate input gives nan.
    """
    if ic.is_finite:
        return math.atan2(ic.z - ref_z, sign * (ic.h - ref_h))
    dh, dz = float(ic.direction[0]) * sign, float(ic.direction[1])
    if dh < 0:
        dh, dz = -dh, -dz
    return math.atan2(dz, dh)


def _inboard(corner: Corner) -> float:
    return -1.0 if corner.side == "left" else 1.0


def _anti_sign(corner: Corner) -> float:
    """Sign of the x direction of the anti side: rearward (-1) at the front, forward at the rear."""
    return -1.0 if corner.axle == "front" else 1.0


def fv_swing_angle(state: CornerState, corner: Corner) -> float:
    """Angle [rad] of the line contact patch -> FVIC.

    Positive when the FVIC is above the patch; equals ``atan2(dz, inboard * dy)``
    (inboard positive), range (-pi, pi]. An inboard IC gives |angle| < pi/2.
    """
    cp = state.points["contact_patch"]
    return _angle(front_view_ic(state, corner), float(cp[1]), float(cp[2]), _inboard(corner))


def sv_swing_angle(state: CornerState, corner: Corner) -> float:
    """Signed angle [rad] of the line contact patch -> SVIC (feeds anti-features).

    Positive means anti-geometry for the axle: front axle, SVIC rearward of and above
    the patch; rear axle, SVIC forward of and above it. Equals
    ``atan2(z_ic - z_cp, s * (x_cp - x_ic))`` with s = +1 front, -1 rear; range
    (-pi, pi]. ``tan`` of it is the swing-arm slope used for anti %.
    """
    cp = state.points["contact_patch"]
    return _angle(side_view_ic(state, corner), float(cp[0]), float(cp[2]), _anti_sign(corner))


def sv_wc_angle(state: CornerState, corner: Corner) -> float:
    """Signed angle [rad] of the line wheel centre -> SVIC; same sign rule as above."""
    wc = state.points["wheel_center"]
    return _angle(side_view_ic(state, corner), float(wc[0]), float(wc[2]), _anti_sign(corner))


def _roll_line(state: CornerState) -> tuple[Vec, Vec]:
    """Front-view line (point, unit direction) from the contact patch through the FVIC."""
    cp = state.points["contact_patch"]
    p0 = np.array([cp[1], cp[2]], dtype=np.float64)
    ic = _ic(state, 0)
    if ic.is_finite:
        d = np.array([ic.h, ic.z]) - p0
        return p0, d / np.linalg.norm(d)
    return p0, ic.direction


def roll_centre_symmetric(state: CornerState, corner: Corner) -> float:
    """Roll-centre height z [m]: line contact patch -> FVIC cut with the centreline y = 0.

    Returns nan if that line is parallel to y = 0 (vertical in the front view).
    """
    p0, d = _roll_line(state)
    if abs(d[0]) < 1e-12:
        return _NAN
    return float(p0[1] + d[1] / d[0] * (0.0 - p0[0]))


def roll_centre_pair(left: CornerState, right: CornerState) -> tuple[float, float]:
    """Front-view roll centre (y, z) of one axle: both contact patch -> FVIC lines intersected.

    Coincident lines (e.g. parallel horizontal wishbones) return the point on the line
    at y = 0, as ``roll_centre_symmetric`` does. Returns ``(nan, nan)`` for parallel
    distinct lines or degenerate input.
    """
    segs = [_roll_line(st) for st in (left, right)]
    if not all(bool(np.all(np.isfinite(d))) for _, d in segs):
        return (_NAN, _NAN)
    lines = [line_through_2d(p0, p0 + d) for p0, d in segs]
    hit = intersect_lines_2d(lines[0], lines[1])
    if hit is None:
        n2, off2 = lines[1]
        p0, d = segs[0]
        if abs(float(np.asarray(n2) @ p0) - off2) > 1e-9 or abs(d[0]) < 1e-12:
            return (_NAN, _NAN)
        return (0.0, float(p0[1] + d[1] / d[0] * (0.0 - p0[0])))
    return (float(hit[0]), float(hit[1]))


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
