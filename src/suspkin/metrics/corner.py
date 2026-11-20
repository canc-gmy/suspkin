"""Corner-level helpers: steer angle, wheel/ride rates, instant centres and joint angles.

Everything here works on one :class:`~suspkin.kinematics.corner.CornerState` (or a list of
them for the joint-angle frame) and returns SI floats (m, rad, N, N/m). The per-pose metric
columns are produced by the vectorised :mod:`suspkin.metrics.arrays`; scalar reference
versions of them live in the test suite (``tests/unit/_scalar_oracles.py``). Sign conventions:
``docs/concepts/conventions.md`` sections 5.2 to 5.7; column definitions:
``docs/concepts/metrics.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from suspkin.geometry.corners import Corner
from suspkin.geometry.vectors import (
    intersect_lines_2d,
    line_direction_2d,
    line_through_2d,
    plane_from_points,
    plane_section_2d,
    rotation_between,
    unit,
)
from suspkin.kinematics.corner import CornerState

Vec = NDArray[np.float64]
_NAN = float("nan")


# --------------------------------------------------------------------------- alignment
# Steer angle: road-wheel heading, ISO yaw sign (positive = wheel points left), hence
# ``steer = toe`` on right corners and ``steer = -toe`` on left corners. The other alignment
# metrics are computed in :mod:`suspkin.metrics.arrays`.


def _out(corner: Corner) -> float:
    """+1 for left (outboard = +y), -1 for right."""
    return 1.0 if corner.side == "left" else -1.0


def steer_angle(state: CornerState, corner: Corner) -> float:
    """Road-wheel heading [rad], ISO yaw sign (positive = wheel points left).

    ``steer = atan2(-o n_x, o n_y)`` with ``n`` the outboard wheel spin axis and ``o`` = +1
    (left) or -1 (right); so ``steer = -toe`` on left and ``+toe`` on right corners.

    Args:
        state: Solved corner (provides ``spin_axis``).
        corner: Its side-resolved corner (provides the side).

    Returns:
        Steer angle in rad.
    """
    n = state.spin_axis
    o = _out(corner)
    return math.atan2(-o * float(n[0]), o * float(n[1]))


# --------------------------------------------------------------------------- actuation
# Wheel and ride rates (docs/concepts/conventions.md section 5.7).
# Virtual work links wheel and spring: with
# spring motion ratio ``MR = -dL/dz`` the wheel load of the spring is ``F_w = F_s * MR`` and
# the wheel rate is ``dF_w/dz = k * MR**2 + F_s * dMR/dz``.


def wheel_load_spring(force: float, mr_spring: float) -> float:
    """Vertical wheel load [N] of the spring, ``F_s * MR`` (virtual work).

    Args:
        force: Spring force in N (compressive positive).
        mr_spring: Spring motion ratio ``MR = -dL/dz`` (dimensionless, m/m).

    Returns:
        Wheel load in N.
    """
    return force * mr_spring


def wheel_rate_linear(spring_rate: float, mr_spring: float) -> float:
    """Linearised wheel rate [N/m], ``k * MR**2``.

    Args:
        spring_rate: Spring rate ``k`` in N/m.
        mr_spring: Spring motion ratio (m/m).

    Returns:
        Wheel rate in N/m (ignores the motion-ratio progression term).
    """
    return spring_rate * mr_spring**2


def wheel_rate(spring_rate: float, mr_spring: float, force: float, progression: float) -> float:
    """Wheel rate [N/m] from virtual work, ``k * MR**2 + F_s * dMR/dz``.

    Args:
        spring_rate: Spring rate ``k`` in N/m.
        mr_spring: Spring motion ratio (m/m).
        force: Spring force ``F_s`` in N.
        progression: Motion-ratio progression ``dMR/dz`` in 1/m.

    Returns:
        Wheel rate in N/m.
    """
    return spring_rate * mr_spring**2 + force * progression


def ride_rate(wheel_rate_value: float, tyre_stiffness: float) -> float:
    """Ride rate [N/m]: wheel rate in series with the vertical tyre stiffness.

    Args:
        wheel_rate_value: Wheel rate in N/m.
        tyre_stiffness: Tyre vertical stiffness in N/m.

    Returns:
        ``k_w k_t / (k_w + k_t)`` in N/m.
    """
    return wheel_rate_value * tyre_stiffness / (wheel_rate_value + tyre_stiffness)


# --------------------------------------------------------------------------- centres
# Instant centres, swing arms and kinematic roll centre (corner level).
#
# Plane method: each wishbone plane passes through its two chassis pivots and its
# current outer ball joint. Front view: both planes are cut with the transverse plane
# ``x = contact_patch.x`` and the two section lines (in y-z) are intersected -> FVIC.
# Side view: cut with ``y = contact_patch.y`` (lines in x-z) -> SVIC.
#
# Parallel section lines mean an instant centre at infinity: ``is_finite=False`` with
# nan coordinates (never an exception); the common line direction is kept in
# ``InstantCentre.direction`` where a direction is needed (swing angles, roll centre).
#
# Angle sign: angles are signed, range (-pi, pi]. Front view: positive = FVIC above the
# contact patch (inboard side is the positive horizontal direction). Side view: positive =
# anti-geometry (front axle: SVIC rearward of and above the reference point; rear axle:
# forward of and above). For an IC at infinity the line direction is oriented to the same
# reference side.


@dataclass(frozen=True)
class InstantCentre:
    """Instant centre in a view plane.

    ``h`` is the horizontal coordinate (y in front view, x in side view), ``z`` the
    height [m]; both nan if ``is_finite`` is False. ``direction`` is the unit
    direction (h, z) of the section lines (the common one when parallel).

    Attributes:
        h: Horizontal coordinate in m (``y`` front view, ``x`` side view); NaN at infinity.
        z: Height in m (vehicle frame); NaN at infinity.
        is_finite: False for an instant centre at infinity or degenerate input.
        direction: Unit direction ``(h, z)`` of the section lines, shape ``(2,)``.
    """

    h: float
    z: float
    is_finite: bool
    direction: Vec


# kept separate: float ordering (scalar and array plane methods differ in degenerate-case handling)
def _ic(state: CornerState, axis: int) -> InstantCentre:
    """Plane-method instant centre of one state (``axis`` 0 = front view, 1 = side view)."""
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
    """Front-view instant centre (h = y, z) of the corner.

    Args:
        state: Solved corner.
        corner: Its side-resolved corner (unused; kept for a uniform signature).

    Returns:
        :class:`InstantCentre` in the ``(y, z)`` plane, m; ``is_finite=False`` for parallel
        wishbone planes (never raises).
    """
    return _ic(state, 0)


def side_view_ic(state: CornerState, corner: Corner) -> InstantCentre:
    """Side-view instant centre (h = x, z) of the corner.

    Args:
        state: Solved corner.
        corner: Its side-resolved corner (unused; kept for a uniform signature).

    Returns:
        :class:`InstantCentre` in the ``(x, z)`` plane, m; ``is_finite=False`` for parallel
        wishbone planes (never raises).
    """
    return _ic(state, 1)


# kept separate: float ordering (see ``arrays._swing``)
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


def _anti_sign(corner: Corner) -> float:
    """Sign of the x direction of the anti side: rearward (-1) front, forward (+1) rear."""
    return -1.0 if corner.axle == "front" else 1.0


def sv_swing_angle(state: CornerState, corner: Corner) -> float:
    """Signed angle [rad] of the line contact patch -> SVIC (feeds anti-features).

    Positive means anti-geometry for the axle: front axle, SVIC rearward of and above
    the patch; rear axle, SVIC forward of and above it. Equals
    ``atan2(z_ic - z_cp, s * (x_cp - x_ic))`` with s = +1 front, -1 rear; range
    (-pi, pi]. ``tan`` of it is the swing-arm slope used for anti %.

    Args:
        state: Solved corner (provides ``contact_patch`` and the wishbone points).
        corner: Its side-resolved corner (provides the axle).

    Returns:
        Angle in rad; NaN for degenerate input.
    """
    cp = state.points["contact_patch"]
    return _angle(side_view_ic(state, corner), float(cp[0]), float(cp[2]), _anti_sign(corner))


def sv_wc_angle(state: CornerState, corner: Corner) -> float:
    """Signed angle [rad] of the line wheel centre -> SVIC; same sign rule as above.

    Args:
        state: Solved corner (provides ``wheel_center`` and the wishbone points).
        corner: Its side-resolved corner (provides the axle).

    Returns:
        Angle in rad (feeds the anti-squat and inboard-brake anti-features).
    """
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


def roll_centre_pair(left: CornerState, right: CornerState) -> tuple[float, float]:
    """Front-view roll centre (y, z) of one axle: both contact patch -> FVIC lines intersected.

    Coincident lines (e.g. parallel horizontal wishbones) return the point on the line
    at y = 0. Returns ``(nan, nan)`` for parallel distinct lines or degenerate input.

    Args:
        left: Solved left corner of the axle.
        right: Solved right corner of the axle.

    Returns:
        ``(y, z)`` of the roll centre in m, vehicle frame (``y`` positive toward the left).
        The height is measured in the body frame; the vehicle sweeps rotate it to the ground.
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


# --------------------------------------------------------------------------- joints
# Joint misalignment angles over a corner sweep (neutral = static pose).
#
# Body rotations relative to the static pose come from a Kabsch fit (SVD with a
# determinant fix) of each body's hardpoint set; the chassis is the identity.
#
# Definitions (angle = ``atan2(|a x b|, a . b)``):
#
# * Rod ends (two-force links ``tie``, ``pushrod``, ``coilover``): the link direction
#   ``d = unit(b - a)`` is compared with the static direction carried by the housing
#   body, ``angle(d_now, R_housing @ d_static)``.
# * Ball joints ``uca_outer`` / ``lca_outer``: the stud axis ``a0`` (static kingpin
#   ``lca_outer -> uca_outer``) is carried by the upright; the housing is the wishbone.
#   Angle = ``angle(R_upright @ a0, R_wishbone @ a0)``.
# * ``axis`` override: a static vector replacing ``a0``. For ball joints the formula
#   is unchanged. For rod ends the carrier rotation is the minimal rotation taking
#   ``d_static`` onto ``d_now`` and the angle is ``angle(R_carrier @ a0, R_housing @ a0)``.


JOINTS: tuple[str, ...] = (
    "tie_inner",
    "tie_outer",
    "pushrod_outer",
    "pushrod_inner",
    "spring_chassis",
    "rocker_spring",
    "uca_outer",
    "lca_outer",
)

# joint -> (link end a, link end b) for rod ends; direction a -> b
_LINKS: dict[str, tuple[str, str]] = {
    "tie_inner": ("tie_inner", "tie_outer"),
    "tie_outer": ("tie_inner", "tie_outer"),
    "pushrod_outer": ("pushrod_outer", "pushrod_inner"),
    "pushrod_inner": ("pushrod_outer", "pushrod_inner"),
    "spring_chassis": ("spring_chassis", "rocker_spring"),
    "rocker_spring": ("spring_chassis", "rocker_spring"),
}
_BODY_POINTS: dict[str, tuple[str, ...]] = {
    "upright": ("uca_outer", "lca_outer", "tie_outer", "wheel_center"),
    "uca": ("uca_front", "uca_rear", "uca_outer"),
    "lca": ("lca_front", "lca_rear", "lca_outer"),
    "rocker": ("rocker_pivot", "rocker_spring", "pushrod_inner"),
}


_DEGENERATE_FIT_MSG = "degenerate point set for rotation fit (collinear or coincident points)"


def kabsch_batch(p0: Vec, p1: Vec) -> Vec:
    """Batched Kabsch fit: ``p0`` ``(k, 3)`` static points, ``p1`` ``(N, k, 3)``.

    Args:
        p0: Static body points, shape ``(k, 3)``, m.
        p1: Moved body points for N states, shape ``(N, k, 3)``, m.

    Returns:
        The proper rotations ``(N, 3, 3)`` minimising ``|R (p0 - c0) - (p1 - c1)|`` (SVD with
        a determinant fix), with ``c0``/``c1`` the centroids.

    Raises:
        ValueError: For a degenerate point set (collinear or coincident points).
    """
    a = p0 - p0.mean(axis=0)
    b = p1 - p1.mean(axis=1, keepdims=True)
    h = np.einsum("pi,npj->nij", a, b)
    u, sv, vt = np.linalg.svd(h)
    if np.any((sv[:, 1] < 1e-9 * sv[:, 0]) | (sv[:, 0] <= 0.0)):
        raise ValueError(_DEGENERATE_FIT_MSG)
    ut = u.swapaxes(-1, -2)
    v = vt.swapaxes(-1, -2)
    d = np.sign(np.linalg.det(v @ ut))
    d = np.where(d != 0.0, d, 1.0)
    fix = np.ones((h.shape[0], 3))
    fix[:, 2] = d
    out: Vec = (v * fix[:, None, :]) @ ut
    return out


def kabsch(p0: Vec, p1: Vec) -> Vec:
    """Proper rotation R minimising ``|R (p0 - c0) - (p1 - c1)|`` for ``(n, 3)`` point sets.

    Args:
        p0: Static body points, shape ``(n, 3)``, m.
        p1: Moved body points, shape ``(n, 3)``, m.

    Returns:
        Rotation matrix, shape ``(3, 3)``.

    Raises:
        ValueError: For a degenerate point set.
    """
    out: Vec = kabsch_batch(p0, p1[None])[0]
    return out


def _rot(static: CornerState, state: CornerState, names: Sequence[str]) -> Vec:
    """Kabsch rotation of the named hardpoints from ``static`` to ``state``."""
    p0 = np.array([static.points[n] for n in names], dtype=np.float64)
    p1 = np.array([state.points[n] for n in names], dtype=np.float64)
    return kabsch(p0, p1)


def body_rotations(state: CornerState, static: CornerState, corner: Corner) -> dict[str, Vec]:
    """Rotation of each body from the static pose to ``state``.

    Keys: ``upright``, ``uca``, ``lca``, ``rocker``, ``chassis`` (identity). Each is
    a Kabsch fit of the body's hardpoints (``rocker_arb`` joins the rocker set if
    present). The pushrod mount body needs no entry of its own: ``pushrod_outer``
    moves with the body named by ``corner.axle_cfg.pushrod_mount``.

    Args:
        state: Solved corner at the current pose.
        static: Solved corner at the static pose (reference).
        corner: Side-resolved corner (unused here; kept for a uniform signature).

    Returns:
        Body name -> rotation matrix ``(3, 3)``.

    Raises:
        ValueError: If a body's point set is degenerate.
    """
    names = dict(_BODY_POINTS)
    if "rocker_arb" in state.points:
        names["rocker"] = (*names["rocker"], "rocker_arb")
    out = {k: _rot(static, state, v) for k, v in names.items()}
    out["chassis"] = np.eye(3)
    return out


def _housing(corner: Corner, joint: str) -> str:
    """Name of the body whose bore or socket carries ``joint`` (the chassis for chassis joints)."""
    if joint in ("tie_inner", "spring_chassis"):
        return "chassis"
    if joint == "tie_outer":
        return "upright"
    if joint == "pushrod_outer":
        return str(corner.axle_cfg.pushrod_mount)
    if joint in ("pushrod_inner", "rocker_spring"):
        return "rocker"
    return "uca" if joint == "uca_outer" else "lca"


def angle_between(a: ArrayLike, b: ArrayLike) -> Vec:
    """Angle [rad] ``atan2(|a x b|, a . b)`` over the last axis (``(3,)`` or ``(N, 3)`` inputs).

    Args:
        a: Vector(s), shape ``(3,)`` or ``(N, 3)``.
        b: Vector(s), broadcastable against ``a``.

    Returns:
        Unsigned angle(s) in rad in ``[0, pi]``, shape ``()`` or ``(N,)``.
    """
    u = np.asarray(a, dtype=np.float64)
    v = np.asarray(b, dtype=np.float64)
    c = np.cross(u, v)
    out: Vec = np.arctan2(
        np.sqrt(np.einsum("...i,...i->...", c, c)), np.einsum("...i,...i->...", u, v)
    )
    return out


def joint_angle(
    state: CornerState,
    static: CornerState,
    corner: Corner,
    joint: str,
    axis: ArrayLike | None = None,
    _rots: dict[str, Vec] | None = None,
) -> float:
    """Misalignment angle [rad] of ``joint`` at ``state`` relative to ``static``.

    The definitions are in the comment block above (rod ends compare the link direction with
    the one carried by the housing body; ball joints compare the stud axis carried by the
    upright with the same axis carried by the wishbone). ``axis`` is an optional static
    vector (normalised internally) replacing the default stud/bore axis. For rod
    ends the override uses a minimal-rotation carrier convention (the carrier turns
    ``d_static`` onto ``d_now`` with no spin about the link), so it is approximate.
    ``_rots`` is an internal cache of ``body_rotations`` for this state.

    Args:
        state: Solved corner at the current pose.
        static: Solved corner at the static pose.
        corner: Side-resolved corner (provides the pushrod mount).
        joint: One of :data:`JOINTS`.
        axis: Optional static stud/bore axis, shape ``(3,)``.

    Returns:
        Unsigned misalignment angle in rad, in ``[0, pi]``.

    Raises:
        ValueError: If ``joint`` is unknown or a point set is degenerate.
    """
    if joint not in JOINTS:
        raise ValueError(f"unknown joint {joint!r}; expected one of {JOINTS}")
    rots = body_rotations(state, static, corner) if _rots is None else _rots
    r_h = rots[_housing(corner, joint)]
    if joint in ("uca_outer", "lca_outer"):
        a0 = (
            unit(np.asarray(static.points["uca_outer"]) - np.asarray(static.points["lca_outer"]))
            if axis is None
            else unit(axis)
        )
        return float(angle_between(rots["upright"] @ a0, r_h @ a0))
    a_name, b_name = _LINKS[joint]
    d0 = unit(np.asarray(static.points[b_name]) - np.asarray(static.points[a_name]))
    d1 = unit(np.asarray(state.points[b_name]) - np.asarray(state.points[a_name]))
    if axis is None:
        return float(angle_between(d1, r_h @ d0))
    a0 = unit(axis)
    return float(angle_between(rotation_between(d0, d1) @ a0, r_h @ a0))


def joint_angles_frame(states: Sequence[CornerState], corner: Corner) -> pd.DataFrame:
    """Joint angles [rad] per state, columns ``joint_<name>`` for every joint in ``JOINTS``.

    The static pose is the state with ``travel ~ 0`` and ``rack ~ 0`` if present,
    else it is solved with ``solve_corner``.

    Args:
        states: Solved corner states (any order).
        corner: Side-resolved corner.

    Returns:
        DataFrame with one row per state (rad); empty with the column names for no states.
    """
    # lazy: suspkin.metrics.arrays imports this module at module level (import cycle otherwise)
    from suspkin.metrics.arrays import joint_angle_arrays

    cols = [f"joint_{j}" for j in JOINTS]
    if len(states) == 0:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(joint_angle_arrays(states, corner))
