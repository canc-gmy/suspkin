"""Vectorised per-state metrics: arrays over N corner states, no per-state Python calls.

``corner_metric_arrays`` returns exactly the columns of
:func:`suspkin.metrics.table.metrics_frame` and ``joint_angle_arrays`` the ``joint_*``
columns of :func:`suspkin.metrics.corner.joint_angles_frame`. The scalar per-state reference
implementations live in ``tests/unit/_scalar_oracles.py``; every value here follows the same
formula (differences are round-off) and degenerate cases give the same nan / inf values
(instant centre at infinity, ...).

Conventions: columns are SI (m, rad, N, N/m), signs per ``docs/concepts/conventions.md``
section 5; column-by-column definitions in ``docs/concepts/metrics.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.geometry.vectors import dot_rows, norm_rows
from suspkin.kinematics.corner import CornerState, solve_corner
from suspkin.kinematics.ic_arrays import Ic, ic_arrays, roll_line_arrays, swing_angle
from suspkin.metrics.corner import (
    _BODY_POINTS,
    _LINKS,
    JOINTS,
    _housing,
    angle_between,
    kabsch_batch,
)

Arr = NDArray[np.float64]
_NAN = float("nan")


def stack_points(states: Sequence[CornerState], name: str) -> Arr:
    """Stack one hardpoint over the states into an ``(N, 3)`` array.

    Args:
        states: ``N`` solved corner states.
        name: Hardpoint name present in every state's ``points`` (e.g. ``"wheel_center"``).

    Returns:
        Array of shape ``(N, 3)`` in m.

    Raises:
        KeyError: If a state has no point ``name``.
    """
    return np.array([s.points[name] for s in states], dtype=np.float64).reshape(-1, 3)


def corner_arrays(
    states: Sequence[CornerState],
    corner: Corner,
    static: CornerState | None = None,
    *,
    contact_patch: Arr | None = None,
) -> tuple[dict[str, Arr], Ic]:
    """Implementation of :func:`corner_metric_arrays`; also returns the front-view IC.

    Args:
        states: ``N`` solved corner states of ``corner``.
        corner: Side-resolved corner (provides side, axle and coilover/tyre data).
        static: Reference pose for the "change from static" columns (default
            ``solve_corner(corner, 0.0)``).
        contact_patch: Optional ``(N, 3)`` contact patches in m replacing the states'
            ``contact_patch`` point (used when the ground is tilted).

    Returns:
        ``(columns, fv)``: column name -> ``(N,)`` array in SI units, and the front-view
        :class:`~suspkin.kinematics.ic_arrays.Ic`. Degenerate cases give NaN / inf, never
        an exception.
    """
    if static is None:
        static = solve_corner(corner, 0.0)
    cache: dict[str, Arr] = {}

    def pts(name: str) -> Arr:
        if name not in cache:
            cache[name] = (
                contact_patch
                if name == "contact_patch" and contact_patch is not None
                else stack_points(states, name)
            )
        return cache[name]

    n = len(states)
    o = 1.0 if corner.side == "left" else -1.0
    cfg = corner.axle_cfg
    coil = cfg.coilover
    spin = np.array([s.spin_axis for s in states], dtype=np.float64).reshape(-1, 3)
    travel = np.array([float(s.travel) for s in states], dtype=np.float64)
    rack = np.array([float(s.rack) for s in states], dtype=np.float64)
    length = np.array([float(s.coilover_length) for s in states], dtype=np.float64)
    rocker = np.array([float(s.rocker_angle) for s in states], dtype=np.float64)

    out: dict[str, Arr] = {"travel": travel, "rack": rack}
    with np.errstate(divide="ignore", invalid="ignore"):
        cp, wc, lbj = pts("contact_patch"), pts("wheel_center"), pts("lca_outer")
        u = pts("uca_outer") - lbj
        scp = np.asarray(static.points["contact_patch"], dtype=np.float64)
        swc = np.asarray(static.points["wheel_center"], dtype=np.float64)

        def ground(z: Arr) -> Arr:
            t = (z - lbj[:, 2]) / u[:, 2]
            g: Arr = lbj + t[:, None] * u
            return g

        g_cp = ground(cp[:, 2])
        g_wc = ground(wc[:, 2])
        un = u / norm_rows(u)[:, None]
        r = wc - lbj
        perp = r - dot_rows(r, un)[:, None] * un
        dx = cp[:, 0] - scp[0]
        ht = np.abs(cp[:, 1])
        out["camber"] = -np.arcsin(np.clip(spin[:, 2], -1.0, 1.0))
        out["toe"] = np.arctan2(spin[:, 0], o * spin[:, 1])
        out["steer_angle"] = np.arctan2(-o * spin[:, 0], o * spin[:, 1])
        out["caster"] = np.arctan2(-u[:, 0], u[:, 2])
        out["kpi"] = np.arctan2(-o * u[:, 1], u[:, 2])
        out["scrub_radius"] = o * (cp[:, 1] - g_cp[:, 1])
        out["mechanical_trail"] = g_cp[:, 0] - cp[:, 0]
        out["kingpin_offset_wc"] = o * (wc[:, 1] - g_wc[:, 1])
        out["spindle_length"] = norm_rows(perp)
        out["half_track"] = ht
        out["half_track_change"] = ht - abs(float(scp[1]))
        out["contact_patch_dx"] = dx
        out["contact_patch_dy"] = o * (cp[:, 1] - scp[1])
        out["wheel_center_recession"] = wc[:, 0] - swc[0]
        out["wheelbase_change_contribution"] = dx if corner.axle == "front" else -dx

        fv = ic_arrays(pts, 0, cp[:, 0])
        sv = ic_arrays(pts, 1, cp[:, 1])
        inboard = -1.0 if corner.side == "left" else 1.0
        anti = -1.0 if corner.axle == "front" else 1.0
        out["fvic_y"] = fv.h
        out["fvic_z"] = fv.z
        out["fvsa"] = np.where(fv.finite, np.abs(fv.h - cp[:, 1]), math.inf)
        out["fv_swing_angle"] = swing_angle(fv, cp[:, 1], cp[:, 2], inboard)
        out["svic_x"] = sv.h
        out["svic_z"] = sv.z
        out["svsa"] = np.where(sv.finite, np.abs(sv.h - cp[:, 0]), math.inf)
        out["sv_swing_angle"] = swing_angle(sv, cp[:, 0], cp[:, 2], anti)
        out["sv_wc_angle"] = swing_angle(sv, wc[:, 0], wc[:, 2], anti)
        rd0, rd1 = roll_line_arrays(cp[:, 1], cp[:, 2], fv)
        out["rc_height"] = np.where(
            np.abs(rd0) < 1e-12, _NAN, cp[:, 2] + rd1 / rd0 * (0.0 - cp[:, 1])
        )

        rod = pts("pushrod_inner") - pts("pushrod_outer")
        out["pushrod_angle"] = np.arctan2(rod[:, 2], np.hypot(rod[:, 0], rod[:, 1]))
        out["pushrod_angle_fv"] = np.arctan2(rod[:, 2], np.abs(rod[:, 1]))
        out["rocker_rotation"] = rocker - static.rocker_angle
        out["coilover_compression"] = static.coilover_length - length
        out["bump_stroke_remaining"] = length - coil.length_min
        out["droop_stroke_remaining"] = coil.length_max - length
        out["spring_force"] = coil.preload + coil.spring_rate * (coil.free_length - length)
    out["spring_rate"] = np.full(n, float(coil.spring_rate))
    out["tyre_stiffness"] = np.full(n, float(cfg.wheel.tyre_vertical_stiffness))
    return out, fv


def corner_metric_arrays(
    states: Sequence[CornerState], corner: Corner, static: CornerState | None = None
) -> dict[str, NDArray[np.float64]]:
    """Per-state metrics as arrays; same columns, order and values as ``metrics_frame``.

    ``static`` is the reference pose (default ``solve_corner(corner, 0.0)``).

    Args:
        states: ``N`` solved corner states of ``corner``.
        corner: Side-resolved corner.
        static: Reference pose for the "change from static" columns.

    Returns:
        Column name -> ``(N,)`` float array in SI units (m, rad, N, N/m; ``inf`` for swing
        arms at infinity, NaN where undefined).
    """
    return corner_arrays(states, corner, static)[0]


# --------------------------------------------------------------------------- joint angles
def _unit_checked(v: Arr) -> Arr:
    """Row-wise normalise ``(N, 3)``; ``ValueError`` if any row is near zero."""
    nrm = norm_rows(v)
    if np.any(nrm < 1e-12):
        raise ValueError("cannot normalize a near-zero vector")
    out: Arr = v / nrm[:, None]
    return out


def joint_angle_arrays(
    states: Sequence[CornerState], corner: Corner, static: CornerState | None = None
) -> dict[str, NDArray[np.float64]]:
    """Joint misalignment angles [rad] as arrays; the ``joint_<name>`` columns of the frame.

    ``static`` defaults like in ``joint_angles_frame``: the state with ``travel ~ 0`` and
    ``rack ~ 0`` if present, else ``solve_corner(corner, 0.0)``.

    Args:
        states: ``N`` solved corner states of ``corner``.
        corner: Side-resolved corner (provides the pushrod mount).
        static: Static-pose reference state.

    Returns:
        ``joint_<name>`` -> ``(N,)`` unsigned angles in rad, for every name in
        :data:`suspkin.metrics.corner.JOINTS`.

    Raises:
        ValueError: If a body point set or link is degenerate.
    """
    if static is None:
        static = next((s for s in states if abs(s.travel) < 1e-12 and abs(s.rack) < 1e-12), None)
        if static is None:
            static = solve_corner(corner, 0.0)
    n = len(states)
    if n == 0:
        return {f"joint_{j}": np.empty(0) for j in JOINTS}
    names = dict(_BODY_POINTS)
    if "rocker_arb" in states[0].points:
        names["rocker"] = (*names["rocker"], "rocker_arb")
    rots: dict[str, Arr] = {}
    for body, pn in names.items():
        p0 = np.array([static.points[k] for k in pn], dtype=np.float64)
        p1 = np.stack([stack_points(states, k) for k in pn], axis=1)
        rots[body] = kabsch_batch(p0, p1)
    rots["chassis"] = np.broadcast_to(np.eye(3), (n, 3, 3))

    def sp(name: str) -> Arr:
        return np.asarray(static.points[name], dtype=np.float64)

    out: dict[str, Arr] = {}
    for joint in JOINTS:
        rh = rots[_housing(corner, joint)]
        if joint in ("uca_outer", "lca_outer"):
            a0 = sp("uca_outer") - sp("lca_outer")
            a0 = a0 / np.linalg.norm(a0)
            val = angle_between(
                np.einsum("nij,j->ni", rots["upright"], a0), np.einsum("nij,j->ni", rh, a0)
            )
        else:
            a_name, b_name = _LINKS[joint]
            d0 = sp(b_name) - sp(a_name)
            d0 = d0 / np.linalg.norm(d0)
            d1 = _unit_checked(stack_points(states, b_name) - stack_points(states, a_name))
            val = angle_between(d1, np.einsum("nij,j->ni", rh, d0))
        out[f"joint_{joint}"] = val
    return out
