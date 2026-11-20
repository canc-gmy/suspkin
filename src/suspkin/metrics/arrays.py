"""Vectorised per-state metrics: arrays over N corner states, no per-state Python calls.

``corner_metric_arrays`` returns exactly the columns of
:func:`suspkin.metrics.table.metrics_frame` and ``joint_angle_arrays`` the ``joint_*``
columns of :func:`suspkin.metrics.corner.joint_angles_frame`. The per-state functions in
``metrics.corner`` remain the reference
implementation; every value here follows the same formula (differences are round-off) and
degenerate cases give the same nan / inf values (instant centre at infinity, ...).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.kinematics.corner import CornerState, solve_corner
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
_PARALLEL_TOL = 1e-9


def stack_points(states: Sequence[CornerState], name: str) -> Arr:
    """Stack one hardpoint over the states into an ``(N, 3)`` array."""
    return np.array([s.points[name] for s in states], dtype=np.float64).reshape(-1, 3)


def _dot(a: Arr, b: Arr) -> Arr:
    out: Arr = np.einsum("ij,ij->i", a, b)
    return out


def _norm(a: Arr) -> Arr:
    out: Arr = np.sqrt(np.einsum("ij,ij->i", a, a))
    return out


class _Ic:
    """Instant centres of N states in one view (see ``corner._ic``)."""

    def __init__(self, h: Arr, z: Arr, finite: NDArray[np.bool_], d0: Arr, d1: Arr) -> None:
        self.h, self.z, self.finite, self.d0, self.d1 = h, z, finite, d0, d1


# kept separate: float ordering (scalar twin: ``corner._ic``)
def _ic_arrays(pts: Callable[[str], Arr], axis: int, cut: Arr) -> _Ic:
    """Plane-method instant centre for every state; ``axis`` 0 = front view, 1 = side view."""
    keep = [i for i in range(3) if i != axis]
    n = cut.shape[0]
    bad = np.zeros(n, dtype=bool)
    n2s: list[Arr] = []
    offs: list[Arr] = []
    for w in ("uca", "lca"):
        a = pts(f"{w}_front")
        nv = np.cross(pts(f"{w}_rear") - a, pts(f"{w}_outer") - a)
        nn = _norm(nv)
        bad |= nn < 1e-12
        nv = nv / nn[:, None]
        n2 = nv[:, keep]
        norm2 = _norm(n2)
        bad |= norm2 < 1e-12
        rhs = _dot(nv, a) - nv[:, axis] * cut
        n2s.append(n2 / norm2[:, None])
        offs.append(rhs / norm2)
    n1, n2 = n2s
    d0 = n1[:, 1].copy()
    d1 = -n1[:, 0]
    d0[bad] = _NAN
    d1[bad] = _NAN
    det = n1[:, 0] * n2[:, 1] - n1[:, 1] * n2[:, 0]
    parallel = np.abs(det) < _PARALLEL_TOL * (_norm(n1) * _norm(n2))
    hit = ~bad & ~parallel
    h = np.full(n, _NAN)
    z = np.full(n, _NAN)
    if hit.any():
        mats = np.stack([n1[hit], n2[hit]], axis=1)
        rhs2 = np.stack([offs[0][hit], offs[1][hit]], axis=1)[..., None]
        sol = np.linalg.solve(mats, rhs2)[..., 0]
        h[hit] = sol[:, 0]
        z[hit] = sol[:, 1]
    return _Ic(h, z, hit, d0, d1)


def _swing(ic: _Ic, ref_h: Arr, ref_z: Arr, sign: float) -> Arr:
    """Vectorised ``corner._angle``."""
    fin = np.arctan2(ic.z - ref_z, sign * (ic.h - ref_h))
    dh = ic.d0 * sign
    dz = ic.d1
    flip = dh < 0
    dh = np.where(flip, -dh, dh)
    dz = np.where(flip, -dz, dz)
    out: Arr = np.where(ic.finite, fin, np.arctan2(dz, dh))
    return out


def roll_line_arrays(cp_y: Arr, cp_z: Arr, ic: _Ic) -> tuple[Arr, Arr]:
    """Direction ``(d_y, d_z)`` of the contact patch -> FVIC line (vectorised ``_roll_line``)."""
    dy = ic.h - cp_y
    dz = ic.z - cp_z
    nrm = np.sqrt(dy * dy + dz * dz)
    return (
        np.where(ic.finite, dy / nrm, ic.d0),
        np.where(ic.finite, dz / nrm, ic.d1),
    )


def corner_arrays(
    states: Sequence[CornerState],
    corner: Corner,
    static: CornerState | None = None,
    *,
    contact_patch: Arr | None = None,
) -> tuple[dict[str, Arr], _Ic]:
    """Implementation of :func:`corner_metric_arrays`; also returns the front-view IC."""
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
        un = u / _norm(u)[:, None]
        r = wc - lbj
        perp = r - _dot(r, un)[:, None] * un
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
        out["spindle_length"] = _norm(perp)
        out["half_track"] = ht
        out["half_track_change"] = ht - abs(float(scp[1]))
        out["contact_patch_dx"] = dx
        out["contact_patch_dy"] = o * (cp[:, 1] - scp[1])
        out["wheel_center_recession"] = wc[:, 0] - swc[0]
        out["wheelbase_change_contribution"] = dx if corner.axle == "front" else -dx

        fv = _ic_arrays(pts, 0, cp[:, 0])
        sv = _ic_arrays(pts, 1, cp[:, 1])
        inboard = -1.0 if corner.side == "left" else 1.0
        anti = -1.0 if corner.axle == "front" else 1.0
        out["fvic_y"] = fv.h
        out["fvic_z"] = fv.z
        out["fvsa"] = np.where(fv.finite, np.abs(fv.h - cp[:, 1]), math.inf)
        out["fv_swing_angle"] = _swing(fv, cp[:, 1], cp[:, 2], inboard)
        out["svic_x"] = sv.h
        out["svic_z"] = sv.z
        out["svsa"] = np.where(sv.finite, np.abs(sv.h - cp[:, 0]), math.inf)
        out["sv_swing_angle"] = _swing(sv, cp[:, 0], cp[:, 2], anti)
        out["sv_wc_angle"] = _swing(sv, wc[:, 0], wc[:, 2], anti)
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
    """
    return corner_arrays(states, corner, static)[0]


# --------------------------------------------------------------------------- joint angles
def _unit_rows(v: Arr) -> Arr:
    nrm = _norm(v)
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
            d1 = _unit_rows(stack_points(states, b_name) - stack_points(states, a_name))
            val = angle_between(d1, np.einsum("nij,j->ni", rh, d0))
        out[f"joint_{joint}"] = val
    return out
