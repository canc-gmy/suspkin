"""Sweeps and their tidy result tables.

* corner level: :func:`states_to_frame` (one row per :class:`CornerState`),
* vehicle level: :func:`sweep_vehicle` (heave, roll, pitch or rack) and :func:`vehicle_frame`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from suspkin.geometry.corners import Corner, CornerId, build_corners
from suspkin.geometry.model import Vehicle
from suspkin.geometry.vectors import body_rotation_batch
from suspkin.kinematics.corner import CornerState, _Ctx
from suspkin.kinematics.ic_arrays import roll_line_arrays
from suspkin.kinematics.rocker import KinematicsError
from suspkin.kinematics.vehicle import VehicleState, solve_vehicle
from suspkin.metrics.arrays import corner_arrays
from suspkin.metrics.table import metrics_frame

_SCALARS = (
    "travel",
    "rack",
    "theta_uca",
    "theta_lca",
    "psi",
    "rocker_angle",
    "coilover_length",
    "toggle_margin",
    "max_residual",
)


def states_to_frame(
    states: Sequence[CornerState], corner: Corner | None = None, metrics: bool = False
) -> pd.DataFrame:
    """One row per state (SI units: m, rad).

    Columns: ``corner``, scalar results, ``spin_x/y/z``, ``<point>_x/_y/_z`` for
    every point (incl. ``contact_patch``) and the health flag ``coilover_in_range``.
    With ``metrics=True`` and a ``corner``, the alignment and centre metric columns of
    :func:`suspkin.metrics.metrics_frame` are appended (existing columns are kept).
    """
    cols: dict[str, object] = {"corner": [s.corner_id for s in states]}
    for name in _SCALARS:
        cols[name] = np.array([float(getattr(s, name)) for s in states], dtype=np.float64)
    if states:
        spin = np.array([s.spin_axis for s in states], dtype=np.float64).reshape(-1, 3)
        for i, axis in enumerate("xyz"):
            cols[f"spin_{axis}"] = spin[:, i]
        keys = list(states[0].points)
        # states of one corner share their point names; mixed lists fall back to the union
        for s in states[1:]:
            if s.points.keys() != states[0].points.keys():
                keys = list(dict.fromkeys(k for st in states for k in st.points))
                break
        for pname in keys:
            if all(pname in s.points for s in states):
                arr = np.array([s.points[pname] for s in states], dtype=np.float64)
                for i, axis in enumerate("xyz"):
                    cols[f"{pname}_{axis}"] = arr[:, i]
            else:  # point missing in some states: NaN there, as pandas fills missing keys
                for i, axis in enumerate("xyz"):
                    cols[f"{pname}_{axis}"] = [
                        float(s.points[pname][i]) if pname in s.points else float("nan")
                        for s in states
                    ]
        cols["coilover_in_range"] = np.array([bool(s.coilover_in_range) for s in states])
    df = pd.DataFrame(cols) if states else pd.DataFrame()
    if metrics and corner is not None:
        mf = metrics_frame(states, corner).drop(columns=["travel", "rack"])
        df = pd.concat([df, mf], axis=1)
    return df


# --------------------------------------------------------------------------- vehicle sweeps
_VARS = ("heave", "roll", "pitch", "rack")
_CORNERS: tuple[CornerId, ...] = ("FL", "FR", "RL", "RR")
_MAX_HALVINGS = 6
_PARALLEL_TOL = 1e-9
Arr = NDArray[np.float64]


def _reach(
    vehicle: Vehicle,
    corners: dict[CornerId, Corner],
    prev: VehicleState,
    target: dict[str, float],
    pivot: ArrayLike | None = None,
    ctxs: Mapping[CornerId, _Ctx] | None = None,
) -> VehicleState:
    """Solve ``target`` from ``prev``; halve the step in input space on failure."""

    def go(src: VehicleState, tgt: dict[str, float], depth: int) -> VehicleState:
        try:
            return solve_vehicle(vehicle, corners, guess=src, pivot=pivot, _ctxs=ctxs, **tgt)
        except KinematicsError:
            if depth >= _MAX_HALVINGS:
                raise
        cur = {k: float(getattr(src, k)) for k in _VARS}
        mid = {k: 0.5 * (cur[k] + tgt[k]) for k in _VARS}
        return go(go(src, mid, depth + 1), tgt, depth + 1)

    return go(prev, target, 0)


def sweep_vehicle(
    vehicle: Vehicle,
    *,
    heave: ArrayLike | None = None,
    roll: ArrayLike | None = None,
    pitch: ArrayLike | None = None,
    rack: ArrayLike | None = None,
    base: Mapping[str, float] | None = None,
    pivot: ArrayLike | None = None,
) -> list[VehicleState]:
    """Sweep exactly one body input; ``base`` sets the others (default 0).

    Continuation runs outward from the sweep value nearest to 0. Returns one state per
    sweep value, in input order. ``pivot`` is the body rotation centre (default: the CG).
    """
    given = {"heave": heave, "roll": roll, "pitch": pitch, "rack": rack}
    swept = [k for k, v in given.items() if v is not None]
    if len(swept) != 1:
        raise ValueError("exactly one of heave, roll, pitch, rack must be an array")
    name = swept[0]
    fixed = {k: 0.0 for k in _VARS}
    for k, v in (base or {}).items():
        if k not in _VARS:
            raise ValueError(f"unknown base key {k!r}; expected one of {_VARS}")
        fixed[k] = float(v)
    vals = np.atleast_1d(np.asarray(given[name], dtype=np.float64))
    if vals.size == 0:
        return []
    corners = build_corners(vehicle)
    ctxs = {cid: _Ctx(c) for cid, c in corners.items()}
    start = int(np.argmin(np.abs(vals)))
    out: dict[int, VehicleState] = {}
    first = {**fixed, name: float(vals[start])}
    try:
        out[start] = solve_vehicle(vehicle, corners, pivot=pivot, _ctxs=ctxs, **first)  # type: ignore[arg-type]
    except KinematicsError:
        out[start] = _reach(
            vehicle,
            corners,
            solve_vehicle(vehicle, corners, pivot=pivot, _ctxs=ctxs),
            first,
            pivot,
            ctxs,
        )
    for direction in (1, -1):
        prev = out[start]
        i = start + direction
        while 0 <= i < vals.size:
            prev = _reach(vehicle, corners, prev, {**fixed, name: float(vals[i])}, pivot, ctxs)
            out[i] = prev
            i += direction
    return [out[i] for i in range(vals.size)]


def vehicle_frame(states: Sequence[VehicleState], vehicle: Vehicle) -> pd.DataFrame:
    """One row per state (SI units).

    Columns: ``heave``, ``roll``, ``pitch``, ``rack``; per corner (prefix ``FL_`` ...)
    the :func:`suspkin.metrics.metrics_frame` columns (``FL_travel`` included, the
    corner ``rack`` column replaced by the shared ``rack``), ``<id>_cp_x/_y/_z`` (contact
    point, body frame); ``<id>_camber_ground`` (camber to the ground normal, negative = top
    toward the centreline) and ``<id>_steer_ground`` (wheel heading in the ground plane,
    ISO yaw sign); ``rc_front_y/_z`` and ``rc_rear_y/_z`` (roll centre: y in the body frame, i.e.
    relative to the car centreline; z = height above ground; NaN if undefined); and
    ``arb_front_elastic`` / ``arb_rear_elastic`` (NaN without ARB). The ground frame is the
    world frame with the
    body rotated about the state's ``pivot``. Per-corner metrics use the ground contact
    point of the state (tilted ground), not the body-vertical lowest point.
    """
    corners = build_corners(vehicle)
    cols: dict[str, Arr] = {
        k: np.array([float(getattr(s, k)) for s in states], dtype=np.float64) for k in _VARS
    }
    fv: dict[CornerId, tuple[Arr, Arr, Arr, Arr, Arr]] = {}
    for cid in _CORNERS:
        cp = np.array([s.contact_points[cid] for s in states], dtype=np.float64).reshape(-1, 3)
        # per-corner metrics use the ground contact point of the state (tilted ground)
        mf, ic = corner_arrays([s.corners[cid] for s in states], corners[cid], contact_patch=cp)
        for k, v in mf.items():
            if k != "rack":
                cols[f"{cid}_{k}"] = v
        for i, a in enumerate("xyz"):
            cols[f"{cid}_cp_{a}"] = cp[:, i]
        rd = roll_line_arrays(cp[:, 1], cp[:, 2], ic)
        fv[cid] = (cp[:, 0], cp[:, 1], cp[:, 2], rd[0], rd[1])
    cols.update(_ground_arrays(states, corners, fv))
    for axle in ("front", "rear"):
        cols[f"arb_{axle}_elastic"] = np.array(
            [
                np.nan if s.arb[axle] is None else float(s.arb[axle].elastic)  # type: ignore[union-attr]
                for s in states
            ],
            dtype=np.float64,
        )
    return pd.DataFrame(cols)


def _ground_arrays(
    states: Sequence[VehicleState],
    corners: dict[CornerId, Corner],
    fv: dict[CornerId, tuple[Arr, Arr, Arr, Arr, Arr]],
) -> dict[str, Arr]:
    """Ground-frame camber, steer and roll centre columns (vectorised over the states)."""
    cols: dict[str, Arr] = {}
    n = len(states)
    roll = np.array([s.roll for s in states], dtype=np.float64)
    pitch = np.array([s.pitch for s in states], dtype=np.float64)
    heave = np.array([s.heave for s in states], dtype=np.float64)
    pivot = np.array([s.pivot for s in states], dtype=np.float64).reshape(-1, 3)
    rot = body_rotation_batch(roll, pitch)
    for cid in _CORNERS:
        spin = np.array([s.corners[cid].spin_axis for s in states], dtype=np.float64)
        sw = np.einsum("nij,nj->ni", rot, spin.reshape(-1, 3))
        o = 1.0 if corners[cid].side == "left" else -1.0
        cols[f"{cid}_camber_ground"] = -np.arcsin(np.clip(sw[:, 2], -1.0, 1.0))
        cols[f"{cid}_steer_ground"] = np.arctan2(-o * sw[:, 0], o * sw[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        pairs: tuple[tuple[str, CornerId, CornerId], ...] = (
            ("front", "FL", "FR"),
            ("rear", "RL", "RR"),
        )
        for axle, lc, rc in pairs:
            y_b, z_b = _roll_centre_pair_arrays(fv[lc], fv[rc])
            ok = np.isfinite(y_b) & np.isfinite(z_b)
            x_cp = 0.5 * (fv[lc][0] + fv[rc][0])
            v = np.stack([x_cp, np.where(ok, y_b, 0.0), np.where(ok, z_b, 0.0)], axis=1)
            w_z = np.einsum("nj,nj->n", rot[:, 2, :], v - pivot) + pivot[:, 2] + heave
            cols[f"rc_{axle}_y"] = np.where(ok, y_b, np.nan)
            cols[f"rc_{axle}_z"] = np.where(ok, w_z, np.nan)
    assert all(v.shape == (n,) for v in cols.values())
    return cols


def _roll_centre_pair_arrays(
    left: tuple[Arr, Arr, Arr, Arr, Arr], right: tuple[Arr, Arr, Arr, Arr, Arr]
) -> tuple[Arr, Arr]:
    """Vectorised ``roll_centre_pair`` from the (x, y, z, d_y, d_z) roll lines of both wheels."""
    p1y, p1z, d1y, d1z = left[1], left[2], left[3], left[4]
    p2y, p2z, d2y, d2z = right[1], right[2], right[3], right[4]
    n = p1y.shape[0]
    y = np.full(n, np.nan)
    z = np.full(n, np.nan)
    valid = np.isfinite(d1y) & np.isfinite(d1z) & np.isfinite(d2y) & np.isfinite(d2z)

    def line(py: Arr, pz: Arr, dy: Arr, dz: Arr) -> tuple[Arr, Arr, Arr]:
        # line_through_2d(p, p + d): the direction is re-derived from the rounded points
        ey = (py + dy) - py
        ez = (pz + dz) - pz
        nrm = np.sqrt(ey * ey + ez * ez)
        ny, nz = -ez / nrm, ey / nrm
        return ny, nz, ny * py + nz * pz

    a1, b1, c1 = line(p1y, p1z, d1y, d1z)
    a2, b2, c2 = line(p2y, p2z, d2y, d2z)
    det = a1 * b2 - b1 * a2
    parallel = np.abs(det) < _PARALLEL_TOL * (
        np.sqrt(a1 * a1 + b1 * b1) * np.sqrt(a2 * a2 + b2 * b2)
    )
    hit = valid & ~parallel
    if hit.any():
        mats = np.stack(
            [np.stack([a1[hit], b1[hit]], axis=1), np.stack([a2[hit], b2[hit]], axis=1)], axis=1
        )
        sol = np.linalg.solve(mats, np.stack([c1[hit], c2[hit]], axis=1)[..., None])[..., 0]
        y[hit] = sol[:, 0]
        z[hit] = sol[:, 1]
    coincident = (
        valid
        & parallel
        & ~((np.abs(a2 * p1y + b2 * p1z - c2) > _PARALLEL_TOL) | (np.abs(d1y) < 1e-12))
    )
    y[coincident] = 0.0
    z[coincident] = (p1z + d1z / d1y * (0.0 - p1y))[coincident]
    return y, z
