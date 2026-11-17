"""Vehicle-level sweeps (heave, roll, pitch or rack) and their result table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from suspkin.geometry.corners import Corner, CornerId, build_corners
from suspkin.geometry.models import Vehicle
from suspkin.kinematics.corner import CornerState
from suspkin.kinematics.rocker import KinematicsError
from suspkin.kinematics.vehicle import VehicleState, body_rotation, solve_vehicle
from suspkin.metrics.centres import roll_centre_pair
from suspkin.metrics.table import metrics_frame

_VARS = ("heave", "roll", "pitch", "rack")
_CORNERS: tuple[CornerId, ...] = ("FL", "FR", "RL", "RR")
_MAX_HALVINGS = 6


def _reach(
    vehicle: Vehicle,
    corners: dict[CornerId, Corner],
    prev: VehicleState,
    target: dict[str, float],
    pivot: ArrayLike | None = None,
) -> VehicleState:
    """Solve ``target`` from ``prev``; halve the step in input space on failure."""

    def go(src: VehicleState, tgt: dict[str, float], depth: int) -> VehicleState:
        try:
            return solve_vehicle(vehicle, corners, guess=src, pivot=pivot, **tgt)
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
    start = int(np.argmin(np.abs(vals)))
    out: dict[int, VehicleState] = {}
    first = {**fixed, name: float(vals[start])}
    try:
        out[start] = solve_vehicle(vehicle, corners, pivot=pivot, **first)  # type: ignore[arg-type]
    except KinematicsError:
        out[start] = _reach(
            vehicle, corners, solve_vehicle(vehicle, corners, pivot=pivot), first, pivot
        )
    for direction in (1, -1):
        prev = out[start]
        i = start + direction
        while 0 <= i < vals.size:
            prev = _reach(vehicle, corners, prev, {**fixed, name: float(vals[i])}, pivot)
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
    data: dict[str, object] = {k: [float(getattr(s, k)) for s in states] for k in _VARS}
    parts = [pd.DataFrame(data)]
    for cid in _CORNERS:
        mf = metrics_frame([_on_ground(s, cid) for s in states], corners[cid]).drop(
            columns=["rack"]
        )
        mf = mf.add_prefix(f"{cid}_")
        cp = np.array([s.contact_points[cid] for s in states], dtype=np.float64).reshape(-1, 3)
        cpf = pd.DataFrame({f"{cid}_cp_{a}": cp[:, i] for i, a in enumerate("xyz")})
        parts += [mf, cpf]
    parts.append(_ground_frame(states, corners))
    for axle in ("front", "rear"):
        parts.append(
            pd.DataFrame(
                {
                    f"arb_{axle}_elastic": [
                        np.nan if s.arb[axle] is None else float(s.arb[axle].elastic)  # type: ignore[union-attr]
                        for s in states
                    ]
                }
            )
        )
    return pd.concat(parts, axis=1)


def _on_ground(s: VehicleState, cid: CornerId) -> CornerState:
    """Corner state whose ``contact_patch`` is the vehicle (tilted-ground) contact point."""
    st = s.corners[cid]
    return replace(st, points={**st.points, "contact_patch": s.contact_points[cid]})


def _ground_frame(states: Sequence[VehicleState], corners: dict[CornerId, Corner]) -> pd.DataFrame:
    """Ground-frame camber, steer and roll centre columns."""
    cols: dict[str, list[float]] = {}

    def add(name: str, v: float) -> None:
        cols.setdefault(name, []).append(v)

    for s in states:
        rot = body_rotation(s.roll, s.pitch)
        for cid in _CORNERS:
            st = s.corners[cid]
            sw = rot @ np.asarray(st.spin_axis, dtype=np.float64)
            o = 1.0 if corners[cid].side == "left" else -1.0
            add(f"{cid}_camber_ground", -float(np.arcsin(np.clip(sw[2], -1.0, 1.0))))
            add(f"{cid}_steer_ground", float(np.arctan2(-o * sw[0], o * sw[1])))
        for axle, (lc, rc) in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
            y_b, z_b = roll_centre_pair(_on_ground(s, lc), _on_ground(s, rc))  # type: ignore[arg-type]
            if np.isfinite(y_b) and np.isfinite(z_b):
                x_cp = 0.5 * (s.contact_points[lc][0] + s.contact_points[rc][0])  # type: ignore[index]
                w = (
                    rot @ (np.array([x_cp, y_b, z_b]) - s.pivot)
                    + s.pivot
                    + np.array([0.0, 0.0, s.heave])
                )
                add(f"rc_{axle}_y", float(y_b))
                add(f"rc_{axle}_z", float(w[2]))
            else:
                add(f"rc_{axle}_y", float("nan"))
                add(f"rc_{axle}_z", float("nan"))
    return pd.DataFrame(cols)
