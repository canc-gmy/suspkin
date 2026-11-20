"""Network training target of one design: channel tensor and validity mask of a corner.

The left corner of one axle is solved on the fixed :class:`AxleGrid`. Kinematic failures
never raise: an unsolved point is invalid (mask false, values NaN). Programming errors
(bad arguments, missing data) still raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner, build_corners
from suspkin.geometry.model import Vehicle
from suspkin.kinematics.corner import CornerState, solve_corner, sweep_corner
from suspkin.kinematics.rocker import KinematicsError
from suspkin.metrics.arrays import corner_metric_arrays, joint_angle_arrays
from suspkin.surrogate.channels import channels_for, wrap_line_angle
from suspkin.surrogate.grid import AxleGrid


@dataclass(frozen=True)
class CornerSample:
    """Training target of one design and axle.

    ``Y`` is ``(C, R, T)`` (NaN where invalid); ``mask`` is ``(R, T)``: kinematically solved
    with finite channel values. Coilover range and toggle margin are not masked (the evaluator
    checks them; ``toggle_margin`` is a channel). ``feasible`` is true when the mask is full.
    """

    Y: NDArray[np.float64]
    mask: NDArray[np.bool_]
    feasible: bool
    channels: tuple[str, ...]


def _metric_arrays(states: list[CornerState], corner: Corner) -> dict[str, NDArray[np.float64]]:
    """Per-state column arrays of every source column the channels use.

    Single adapter between the extractor and the metrics: the vectorised batch kernels
    (``corner_metric_arrays`` and ``joint_angle_arrays``) plus the solver's coilover length
    and toggle margin. ``states`` must be non-empty.
    """
    out = dict(corner_metric_arrays(states, corner))
    out.update(joint_angle_arrays(states, corner))
    out["coilover_length"] = np.array([s.coilover_length for s in states], dtype=np.float64)
    out["toggle_margin"] = np.array([s.toggle_margin for s in states], dtype=np.float64)
    for name in ("theta_uca", "theta_lca", "psi"):
        out[name] = np.array([getattr(s, name) for s in states], dtype=np.float64)
    return out


def _solve_line(
    corner: Corner, travel: NDArray[np.float64], rack: float
) -> list[CornerState | None]:
    """States along a travel line; ``None`` where unsolved (never raises on kinematics)."""
    try:
        return list(sweep_corner(corner, travel, rack=rack))
    except KinematicsError:
        pass
    out: list[CornerState | None] = [None] * travel.size
    start = int(np.argmin(np.abs(travel)))
    try:
        first = solve_corner(corner, float(travel[start]), rack)
    except KinematicsError:
        return out
    out[start] = first
    for step in (1, -1):
        prev = first
        i = start + step
        while 0 <= i < travel.size:
            try:
                prev = solve_corner(corner, float(travel[i]), rack, guess=prev)
            except KinematicsError:
                break
            out[i] = prev
            i += step
    return out


def extract_corner(
    vehicle: Vehicle, axle: Literal["front", "rear"], grid: AxleGrid
) -> CornerSample:
    """Solve the left corner (FL or RL) of ``axle`` on ``grid`` and build the channel tensor."""
    corner = build_corners(vehicle)["FL" if axle == "front" else "RL"]
    chans = channels_for(axle)
    n_r, n_t = grid.shape
    y = np.full((len(chans), n_r, n_t), np.nan)
    mask = np.zeros((n_r, n_t), dtype=bool)

    try:
        solve_corner(corner, 0.0)
    except KinematicsError:
        return CornerSample(y, mask, False, tuple(c.name for c in chans))

    solved: list[tuple[int, int, CornerState]] = []
    for r, rack in enumerate(grid.rack):
        for t, st in enumerate(_solve_line(corner, grid.travel, float(rack))):
            if st is not None:
                solved.append((r, t, st))

    if solved:
        cols = _metric_arrays([s for _, _, s in solved], corner)
        ri = np.array([r for r, _, _ in solved])
        ti = np.array([t for _, t, _ in solved])
        vals = np.empty((len(chans), len(solved)))
        for k, ch in enumerate(chans):
            v = cols[ch.source]
            vals[k] = wrap_line_angle(v) if ch.encoding == "line_angle" else v
        ok = np.all(np.isfinite(vals), axis=0)
        mask[ri[ok], ti[ok]] = True
        y[:, ri[ok], ti[ok]] = vals[:, ok]
    return CornerSample(y, mask, bool(mask.all()), tuple(c.name for c in chans))


def extract_axles(vehicle: Vehicle, grids: dict[str, AxleGrid]) -> dict[str, CornerSample]:
    """:func:`extract_corner` for every axle in ``grids`` (keys ``front`` / ``rear``)."""
    return {axle: extract_corner(vehicle, g.axle, g) for axle, g in grids.items()}
