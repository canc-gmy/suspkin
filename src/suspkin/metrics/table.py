"""Metric tables: per-state metrics, gradients vs the sweep variable, static summary."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.kinematics.corner import CornerState, solve_corner
from suspkin.metrics.alignment import alignment_metrics
from suspkin.metrics.centres import centre_metrics

# metric column -> gradient column, per sweep variable
GRADIENTS: dict[str, dict[str, str]] = {
    "travel": {
        "camber": "camber_gain",
        "toe": "bump_steer",
        "caster": "caster_change",
        "half_track_change": "track_change_rate",
        "rc_height": "rc_migration_rate",
    },
    "rack": {"steer_angle": "steer_gain"},
}


def metrics_frame(states: Sequence[CornerState], corner: Corner) -> pd.DataFrame:
    """One row per state: ``travel``, ``rack`` and all alignment and centre metrics (SI)."""
    static = solve_corner(corner, 0.0)
    rows: list[dict[str, float]] = []
    for s in states:
        row = {"travel": float(s.travel), "rack": float(s.rack)}
        row.update(alignment_metrics(s, corner, static))
        row.update(centre_metrics(s, corner))
        rows.append(row)
    return pd.DataFrame(rows)


def _check_axis(df: pd.DataFrame, x: str) -> NDArray[np.float64]:
    if x not in GRADIENTS:
        raise ValueError(f"x must be one of {sorted(GRADIENTS)}, got {x!r}")
    if len(df) < 3:
        raise ValueError("at least 3 rows are required")
    xv = df[x].to_numpy(dtype=float)
    d = np.diff(xv)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError(f"{x!r} must be strictly monotonic")
    return xv


def add_gradients(df: pd.DataFrame, x: str = "travel") -> pd.DataFrame:
    """Return a copy of ``df`` with ``d<metric>/d<x>`` columns (``np.gradient``).

    ``x="travel"`` adds ``camber_gain``, ``bump_steer``, ``caster_change``,
    ``track_change_rate`` and ``rc_migration_rate`` (rad/m or m/m); ``x="rack"`` adds
    ``steer_gain`` (rad/m). Requires strictly monotonic ``x`` and at least 3 rows.
    """
    xv = _check_axis(df, x)
    out = df.copy()
    for metric, name in GRADIENTS[x].items():
        out[name] = np.gradient(df[metric].to_numpy(dtype=float), xv)
    return out


def static_summary(df: pd.DataFrame, x: str = "travel") -> dict[str, float]:
    """Value of every metric and gradient column at ``x == 0`` (linear interpolation)."""
    xv = _check_axis(df, x)
    order = np.argsort(xv)
    cols = [c for c in df.columns if c not in ("travel", "rack") and df[c].dtype.kind == "f"]
    if not float(np.min(xv)) <= 0.0 <= float(np.max(xv)):
        raise ValueError(f"{x} = 0 lies outside the sweep range")
    return {c: float(np.interp(0.0, xv[order], df[c].to_numpy(dtype=float)[order])) for c in cols}
