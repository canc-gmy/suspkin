"""Metric tables: per-state metrics, gradients vs the sweep variable, static summary.

All tables are SI (m, rad, N, N/m); the display conversion is done by the plotting and CLI
layers (``docs/concepts/conventions.md`` section 3). Column definitions:
``docs/concepts/metrics.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.kinematics.batch import GRADIENTS
from suspkin.kinematics.corner import CornerState
from suspkin.metrics.arrays import corner_metric_arrays
from suspkin.metrics.corner import ride_rate, wheel_load_spring, wheel_rate, wheel_rate_linear


def metrics_frame(states: Sequence[CornerState], corner: Corner) -> pd.DataFrame:
    """One row per state: ``travel``, ``rack``, alignment, centre and actuation metrics (SI).

    The constant columns ``spring_rate`` and ``tyre_stiffness`` carry the rates that
    :func:`add_gradients` needs for the wheel and ride rate.

    Args:
        states: Solved corner states of ``corner`` (e.g. from ``sweep_corner``).
        corner: Side-resolved corner.

    Returns:
        DataFrame, one row per state (empty DataFrame for no states), SI units.
    """
    if len(states) == 0:
        return pd.DataFrame()
    return pd.DataFrame(corner_metric_arrays(states, corner))


def _check_axis(df: pd.DataFrame, x: str) -> NDArray[np.float64]:
    """Validate the gradient axis column (known name, >= 3 rows, strictly monotonic)."""
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

    For ``x="travel"`` with the actuation columns present it also adds ``mr_spring``
    (``d(coilover_compression)/dz``, equal to ``-dL/dz``), ``mr_progression``
    (``d(mr_spring)/dz``), ``wheel_load_spring``, ``wheel_rate_linear``, ``wheel_rate``
    and ``ride_rate`` (N/m), all from the table values.

    Args:
        df: Table from :func:`metrics_frame` (SI).
        x: Sweep variable column: ``"travel"`` (m) or ``"rack"`` (m).

    Returns:
        Copy of ``df`` with the gradient columns added (per metre).

    Raises:
        ValueError: If ``x`` is not ``"travel"`` or ``"rack"``, there are fewer than 3 rows,
            or ``x`` is not strictly monotonic.
    """
    xv = _check_axis(df, x)
    out = df.copy()
    for metric, name in GRADIENTS[x].items():
        out[name] = np.gradient(df[metric].to_numpy(dtype=float), xv)
    needed = {"coilover_compression", "spring_force", "spring_rate", "tyre_stiffness"}
    if x == "travel" and needed <= set(df.columns):
        mr = np.gradient(df["coilover_compression"].to_numpy(dtype=float), xv)
        prog = np.gradient(mr, xv)
        force = df["spring_force"].to_numpy(dtype=float)
        k = df["spring_rate"].to_numpy(dtype=float)
        k_t = df["tyre_stiffness"].to_numpy(dtype=float)
        wr = wheel_rate(k, mr, force, prog)  # type: ignore[arg-type]
        out["mr_spring"] = mr
        out["mr_progression"] = prog
        out["wheel_load_spring"] = wheel_load_spring(force, mr)  # type: ignore[arg-type]
        out["wheel_rate_linear"] = wheel_rate_linear(k, mr)  # type: ignore[arg-type]
        out["wheel_rate"] = wr
        out["ride_rate"] = ride_rate(wr, k_t)  # type: ignore[arg-type]
    return out


def static_summary(df: pd.DataFrame, x: str = "travel") -> dict[str, float]:
    """Value of every metric and gradient column at ``x == 0`` (linear interpolation).

    Args:
        df: Table from :func:`metrics_frame` / :func:`add_gradients` (SI).
        x: Sweep variable column (``"travel"`` or ``"rack"``).

    Returns:
        Column name -> value at ``x = 0`` (float columns only, ``travel`` and ``rack``
        excluded), in SI units.

    Raises:
        ValueError: If ``x`` is invalid, the table has fewer than 3 rows, ``x`` is not
            strictly monotonic, or ``x = 0`` lies outside the sweep range.
    """
    xv = _check_axis(df, x)
    order = np.argsort(xv)
    cols = [c for c in df.columns if c not in ("travel", "rack") and df[c].dtype.kind == "f"]
    if not float(np.min(xv)) <= 0.0 <= float(np.max(xv)):
        raise ValueError(f"{x} = 0 lies outside the sweep range")
    return {c: float(np.interp(0.0, xv[order], df[c].to_numpy(dtype=float)[order])) for c in cols}
