"""Tidy result tables from corner solver states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from suspkin.geometry.corners import Corner
from suspkin.kinematics.corner import CornerState

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
        from suspkin.metrics.table import metrics_frame

        mf = metrics_frame(states, corner).drop(columns=["travel", "rack"])
        df = pd.concat([df, mf], axis=1)
    return df
