"""Tidy result tables from corner solver states."""

from __future__ import annotations

from collections.abc import Sequence

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
    rows: list[dict[str, object]] = []
    for s in states:
        row: dict[str, object] = {"corner": s.corner_id}
        for name in _SCALARS:
            row[name] = float(getattr(s, name))
        for axis, v in zip("xyz", s.spin_axis, strict=True):
            row[f"spin_{axis}"] = float(v)
        for pname, p in s.points.items():
            for axis, v in zip("xyz", p, strict=True):
                row[f"{pname}_{axis}"] = float(v)
        row["coilover_in_range"] = bool(s.coilover_in_range)
        rows.append(row)
    df = pd.DataFrame(rows)
    if metrics and corner is not None:
        from suspkin.metrics.table import metrics_frame

        mf = metrics_frame(states, corner).drop(columns=["travel", "rack"])
        df = pd.concat([df, mf], axis=1)
    return df
