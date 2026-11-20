"""Declarative figure spec: what ``plotting.figures`` draws, as data (for a generic renderer).

The spec is built from the very panel definitions of ``figures.py`` (``_PANELS``,
``_ROLL_PANELS``, ``_STEERING_PANELS`` ...), so the matplotlib figures and the spec cannot
drift apart. A renderer (the web app) overlays N result bundles on every panel; it never
computes physics: every plotted column exists in the study table, after
:func:`derived_columns` for the studies that need derived curves (``rack``).

Structure decisions
-------------------
* One :class:`FigureSpec` per figure and study. The corner figures exist twice
  (``corner_alignment_front`` on study ``front_corner``, ``corner_alignment_rear`` on
  ``rear_corner``); ``heave_pitch`` is split into ``heave_pitch_heave`` (study ``heave``)
  and ``heave_pitch_pitch`` (study ``pitch``). Each figure then reads exactly one table
  per dataset, so overlaying N datasets is a plain loop.
* Tables and the spec's columns are SI. Display factor and unit are resolved per series
  and axis from the dimension (``optimize.registry`` + ``units.display_*``); the
  few series that figures.py shows in another unit (e.g. FVIC in m) carry a ``display``
  override naming the figures.py unit kind.
* A panel may override the x axis (``Panel.x``): the Ackermann panel plots against the
  outer steer angle.
* Derived columns of the ``rack`` table: ``d_left``/``d_right`` (steer angle relative to
  the static pose), ``outer``, ``ackermann`` [%], ``ratio`` (steering ratio, needs the
  ``steer_c_factor`` column).
* ``SummaryRow`` mirrors ``units.SUMMARY_ROWS`` (path into the summary dict, display factor,
  unit and ``decimals``; ``dim`` from the registry for information). A renderer shows
  ``value * factor`` with ``decimals`` fixed decimals; a null / non-finite value shows
  as ``"-"``; a value that rounds to zero is shown without a minus sign ("-0.0" -> "0.0").
  A row whose path is absent from the summary is left out.
* Units: the dimensionless unit is ``""`` everywhere in :func:`spec_json` (the registry
  and figures.py may spell it ``"-"``; it is normalised in one place, ``_norm_unit``).

Renderer rules (what ``plotting.figures`` does, to be matched by any other renderer)
-----------------------------------------------------------------------------------
* Dataset ``i`` (overlay order) is drawn in ``palette[i % len(palette)]``; ``spec_json``
  carries the palette as the hex list ``palette``.
* Series ``k`` of a panel gets the line style ``linestyles[k % len(linestyles)]``
  (``side_linestyle(k)`` in ``plotting.style``; ``spec_json["linestyles"]``): solid =
  left / front, dashed = right / rear, then dotted and dash-dot.
* A panel legend (series labels in neutral grey with their line style) is shown only
  when the panel has more than one drawn series; a dataset legend (colour per dataset
  label) only when more than one dataset is overlaid.
* Panel selection per figure, given the columns of each dataset (after
  :func:`derived_columns`): a series is drawn when its column exists in the dataset; a
  panel is drawn when at least one of its series is drawn, its x column (``Panel.x`` or
  the figure x) exists, and every ``Panel.requires`` column exists in at least one
  dataset. A panel with ``alternative_to`` set is an alternative of the panel with that
  id: it is drawn only when that panel is not drawn (the first panel of an alternative
  group whose requirements hold wins). The steering figure ends with ``steering_ratio``
  (requires ``ratio``, i.e. a ``steer_c_factor`` column in the rack table), else
  ``steer_gain`` (requires ``gain``).
* Non-finite values are gaps (nan); a panel with ``ic=True`` notes the number of dropped
  points as "IC at infinity".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from suspkin.optimize.registry import (
    corner_metric_dim,
    summary_metric_dim,
    vehicle_metric_dim,
)
from suspkin.plotting import figures as _f
from suspkin.plotting import units as display_units
from suspkin.plotting.style import PALETTE, SIDE_LINESTYLES
from suspkin.units import Dim, display_factor, display_unit

__all__ = [
    "FIGURES",
    "SPEC_SCHEMA",
    "SUMMARY_ROWS",
    "Axis",
    "FigureSpec",
    "Panel",
    "Series",
    "SummaryRow",
    "derived_columns",
    "spec_json",
]

SPEC_SCHEMA = 1
Side = Literal["left", "right", "front", "rear"]
_SIDES: tuple[Side, ...] = ("left", "right", "front", "rear")

# dimensions of the derived columns of the rack table
_DERIVED_DIMS: dict[str, Dim] = {
    "d_left": "angle",
    "d_right": "angle",
    "outer": "angle",
    "ackermann": "percent",
    "ratio": "ratio",
    "gain": "angle/length",
}


@dataclass(frozen=True)
class Axis:
    """An x axis: table column, label and dimension (SI -> display via ``dim``)."""

    column: str
    label: str
    dim: Dim


@dataclass(frozen=True)
class Series:
    """One curve per dataset: a table column.

    ``side`` ("left"/"right" or "front"/"rear") and ``corner`` ("FL".."RR") tell the
    renderer how to style it (line style per series index); ``display`` is the figures.py
    unit kind when it differs from the default display unit of ``dim``.
    """

    column: str
    label: str
    dim: Dim
    side: Side | None = None
    corner: str | None = None
    display: str | None = None


@dataclass(frozen=True)
class Panel:
    """One axes: series sharing a y axis. ``x`` overrides the figure's x axis.

    ``requires``: columns (after :func:`derived_columns`) that must exist, in at least one
    dataset, for the panel to be drawn. ``alternative_to``: id of the panel this one
    replaces when that panel is not drawn (see the module docstring).
    """

    id: str
    title: str
    series: tuple[Series, ...]
    y_label: str | None = None  # explicit y label, default "[unit of the first series]"
    ic: bool = False  # non-finite points are instant centres at infinity
    x: Axis | None = None
    requires: tuple[str, ...] = ()
    alternative_to: str | None = None

    @property
    def y_dim(self) -> Dim:
        """Dimension of the first series."""
        return self.series[0].dim


@dataclass(frozen=True)
class FigureSpec:
    """One figure: panels over one study table, overlaid for N datasets."""

    id: str
    group: str  # FIGURE_SET key
    title: str
    study: str  # StudyResults table name
    x: Axis
    panels: tuple[Panel, ...]
    columns: int = 1  # panel grid columns of the matplotlib figure


@dataclass(frozen=True)
class SummaryRow:
    """One row of the summary table."""

    label: str
    path: tuple[str, ...]
    dim: Dim | None
    factor: float
    unit: str
    fmt: str  # figures.py format string ("{:.2f}"); the spec exposes ``decimals``

    @property
    def decimals(self) -> int:
        """Number of decimals of ``fmt`` (``"{:.2f}"`` -> 2)."""
        m = re.fullmatch(r"\{:\.(\d+)f\}", self.fmt)
        if m is None:
            raise ValueError(f"unsupported summary format {self.fmt!r}")
        return int(m.group(1))


# ------------------------------------------------------------------------------- builders
def _dim(study: str, column: str) -> Dim:
    if column in _DERIVED_DIMS and study == "rack":
        return _DERIVED_DIMS[column]
    return corner_metric_dim(column) if study.endswith("_corner") else vehicle_metric_dim(column)


def _norm_unit(unit: str) -> str:
    """Dimensionless is ``""`` (figures.py / the registry may spell it ``"-"``)."""
    return "" if unit == "-" else unit


def _unit_of(dim: Dim) -> tuple[float, str]:
    return display_factor(dim), _norm_unit(display_unit(dim))


def _kind_unit(kind: str) -> tuple[float, str]:
    fac, unit = _f._KINDS[kind]
    return fac, _norm_unit(unit)


def _same_factor(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12 * max(1.0, abs(b))


def _series(study: str, s: _f._Series) -> Series:
    dim = _dim(study, s.col)
    fac, unit = _unit_of(dim)
    kfac, kunit = _kind_unit(s.kind)
    display = None if _same_factor(fac, kfac) and unit == kunit else s.kind
    m = re.match(r"^(FL|FR|RL|RR)_", s.col)
    side: Side | None = None
    if s.side is not None:
        side = next((c for c in _SIDES if c == s.side), None)
        if side is None:
            raise ValueError(f"series {s.col!r}: unknown side {s.side!r}")
    return Series(s.col, s.label, dim, side, m.group(1) if m else None, display)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _axis(study: str, column: str, kind: str, name: str) -> Axis:
    ax = Axis(column, name, _dim(study, column))
    if not _same_factor(display_factor(ax.dim), _f._KINDS[kind][0]):
        raise ValueError(f"axis {column!r}: registry unit differs from figures kind {kind}")
    return ax


def _panel(
    study: str,
    p: _f._Panel,
    fig_x: Axis,
    x: Axis | None = None,
    *,
    requires: tuple[str, ...] = (),
    alternative_to: str | None = None,
) -> Panel:
    return Panel(
        id=_slug(p.title),
        title=p.title,
        series=tuple(_series(study, s) for s in p.series),
        y_label=p.ylabel,
        ic=p.ic,
        x=x if x is not None and x != fig_x else None,
        requires=requires,
        alternative_to=alternative_to,
    )


def _build() -> tuple[FigureSpec, ...]:
    out: list[FigureSpec] = []
    corner_figs = (
        ("corner_alignment", "fig_corner_alignment", "alignment", 3),
        ("corner_position", "fig_corner_position", "wheel position", 2),
        ("corner_centres", "fig_corner_centres", "centres", 3),
        ("actuation", "fig_actuation", "actuation", 3),
    )
    for group, key, name, ncols in corner_figs:
        for axle in ("front", "rear"):
            study = f"{axle}_corner"
            x = _axis(study, "travel", "mm", "wheel travel")
            panels = tuple(_panel(study, p, x) for p in _f._PANELS[key])
            out.append(
                FigureSpec(
                    f"{group}_{axle}", group, f"{axle.capitalize()} corner: {name}",
                    study, x, panels, ncols,
                )
            )  # fmt: skip
    # steering (rack study): the last panel is the steering ratio when the table has
    # ``steer_c_factor`` (derived column ``ratio``), else the steer gain (``gain``)
    xr = _axis("rack", "rack", "mm", "rack travel")
    ratio_id = _slug(_f._STEER_RATIO[0].title)
    panels = tuple(
        _panel("rack", p, xr, _axis("rack", col, kind, name))
        for p, col, kind, name in _f._STEERING_PANELS
    )
    for (p, col, kind, name), req, alt in (
        (_f._STEER_RATIO, ("ratio",), None),
        (_f._STEER_GAIN, ("gain",), ratio_id),
    ):
        panels += (
            _panel("rack", p, xr, _axis("rack", col, kind, name), requires=req, alternative_to=alt),
        )
    out.append(FigureSpec("steering", "steering", "Steering", "rack", xr, panels, 2))
    xroll = _axis("roll", "roll", "deg", "body roll")
    out.append(
        FigureSpec(
            "roll", "roll", "Roll sweep", "roll", xroll,
            tuple(_panel("roll", p, xroll) for p in _f._ROLL_PANELS), 3,
        )
    )  # fmt: skip
    xh = _axis("heave", "heave", "mm", "heave")
    out.append(
        FigureSpec(
            "heave_pitch_heave", "heave_pitch", "Heave sweep", "heave", xh,
            tuple(_panel("heave", p, xh) for p in _f._HEAVE_PANELS), 2,
        )
    )  # fmt: skip
    xp = _axis("pitch", "pitch", "deg", "body pitch")
    out.append(
        FigureSpec(
            "heave_pitch_pitch", "heave_pitch", "Pitch sweep", "pitch", xp,
            tuple(_panel("pitch", p, xp) for p in _f._PITCH_PANELS), 2,
        )
    )  # fmt: skip
    return tuple(out)


def _summary_rows() -> tuple[SummaryRow, ...]:
    rows = []
    for label, path, factor, unit, fmt in display_units.SUMMARY_ROWS:
        rows.append(SummaryRow(label, path, summary_metric_dim(".".join(path)), factor, unit, fmt))
    for r in rows:
        _ = r.decimals  # validate the format strings at import time
    return tuple(rows)


FIGURES: tuple[FigureSpec, ...] = _build()
SUMMARY_ROWS: tuple[SummaryRow, ...] = _summary_rows()


# ------------------------------------------------------------------------------- derived
def derived_columns(study: str, df: pd.DataFrame) -> pd.DataFrame:
    """``df`` plus the derived curve columns of ``study`` (a copy; SI).

    Only the ``rack`` study has any (``d_left``, ``d_right``, ``outer``, ``ackermann``,
    ``ratio``, see the module docstring); other studies are returned unchanged. The maths
    is that of ``plotting.figures`` (``_steering_frame``).
    """
    if study != "rack" or len(df) == 0:
        return df
    derived = _f._steering_frame(df, None, None, None)
    if derived.empty:
        return df
    extra = derived[[c for c in derived.columns if c not in df.columns]]
    return pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


# ------------------------------------------------------------------------------------ JSON
def _axis_json(ax: Axis) -> dict[str, Any]:
    fac, unit = _unit_of(ax.dim)
    return {"column": ax.column, "label": ax.label, "dim": ax.dim, "factor": fac, "unit": unit}


def _series_json(s: Series) -> dict[str, Any]:
    if s.display is not None:
        fac, unit = _kind_unit(s.display)
    else:
        fac, unit = _unit_of(s.dim)
    return {
        "column": s.column,
        "label": s.label,
        "dim": s.dim,
        "side": s.side,
        "corner": s.corner,
        "factor": fac,
        "unit": unit,
    }


def _panel_json(p: Panel) -> dict[str, Any]:
    first = _series_json(p.series[0])
    y_label = (
        p.y_label if p.y_label is not None else (f"[{first['unit']}]" if first["unit"] else "")
    )
    return {
        "id": p.id,
        "title": p.title,
        "y_label": y_label,
        "y_dim": p.y_dim,
        "ic": p.ic,
        "requires": list(p.requires),
        "alternative_to": p.alternative_to,
        "x": None if p.x is None else _axis_json(p.x),
        "series": [_series_json(s) for s in p.series],
    }


def spec_json() -> dict[str, Any]:
    """The whole spec as a JSON-serialisable dict (display factor/unit resolved)."""
    return {
        "schema": SPEC_SCHEMA,
        "palette": list(PALETTE),
        "linestyles": list(SIDE_LINESTYLES),
        "figures": [
            {
                "id": f.id,
                "group": f.group,
                "title": f.title,
                "study": f.study,
                "columns": f.columns,
                "x": _axis_json(f.x),
                "panels": [_panel_json(p) for p in f.panels],
            }
            for f in FIGURES
        ],
        "summary_rows": [
            {
                "label": r.label,
                "path": list(r.path),
                "dim": r.dim,
                "factor": r.factor,
                "unit": _norm_unit(r.unit),
                "decimals": r.decimals,
            }
            for r in SUMMARY_ROWS
        ],
    }
