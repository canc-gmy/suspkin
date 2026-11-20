"""Report-quality 2D figure sets (matplotlib) built from result tables.

Plotting consumes tidy result tables only; it never calls the solver. Every ``fig_*``
function returns a :class:`matplotlib.figure.Figure` (created without pyplot, never
shown) and takes ``datasets``: a sequence of ``(label, frame)`` pairs, one per design, so
that N designs overlay on the same axes (colour = design). Tables are in SI; the figures
display mm and deg. Non-finite values (nan, inf) are dropped from the lines.

Expected frame kinds:

* corner frame: ``metrics_frame`` + ``add_gradients`` of a travel sweep of ONE corner
  (e.g. the front-left and/or rear-left corner); ``states_to_frame(metrics=True)``
  frames work too and add ``coilover_length`` / ``toggle_margin``.
* vehicle frame: ``vehicle_frame`` of a roll, heave, pitch or rack sweep.

Panels whose columns are absent from every frame are left out.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from suspkin.metrics import vehicle as vm
from suspkin.plotting.style import (
    axis_label,
    design_color,
    figure_size,
    side_linestyle,
    use_style,
    zero_lines,
)
from suspkin.plotting.units import SUMMARY_ROWS, format_value

Datasets = Sequence[tuple[str, pd.DataFrame]]

# kind -> (factor from SI, display unit)
_KINDS: dict[str, tuple[float, str]] = {
    "mm": (1e3, "mm"),
    "m": (1.0, "m"),
    "deg": (180.0 / math.pi, "deg"),
    "N": (1.0, "N"),
    "N/mm": (1e-3, "N/mm"),
    "1/mm": (1e-3, "1/mm"),
    "pct": (1.0, "%"),
    "ratio": (1.0, ""),
    "raw": (1.0, ""),
}
_CORNER_IDS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class _Series:
    col: str
    label: str
    kind: str
    side: str | None = None  # "left"/"right" or "front"/"rear" (informational, for the spec)


@dataclass(frozen=True)
class _Panel:
    title: str
    series: tuple[_Series, ...]
    ic: bool = False  # annotate dropped non-finite points as "IC at infinity"
    ylabel: str | None = None  # override of the y-axis label (default: unit of series 0)


@dataclass(frozen=True)
class _Item:
    panel: _Panel
    datasets: Datasets
    xcol: str
    xkind: str
    xname: str


def _s(col: str, label: str, kind: str, side: str | None = None) -> _Series:
    return _Series(col, label, kind, side)


def _check(datasets: Datasets, name: str, allow_empty: bool = False) -> None:
    if not datasets and not allow_empty:
        raise ValueError(f"{name} must contain at least one (label, frame) pair")
    for label, df in datasets:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"dataset {label!r} is not a DataFrame")


def _values(df: pd.DataFrame, col: str, kind: str) -> tuple[np.ndarray[Any, Any], int]:
    """Display values of a column with non-finite entries as nan, and their count."""
    raw = df[col].to_numpy(dtype=float) * _KINDS[kind][0]
    bad = ~np.isfinite(raw)
    return np.where(bad, np.nan, raw), int(bad.sum())


def _ylabel(p: _Panel) -> str:
    if p.ylabel is not None:
        return p.ylabel
    unit = _KINDS[p.series[0].kind][1]
    return f"[{unit}]" if unit else ""


def _draw_panel(ax: Axes, it: _Item) -> None:
    p = it.panel
    n_ic = 0
    for si, s in enumerate(p.series):
        for di, (_, df) in enumerate(it.datasets):
            if s.col not in df.columns or it.xcol not in df.columns:
                continue
            x, _ = _values(df, it.xcol, it.xkind)
            y, bad = _values(df, s.col, s.kind)
            n_ic += bad
            ax.plot(x, y, color=design_color(di), ls=side_linestyle(si))
    zero_lines(ax)
    ax.set_title(p.title)
    ax.set_ylabel(_ylabel(p))
    ax.set_xlabel(axis_label(it.xname, _KINDS[it.xkind][1]))
    if len(p.series) > 1:
        proxies = [
            Line2D([], [], color="#555555", ls=side_linestyle(i), label=s.label)
            for i, s in enumerate(p.series)
        ]
        ax.legend(handles=proxies, loc="best")
    if p.ic and n_ic:
        ax.text(
            0.02,
            0.03,
            f"IC at infinity ({n_ic} pts dropped)",
            transform=ax.transAxes,
            fontsize=6,
            color="#8c8c8c",
        )


def _keep(items: Sequence[_Item]) -> list[_Item]:
    """Drop series and panels whose columns are absent from every dataset."""
    keep: list[_Item] = []
    for it in items:
        present = tuple(
            s for s in it.panel.series if any(s.col in df.columns for _, df in it.datasets)
        )
        has_x = any(it.xcol in df.columns for _, df in it.datasets)
        if present and has_x:
            p = _Panel(it.panel.title, present, it.panel.ic, it.panel.ylabel)
            keep.append(_Item(p, it.datasets, it.xcol, it.xkind, it.xname))
    return keep


def _make(items: Sequence[_Item], ncols: int, title: str, preset: str = "a4_landscape") -> Figure:
    keep = _keep(items)
    if not keep:
        raise ValueError("none of the expected columns are present in the datasets")
    nrows = math.ceil(len(keep) / ncols)
    with use_style():
        fig = Figure(figsize=figure_size(preset, rows=nrows), layout="constrained")
        gs = fig.add_gridspec(nrows, ncols)
        designs: dict[str, int] = {}
        for k, it in enumerate(keep):
            ax = fig.add_subplot(gs[k // ncols, k % ncols])
            _draw_panel(ax, it)
            for i, (lab, _) in enumerate(it.datasets):
                designs.setdefault(lab, i)
        if len(designs) > 1:
            handles = [
                Line2D([], [], color=design_color(i), label=lab) for lab, i in designs.items()
            ]
            fig.legend(
                handles=handles, loc="outside lower center", ncol=min(len(handles), 4), fontsize=7
            )
        fig.suptitle(title, fontsize=10)
    return fig


def _corner_items(datasets: Datasets, panels: Sequence[_Panel]) -> list[_Item]:
    return [_Item(p, datasets, "travel", "mm", "wheel travel") for p in panels]


def _axle_name(axle: str) -> str:
    if axle not in ("front", "rear"):
        raise ValueError(f"axle must be 'front' or 'rear', got {axle!r}")
    return axle.capitalize()


# --------------------------------------------------------------------------- panel layouts
# Panel definitions are module-level so that ``plotting.spec`` mirrors them exactly.
_PANELS: dict[str, list[_Panel]] = {
    "fig_corner_alignment": [
        _Panel("Camber", (_s("camber", "camber", "deg"),)),
        _Panel("Toe (bump steer)", (_s("toe", "toe", "deg"),)),
        _Panel("Caster", (_s("caster", "caster", "deg"),)),
        _Panel("Kingpin inclination", (_s("kpi", "KPI", "deg"),)),
        _Panel("Scrub radius", (_s("scrub_radius", "scrub radius", "mm"),)),
        _Panel("Mechanical trail", (_s("mechanical_trail", "trail", "mm"),)),
    ],
    "fig_corner_position": [
        _Panel("Half-track change", (_s("half_track_change", "half-track", "mm"),)),
        _Panel(
            "Contact-patch shift",
            (
                _s("contact_patch_dx", "dx (long.)", "mm"),
                _s("contact_patch_dy", "dy (lat.)", "mm"),
            ),
        ),
        _Panel("Wheel-centre recession", (_s("wheel_center_recession", "recession", "mm"),)),
        _Panel(
            "Wheelbase change (contribution)",
            (_s("wheelbase_change_contribution", "wheelbase", "mm"),),
        ),
    ],
    "fig_corner_centres": [
        _Panel("Roll-centre height", (_s("rc_height", "RC height", "mm"),), ic=True),
        _Panel("FVIC lateral (y)", (_s("fvic_y", "FVIC y", "m"),), ic=True),
        _Panel("FVIC height (z)", (_s("fvic_z", "FVIC z", "mm"),), ic=True),
        _Panel("Front-view swing-arm length", (_s("fvsa", "FVSA", "m"),), ic=True),
        _Panel("Front-view swing angle", (_s("fv_swing_angle", "FV swing", "deg"),), ic=True),
        _Panel(
            "Side-view angles",
            (
                _s("sv_swing_angle", "SV swing (patch)", "deg"),
                _s("sv_wc_angle", "SV swing (wheel centre)", "deg"),
            ),
            ic=True,
        ),
    ],
    "fig_actuation": [
        _Panel("Coilover compression", (_s("coilover_compression", "compression", "mm"),)),
        _Panel("Coilover length", (_s("coilover_length", "length", "mm"),)),
        _Panel("Spring motion ratio", (_s("mr_spring", "MR spring/wheel", "ratio"),)),
        _Panel("MR progression", (_s("mr_progression", "d(MR)/dz", "1/mm"),)),
        _Panel(
            "Wheel and ride rate",
            (_s("wheel_rate", "wheel rate", "N/mm"), _s("ride_rate", "ride rate", "N/mm")),
        ),
        _Panel("Spring force", (_s("spring_force", "spring force", "N"),)),
        _Panel(
            "Pushrod and rocker angle",
            (_s("pushrod_angle", "pushrod angle", "deg"), _s("rocker_rotation", "rocker", "deg")),
        ),
        _Panel("Toggle margin", (_s("toggle_margin", "toggle margin", "deg"),)),
        _Panel(
            "Stroke remaining",
            (
                _s("bump_stroke_remaining", "bump", "mm"),
                _s("droop_stroke_remaining", "droop", "mm"),
            ),
        ),
    ],
}


# ------------------------------------------------------------------------------ corner sets
def fig_corner_alignment(datasets: Datasets, axle: str = "front") -> Figure:
    """Camber, toe (bump steer), caster, KPI, scrub radius, mechanical trail vs wheel travel.

    ``datasets``: corner frames (one corner per frame; ``axle`` only labels the title).

    Args:
        datasets: Sequence of ``(label, DataFrame)``; one curve per dataset (SI columns).
        axle: ``"front"`` or ``"rear"``; only used in the title.

    Returns:
        Matplotlib figure (angles shown in deg, lengths in mm).
    """
    _check(datasets, "datasets")
    panels = _PANELS["fig_corner_alignment"]
    return _make(_corner_items(datasets, panels), 3, f"{_axle_name(axle)} corner: alignment")


def fig_corner_position(datasets: Datasets, axle: str = "front") -> Figure:
    """Half-track change, contact-patch shift, wheel-centre recession, wheelbase change.

    ``datasets``: corner frames (one corner per frame).

    Args:
        datasets: Sequence of ``(label, DataFrame)``; one curve per dataset (SI columns).
        axle: ``"front"`` or ``"rear"``; only used in the title.

    Returns:
        Matplotlib figure (lengths in mm).
    """
    _check(datasets, "datasets")
    panels = _PANELS["fig_corner_position"]
    return _make(_corner_items(datasets, panels), 2, f"{_axle_name(axle)} corner: wheel position")


def fig_corner_centres(datasets: Datasets, axle: str = "front") -> Figure:
    """Roll-centre height, FVIC y/z, FVSA and side-view swing angles vs wheel travel.

    ``datasets``: corner frames. Instant centres at infinity (nan/inf) are dropped and
    annotated.

    Args:
        datasets: Sequence of ``(label, DataFrame)``; one curve per dataset (SI columns).
        axle: ``"front"`` or ``"rear"``; only used in the title.

    Returns:
        Matplotlib figure (lengths in mm, angles in deg).
    """
    _check(datasets, "datasets")
    panels = _PANELS["fig_corner_centres"]
    return _make(_corner_items(datasets, panels), 3, f"{_axle_name(axle)} corner: centres")


def fig_actuation(datasets: Datasets, axle: str = "front") -> Figure:
    """Coilover, motion ratios, wheel/ride rate, spring force, toggle margin, stroke left.

    ``datasets``: corner frames after ``add_gradients`` (``mr_spring``, ``wheel_rate`` ...);
    ``coilover_length`` and ``toggle_margin`` appear when the frame has them
    (``states_to_frame`` columns merged in).

    Args:
        datasets: Sequence of ``(label, DataFrame)``; one curve per dataset (SI columns).
        axle: ``"front"`` or ``"rear"``; only used in the title.

    Returns:
        Matplotlib figure.
    """
    _check(datasets, "datasets")
    panels = _PANELS["fig_actuation"]
    return _make(_corner_items(datasets, panels), 3, f"{_axle_name(axle)} corner: actuation")


# ----------------------------------------------------------------------------- vehicle sets
def _steering_frame(
    df: pd.DataFrame, wheelbase: float | None, track: float | None, c_factor: float | None
) -> pd.DataFrame:
    """Derived steering table from a rack-sweep vehicle frame (SI, steer angles relative)."""
    need = {"rack", "FL_steer_angle", "FR_steer_angle"}
    if not need <= set(df.columns):
        return pd.DataFrame()
    rack = df["rack"].to_numpy(dtype=float)
    i0 = int(np.argmin(np.abs(rack)))
    dl = df["FL_steer_angle"].to_numpy(dtype=float)
    dr = df["FR_steer_angle"].to_numpy(dtype=float)
    dl, dr = dl - dl[i0], dr - dr[i0]
    out = pd.DataFrame({"rack": rack, "d_left": dl, "d_right": dr})
    for c in ("FL_toe", "FR_toe"):
        if c in df.columns:
            out[c] = df[c].to_numpy(dtype=float)
    mean = 0.5 * (dl + dr)
    outer = np.full(rack.size, np.nan)
    inner = np.full(rack.size, np.nan)
    for i in range(rack.size):
        if np.isfinite(dl[i]) and np.isfinite(dr[i]):
            inner[i], outer[i] = vm.inner_outer(float(dl[i]), float(dr[i]))
    sgn = np.where(mean >= 0.0, 1.0, -1.0)
    out["outer"] = np.where(np.abs(outer) < 1e-4, np.nan, sgn * outer)
    if "wheelbase" in df.columns:
        wheelbase = float(df["wheelbase"].iloc[0])
    if "steer_c_factor" in df.columns:
        c_factor = float(df["steer_c_factor"].iloc[0])
    if track is None and {"FL_cp_y", "FR_cp_y"} <= set(df.columns):
        track = float(df["FL_cp_y"].iloc[i0] - df["FR_cp_y"].iloc[i0])
    if wheelbase is not None and track is not None:
        out["ackermann"] = [
            vm.ackermann_percent(float(i), float(o), wheelbase, track)
            if np.isfinite(i) and np.isfinite(o)
            else np.nan
            for i, o in zip(inner, outer, strict=True)
        ]
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.gradient(mean, rack) if rack.size > 1 else np.full(rack.size, np.nan)
        if c_factor is not None:
            out["ratio"] = np.where(np.abs(gain) > 1e-9, (2.0 * math.pi / c_factor) / gain, np.nan)
        else:
            out["gain"] = gain * (180.0 / math.pi) / 1000.0
    return out


# (panel, x column, x kind, x name) of the steering figure; the last panel is the steering
# ratio when a c_factor is known, else the steer gain.
_STEERING_PANELS: tuple[tuple[_Panel, str, str, str], ...] = (
    (
        _Panel(
            "Steer angle",
            (_s("d_left", "left", "deg", "left"), _s("d_right", "right", "deg", "right")),
        ),
        "rack",
        "mm",
        "rack travel",
    ),
    (
        _Panel("Ackermann", (_s("ackermann", "Ackermann", "pct"),)),
        "outer",
        "deg",
        "outer steer angle",
    ),
    (
        _Panel("Toe", (_s("FL_toe", "left", "deg", "left"), _s("FR_toe", "right", "deg", "right"))),
        "rack",
        "mm",
        "rack travel",
    ),
)
_STEER_RATIO = (
    _Panel("Steering ratio", (_s("ratio", "ratio", "ratio"),)),
    "rack",
    "mm",
    "rack travel",
)
_STEER_GAIN = (
    _Panel("Steer gain", (_s("gain", "mean steer gain", "raw"),), ylabel="[deg/mm]"),
    "rack",
    "mm",
    "rack travel",
)


def fig_steering(
    datasets: Datasets,
    *,
    wheelbase: float | None = None,
    track: float | None = None,
    c_factor: float | None = None,
) -> Figure:
    """Steer angle vs rack, Ackermann % vs outer steer, toe vs rack, steering ratio vs rack.

    ``datasets``: vehicle frames of a rack sweep (``rack``, ``FL_/FR_steer_angle``,
    ``FL_/FR_toe``, ``FL_/FR_cp_y``). Steer angles are shown relative to the static pose.
    ``wheelbase`` [m] and ``c_factor`` [m/rev] are read per dataset from the constant
    columns ``wheelbase`` and ``steer_c_factor`` when present (so designs of different
    wheelbase overlay correctly), else from the keyword arguments.
    The Ackermann panel needs a wheelbase (``track`` defaults to the front contact
    point spacing of the frame). With ``c_factor`` [m/rev] the last panel shows the
    differential steering ratio (handwheel / mean road wheel), otherwise the steer gain
    [deg/mm].

    Args:
        datasets: Sequence of ``(label, DataFrame)`` of vehicle rack-sweep frames (SI).
        wheelbase: Wheelbase [m] used when a frame has no ``wheelbase`` column.
        track: Front track [m]; default the contact-point spacing of the frame.
        c_factor: Rack travel per handwheel revolution [m/rev]; selects the steering-ratio panel.

    Returns:
        Matplotlib figure.
    """
    _check(datasets, "datasets")
    derived: list[tuple[str, pd.DataFrame]] = [
        (lab, _steering_frame(df, wheelbase, track, c_factor)) for lab, df in datasets
    ]
    last = _STEER_RATIO if any("ratio" in d.columns for _, d in derived) else _STEER_GAIN
    items = [
        _Item(panel, derived, xcol, xkind, xname)
        for panel, xcol, xkind, xname in (*_STEERING_PANELS, last)
    ]
    return _make(items, 2, "Steering")


def _corners(col: str, kind: str) -> tuple[_Series, ...]:
    return tuple(_s(f"{c}_{col}", c, kind) for c in _CORNER_IDS)


_ROLL_PANELS = [
    _Panel("Corner travel", _corners("travel", "mm")),
    _Panel("Camber to ground", _corners("camber_ground", "deg")),
    _Panel("Steer to ground", _corners("steer_ground", "deg")),
    _Panel(
        "Roll-centre height",
        (_s("rc_front_z", "front", "mm", "front"), _s("rc_rear_z", "rear", "mm", "rear")),
        ic=True,
    ),
    _Panel(
        "Roll-centre lateral position",
        (_s("rc_front_y", "front", "mm", "front"), _s("rc_rear_y", "rear", "mm", "rear")),
        ic=True,
    ),
    _Panel(
        "ARB elastic coordinate",
        (
            _s("arb_front_elastic", "front", "raw", "front"),
            _s("arb_rear_elastic", "rear", "raw", "rear"),
        ),
        ylabel="[rad or m]",
    ),
]

_HEAVE_PANELS = (
    _Panel("Heave: corner travel", _corners("travel", "mm")),
    _Panel(
        "Heave: roll-centre height",
        (_s("rc_front_z", "front", "mm", "front"), _s("rc_rear_z", "rear", "mm", "rear")),
        ic=True,
    ),
)

_PITCH_PANELS = (
    _Panel(
        "Pitch: axle travel",
        (_s("FL_travel", "front (FL)", "mm", "front"), _s("RL_travel", "rear (RL)", "mm", "rear")),
    ),
    _Panel(
        "Pitch: side-view swing angle",
        (
            _s("FL_sv_swing_angle", "front (FL)", "deg", "front"),
            _s("RL_sv_swing_angle", "rear (RL)", "deg", "rear"),
        ),
        ic=True,
    ),
)


def fig_roll(datasets: Datasets) -> Figure:
    """Corner travel, ground camber and steer, RC height and lateral position, ARB vs roll.

    ``datasets``: vehicle frames of a roll sweep. Line style per corner: FL solid,
    FR dashed, RL dotted, RR dash-dot; front/rear axle panels use solid = front,
    dashed = rear.
    """
    _check(datasets, "datasets")
    panels = _ROLL_PANELS
    items = [_Item(p, datasets, "roll", "deg", "body roll") for p in panels]
    return _make(items, 3, "Roll sweep")


def fig_heave_pitch(datasets_heave: Datasets, datasets_pitch: Datasets) -> Figure:
    """Heave: corner travel and RC heights; pitch: axle travel and side-view swing angles.

    ``datasets_heave`` / ``datasets_pitch``: vehicle frames of a heave / pitch sweep (either
    may be empty, not both). Panel line styles as in :func:`fig_roll`.

    Args:
        datasets_heave: Sequence of ``(label, DataFrame)`` of heave-sweep frames (SI).
        datasets_pitch: Sequence of ``(label, DataFrame)`` of pitch-sweep frames (SI).

    Returns:
        Matplotlib figure.
    """
    _check(datasets_heave, "datasets_heave", allow_empty=True)
    _check(datasets_pitch, "datasets_pitch", allow_empty=True)
    if not datasets_heave and not datasets_pitch:
        raise ValueError("at least one of datasets_heave, datasets_pitch must be non-empty")
    items: list[_Item] = []
    if datasets_heave:
        for p in _HEAVE_PANELS:
            items.append(_Item(p, datasets_heave, "heave", "mm", "heave"))
    if datasets_pitch:
        for p in _PITCH_PANELS:
            items.append(_Item(p, datasets_pitch, "pitch", "deg", "body pitch"))
    return _make(items, 2, "Heave and pitch sweeps")


# ---------------------------------------------------------------------------- summary table
def _lookup(summary: object, path: Sequence[str]) -> float | None:
    cur: Any = summary
    for key in path:
        cur = cur.get(key) if isinstance(cur, Mapping) else getattr(cur, key, None)
        if cur is None:
            return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def fig_summary_table(summary: object, title: str = "Vehicle summary") -> Figure:
    """Key static scalars as a table figure.

    ``summary`` is a ``VehicleSummary`` (or any dataclass) or its ``to_dict()`` mapping
    (nested ``front`` / ``rear``). Missing fields are left out, nan values show as "-".

    Args:
        summary: ``VehicleSummary``, other dataclass, or its ``to_dict()`` mapping.
        title: Figure title.

    Returns:
        Matplotlib figure containing the table.
    """
    if not (isinstance(summary, Mapping) or is_dataclass(summary)):
        raise TypeError("summary must be a dataclass or a mapping")
    rows: list[list[str]] = []
    for label, path, factor, unit, fmt in SUMMARY_ROWS:
        v = _lookup(summary, path)
        if v is None:
            continue
        rows.append([label, format_value(fmt, v * factor) if math.isfinite(v) else "-", unit])
    if not rows:
        raise ValueError("summary has none of the expected fields")
    half = math.ceil(len(rows) / 2)
    with use_style():
        fig = Figure(figsize=(figure_size("a4_width")[0], 0.28 * half + 0.9), layout="constrained")
        for k, chunk in enumerate((rows[:half], rows[half:])):
            ax = fig.add_subplot(1, 2, k + 1)
            ax.axis("off")
            if not chunk:
                continue
            tab = ax.table(
                cellText=chunk,
                colLabels=["Quantity", "Value", "Unit"],
                colWidths=[0.55, 0.22, 0.23],
                cellLoc="left",
                loc="upper center",
            )
            tab.auto_set_font_size(False)
            tab.set_fontsize(7)
            tab.scale(1.0, 1.25)
            for (r, _c), cell in tab.get_celld().items():
                cell.set_linewidth(0.3)
                cell.set_edgecolor("#d9d9d9")
                if r == 0:
                    cell.set_text_props(weight="bold")
        fig.suptitle(title, fontsize=10)
    return fig


# ------------------------------------------------------------------------------- registry
FIGURE_SET: dict[str, Callable[..., Figure]] = {
    "corner_alignment": fig_corner_alignment,
    "corner_position": fig_corner_position,
    "corner_centres": fig_corner_centres,
    "actuation": fig_actuation,
    "steering": fig_steering,
    "roll": fig_roll,
    "heave_pitch": fig_heave_pitch,
    "summary_table": fig_summary_table,
}

# Frame kind(s) each figure consumes (for the report generator).
FIGURE_INPUTS: dict[str, str] = {
    "corner_alignment": "corner_frames",
    "corner_position": "corner_frames",
    "corner_centres": "corner_frames",
    "actuation": "corner_frames",
    "steering": "vehicle_frames_rack",
    "roll": "vehicle_frames_roll",
    "heave_pitch": "vehicle_frames_heave + vehicle_frames_pitch",
    "summary_table": "vehicle_summary",
}


def save_figure(
    fig: Figure,
    path: str | Path,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 200,
) -> list[Path]:
    """Save ``fig`` as ``<path stem>.<fmt>`` for every format; returns the written paths.

    Args:
        fig: Matplotlib figure.
        path: Target path; any suffix is replaced by each format's extension.
        formats: Image formats, e.g. ``("png", "svg")``.
        dpi: Raster resolution in dots per inch.
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        ext = fmt.lstrip(".")
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, format=ext, dpi=dpi, bbox_inches="tight")
        written.append(out)
    return written
