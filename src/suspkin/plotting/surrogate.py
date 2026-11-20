"""Surrogate validation plots (experimental).

Pure consumers of the tables written by ``suspkin surrogate validate``:

- ``validation.csv``: ``channel, unit, rmse, tol, ratio, passed``;
- ``parity.csv``: ``channel, unit, fom, pred`` (one row per valid test grid point);
- ``worst_curves.csv``: ``rank, sample, channel, unit, travel_mm, fom, pred``.

Nothing here calls the model, the solver or the evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

from suspkin.plotting.figures import save_figure
from suspkin.plotting.style import design_color, figure_size, side_linestyle, use_style

__all__ = [
    "plot_al_progress",
    "plot_al_radius",
    "plot_al_residual",
    "plot_parity",
    "plot_rmse_ratio",
    "plot_worst_curves",
    "write_surrogate_figures",
]

_PASS = "#0072B2"
_FAIL = "#D55E00"


def _need(df: pd.DataFrame, cols: Sequence[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} table lacks columns {missing}")


def plot_rmse_ratio(table: pd.DataFrame, title: str = "") -> Figure:
    """Horizontal bars of ``RMSE / tolerance`` per channel with the 1.0 pass line."""
    _need(table, ("channel", "ratio"), "validation")
    names = [str(c) for c in table["channel"]]
    ratio = table["ratio"].to_numpy(dtype=float)
    with use_style():
        fig = Figure(figsize=(6.3, 0.9 + 0.2 * len(names)), layout="constrained")
        ax = fig.subplots()
        colors = [_PASS if r <= 1.0 else _FAIL for r in ratio]
        ax.barh(np.arange(len(names)), ratio, color=colors)
        ax.axvline(1.0, color="#333333", lw=1.0, ls="--", label="tolerance")
        ax.set_yticks(np.arange(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlabel("RMSE / tolerance")
        top = float(np.nanmax(ratio)) if ratio.size else 1.0
        if math.isfinite(top) and top > 20.0:
            ax.set_xscale("log")
        ax.set_title(title or "Test error per channel")
        ax.legend(loc="lower right")
    return fig


def plot_parity(
    parity: pd.DataFrame, channels: Sequence[str] | None = None, title: str = ""
) -> Figure:
    """Predicted vs FOM scatter with the identity line, one panel per channel."""
    _need(parity, ("channel", "fom", "pred"), "parity")
    names = list(channels) if channels is not None else list(dict.fromkeys(parity["channel"]))
    if not names:
        raise ValueError("parity table has no channel")
    ncol = min(3, len(names))
    nrow = math.ceil(len(names) / ncol)
    with use_style():
        fig = Figure(figsize=figure_size("a4_width", rows=nrow), layout="constrained")
        axes = np.atleast_1d(fig.subplots(nrow, ncol, squeeze=False)).ravel()
        for ax, name in zip(axes, names, strict=False):
            sub = parity[parity["channel"] == name]
            fom = sub["fom"].to_numpy(dtype=float)
            pred = sub["pred"].to_numpy(dtype=float)
            ax.scatter(fom, pred, s=3, alpha=0.3, color=design_color(0), rasterized=True)
            if fom.size:
                lo, hi = float(min(fom.min(), pred.min())), float(max(fom.max(), pred.max()))
                ax.plot([lo, hi], [lo, hi], color="#333333", lw=0.8)
            unit = str(sub["unit"].iloc[0]) if "unit" in sub.columns and len(sub) else ""
            lab = f" [{unit}]" if unit and unit != "-" else ""
            ax.set_title(name)
            ax.set_xlabel(f"FOM{lab}")
            ax.set_ylabel(f"predicted{lab}")
        for ax in axes[len(names) :]:
            ax.set_visible(False)
        fig.suptitle(title or "Parity (test split)", fontsize=9)
    return fig


def plot_worst_curves(curves: pd.DataFrame, title: str = "") -> Figure:
    """FOM (solid) and predicted (dashed) curves of the worst test samples on one channel."""
    _need(curves, ("rank", "channel", "travel_mm", "fom", "pred"), "worst_curves")
    ranks = sorted({int(r) for r in curves["rank"]})
    if not ranks:
        raise ValueError("worst_curves table is empty")
    channel = str(curves["channel"].iloc[0])
    unit = str(curves["unit"].iloc[0]) if "unit" in curves.columns else ""
    lab = f" [{unit}]" if unit and unit != "-" else ""
    with use_style():
        fig = Figure(figsize=figure_size("a4_width"), layout="constrained")
        ax = fig.subplots()
        for k, rank in enumerate(ranks):
            sub = curves[curves["rank"] == rank].sort_values("travel_mm")
            col = design_color(k)
            lbl = f"worst #{rank + 1}"
            if "sample" in sub.columns and len(sub):
                lbl += f" (sample {int(sub['sample'].iloc[0])})"
            ax.plot(sub["travel_mm"], sub["fom"], color=col, ls=side_linestyle(0), label=lbl)
            ax.plot(sub["travel_mm"], sub["pred"], color=col, ls=side_linestyle(1))
        ax.set_xlabel("wheel travel [mm]")
        ax.set_ylabel(f"{channel}{lab}")
        ax.set_title(title or f"Worst test samples on {channel} (solid FOM, dashed network)")
        ax.legend(loc="best")
    return fig


# --------------------------------------------------------------------------- active learning


def _finite(a: NDArray[np.float64]) -> NDArray[np.float64]:
    return a[np.isfinite(a)]


def plot_al_progress(al: pd.DataFrame, title: str = "") -> Figure:
    """Best verified ``f`` and ``f_nn`` / ``f_fom`` of the infill optimum per iteration."""
    _need(al, ("k", "f_nn", "f_fom", "best_f"), "al")
    k = al["k"].to_numpy(dtype=float)
    cols = (
        ("best_f", "best FOM-verified"),
        ("f_fom", "infill optimum, FOM"),
        ("f_nn", "infill optimum, network"),
    )
    vals = np.concatenate([_finite(al[c].to_numpy(dtype=float)) for c, _ in cols])
    with use_style():
        fig = Figure(figsize=figure_size("a4_width"), layout="constrained")
        ax = fig.subplots()
        for i, (col, lab) in enumerate(cols):
            ax.plot(
                k,
                al[col].to_numpy(dtype=float),
                color=design_color(i),
                ls=side_linestyle(1 if col == "f_nn" else 0),
                marker="o",
                ms=3,
                label=lab,
            )
        if vals.size and float(np.ptp(vals)) > 1e3 * max(1.0, float(np.min(np.abs(vals)))):
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("iteration")
        ax.set_ylabel("f_total")
        ax.set_title(title or "Active learning: objective per iteration")
        ax.legend(loc="best")
    return fig


def plot_al_residual(al: pd.DataFrame, title: str = "") -> Figure:
    """Curve residual (mean RMSE / tolerance at the infill optimum) per axle and iteration."""
    _need(al, ("k",), "al")
    cols = [c for c in ("resid_front", "resid_rear") if c in al.columns]
    with use_style():
        fig = Figure(figsize=figure_size("a4_width"), layout="constrained")
        ax = fig.subplots()
        for i, c in enumerate(cols):
            v = al[c].to_numpy(dtype=float)
            if np.isfinite(v).any():
                ax.plot(al["k"], v, color=design_color(i), marker="o", ms=3, label=c[6:])
        ax.axhline(1.0, color="#333333", lw=1.0, ls="--", label="tolerance")
        allv = [al[c].to_numpy(dtype=float) for c in cols]
        top = _finite(np.concatenate(allv)) if allv else np.zeros(0)
        if top.size and float(np.max(top)) > 20.0:
            ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("RMSE / tolerance (before fine-tuning)")
        ax.set_title(title or "Active learning: network residual per iteration")
        ax.legend(loc="best")
    return fig


def plot_al_radius(al: pd.DataFrame, title: str = "") -> Figure:
    """Trust-region radius per iteration."""
    _need(al, ("k", "radius"), "al")
    with use_style():
        fig = Figure(figsize=figure_size("a4_width"), layout="constrained")
        ax = fig.subplots()
        ax.step(al["k"], al["radius"], where="mid", color=design_color(0), marker="o", ms=3)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("trust-region radius (unit cube)")
        ax.set_title(title or "Active learning: trust region")
    return fig


def write_surrogate_figures(
    fig_dir: str | Path,
    *,
    al: pd.DataFrame | None = None,
    validation: pd.DataFrame | None = None,
    parity: pd.DataFrame | None = None,
    curves: pd.DataFrame | None = None,
    axle: str = "",
    dpi: int = 200,
) -> None:
    """Save the surrogate PNG figures of the tables given, into ``fig_dir``.

    ``al`` (the active-learning table, skipped if empty) gives ``al_progress``,
    ``al_residual`` and ``al_radius``. ``validation`` gives ``rmse_ratio`` (titled with
    ``axle``); ``parity`` and ``curves`` give ``parity`` and ``worst_curves`` when non-empty.
    """
    fdir = Path(fig_dir)
    if al is not None and len(al):
        save_figure(plot_al_progress(al), fdir / "al_progress", ("png",), dpi)
        save_figure(plot_al_residual(al), fdir / "al_residual", ("png",), dpi)
        save_figure(plot_al_radius(al), fdir / "al_radius", ("png",), dpi)
    if validation is not None:
        title = f"{axle}: test error per channel"
        save_figure(plot_rmse_ratio(validation, title), fdir / "rmse_ratio", ("png",), dpi)
    if parity is not None and len(parity):
        title = f"{axle}: parity (test split)"
        save_figure(plot_parity(parity, title=title), fdir / "parity", ("png",), dpi)
    if curves is not None and len(curves):
        save_figure(plot_worst_curves(curves), fdir / "worst_curves", ("png",), dpi)
