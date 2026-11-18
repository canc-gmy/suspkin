"""Optimization plots (convergence, Pareto front, parallel coordinates, constraint margins).

Pure consumers of the run's ``history.csv`` table (columns ``eval_id``, ``x_*``, ``p_*``,
``f_*``, ``f_total``, ``g_*``, ``max_g``, ``feasible``, ...) and of ``Evaluation.frame()``
tables. Nothing here calls the solver or the evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from suspkin.plotting.style import design_color, figure_size, use_style

__all__ = [
    "G_CLIP",
    "best_so_far",
    "knee_index",
    "pareto_mask",
    "plot_constraint_margins",
    "plot_convergence",
    "plot_parallel_coordinates",
    "plot_pareto",
]

G_CLIP = 50.0
_GREY = "#9a9a9a"
_FRONT = "#D55E00"
_FEAS = "#0072B2"
_KNEE = "#CC79A7"


# --------------------------------------------------------------------------- helpers


def pareto_mask(objs: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Non-dominated mask of the rows of ``objs`` (minimisation, shape ``(n, m)``).

    Row ``i`` is dominated when some row is <= in every objective and < in at least one.
    Duplicated points are all kept. Callers pass feasible points only.
    """
    pts = np.asarray(objs, dtype=float)
    if pts.ndim != 2:
        raise ValueError("objs must be a 2D array (n_points, n_objectives)")
    n = pts.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        le = np.all(pts <= pts[i], axis=1)
        lt = np.any(pts < pts[i], axis=1)
        if np.any(le & lt):
            mask[i] = False
    return mask


def knee_index(objs: np.ndarray[Any, Any]) -> int:
    """Index (into ``objs``) of the knee of the non-dominated front.

    Each objective is normalised to [0, 1] over the front; the knee is the front point
    closest (Euclidean) to the origin (the ideal point).
    """
    pts = np.asarray(objs, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("objs must be a non-empty 2D array")
    front = np.flatnonzero(pareto_mask(pts))
    sub = pts[front]
    lo, hi = sub.min(axis=0), sub.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    dist = np.linalg.norm((sub - lo) / span, axis=1)
    return int(front[int(np.argmin(dist))])


def best_so_far(
    values: np.ndarray[Any, Any], valid: np.ndarray[Any, Any] | None = None
) -> np.ndarray[Any, Any]:
    """Running minimum of ``values`` over the ``valid`` entries (NaN before the first one)."""
    v = np.asarray(values, dtype=float).copy()
    if valid is not None:
        v[~np.asarray(valid, dtype=bool)] = np.inf
    v[np.isnan(v)] = np.inf
    out = np.minimum.accumulate(v)
    out[np.isinf(out)] = np.nan
    return out


def _cols(history: pd.DataFrame, prefix: str, exclude: Sequence[str] = ()) -> list[str]:
    return [c for c in history.columns if c.startswith(prefix) and c not in exclude]


def _feasible(history: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.bool_]]:
    return history["feasible"].to_numpy(dtype=bool)


def _set_scale(ax: Axes, values: np.ndarray[Any, Any]) -> None:
    """Log if all > 0 and spanning more than 2 decades, symlog if mixed signs, else linear."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return
    if np.all(v > 0):
        if v.max() / v.min() > 100.0:
            ax.set_yscale("log")
        return
    nz = np.abs(v[v != 0])
    if nz.size and np.abs(v).max() / nz.min() > 100.0:
        ax.set_yscale("symlog", linthresh=float(nz.min()))


# --------------------------------------------------------------------------- convergence


def plot_convergence(history: pd.DataFrame) -> Figure:
    """Best feasible ``f_total`` so far (step line) and best ``max_g`` so far, vs ``eval_id``."""
    h = history.sort_values("eval_id")
    eid = h["eval_id"].to_numpy(dtype=float)
    ft = h["f_total"].to_numpy(dtype=float)
    mg = h["max_g"].to_numpy(dtype=float)
    feas = _feasible(h)
    best_f = best_so_far(ft, feas)
    best_g = best_so_far(mg)
    with use_style():
        fig = Figure(figsize=figure_size("a4_width", rows=2), layout="constrained")
        ax, ag = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
        ax.scatter(
            eid[~feas], ft[~feas], s=6, marker="x", color=_GREY, alpha=0.4, label="infeasible"
        )
        ax.scatter(eid[feas], ft[feas], s=6, color=_FEAS, alpha=0.3, label="feasible")
        ok = np.isfinite(best_f)
        ax.step(eid[ok], best_f[ok], where="post", color=_FRONT, lw=1.8, label="best feasible")
        ax.set_ylabel("f_total")
        _set_scale(ax, ft[np.isfinite(ft)])
        ax.set_title("Convergence")
        ax.legend(loc="upper right")
        okg = np.isfinite(best_g)
        ag.step(eid[okg], best_g[okg], where="post", color=design_color(2))
        ag.axhline(0.0, color="#8c8c8c", lw=0.7)
        ag.set_ylabel("best max_g")
        ag.set_xlabel("evaluation")
        _set_scale(ag, mg[np.isfinite(mg)])
    return fig


# --------------------------------------------------------------------------- Pareto


def _objective_columns(history: pd.DataFrame, objectives: Sequence[str] | None) -> list[str]:
    if objectives is None:
        return _cols(history, "f_", exclude=("f_total",))
    cols = [o if o in history.columns else f"f_{o}" for o in objectives]
    missing = [c for c in cols if c not in history.columns]
    if missing:
        raise ValueError(f"objective columns not in history: {missing}")
    return cols


def plot_pareto(
    history: pd.DataFrame,
    objectives: Sequence[str] | None = None,
    front_mask: np.ndarray[Any, Any] | None = None,
) -> Figure:
    """Pareto view of the feasible points (infeasible in grey).

    2 objectives: scatter with the connected front. 3: 3D scatter. More: scatter matrix.
    ``objectives`` are history column names (with or without the ``f_`` prefix);
    ``front_mask`` is a boolean per history row (default: computed over the feasible rows).
    """
    cols = _objective_columns(history, objectives)
    if len(cols) < 2:
        raise ValueError("a Pareto plot needs at least 2 objectives")
    objs = history[cols].to_numpy(dtype=float)
    feas = _feasible(history)
    if front_mask is None:
        front = np.zeros(len(history), dtype=bool)
        idx = np.flatnonzero(feas)
        if idx.size:
            front[idx[pareto_mask(objs[idx])]] = True
    else:
        front = np.asarray(front_mask, dtype=bool) & feas
    labels = [c[2:] if c.startswith("f_") else c for c in cols]
    knee: int | None = None
    if front.any():
        fidx = np.flatnonzero(front)
        knee = int(fidx[knee_index(objs[fidx])])
    nonfront = feas & ~front

    def draw(ax: Axes, i: int, j: int) -> None:
        ax.scatter(objs[~feas, i], objs[~feas, j], s=8, color=_GREY, alpha=0.35, label="infeasible")
        ax.scatter(
            objs[nonfront, i], objs[nonfront, j], s=10, color=_FEAS, alpha=0.5, label="feasible"
        )
        if front.any():
            order = np.flatnonzero(front)[np.argsort(objs[front, i])]
            ax.plot(
                objs[order, i], objs[order, j], "-o", color=_FRONT, ms=3.5, lw=1.2, label="front"
            )
        if knee is not None:
            ax.scatter(
                [objs[knee, i]],
                [objs[knee, j]],
                marker="*",
                s=90,
                color=_KNEE,
                zorder=5,
                label="knee",
            )

    with use_style():
        n = len(cols)
        if n == 2:
            fig = Figure(figsize=figure_size("a4_width"), layout="constrained")
            ax = fig.add_subplot()
            draw(ax, 0, 1)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.legend()
            ax.set_title("Pareto front")
        elif n == 3:
            fig = Figure(figsize=figure_size("a4_width"), layout="constrained")
            ax3: Any = fig.add_subplot(projection="3d")
            ax3.scatter(objs[~feas, 0], objs[~feas, 1], objs[~feas, 2], s=8, color=_GREY, alpha=0.3)
            ax3.scatter(
                objs[nonfront, 0],
                objs[nonfront, 1],
                objs[nonfront, 2],
                s=10,
                color=_FEAS,
                alpha=0.5,
            )
            ax3.scatter(
                objs[front, 0], objs[front, 1], objs[front, 2], s=22, color=_FRONT, label="front"
            )
            if knee is not None:
                ax3.scatter(
                    [objs[knee, 0]], [objs[knee, 1]], [objs[knee, 2]], marker="*", s=90, color=_KNEE
                )
            ax3.set_xlabel(labels[0])
            ax3.set_ylabel(labels[1])
            ax3.set_zlabel(labels[2])
            ax3.set_title("Pareto front")
            ax3.legend()
        else:
            side = 1.6 * n + 0.8
            fig = Figure(figsize=(side, side), layout="constrained")
            axes = fig.subplots(n, n, squeeze=False)
            for r in range(n):
                for c in range(n):
                    a = axes[r][c]
                    if r == c:
                        vals = objs[feas, r]
                        if vals.size:
                            a.hist(vals[np.isfinite(vals)], bins=15, color=_FEAS, alpha=0.6)
                    elif r > c:
                        draw(a, c, r)
                    else:
                        a.set_axis_off()
                    if r == n - 1:
                        a.set_xlabel(labels[c])
                    if c == 0 and r != 0:
                        a.set_ylabel(labels[r])
            fig.suptitle("Pareto front (scatter matrix)", fontsize=10)
    return fig


# --------------------------------------------------------------------------- parallel coordinates


def plot_parallel_coordinates(
    history: pd.DataFrame, highlight: int | Sequence[int] | None = None
) -> Figure:
    """Design variables (``x_*``, unit cube) as parallel axes, coloured by ``f_total``.

    Feasible rows use a colour scale; infeasible rows are faint grey. ``highlight`` is one
    ``eval_id`` or a sequence of them, drawn thick on top.
    """
    xcols = _cols(history, "x_")
    if not xcols:
        raise ValueError("history has no x_ columns")
    xs = history[xcols].to_numpy(dtype=float)
    feas = _feasible(history)
    ft = history["f_total"].to_numpy(dtype=float)
    pos = np.arange(len(xcols))
    with use_style():
        fig = Figure(figsize=figure_size("a4_width"), layout="constrained")
        ax = fig.add_subplot()
        ax.grid(False)
        for k in pos:
            ax.axvline(float(k), color="#8c8c8c", lw=0.8, zorder=0.5)
        for row in xs[~feas]:
            ax.plot(pos, row, color=_GREY, alpha=0.12, lw=0.6, zorder=1)
        fvals = ft[feas & np.isfinite(ft)]
        if fvals.size:
            lo, hi = float(fvals.min()), float(fvals.max())
            norm = Normalize(lo, hi) if hi > lo else Normalize(lo - 0.5, hi + 0.5)
            sm = ScalarMappable(norm=norm, cmap="viridis_r")
            for row, val in zip(xs[feas], ft[feas], strict=True):
                if np.isfinite(val):
                    ax.plot(pos, row, color=sm.to_rgba(val), alpha=0.6, lw=0.9, zorder=2)
            fig.colorbar(sm, ax=ax, label="f_total", pad=0.02)
        if highlight is not None:
            ids = [highlight] if isinstance(highlight, int | np.integer) else list(highlight)
            for eid in ids:
                sel = np.flatnonzero(history["eval_id"].to_numpy() == eid)
                if sel.size == 0:
                    raise ValueError(f"highlight eval_id {eid} not in history")
                ax.plot(pos, xs[sel[0]], color="black", lw=2.6, zorder=4)
        ax.set_xticks(pos, [c[2:] for c in xcols], rotation=45, ha="right")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("normalised value [0, 1]")
        ax.set_title("Parallel coordinates")
    return fig


# --------------------------------------------------------------------------- constraint margins


def plot_constraint_margins(baseline: pd.DataFrame, best: pd.DataFrame) -> Figure:
    """Horizontal bars of ``g`` per constraint, baseline vs best (``g > 0`` is infeasible).

    Bars are clipped at ``g = +-G_CLIP`` and annotated "clipped" with the true value.
    """
    names = [str(n) for n in baseline["name"]]
    gb = pd.Series(baseline["g"].to_numpy(dtype=float), index=names)
    ga = pd.Series(best["g"].to_numpy(dtype=float), index=[str(n) for n in best["name"]])
    ga = ga.reindex(names)
    n = len(names)
    y = np.arange(n, dtype=float)
    h = 0.38
    with use_style():
        fig = Figure(figsize=(figure_size("a4_width")[0], 0.5 * n + 1.6), layout="constrained")
        ax = fig.add_subplot()
        ax.axvspan(0.0, G_CLIP * 1.15, color="#D55E00", alpha=0.08, lw=0)
        ax.axvline(0.0, color="black", lw=1.0)
        for series, off, lab, color in (
            (gb, -h / 2, "baseline", design_color(0)),
            (ga, h / 2, "best", design_color(1)),
        ):
            ax.barh([], [], height=h, color=color, label=lab)  # legend entry
            for k, name in enumerate(names):
                g = float(series[name])
                if math.isnan(g):
                    continue
                shown = float(np.clip(g, -G_CLIP, G_CLIP))
                ax.barh(y[k] + off, shown, height=h, color=color)
                if shown != g:
                    ax.annotate(
                        f"clipped ({g:.3g})",
                        (shown, y[k] + off),
                        xytext=(-3 if shown > 0 else 3, 0),
                        textcoords="offset points",
                        ha="right" if shown > 0 else "left",
                        va="center",
                        fontsize=6,
                        color="white",
                    )
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(-G_CLIP * 1.15, G_CLIP * 1.15)
        ax.set_xlabel("g  (<= 0 feasible, > 0 infeasible)")
        ax.set_title("Constraint margins")
        ax.legend(loc="lower right")
    return fig
