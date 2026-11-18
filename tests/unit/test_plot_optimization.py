"""Tests for the optimization plots (synthetic history, Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from suspkin.plotting.optimization import (
    best_so_far,
    knee_index,
    pareto_mask,
    plot_constraint_margins,
    plot_convergence,
    plot_parallel_coordinates,
    plot_pareto,
)


def make_history(n: int = 80, n_obj: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {"eval_id": np.arange(n)}
    for k in range(3):
        data[f"x_v{k}"] = rng.random(n)
    f = rng.random((n, n_obj)) * 10.0
    for k in range(n_obj):
        data[f"f_o{k}"] = f[:, k]
    data["f_total"] = f.sum(axis=1)
    g = rng.normal(0.0, 1.0, (n, 2))
    data["g_c0"] = g[:, 0]
    data["g_c1"] = g[:, 1]
    data["max_g"] = g.max(axis=1)
    data["feasible"] = data["max_g"] <= 0.0  # type: ignore[operator]
    data["status"] = "ok"
    return pd.DataFrame(data)


def brute_force_mask(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    out = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                out[i] = False
    return out


@pytest.mark.parametrize("m", [2, 3, 4])
def test_pareto_mask_matches_brute_force(m: int) -> None:
    rng = np.random.default_rng(m)
    pts = rng.random((60, m))
    assert np.array_equal(pareto_mask(pts), brute_force_mask(pts))


def test_pareto_mask_keeps_duplicates() -> None:
    pts = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    assert pareto_mask(pts).tolist() == [True, True, False]


def test_knee_index_convex_front() -> None:
    t = np.linspace(0.0, 1.0, 101)
    pts = np.column_stack([t, (1.0 - t) ** 2])
    # Minimum of t^2 + (1 - t)^4 on the grid (t ~ 0.41).
    k = knee_index(pts)
    assert k == int(np.argmin(t**2 + (1.0 - t) ** 4))
    # A dominated point never wins.
    pts2 = np.vstack([pts, [[0.0, 0.0001 + 2.0]]])
    assert knee_index(pts2) == k


def test_best_so_far_monotone() -> None:
    v = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
    valid = np.array([False, True, True, True, True])
    out = best_so_far(v, valid)
    assert np.isnan(out[0])
    assert np.all(np.diff(out[1:]) <= 0)
    assert out[-1] == 1.0


def test_convergence_figure_and_monotone() -> None:
    h = make_history()
    fig = plot_convergence(h)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    step = next(ln for ln in fig.axes[0].lines if ln.get_label() == "best feasible")
    y = np.asarray(step.get_ydata(), dtype=float)
    assert y.size > 0
    assert np.all(np.diff(y) <= 1e-12)


def test_convergence_log_scale_when_wide() -> None:
    h = make_history()
    h["f_total"] = np.logspace(-2, 3, len(h))
    fig = plot_convergence(h)
    assert fig.axes[0].get_yscale() == "log"


@pytest.mark.parametrize(("n_obj", "n_axes"), [(2, 1), (3, 1), (4, 16)])
def test_pareto_axes_count(n_obj: int, n_axes: int) -> None:
    h = make_history(n=120, n_obj=n_obj)
    fig = plot_pareto(h)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == n_axes


def test_pareto_requires_two_objectives() -> None:
    with pytest.raises(ValueError):
        plot_pareto(make_history(n_obj=2), objectives=["o0"])


def test_parallel_coordinates() -> None:
    h = make_history()
    fig = plot_parallel_coordinates(h, highlight=[3, 5])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # main axes + colorbar
    thick = [ln for ln in fig.axes[0].lines if ln.get_linewidth() > 2.0]
    assert len(thick) == 2
    with pytest.raises(ValueError):
        plot_parallel_coordinates(h, highlight=10_000)


def _frame(g: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [f"c{i}" for i in range(len(g))],
            "type": "range",
            "g": g,
            "status": "ok",
        }
    )


def test_constraint_margins_clipping_annotation() -> None:
    base = _frame([-1.0, 3.0, 120.0])
    best = _frame([-2.0, -0.5, -80.0])
    fig = plot_constraint_margins(base, best)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    notes = [t.get_text() for t in fig.axes[0].texts]
    assert len(notes) == 2
    assert all("clipped" in s for s in notes)
    widths = [p.get_width() for c in fig.axes[0].containers for p in c.patches]
    assert max(abs(w) for w in widths) <= 50.0 + 1e-9


def test_constraint_margins_no_clipping() -> None:
    fig = plot_constraint_margins(_frame([-1.0, 2.0]), _frame([-3.0, -1.0]))
    assert not fig.axes[0].texts
