"""Fast helpers of ``optimize.evaluate`` match their original (reference) implementations."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from suspkin.optimize import DesignSpace
from suspkin.optimize.evaluate import evaluate
from suspkin.optimize.spec import load_study
from suspkin.optimize.spec.variables import default_scale
from suspkin.units import normalize_dim

ev: Any = importlib.import_module("suspkin.optimize.evaluate")
ROOT = Path(__file__).parents[2]


def _ref_g_array(v: Any, lo: Any, hi: Any, s: float) -> Any:
    g = np.full(v.shape, -np.inf)
    if hi is not None:
        g = np.maximum(g, (v - hi) / s)
    if lo is not None:
        g = np.maximum(g, (lo - v) / s)
    return g


def _ref_worst(v: Any, lo: Any, hi: Any, s: float, signed: float = 1.0) -> tuple[float, float]:
    arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
    g = signed * _ref_g_array(arr, lo, hi, s)
    i = int(np.argmax(g))
    return float(g[i]), float(arr[i])


def _ref_window(x: Any, y: Any, where: Any) -> tuple[Any, Any]:
    order = np.argsort(x)
    x, y = x[order], y[order]
    if where is None:
        return x, y
    lo, hi = where
    inner = (x > lo) & (x < hi)
    xs = np.concatenate([[lo], x[inner], [hi]])
    ys = np.concatenate([[np.interp(lo, x, y)], y[inner], [np.interp(hi, x, y)]])
    return xs, ys


_lim = st.one_of(st.none(), st.floats(-5, 5))


@settings(max_examples=200, deadline=None)
@given(
    v=st.lists(st.floats(-10, 10), min_size=1, max_size=12),
    lo=_lim,
    hi=_lim,
    s=st.floats(1e-3, 10),
    signed=st.sampled_from([1.0, -1.0]),
)
def test_worst_matches_reference(
    v: list[float], lo: float | None, hi: float | None, s: float, signed: float
) -> None:
    assert ev._worst(np.array(v), lo, hi, s, signed) == _ref_worst(np.array(v), lo, hi, s, signed)
    assert ev._worst(v[0], lo, hi, s, signed) == _ref_worst(v[0], lo, hi, s, signed)


def test_worst_non_finite_unsolved() -> None:
    for bad in (float("nan"), float("inf"), np.array([1.0, np.nan])):
        with pytest.raises(ev._Unsolved):
            ev._worst(bad, 0.0, 1.0, 1.0)


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(1, 15),
    seed=st.integers(0, 10_000),
    shuffle=st.booleans(),
    where=st.one_of(st.none(), st.tuples(st.floats(-1, 2), st.floats(-1, 2)).map(sorted)),
)
def test_window_matches_reference(n: int, seed: int, shuffle: bool, where: Any) -> None:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    if shuffle:
        x = rng.permutation(x)
    y = rng.normal(size=n)
    w = None if where is None else (where[0], where[1])
    got, ref = ev._window(x, y, w), _ref_window(x, y, w)
    np.testing.assert_array_equal(got[0], ref[0])
    np.testing.assert_array_equal(got[1], ref[1])


def test_window_ties_and_single_point() -> None:
    x, y = np.array([0.0, 0.5, 0.5, 1.0]), np.array([1.0, 2.0, 3.0, 4.0])
    for w in (None, (0.2, 0.8), (0.5, 1.0)):
        got, ref = ev._window(x, y, w), _ref_window(x, y, w)
        np.testing.assert_array_equal(got[0], ref[0])
        np.testing.assert_array_equal(got[1], ref[1])
    got, ref = ev._window(x[:1], y[:1], None), _ref_window(x[:1], y[:1], None)
    np.testing.assert_array_equal(got[1], ref[1])


@settings(max_examples=200, deadline=None)
@given(n=st.integers(2, 30), seed=st.integers(0, 10_000), off=st.floats(-1e3, 1e3))
def test_fit_line_matches_polyfit(n: int, seed: int, off: float) -> None:
    rng = np.random.default_rng(seed)
    xs = np.sort(rng.uniform(-0.05, 0.05, n)) + off * 1e-3
    ys = 3.0 * xs + rng.normal(scale=0.01, size=n)
    if np.unique(xs).size < 2:
        return
    np.testing.assert_allclose(ev._fit_line(xs, ys), np.polyfit(xs, ys, 1), rtol=1e-9, atol=1e-12)


def test_fit_line_two_points_and_degenerate() -> None:
    out = ev._fit_line(np.array([1.0, 3.0]), np.array([2.0, 8.0]))
    np.testing.assert_allclose(out, [3.0, -1.0], atol=1e-12)
    with pytest.raises(ev._Unsolved):
        ev._fit_line(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ev._Unsolved):
        ev._fit_line(np.array([1.0]), np.array([1.0]))


def test_default_scale_aliases() -> None:
    for d in ("length", "angle", "ratio", "percent", "force", "stiffness", "force/length"):
        assert default_scale(d) == default_scale(normalize_dim(d))
    assert default_scale("stiffness") == 1000.0


@pytest.mark.perf
@pytest.mark.parametrize("study_name", ["fs_reference_optimize", "fs_reference_constraints"])
def test_logic_time_per_design(study_name: str) -> None:
    study, veh = load_study(ROOT / "configs" / "studies" / f"{study_name}.yaml")
    sp = DesignSpace(study, veh)
    ctx = ev._Ctx(dict(study.sweeps), veh)
    first = evaluate(sp, ctx=ctx)  # fills the sweep memo; later calls time only the logic
    assert all(r.status != "unsolved" for r in first.results)
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(50):
            evaluate(sp, ctx=ctx)
        best = min(best, (time.perf_counter() - t) / 50)
    assert best < 0.05  # generous guard; typical 1-4 ms


def test_hull_built_once_per_region(monkeypatch: pytest.MonkeyPatch) -> None:
    regions: Any = importlib.import_module("suspkin.optimize.regions")
    study, veh = load_study(ROOT / "configs" / "studies" / "fs_reference_constraints.yaml")
    sp = DesignSpace(study, veh)
    ctx = ev._Ctx(dict(study.sweeps), veh)
    evaluate(sp, ctx=ctx)  # warm the hull cache
    calls = []
    real = regions.ConvexHull
    monkeypatch.setattr(regions, "ConvexHull", lambda v: calls.append(1) or real(v))
    for _ in range(3):
        evaluate(sp, ctx=ctx)
    assert calls == []
