"""Batched decode / history / cache-key paths equal the per-design ones (speed refactor)."""
# ruff: noqa: N806

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from suspkin.optimize import read_history
from suspkin.optimize.design import DesignSpace, InfeasibleDesign
from suspkin.optimize.history import cache_key, cache_keys
from suspkin.optimize.problem import Problem
from suspkin.optimize.spec import load_study

TINY = Path(__file__).parents[1] / "fixtures" / "studies" / "tiny_optimize.yaml"


def _space() -> DesignSpace:
    study, veh = load_study(TINY)
    return DesignSpace(study, veh)


@pytest.mark.parametrize("quantise", [True, False])
def test_points_batch_equals_points(quantise: bool) -> None:
    sp = _space()
    X = np.random.default_rng(3).random((7, sp.n))
    pb = sp.points_batch(X, quantise=quantise)
    for i, x in enumerate(X):
        for lab, p in sp.points(x, quantise=quantise).items():
            np.testing.assert_allclose(pb[lab][i], p, rtol=1e-12, atol=1e-15)


def test_points_batch_rejects_bad_input() -> None:
    sp = _space()
    with pytest.raises(ValueError):
        sp.points_batch(np.full((2, sp.n), 1.5))
    with pytest.raises(ValueError):
        sp.points_batch(np.zeros((2, sp.n + 1)))


def test_vehicle_from_points_equals_to_vehicle() -> None:
    sp = _space()
    X = sp.quantise_x(np.random.default_rng(4).random((5, sp.n)))
    pb = sp.points_batch(X)
    for i, x in enumerate(X):
        try:
            ref = sp.to_vehicle(x)
        except InfeasibleDesign:
            with pytest.raises(InfeasibleDesign):
                sp.vehicle_from_points({k: v[i] for k, v in pb.items()})
            continue
        got = sp.vehicle_from_points({k: v[i] for k, v in pb.items()})
        assert got.model_dump() == ref.model_dump()


def test_from_vehicle_baseline_cache_is_a_copy() -> None:
    sp = _space()
    a = sp.from_vehicle()
    a[:] = -1.0
    b = sp.from_vehicle()
    np.testing.assert_allclose(b, sp.from_vehicle(sp.baseline))


def test_cache_keys_equal_cache_key() -> None:
    X = np.random.default_rng(5).random((6, 9))
    assert cache_keys(X) == [cache_key(x) for x in X]
    assert cache_keys(X[0]) == [cache_key(X[0])]


def test_history_points_columns_match_per_design(tmp_path: Path) -> None:
    with Problem(TINY, tmp_path, workers=1) as p:
        X = np.random.default_rng(6).random((4, p.n))
        p.evaluate_batch(X)
        xm = p.space.quantise_x(X)
    df = read_history(tmp_path / "history.csv")
    p_cols = [c for c in df.columns if c.startswith("p_")]
    sp = _space()
    for i, x in enumerate(xm):
        if df.loc[i, "status"] == "infeasible":
            continue
        ref = np.concatenate(list(sp.points(x).values()))
        np.testing.assert_allclose(df.loc[i, p_cols].to_numpy(float), ref, rtol=1e-12, atol=1e-15)
