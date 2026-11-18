"""Problem: caching, budgets, resume, finite-difference Jacobian, process pool."""
# ruff: noqa: N806

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from suspkin.optimize import BudgetExhausted, DesignSpace, Problem, read_history
from suspkin.optimize import problem as problem_mod
from suspkin.optimize.design import InfeasibleDesign
from suspkin.optimize.evaluate import F_FAIL, G_FAIL
from suspkin.optimize.history import history_columns
from suspkin.optimize.spec import load_study

FIX = Path(__file__).parents[1] / "fixtures" / "studies" / "objectives_study.yaml"
X1 = np.array([0.3, 0.5, 0.7])
X2 = np.array([0.6, 0.2, 0.4])
X3 = np.array([0.1, 0.9, 0.5])


def _prob(tmp_path: Path, **kw: Any) -> Problem:
    kw.setdefault("workers", 1)
    kw.setdefault("evaluations", 1000)
    return Problem(FIX, tmp_path, **kw)


def _count_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = problem_mod.evaluate

    def counting(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(problem_mod, "evaluate", counting)
    return calls


def test_columns_and_history(tmp_path: Path) -> None:
    with _prob(tmp_path) as p:
        p.evaluate_one(X1, "init")
        p.evaluate_one(X2, "de")
    study, vehicle = load_study(FIX)
    df = read_history(tmp_path / "history.csv")
    assert list(df.columns) == history_columns(DesignSpace(study, vehicle), study)
    assert list(df["eval_id"]) == [0, 1]
    assert list(df["source"]) == ["init", "de"]


def test_cache_avoids_reevaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_calls(monkeypatch)
    with _prob(tmp_path) as p:
        r1 = p.evaluate_one(X1)
        r2 = p.evaluate_one(X1 + 1e-14)
        assert len(calls) == 1
        assert p.n_evals == 1
        assert r1[2] == r2[2]
    assert len(read_history(tmp_path / "history.csv")) == 1


def test_batch_dedup_and_order(tmp_path: Path) -> None:
    with _prob(tmp_path) as p:
        F, G, ft, mg, fe = p.evaluate_batch(np.array([X1, X2, X1, X2]))
        assert p.n_evals == 2
        assert F.shape == (4, 6)
        assert G.shape == (4, 2)
        assert ft[0] == ft[2]
        assert ft[1] == ft[3]
        assert fe.dtype == bool
        assert mg.shape == (4,)
    df = read_history(tmp_path / "history.csv")
    assert list(df["eval_id"]) == [0, 1]
    np.testing.assert_allclose(df["x_front.tie_inner.u"].to_numpy(), [0.3, 0.6])


def test_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _prob(tmp_path) as p:
        p.evaluate_batch(np.array([X1, X2]))
    calls = _count_calls(monkeypatch)
    with _prob(tmp_path, resume=True, evaluations=2) as p:
        assert len(p.cache) == 2
        assert p.n_evals == 0
        p.evaluate_one(X2)
        assert calls == []
        p.evaluate_one(X3)
        assert p.n_evals == 1
    df = read_history(tmp_path / "history.csv")
    assert list(df["eval_id"]) == [0, 1, 2]


def test_budget_evaluations(tmp_path: Path) -> None:
    with _prob(tmp_path, evaluations=3) as p:
        # a batch crossing the limit completes
        p.evaluate_batch(np.array([X1, X2, X3, X3 * 0.9]))
        assert p.n_evals == 4
        with pytest.raises(BudgetExhausted):
            p.evaluate_one(X1 * 0.5)


def test_budget_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(problem_mod, "_now", lambda: clock[0])
    with _prob(tmp_path, evaluations=None, time=10.0) as p:
        p.evaluate_one(X1)
        clock[0] = 11.0
        assert p.elapsed == pytest.approx(11.0)
        with pytest.raises(BudgetExhausted):
            p.evaluate_one(X2)


def test_fd_jacobian_vs_central_difference(tmp_path: Path) -> None:
    w = np.array([2.0, 1.0, 1.0, 1.0, 0.5, 1.0])  # objective weights of the fixture
    with _prob(tmp_path) as p:
        grad, g_jac, f_jac, (_f0, _g0, ft0) = p.fd_jacobian(X1, step=1e-4)
        assert grad.shape == (3,)
        assert g_jac.shape == (2, 3)
        assert f_jac.shape == (6, 3)
        assert p.n_evals == 4
        h = 1e-4
        for i in range(3):
            xp, xm = X1.copy(), X1.copy()
            xp[i] += h
            xm[i] -= h
            cd = (p.evaluate_one(xp)[2] - p.evaluate_one(xm)[2]) / (2 * h)
            assert grad[i] == pytest.approx(cd, rel=5e-2, abs=1e-3)
        assert ft0 == pytest.approx(p.evaluate_one(X1)[2])
        np.testing.assert_allclose(w @ f_jac, grad, rtol=1e-9, atol=1e-9)


def test_fd_backward_step_at_bound(tmp_path: Path) -> None:
    x = np.array([0.5, 1.0, 0.5])
    with _prob(tmp_path) as p:
        p.fd_jacobian(x, step=1e-3)
    df = read_history(tmp_path / "history.csv")
    v = df["x_front.tie_inner.v"].to_numpy()
    assert v.max() <= 1.0
    assert v.min() == pytest.approx(1.0 - 1e-3)


def test_infeasible_design(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _prob(tmp_path) as p:

        def boom(x: Any) -> Any:
            raise InfeasibleDesign("broken")

        monkeypatch.setattr(p.space, "to_vehicle", boom)
        F, G, _ft, _mg, fe = p.evaluate_one(X1)
        assert not fe
        assert np.all(G == G_FAIL)
        assert np.all(F == F_FAIL)
    df = read_history(tmp_path / "history.csv")
    assert df.loc[0, "status"] == "infeasible"
    assert np.isnan(df.loc[0, "p_front.tie_inner_x"])
    assert df.loc[0, "g_mr_static"] == G_FAIL


@pytest.mark.skipif(os.environ.get("SUSPKIN_SKIP_POOL") == "1", reason="pool test disabled")
def test_pool_matches_serial(tmp_path: Path) -> None:
    X = np.array([X1, X2, X3])
    with _prob(tmp_path / "a", workers=1) as p1:
        r1 = p1.evaluate_batch(X)
    with _prob(tmp_path / "b", workers=2) as p2:
        r2 = p2.evaluate_batch(X)
        assert p2.n_evals == 3
    for a, b in zip(r1, r2, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12)
    df = read_history(tmp_path / "b" / "history.csv")
    assert list(df["eval_id"]) == [0, 1, 2]
