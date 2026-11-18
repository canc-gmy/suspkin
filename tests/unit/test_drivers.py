"""Optimizer drivers and run folder on a tiny study (minimal vehicle, 2 parameters)."""

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from suspkin.optimize import problem as problem_mod
from suspkin.optimize import read_history, run_optimization
from suspkin.optimize.drivers import RunError, RunResult

FIX_DIR = Path(__file__).parents[1] / "fixtures"
FIX = FIX_DIR / "studies" / "tiny_optimize.yaml"
META_KEYS = {
    "suspkin_version",
    "git_hash",
    "seed",
    "method",
    "mode",
    "start",
    "end",
    "n_evals",
    "workers",
    "resumed",
}


def _variant(tmp_path: Path, *repl: tuple[str, str]) -> Path:
    """Copy of the tiny study with text replacements and an absolute vehicle path."""
    text = FIX.read_text(encoding="utf-8")
    text = text.replace(
        "vehicle: ../minimal_vehicle.yaml",
        f"vehicle: {(FIX_DIR / 'minimal_vehicle.yaml').as_posix()}",
    )
    for old, new in repl:
        assert old in text
        text = text.replace(old, new)
    out = tmp_path / "variant.yaml"
    out.write_text(text, encoding="utf-8")
    return out


def _run(tmp_path: Path, **kw: Any) -> RunResult:
    kw.setdefault("workers", 1)
    kw.setdefault("evaluations", 40)
    kw.setdefault("report", False)
    return run_optimization(FIX, tmp_path / "run", **kw)


def test_de_improves_and_is_feasible(tmp_path: Path) -> None:
    res = _run(tmp_path, method="de")
    assert res.feasible
    assert res.f_total <= res.baseline.f_total
    assert res.n_evals <= 40 + 8  # budget plus one population batch


def test_slsqp_improves_or_equals(tmp_path: Path) -> None:
    res = _run(tmp_path, method="slsqp")
    assert res.feasible
    assert res.f_total <= res.baseline.f_total + 1e-12


def test_trust_constr_runs(tmp_path: Path) -> None:
    res = _run(tmp_path, method="trust-constr", evaluations=30)
    assert res.feasible
    assert res.f_total <= res.baseline.f_total + 1e-12


def test_de_slsqp_respects_budget(tmp_path: Path) -> None:
    res = _run(tmp_path, method="de+slsqp", evaluations=40)
    df = read_history(res.out_dir / "history.csv")
    assert res.n_evals == len(df)
    # one population (8) or one finite-difference gradient (3) may cross the limit
    assert res.n_evals <= 40 + 8
    assert set(df["source"]) <= {"de", "slsqp", "lookup"}
    assert res.f_total <= res.baseline.f_total + 1e-12


def test_run_folder_and_meta(tmp_path: Path) -> None:
    res = _run(tmp_path, method="de+slsqp")
    out = res.out_dir
    names = ("study.yaml", "vehicle.yaml", "meta.json", "history.csv", "best.yaml", "result.json")
    for name in names:
        assert (out / name).is_file(), name
    for fig in ("convergence", "parallel", "constraint_margins"):
        assert (out / "figures" / f"{fig}.png").is_file(), fig
    assert not (out / "report").exists()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert set(meta) >= META_KEYS
    assert meta["seed"] == 1
    assert meta["method"] == "de+slsqp"
    assert meta["mode"] == "weighted"
    assert meta["workers"] == 1
    assert meta["resumed"] is False
    assert meta["n_evals"] == res.n_evals
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert set(result["best"]["g"]) == {"mr_static", "camber_band"}
    assert set(result["best"]["f"]) == {"camber_curve", "mr_target"}
    assert result["baseline"]["feasible"] is True
    assert len(result["best"]["constraints"]) == 2


def test_resume_continues_ids_and_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run(tmp_path, method="de", evaluations=20)
    h1 = read_history(first.out_dir / "history.csv")
    calls: list[int] = []
    real = problem_mod.evaluate

    def counting(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(problem_mod, "evaluate", counting)
    second = _run(tmp_path, method="de", evaluations=20, resume=True)
    h2 = read_history(second.out_dir / "history.csv")
    assert second.resumed
    assert list(h2["eval_id"]) == list(range(len(h2)))
    assert len(h2) > len(h1)
    assert len(calls) == second.n_evals == len(h2) - len(h1)
    xcols = [c for c in h2.columns if c.startswith("x_")]
    assert not h2.duplicated(subset=xcols).any()  # cached points are not re-evaluated
    assert second.f_total <= first.f_total + 1e-12


def test_resume_needs_history(tmp_path: Path) -> None:
    with pytest.raises(RunError):
        run_optimization(FIX, tmp_path / "none", resume=True, workers=1, report=False)
    with pytest.raises(RunError):
        run_optimization(FIX, None, resume=True, workers=1, report=False)


def test_pareto_mode_needs_nsga2(tmp_path: Path) -> None:
    study = _variant(tmp_path, ("mode: weighted", "mode: pareto"), ("de+slsqp", "nsga2"))
    with pytest.raises(RunError):
        run_optimization(study, tmp_path / "r", method="de", workers=1, report=False)


def test_nsga2_weighted(tmp_path: Path) -> None:
    pytest.importorskip("pymoo")
    res = _run(tmp_path, method="nsga2", evaluations=40)
    assert res.feasible
    assert res.f_total <= res.baseline.f_total + 1e-12
    assert not (res.out_dir / "pareto").exists()


def test_nsga2_pareto(tmp_path: Path) -> None:
    pytest.importorskip("pymoo")
    study = _variant(tmp_path, ("mode: weighted", "mode: pareto"), ("de+slsqp", "nsga2"))
    res = run_optimization(study, tmp_path / "run", workers=1, evaluations=60, report=False)
    assert res.mode == "pareto"
    assert res.n_pareto >= 1
    out = res.out_dir
    assert (out / "pareto.csv").is_file()
    assert len(list((out / "pareto").glob("pareto_*.yaml"))) == res.n_pareto
    assert (out / "figures" / "pareto.png").is_file()
    assert res.feasible


def test_report_written(tmp_path: Path) -> None:
    res = _run(tmp_path, method="slsqp", evaluations=15, report=True)
    assert res.report_path is not None
    assert res.report_path.is_file()
    assert (res.out_dir / "report" / "summary.json").is_file()


def test_run_folder_is_self_contained_and_resumable_from_copy(tmp_path: Path) -> None:
    from suspkin.optimize import DesignSpace, evaluate
    from suspkin.optimize.spec import load_study

    res = _run(tmp_path, method="de", evaluations=20)
    copy = res.out_dir / "study.yaml"
    assert "vehicle: vehicle.yaml" in copy.read_text(encoding="utf-8")
    s0, v0 = load_study(FIX)
    s1, v1 = load_study(copy)
    e0, e1 = evaluate(DesignSpace(s0, v0)), evaluate(DesignSpace(s1, v1))
    assert e0.f_total == pytest.approx(e1.f_total)
    assert e0.max_g == pytest.approx(e1.max_g)
    again = run_optimization(
        copy, res.out_dir, resume=True, workers=1, evaluations=10, report=False
    )
    assert again.resumed
    assert again.n_total > res.n_total
