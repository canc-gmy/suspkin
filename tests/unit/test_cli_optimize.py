"""Tests for ``suspkin optimize``."""

import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from typer.testing import CliRunner

from suspkin.cli import app

FIX_DIR = Path(__file__).parents[1] / "fixtures"
FIX = FIX_DIR / "studies" / "tiny_optimize.yaml"
runner = CliRunner()
BASE = ["--workers", "1", "--no-report"]


def test_optimize_ok(tmp_path: Path) -> None:
    out = tmp_path / "run"
    args = ["optimize", str(FIX), "--out", str(out), "--evaluations", "25"]
    res = runner.invoke(app, [*args, "--method", "de", "--time", "5 min", *BASE])
    assert res.exit_code == 0, res.output
    assert "method de, mode weighted, 2 variables, 2 constraints, 2 objectives" in res.output
    assert "f_total: baseline" in res.output
    assert "mr_static" in res.output
    assert str(out) in res.output
    assert (out / "best.yaml").is_file()
    assert not (out / "report").exists()


def test_optimize_resume(tmp_path: Path) -> None:
    out = tmp_path / "run"
    args = ["optimize", str(FIX), "--out", str(out), "--evaluations", "10", "--method", "slsqp"]
    assert runner.invoke(app, [*args, *BASE]).exit_code == 0
    res = runner.invoke(app, [*args, *BASE, "--resume"])
    assert res.exit_code == 0, res.output


def test_optimize_with_report(tmp_path: Path) -> None:
    out = tmp_path / "run"
    args = ["optimize", str(FIX), "--out", str(out), "--evaluations", "10", "--method", "slsqp"]
    res = runner.invoke(app, [*args, "--workers", "1"])
    assert res.exit_code == 0, res.output
    assert (out / "report" / "report.html").is_file()


def test_optimize_study_error_exit_2(tmp_path: Path) -> None:
    res = runner.invoke(app, ["optimize", str(tmp_path / "missing.yaml"), "--no-report"])
    assert res.exit_code == 2


def test_optimize_bad_time_exit_2() -> None:
    res = runner.invoke(app, ["optimize", str(FIX), "--time", "soon", *BASE])
    assert res.exit_code == 2


def test_optimize_resume_without_history_exit_2(tmp_path: Path) -> None:
    res = runner.invoke(
        app, ["optimize", str(FIX), "--out", str(tmp_path / "x"), "--resume", *BASE]
    )
    assert res.exit_code == 2


def test_optimize_no_feasible_exit_1(tmp_path: Path) -> None:
    text = FIX.read_text(encoding="utf-8").replace("min: 0.3, max: 1.5", "min: 0.9, max: 1.5")
    text = text.replace(
        "vehicle: ../minimal_vehicle.yaml",
        f"vehicle: {(FIX_DIR / 'minimal_vehicle.yaml').as_posix()}",
    )
    study = tmp_path / "infeasible.yaml"
    study.write_text(text, encoding="utf-8")
    args = ["optimize", str(study), "--out", str(tmp_path / "r"), "--evaluations", "15"]
    res = runner.invoke(app, [*args, "--method", "de", *BASE])
    assert res.exit_code == 1, res.output


def test_optimize_workers_accepts_auto_and_rejects_junk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import suspkin.optimize.drivers as drv

    seen: list[Any] = []

    def fake(*a: Any, **k: Any) -> Any:
        seen.append(k["workers"])
        raise ValueError("stop")

    monkeypatch.setattr(drv, "run_optimization", fake)
    assert runner.invoke(app, ["optimize", str(FIX), "--workers", "auto"]).exit_code == 2
    assert runner.invoke(app, ["optimize", str(FIX), "--workers", "3"]).exit_code == 2
    assert seen == ["auto", 3]
    assert runner.invoke(app, ["optimize", str(FIX), "--workers", "x"]).exit_code == 2
    assert runner.invoke(app, ["optimize", str(FIX), "--workers", "0"]).exit_code == 2
    assert seen == ["auto", 3]
