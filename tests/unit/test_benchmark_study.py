"""Benchmark study (all hardpoints free) and the argument handling of the benchmark script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from suspkin.optimize.spec import load_study

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "configs" / "studies" / "fs_reference_surrogate.yaml"
SCRIPT = ROOT / "scripts" / "benchmark_surrogate.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_surrogate", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_surrogate"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_tiny_study(dst: Path) -> Path:
    """Copy of the benchmark study sized for a smoke run (absolute vehicle path)."""
    raw: dict[str, Any] = yaml.safe_load(STUDY.read_text(encoding="utf-8"))
    raw["vehicle"] = str((STUDY.parent / raw["vehicle"]).resolve())
    sg = raw["surrogate"]
    sg["grid"] = {
        "travel": {"min": "-30 mm", "max": "30 mm", "points": 11},
        "rack": {"min": "-30 mm", "max": "30 mm", "points": 7},
    }
    sg["dataset"]["n_initial"] = 60
    sg["hpo"] = {"n_trials": 2, "timeout": "2 min", "parallel": "auto"}
    sg["active_learning"]["fom_evaluations"] = 40
    sg["active_learning"]["max_iterations"] = 2
    dst.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return dst


def test_study_loads_with_all_hardpoints_free() -> None:
    study, _ = load_study(STUDY)
    assert len(study.variables) == 2 * 15  # 15 points per axle (rocker_axis is a direction)
    assert study.surrogate is not None
    assert study.surrogate.dataset.n_initial == 20000
    assert study.surrogate.grid.travel.points % 2 == 1
    assert study.surrogate.grid.rack is not None and study.surrogate.grid.rack.points % 2 == 1
    assert study.optimizer is not None and study.optimizer.method == "de"


def test_tiny_study_loads(tmp_path: Path) -> None:
    study, _ = load_study(make_tiny_study(tmp_path / "tiny.yaml"))
    assert study.surrogate is not None and study.surrogate.dataset.n_initial == 60


def test_parser_defaults_and_force() -> None:
    mod = _load_script()
    a = mod.build_parser().parse_args(["--force", "al", "--force", "de_same", "--device", "cpu"])
    assert a.force == ["al", "de_same"] and a.device == "cpu" and not a.dry_run
    assert a.study == mod.DEFAULT_STUDY
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--force", "nonsense"])


def test_dry_run_prints_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    rc = mod.main(["--root", str(tmp_path / "s"), "--run", str(tmp_path / "r"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    for stage in mod.STAGES:
        assert stage in out
    assert not (tmp_path / "r").exists()  # a dry run writes nothing


def test_stage_commands_carry_budgets(tmp_path: Path) -> None:
    mod = _load_script()
    a = mod.build_parser().parse_args(
        ["--root", str(tmp_path / "s"), "--run", str(tmp_path / "r"), "--workers", "3"]
    )
    cmds = mod.stage_commands(a, n_al=120, n_total=45000)
    assert "120" in cmds["de_same"] and "45000" in cmds["de_total"]
    assert "--surrogate" in cmds["al"] and "3" in cmds["sample"]
