"""Tests for ``suspkin check``."""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
from typer.testing import CliRunner

from suspkin.cli import app

ROOT = Path(__file__).parents[2]
STUDY = ROOT / "configs" / "studies" / "fs_reference_constraints.yaml"
VEHICLE = ROOT / "configs" / "vehicles" / "fs_reference.yaml"
PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47])
runner = CliRunner()


def _variant(tmp_path: Path, old: str, new: str) -> Path:
    text = STUDY.read_text(encoding="utf-8")
    assert old in text
    text = text.replace(old, new).replace(
        "vehicle: ../vehicles/fs_reference.yaml", f'vehicle: "{VEHICLE.as_posix()}"'
    )
    out = tmp_path / "study.yaml"
    out.write_text(text, encoding="utf-8")
    return out


def test_check_pass() -> None:
    res = runner.invoke(app, ["check", str(STUDY), "--strict"])
    assert res.exit_code == 0, res.output
    assert "20/20 constraints pass, feasible: yes" in res.output
    assert "PASS" in res.output
    assert "FAIL" not in res.output


def test_check_strict_fails_on_tightened_study(tmp_path: Path) -> None:
    study = _variant(tmp_path, "max: 1.2 deg/g", "max: 0.3 deg/g")
    ok = runner.invoke(app, ["check", str(study)])
    assert ok.exit_code == 0, ok.output
    assert "FAIL" in ok.output
    assert "feasible: no" in ok.output
    res = runner.invoke(app, ["check", str(study), "--strict"])
    assert res.exit_code == 1


def test_check_bad_unit_exits_2(tmp_path: Path) -> None:
    study = _variant(tmp_path, "min: 200 mm", "min: 200 furlong")
    res = runner.invoke(app, ["check", str(study)])
    assert res.exit_code == 2
    assert "furlong" in res.output


def test_check_missing_file_exits_2(tmp_path: Path) -> None:
    res = runner.invoke(app, ["check", str(tmp_path / "nope.yaml")])
    assert res.exit_code == 2


def test_check_writes_csv_and_plot(tmp_path: Path) -> None:
    csv, png = tmp_path / "out" / "c.csv", tmp_path / "out" / "s.png"
    res = runner.invoke(app, ["check", str(STUDY), "--csv", str(csv), "--plot", str(png)])
    assert res.exit_code == 0, res.output
    assert png.read_bytes()[:4] == PNG_MAGIC
    df = pd.read_csv(csv)
    assert len(df) == 20
    assert set(df["status"]) == {"pass"}
    assert {"name", "value", "g", "dim"} <= set(df.columns)
