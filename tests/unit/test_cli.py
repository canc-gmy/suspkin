"""Tests for the CLI."""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from typer.testing import CliRunner

from suspkin.cli import app

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"
runner = CliRunner()


def test_show_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "v.png"
    res = runner.invoke(app, ["show", str(FIXTURE), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.read_bytes()[:4] == b"\x89PNG"
    assert str(out) in res.output


def test_show_invalid_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("vehicle: [unclosed\n")
    res = runner.invoke(app, ["show", str(bad)])
    assert res.exit_code == 1
    assert "Traceback" not in res.output
    assert res.output.strip() != ""
