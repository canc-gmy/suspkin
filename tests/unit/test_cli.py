"""Tests for the CLI."""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
from typer.testing import CliRunner

from suspkin.cli import app

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"
PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47])
runner = CliRunner()


def test_show_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "v.png"
    res = runner.invoke(app, ["show", str(FIXTURE), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.read_bytes()[:4] == PNG_MAGIC
    assert str(out) in res.output


def test_show_invalid_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("vehicle: [unclosed\n")
    res = runner.invoke(app, ["show", str(bad)])
    assert res.exit_code == 1
    assert "Traceback" not in res.output
    assert res.output.strip() != ""


def test_sweep_travel_writes_csv(tmp_path: Path) -> None:
    import pandas as pd

    out = tmp_path / "r.csv"
    res = runner.invoke(
        app,
        ["sweep", str(FIXTURE), "--corner", "RL", "--travel", "0.0", "0.02", "--steps", "7",
         "--out", str(out)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert str(out) in res.output
    df = pd.read_csv(out)
    assert len(df) == 7
    for col in ("corner", "travel", "rack", "theta_uca", "rocker_angle", "coilover_length",
                "toggle_margin", "max_residual", "spin_x", "wheel_center_z",
                "contact_patch_z", "coilover_in_range"):  # fmt: skip
        assert col in df.columns
    assert set(df["corner"]) == {"RL"}


def test_sweep_rack_front(tmp_path: Path) -> None:
    import pandas as pd

    out = tmp_path / "r.csv"
    res = runner.invoke(
        app,
        ["sweep", str(FIXTURE), "--corner", "FL", "--rack-sweep", "-0.01", "0.01", "--steps", "5",
         "--out", str(out)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert len(pd.read_csv(out)) == 5


def test_sweep_requires_exactly_one_mode(tmp_path: Path) -> None:
    out = tmp_path / "r.csv"
    res = runner.invoke(app, ["sweep", str(FIXTURE), "--out", str(out)])
    assert res.exit_code == 2
    res = runner.invoke(
        app,
        ["sweep", str(FIXTURE), "--travel", "-0.01", "0.01", "--rack-sweep", "-0.01", "0.01",
         "--out", str(out)],
    )  # fmt: skip
    assert res.exit_code == 2


def test_sweep_rear_rack_fails_cleanly(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        ["sweep", str(FIXTURE), "--corner", "RL", "--rack-sweep", "-0.01", "0.01",
         "--out", str(tmp_path / "r.csv")],
    )  # fmt: skip
    assert res.exit_code == 1
    assert "Traceback" not in res.output


def test_sweep_metrics_columns(tmp_path: Path) -> None:
    import pandas as pd

    out = tmp_path / "m.csv"
    res = runner.invoke(
        app,
        ["sweep", str(FIXTURE), "--corner", "FL", "--travel", "-0.01", "0.01", "--steps", "5",
         "--metrics", "--out", str(out)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    df = pd.read_csv(out)
    for col in ("camber", "caster", "rc_height", "camber_gain", "bump_steer", "wheel_center_z"):
        assert col in df.columns


def test_summary_output_and_json(tmp_path: Path) -> None:
    import json

    out = tmp_path / "s.json"
    res = runner.invoke(app, ["summary", str(FIXTURE), "--json", str(out)])
    assert res.exit_code == 0, res.output
    for word in ("caster", "kpi", "rc_height", "camber_gain", "FL", "RL", "deg/mm"):
        assert word in res.output
    for word in ("pushrod_angle", "mr_spring", "wheel_rate", "ride_rate", "spring_force", "N/mm"):
        assert word in res.output
    data = json.loads(out.read_text())
    assert set(data) == {"FL", "RL"}
    assert abs(data["FL"]["camber"] - (-0.03)) < 1e-6  # SI (rad) in JSON


def test_vehicle_summary_and_json(tmp_path: Path) -> None:
    import json

    out = tmp_path / "v.json"
    res = runner.invoke(app, ["vehicle", str(FIXTURE), "--json", str(out)])
    assert res.exit_code == 0, res.output
    assert "roll_gradient" in res.output and "deg/g" in res.output
    assert "front.roll_stiffness" in res.output and "N*m/deg" in res.output
    data = json.loads(out.read_text())
    assert data["wheelbase"] > 0 and "front" in data


def test_vsweep_roll_degrees(tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd

    out = tmp_path / "vs.csv"
    res = runner.invoke(
        app,
        ["vsweep", str(FIXTURE), "--roll", "-1", "1", "--steps", "5", "--base-rack", "0.002",
         "--out", str(out)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    df = pd.read_csv(out)
    assert len(df) == 5
    np.testing.assert_allclose(df["roll"].iloc[[0, -1]], np.radians([-1, 1]))
    assert (df["rack"] == 0.002).all()
    assert "FL_camber_ground" in df.columns and "rc_front_z" in df.columns


def test_vsweep_requires_one_variable(tmp_path: Path) -> None:
    res = runner.invoke(app, ["vsweep", str(FIXTURE), "--out", str(tmp_path / "x.csv")])
    assert res.exit_code == 2


def test_vsweep_pivot_option(tmp_path: Path) -> None:
    import pandas as pd

    out_a, out_b = tmp_path / "a.csv", tmp_path / "b.csv"
    base = ["vsweep", str(FIXTURE), "--pitch", "0", "1", "--steps", "3"]
    assert runner.invoke(app, [*base, "--out", str(out_a)]).exit_code == 0
    res = runner.invoke(app, [*base, "--pivot", "0", "0", "0", "--out", str(out_b)])
    assert res.exit_code == 0, res.output
    a, b = pd.read_csv(out_a), pd.read_csv(out_b)
    assert abs(a["FL_travel"].iloc[-1] - b["FL_travel"].iloc[-1]) > 1e-4


def test_show_roll_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "r.png"
    args = ["show", str(FIXTURE), "--roll", "2", "--heave", "5", "--out", str(out)]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    assert out.read_bytes()[:4] == PNG_MAGIC


def test_show_html(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    out = tmp_path / "v.html"
    res = runner.invoke(app, ["show", str(FIXTURE), "--roll", "1", "--html", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "plotly" in out.read_text(encoding="utf-8")


def test_animate_roll_gif(tmp_path: Path) -> None:
    out = tmp_path / "a.gif"
    res = runner.invoke(
        app, ["animate", str(FIXTURE), "--roll", "-1", "1", "--steps", "3", "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert out.read_bytes()[:3] == b"GIF"


def test_animate_requires_one_variable(tmp_path: Path) -> None:
    res = runner.invoke(app, ["animate", str(FIXTURE), "--out", str(tmp_path / "a.gif")])
    assert res.exit_code == 2
