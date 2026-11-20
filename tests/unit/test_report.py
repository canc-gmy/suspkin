"""Standard studies and report builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.report import StudyResults, build_report, run_standard_studies
from suspkin.report.builder import FIGURE_TITLES
from tests.unit.helpers import FIXTURE


@pytest.fixture(scope="module")
def results() -> StudyResults:
    return run_standard_studies(load_vehicle(FIXTURE), steps=5)


def test_studies_frames(results: StudyResults) -> None:
    for df in (
        results.front_corner,
        results.rear_corner,
        results.heave,
        results.roll,
        results.pitch,
    ):
        assert len(df) == 5
    for col in ("coilover_length", "toggle_margin", "camber", "wheel_rate", "ride_rate"):
        assert col in results.front_corner.columns
    assert "FL_camber_ground" in results.roll.columns
    assert "rc_front_z" in results.heave.columns
    assert results.summary.wheelbase > 0
    assert set(results.tables()) >= {"front_corner", "rear_corner", "heave", "roll", "pitch"}
    with pytest.raises(ValueError):
        run_standard_studies(load_vehicle(FIXTURE), steps=2)


def test_build_report(results: StudyResults, tmp_path: Path) -> None:
    html_path = build_report(
        [("A", results), ("B", results)], tmp_path / "rep", formats=("png", "svg"), include_3d=False
    )
    out = tmp_path / "rep"
    text = html_path.read_text(encoding="utf-8")
    assert html_path == out / "report.html"
    assert "data:image/png;base64," in text
    assert "https://" not in text and "src='figures" not in text
    for key in ("corner_alignment", "corner_position", "corner_centres", "actuation"):
        assert f"{FIGURE_TITLES[key]} (front corner)" in text
        assert f"{FIGURE_TITLES[key]} (rear corner)" in text
    for key in ("roll", "heave_pitch"):
        assert FIGURE_TITLES[key] in text
    assert "Summary table: A" in text and "Summary table: B" in text
    assert "<th>A</th><th>B</th>" in text
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(data) == {"A", "B"} and data["A"]["wheelbase"] > 0
    assert (out / "data" / "A_front_corner.csv").stat().st_size > 0
    assert (out / "data" / "B_roll.csv").exists()
    assert (out / "figures" / "roll.png").exists() and (out / "figures" / "roll.svg").exists()
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "figures/roll.png" in md


def test_build_report_validation(results: StudyResults, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_report([], tmp_path)
    with pytest.raises(ValueError):
        build_report([("A", results), ("A", results)], tmp_path)


def test_build_report_3d(results: StudyResults, tmp_path: Path) -> None:
    html_path = build_report([("A", results)], tmp_path / "r3", formats=("png",))
    assert (tmp_path / "r3" / "figures" / "geometry_3d.png").exists()
    assert (tmp_path / "r3" / "figures" / "roll_sweep.gif").read_bytes()[:3] == b"GIF"
    assert "data:image/gif;base64," in html_path.read_text(encoding="utf-8")


def test_format_value_no_negative_zero() -> None:
    from suspkin.plotting.units import format_value

    assert format_value("{:.1f}", -0.0) == "0.0"
    assert format_value("{:.1f}", -0.04) == "0.0"
    assert format_value("{:.1f}", -1.5) == "-1.5"
    assert format_value("{:12.4f}", -0.0).strip() == "0.0000"


def test_steering_per_design_wheelbase(results: StudyResults) -> None:
    from suspkin.plotting.figures import fig_steering

    a = results.rack
    b = a.copy()
    b["wheelbase"] = a["wheelbase"] * 1.2
    b["steer_c_factor"] = a["steer_c_factor"] * 2.0
    fig = fig_steering([("A", a), ("B", b)])
    titles = [ax.get_title() for ax in fig.axes]
    assert "Ackermann" in titles and "Steering ratio" in titles
    ack = next(ax for ax in fig.axes if ax.get_title() == "Ackermann")
    rat = next(ax for ax in fig.axes if ax.get_title() == "Steering ratio")
    ya = [ln.get_ydata() for ln in ack.lines if ln.get_gid() != "zero"]
    assert not np.allclose(ya[0], ya[1], equal_nan=True)
    ra = [ln.get_ydata() for ln in rat.lines if ln.get_gid() != "zero"]
    np.testing.assert_allclose(ra[1], 0.5 * ra[0], equal_nan=True)
