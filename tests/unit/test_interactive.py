"""Tests for the plotly interactive view."""

import sys
from pathlib import Path

import pytest

from suspkin.geometry import load_vehicle
from suspkin.kinematics import solve_vehicle
from suspkin.plotting.interactive import interactive_view

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"


def test_missing_plotly_gives_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(ImportError, match="viz"):
        interactive_view(load_vehicle(FIXTURE), None, tmp_path / "x.html")


def test_html_static_and_posed(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    veh = load_vehicle(FIXTURE)
    a = interactive_view(veh, None, tmp_path / "s.html")
    b = interactive_view(veh, solve_vehicle(veh, roll=0.03), tmp_path / "p.html")
    for f in (a, b):
        text = f.read_text(encoding="utf-8")
        assert "plotly" in text
        assert "cdn.plot.ly" in text


def test_html_inline_is_larger(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    veh = load_vehicle(FIXTURE)
    cdn = interactive_view(veh, None, tmp_path / "c.html")
    inline = interactive_view(veh, None, tmp_path / "i.html", inline=True)
    assert inline.stat().st_size > cdn.stat().st_size + 100_000
