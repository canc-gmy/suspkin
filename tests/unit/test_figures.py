"""2D figure sets: build from reference-car tables, overlay, nan/inf, saving, style."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics.corner import sweep_corner
from suspkin.metrics.table import add_gradients, metrics_frame
from suspkin.metrics.vehicle import vehicle_summary
from suspkin.plotting.figures import (
    FIGURE_SET,
    fig_actuation,
    fig_corner_alignment,
    fig_corner_centres,
    fig_corner_position,
    fig_heave_pitch,
    fig_roll,
    fig_steering,
    fig_summary_table,
    save_figure,
)
from suspkin.plotting.style import design_color, figure_size, use_style
from suspkin.sweeps import sweep_vehicle, vehicle_frame
from tests.unit.helpers import FIXTURE

Frames = dict[str, pd.DataFrame]


@pytest.fixture(scope="module")
def vehicle():  # type: ignore[no-untyped-def]
    return load_vehicle(FIXTURE)


@pytest.fixture(scope="module")
def frames(vehicle) -> Frames:  # type: ignore[no-untyped-def]
    corners = build_corners(vehicle)
    fl = corners["FL"]
    out: Frames = {
        "corner": add_gradients(metrics_frame(sweep_corner(fl, np.linspace(-0.02, 0.02, 9)), fl))
    }
    out["roll"] = vehicle_frame(sweep_vehicle(vehicle, roll=np.linspace(-0.02, 0.02, 7)), vehicle)
    out["heave"] = vehicle_frame(sweep_vehicle(vehicle, heave=np.linspace(-0.01, 0.01, 5)), vehicle)
    out["pitch"] = vehicle_frame(sweep_vehicle(vehicle, pitch=np.linspace(-0.01, 0.01, 5)), vehicle)
    steer = vehicle.front.steering
    assert steer is not None
    lim = 0.5 * min(abs(steer.rack_travel[0]), abs(steer.rack_travel[1]))
    out["rack"] = vehicle_frame(sweep_vehicle(vehicle, rack=np.linspace(-lim, lim, 9)), vehicle)
    return out


def _data_lines(fig: Figure) -> int:
    return sum(1 for ax in fig.axes for ln in ax.lines if ln.get_gid() != "zero")


def _two(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [("A", df), ("B", df)]


CORNER_FUNCS = [fig_corner_alignment, fig_corner_position, fig_corner_centres, fig_actuation]


@pytest.mark.parametrize("func", CORNER_FUNCS)
def test_corner_figures_overlay(func, frames: Frames) -> None:  # type: ignore[no-untyped-def]
    one = func([("A", frames["corner"])])
    two = func(_two(frames["corner"]))
    assert isinstance(one, Figure)
    assert _data_lines(one) > 0
    assert _data_lines(two) == 2 * _data_lines(one)


def test_axle_validation(frames: Frames) -> None:
    with pytest.raises(ValueError):
        fig_corner_alignment([("A", frames["corner"])], axle="middle")
    with pytest.raises(ValueError):
        fig_corner_alignment([])


def test_steering(frames: Frames, vehicle) -> None:  # type: ignore[no-untyped-def]
    one = fig_steering([("A", frames["rack"])], wheelbase=vehicle.wheelbase)
    titles = [ax.get_title() for ax in one.axes]
    assert "Ackermann" in titles and "Steer gain" in titles
    two = fig_steering(_two(frames["rack"]), wheelbase=vehicle.wheelbase, c_factor=0.07)
    assert "Steering ratio" in [ax.get_title() for ax in two.axes]
    assert _data_lines(two) == 2 * _data_lines(one)
    assert "Ackermann" not in [ax.get_title() for ax in fig_steering([("A", frames["rack"])]).axes]


def test_roll_and_heave_pitch(frames: Frames) -> None:
    one = fig_roll([("A", frames["roll"])])
    assert _data_lines(fig_roll(_two(frames["roll"]))) == 2 * _data_lines(one)
    hp = fig_heave_pitch([("A", frames["heave"])], [("A", frames["pitch"])])
    assert len(hp.axes) == 4
    assert len(fig_heave_pitch([("A", frames["heave"])], []).axes) == 2
    with pytest.raises(ValueError):
        fig_heave_pitch([], [])


def test_nan_inf_do_not_crash(frames: Frames) -> None:
    df = frames["corner"].copy()
    df.loc[3, ["fvic_y", "fvic_z", "fvsa", "fv_swing_angle"]] = np.inf
    df.loc[4, ["fvic_y", "fvic_z", "rc_height", "camber"]] = np.nan
    fig = fig_corner_centres([("A", df)])
    assert any("IC at infinity" in t.get_text() for ax in fig.axes for t in ax.texts)
    fig_corner_alignment([("A", df)])
    r = frames["roll"].copy()
    r.loc[2, ["rc_front_y", "rc_front_z", "arb_front_elastic"]] = np.nan
    r.loc[3, ["rc_rear_z"]] = np.inf
    fig_roll([("A", r)])
    q = frames["rack"].copy()
    q.loc[2, ["FL_steer_angle"]] = np.nan
    fig_steering([("A", q)], wheelbase=1.6)


def test_missing_columns_skip_panels(frames: Frames) -> None:
    df = frames["corner"][["travel", "camber", "toe"]]
    fig = fig_corner_alignment([("A", df)])
    assert len(fig.axes) == 2
    with pytest.raises(ValueError):
        fig_corner_alignment([("A", frames["corner"][["travel"]])])


def _cells(fig: Figure) -> list[str]:
    return sorted(
        c.get_text().get_text()
        for ax in fig.axes
        for tb in ax.tables
        for c in tb.get_celld().values()
    )


def test_summary_table(vehicle) -> None:  # type: ignore[no-untyped-def]
    s = vehicle_summary(vehicle)
    f1 = fig_summary_table(s)
    f2 = fig_summary_table(s.to_dict())
    assert isinstance(f1, Figure) and isinstance(f2, Figure)
    texts1 = sorted(
        c.get_text().get_text()
        for ax in f1.axes
        for tb in ax.tables
        for c in tb.get_celld().values()
    )
    texts2 = sorted(
        c.get_text().get_text()
        for ax in f2.axes
        for tb in ax.tables
        for c in tb.get_celld().values()
    )
    assert texts1 == texts2 and texts1
    with pytest.raises(TypeError):
        fig_summary_table(3.0)


def test_registry() -> None:
    assert set(FIGURE_SET) == {
        "corner_alignment",
        "corner_position",
        "corner_centres",
        "actuation",
        "steering",
        "roll",
        "heave_pitch",
        "summary_table",
    }


def test_save_figure(tmp_path, frames: Frames) -> None:  # type: ignore[no-untyped-def]
    fig = fig_corner_alignment([("A", frames["corner"])])
    paths = save_figure(fig, tmp_path / "sub" / "fig", dpi=60)
    assert [p.suffix for p in paths] == [".png", ".svg"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert paths[0].read_bytes()[:4] == b"\x89PNG"


def test_style_does_not_leak() -> None:
    before = dict(mpl.rcParams)
    with use_style():
        assert mpl.rcParams["axes.grid"] is True
        assert mpl.rcParams["axes.spines.top"] is False
    assert dict(mpl.rcParams) == before
    assert design_color(0) != design_color(1)
    assert design_color(8) == design_color(0)
    w, h = figure_size("a4_width")
    assert w < 7 and h > 0
    with pytest.raises(ValueError):
        figure_size("nope")
