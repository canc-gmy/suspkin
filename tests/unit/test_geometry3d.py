"""Tests for the static 3D geometry plot."""

import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from suspkin.geometry import load_vehicle
from suspkin.plotting import plot_vehicle_3d, wheel_spin_axis
from suspkin.plotting.geometry3d import GROUP_COLORS, _wheel_basis

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"
Side = Literal["left", "right"]


def test_plot_returns_figure_with_legend_groups() -> None:
    fig = plot_vehicle_3d(load_vehicle(FIXTURE))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.name == "3d"
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert labels == {
        "wishbones",
        "tie rod",
        "upright",
        "pushrod",
        "rocker",
        "coilover",
        "ARB",
        "wheel",
    }
    assert ax.get_title() == "minimal_fs"


def test_plot_with_labels() -> None:
    fig = plot_vehicle_3d(load_vehicle(FIXTURE), show_labels=True)
    assert len(fig.axes[0].texts) > 0


def test_spin_axis_zero() -> None:
    np.testing.assert_allclose(wheel_spin_axis("left", 0.0, 0.0), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(wheel_spin_axis("right", 0.0, 0.0), [0, -1, 0], atol=1e-12)


def test_spin_axis_unit_and_mirror() -> None:
    left = wheel_spin_axis("left", -0.03, 0.01)
    right = wheel_spin_axis("right", -0.03, 0.01)
    np.testing.assert_allclose(np.linalg.norm(left), 1.0)
    np.testing.assert_allclose(right, left * np.array([1, -1, 1]))


@pytest.mark.parametrize(("side", "sign"), [("left", 1.0), ("right", -1.0)])
def test_negative_camber_top_inboard(side: Side, sign: float) -> None:
    centre_y = sign * 0.6
    up, _ = _wheel_basis(wheel_spin_axis(side, -0.05, 0.0))
    top_y = centre_y + 0.2 * up[1]
    assert abs(top_y) < abs(centre_y)


@pytest.mark.parametrize(("side", "sign"), [("left", 1.0), ("right", -1.0)])
def test_toe_in_front_inboard(side: Side, sign: float) -> None:
    axis = wheel_spin_axis(side, 0.0, 0.02)
    fwd = np.cross(axis, [0.0, 0.0, 1.0])
    fwd = fwd * np.sign(fwd[0])
    assert sign * fwd[1] < 0.0


def test_arb_drawing_beam_and_torsional_bar() -> None:
    ax = plot_vehicle_3d(load_vehicle(FIXTURE)).axes[0]
    arb_lines = [ln for ln in ax.lines if ln.get_color() == GROUP_COLORS["ARB"]]
    ys = [np.asarray(ln.get_data_3d()[1]) for ln in arb_lines]
    # torsional tube: two points at the mirrored pivots (y = +-0.20)
    assert any(len(y) == 2 and np.allclose(y, [0.20, -0.20]) for y in ys)
    # beam: end - centre - end through y = 0
    assert any(len(y) == 3 and np.allclose(y, [0.2342, 0.0, -0.2342]) for y in ys)
