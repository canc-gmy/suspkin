"""Tests for the static 3D geometry plot."""

import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics import solve_vehicle
from suspkin.plotting import plot_vehicle_3d, wheel_spin_axis
from suspkin.plotting.geometry3d import GROUP_COLORS, _wheel_basis, vehicle_segments

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
    # beam: end - centre and centre - end, meeting on y = 0
    assert any(np.allclose(y, [0.2342, 0.0]) for y in ys)
    assert any(np.allclose(y, [-0.2342, 0.0]) for y in ys)


def _has_segment(segs: list[tuple[np.ndarray, np.ndarray]], a: np.ndarray, b: np.ndarray) -> bool:
    return any(np.allclose(p, a) and np.allclose(q, b) for p, q in segs)


def test_static_segments_match_hardpoints() -> None:
    veh = load_vehicle(FIXTURE)
    scene = vehicle_segments(veh)
    for c in build_corners(veh).values():
        p = c.points
        assert _has_segment(scene.segments["wishbones"], p["uca_front"], p["uca_outer"])
        assert _has_segment(scene.segments["tie rod"], p["tie_inner"], p["tie_outer"])
        assert _has_segment(scene.segments["pushrod"], p["pushrod_outer"], p["pushrod_inner"])
        assert _has_segment(scene.segments["coilover"], p["rocker_spring"], p["spring_chassis"])
    assert len(scene.wheels) == 4
    np.testing.assert_allclose(scene.ground_normal, [0, 0, 1])


def test_zero_pose_state_equals_static() -> None:
    veh = load_vehicle(FIXTURE)
    a = vehicle_segments(veh)
    b = vehicle_segments(veh, solve_vehicle(veh))
    for group, segs in a.segments.items():
        for (p0, p1), (q0, q1) in zip(segs, b.segments[group], strict=True):
            np.testing.assert_allclose(p0, q0, atol=1e-6)
            np.testing.assert_allclose(p1, q1, atol=1e-6)
    for ca, cb in zip(a.wheels, b.wheels, strict=True):
        np.testing.assert_allclose(ca, cb, atol=1e-6)


def test_rolled_pose_moves_wheels_in_body_frame() -> None:
    veh = load_vehicle(FIXTURE)
    corners = build_corners(veh)
    st = solve_vehicle(veh, roll=np.radians(2.0))
    # body frame fixed, ground moves: the left (lifted) corner droops, the right bumps
    assert st.corners["FL"].points["wheel_center"][2] < corners["FL"].points["wheel_center"][2]
    assert st.corners["FR"].points["wheel_center"][2] > corners["FR"].points["wheel_center"][2]
    scene = vehicle_segments(veh, st)
    assert abs(scene.ground_normal[1]) > 1e-3
    # chassis points do not move
    p0, p1 = corners["FL"].points, st.corners["FL"].points
    np.testing.assert_allclose(p1["uca_front"], p0["uca_front"])
    # the drawn wheel circle is centred on the moved wheel centre
    ctr = scene.wheels[0][:-1].mean(axis=0)
    np.testing.assert_allclose(ctr, p1["wheel_center"], atol=1e-3)


def test_plot_with_state_returns_figure() -> None:
    veh = load_vehicle(FIXTURE)
    fig = plot_vehicle_3d(veh, state=solve_vehicle(veh, heave=0.01, roll=0.02))
    assert isinstance(fig, Figure)
