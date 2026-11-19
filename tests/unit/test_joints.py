"""Tests for joint misalignment angles."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

from suspkin.geometry import build_corners, load_vehicle
from suspkin.geometry.corners import CornerId
from suspkin.geometry.vectors import axis_angle_matrix
from suspkin.kinematics.corner import CornerState, solve_corner, sweep_corner, sweep_rack
from suspkin.metrics import joints as jt
from tests.unit.helpers import FIXTURE, fixture_corners

CORNERS = fixture_corners()
IDS: tuple[CornerId, ...] = ("FL", "FR", "RL", "RR")


def test_kabsch_recovers_known_rotation() -> None:
    rng = np.random.default_rng(1)
    p0 = rng.normal(size=(5, 3))
    r = axis_angle_matrix([1.0, 2.0, -0.5], 0.7)
    p1 = p0 @ r.T + np.array([0.3, -1.0, 2.0])
    np.testing.assert_allclose(jt._kabsch(p0, p1), r, atol=1e-12)


def test_kabsch_pure_translation_is_identity() -> None:
    p0 = np.random.default_rng(2).normal(size=(4, 3))
    p1 = p0 + np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(jt._kabsch(p0, p1), np.eye(3), atol=1e-12)


@pytest.mark.parametrize("cid", IDS)
def test_static_state_all_zero(cid: CornerId) -> None:
    c = CORNERS[cid]
    st = solve_corner(c, 0.0)
    for j in jt.JOINTS:
        assert jt.joint_angle(st, st, c, j) == pytest.approx(0.0, abs=1e-9)
    df = jt.joint_angles_frame([st], c)
    assert list(df.columns) == [f"joint_{j}" for j in jt.JOINTS]
    assert float(np.abs(df.to_numpy()).max()) < 1e-9


@pytest.mark.parametrize("cid", IDS)
def test_sweep_finite_continuous_growing(cid: CornerId) -> None:
    c = CORNERS[cid]
    travels = np.linspace(-0.025, 0.025, 11)
    states = sweep_corner(c, travels)
    df = jt.joint_angles_frame(states, c)
    arr = df.to_numpy()
    assert np.all(np.isfinite(arr))
    assert np.max(np.abs(np.diff(arr, axis=0))) < 0.05
    mid = 5
    for col in df.columns:
        v = df[col].to_numpy()
        assert v[mid] == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.isfinite(v))
        assert np.max(np.abs(np.diff(v))) < 0.05
    t = df["joint_tie_outer"].to_numpy()
    assert np.all(np.diff(t[mid:]) > 0.0)
    assert np.all(np.diff(t[: mid + 1]) < 0.0)


def test_ball_joint_synthetic_five_degrees() -> None:
    c = CORNERS["FL"]
    st = solve_corner(c, 0.0)
    ang = np.deg2rad(5.0)
    rx = axis_angle_matrix([1.0, 0.0, 0.0], ang)
    ubj, lbj = np.asarray(st.points["uca_outer"]), np.asarray(st.points["lca_outer"])
    pts = {k: np.array(v) for k, v in st.points.items()}
    for n in ("uca_outer", "lca_outer", "tie_outer", "wheel_center"):
        pts[n] = ubj + rx @ (np.asarray(st.points[n]) - ubj)
    moved: CornerState = replace(st, points=pts)
    rots = jt.body_rotations(moved, st, c)
    np.testing.assert_allclose(rots["upright"], rx, atol=1e-9)
    np.testing.assert_allclose(rots["uca"], np.eye(3), atol=1e-9)
    a0 = (ubj - lbj) / np.linalg.norm(ubj - lbj)
    expect = float(np.arccos(np.clip((rx @ a0) @ a0, -1.0, 1.0)))
    assert jt.joint_angle(moved, st, c, "uca_outer") == pytest.approx(expect, abs=1e-9)
    # Stud axis along y (perpendicular to the x rotation axis): exactly 5 deg.
    got = jt.joint_angle(moved, st, c, "uca_outer", axis=[0.0, 1.0, 0.0])
    assert got == pytest.approx(ang, abs=1e-9)
    # Stud axis along the rotation axis: unaffected.
    assert jt.joint_angle(moved, st, c, "uca_outer", axis=[1.0, 0.0, 0.0]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_rod_end_axis_override_and_unknown_joint() -> None:
    c = CORNERS["FL"]
    st = solve_corner(c, 0.0)
    mv = solve_corner(c, 0.02)
    assert jt.joint_angle(mv, st, c, "tie_inner") > 0.0
    assert jt.joint_angle(mv, st, c, "tie_inner", axis=[0.0, 0.0, 1.0]) >= 0.0
    with pytest.raises(ValueError):
        jt.joint_angle(mv, st, c, "nope")


def _rigid_moved(st: CornerState, names: tuple[str, ...], origin: np.ndarray, rot: np.ndarray):
    pts = {k: np.array(v) for k, v in st.points.items()}
    for n in names:
        pts[n] = origin + rot @ (np.asarray(st.points[n]) - origin)
    return replace(st, points=pts)


_UPRIGHT = ("uca_outer", "lca_outer", "tie_outer", "wheel_center")


def test_rod_end_analytical_tie_outer() -> None:
    c = CORNERS["FL"]
    st = solve_corner(c, 0.0)
    to = np.asarray(st.points["tie_outer"])
    d0 = to - np.asarray(st.points["tie_inner"])
    ax = np.cross(d0, [0.0, 0.0, 1.0])
    ang = np.deg2rad(5.0)
    moved = _rigid_moved(st, _UPRIGHT, to, axis_angle_matrix(ax, ang))
    assert jt.joint_angle(moved, st, c, "tie_outer") == pytest.approx(ang, abs=1e-9)


@pytest.mark.parametrize("name", ["uca_outer", "lca_outer"])
def test_spin_about_kingpin_invariant(name: str) -> None:
    c = CORNERS["FL"]
    st = solve_corner(c, 0.0)
    ubj, lbj = np.asarray(st.points["uca_outer"]), np.asarray(st.points["lca_outer"])
    moved = _rigid_moved(st, _UPRIGHT, lbj, axis_angle_matrix(ubj - lbj, 0.5))
    assert jt.joint_angle(moved, st, c, name) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(("left", "right"), [("FL", "FR"), ("RL", "RR")])
def test_left_right_symmetry(left: CornerId, right: CornerId) -> None:
    travels = np.linspace(-0.02, 0.02, 9)
    a = jt.joint_angles_frame(sweep_corner(CORNERS[left], travels), CORNERS[left])
    b = jt.joint_angles_frame(sweep_corner(CORNERS[right], travels), CORNERS[right])
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-7)


@pytest.mark.parametrize("mount", ["upright", "lca", "uca"])
def test_pushrod_mount_body(mount: Literal["upright", "lca", "uca"]) -> None:
    data = load_vehicle(FIXTURE).model_dump()
    data["front"]["pushrod_mount"] = mount
    from suspkin.geometry.model import Vehicle

    c = build_corners(Vehicle.model_validate(data))["FL"]
    st = solve_corner(c, 0.0)
    mv = solve_corner(c, 0.02)
    rots = jt.body_rotations(mv, st, c)
    po0, po1 = np.asarray(st.points["pushrod_outer"]), np.asarray(mv.points["pushrod_outer"])
    if mount == "upright":
        o0, o1 = np.asarray(st.points["lca_outer"]), np.asarray(mv.points["lca_outer"])
    else:
        o0 = o1 = np.asarray(st.points[f"{mount}_front"])
    np.testing.assert_allclose(po1 - o1, rots[mount] @ (po0 - o0), atol=1e-9)
    assert jt.joint_angle(st, st, c, "pushrod_outer") == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(jt.joint_angle(mv, st, c, "pushrod_outer"))


def test_pure_rack_tie_inner_small() -> None:
    c = CORNERS["FL"]
    if c.axle_cfg.steering is None:
        pytest.skip("fixture front axle has no steering")
    st = solve_corner(c, 0.0)
    states = sweep_rack(c, [0.0, 0.003, 0.006])
    df = jt.joint_angles_frame(states, c)
    assert df["joint_tie_inner"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < df["joint_tie_inner"].iloc[-1] < 0.05
    assert st.rack == 0.0


def test_kabsch_proper_rotation_for_planar_reflection() -> None:
    p0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.5, 0.0]])
    p1 = p0 * np.array([1.0, -1.0, 1.0])
    r = jt._kabsch(p0, p1)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_kabsch_degenerate_raises() -> None:
    line = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    with pytest.raises(ValueError):
        jt._kabsch(line, line + 1.0)
