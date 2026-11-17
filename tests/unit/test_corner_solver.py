"""Tests for the corner kinematic solver."""

import numpy as np
import pytest

from suspkin.geom_math import mirror_y, wheel_spin_axis
from suspkin.geometry import Vehicle, build_corners, load_vehicle
from suspkin.geometry.corners import CornerId
from suspkin.kinematics import KinematicsError, solve_corner
from tests.unit.helpers import FIXTURE, good_corners, good_vehicle, max_invariant_error

ALL: tuple[CornerId, ...] = ("FL", "FR", "RL", "RR")


def _with_mount(v: Vehicle, mount: str) -> Vehicle:
    return v.model_copy(
        update={
            "front": v.front.model_copy(update={"pushrod_mount": mount}),
            "rear": v.rear.model_copy(update={"pushrod_mount": mount}),
        }
    )


@pytest.mark.parametrize("mount", ["upright", "lca", "uca"])
@pytest.mark.parametrize("vehicle_kind", ["minimal", "good"])
def test_static_pose_reproduced(mount: str, vehicle_kind: str) -> None:
    v = load_vehicle(FIXTURE) if vehicle_kind == "minimal" else good_vehicle()
    corners = build_corners(_with_mount(v, mount))
    for cid in ALL:
        c = corners[cid]
        s = solve_corner(c, 0.0, 0.0)
        for name, p in c.points.items():
            np.testing.assert_allclose(s.points[name], p, atol=1e-9, err_msg=f"{cid}.{name}")
        assert s.max_residual < 1e-9
        assert s.theta_uca == pytest.approx(0, abs=1e-9)
        assert s.rocker_angle == pytest.approx(0, abs=1e-9)
        cfg = c.axle_cfg
        np.testing.assert_allclose(
            s.spin_axis,
            wheel_spin_axis(c.side, cfg.wheel.static_camber, cfg.wheel.static_toe),
            atol=1e-12,
        )
        assert s.points["contact_patch"][2] == pytest.approx(
            c.points["wheel_center"][2] - cfg.wheel.tyre_radius * np.cos(cfg.wheel.static_camber),
            abs=1e-6,
        )


@pytest.mark.parametrize("mount", ["upright", "lca", "uca"])
@pytest.mark.parametrize("cid", ALL)
def test_link_lengths_constant_over_travel(mount: str, cid: CornerId) -> None:
    c = build_corners(_with_mount(good_vehicle(), mount))[cid]
    for t in np.linspace(-0.03, 0.03, 13):
        s = solve_corner(c, float(t))
        assert max_invariant_error(c, s) < 1e-9
        assert s.max_residual < 1e-9


@pytest.mark.parametrize("mount", ["lca", "uca"])
def test_pushrod_follows_wishbone_rotation(mount: str) -> None:
    c = build_corners(_with_mount(good_vehicle(), mount))["FL"]
    s = solve_corner(c, 0.02)
    fixed = "lca" if mount == "lca" else "uca"
    for end in ("front", "rear"):
        a = c.points[f"{fixed}_{end}"]
        assert np.linalg.norm(s.points["pushrod_outer"] - a) == pytest.approx(
            np.linalg.norm(c.points["pushrod_outer"] - a), abs=1e-9
        )


def test_front_rack_invariants_and_motion() -> None:
    c = good_corners()["FL"]
    for rack in (-0.02, -0.01, 0.01, 0.02):
        s = solve_corner(c, 0.01, rack)
        assert max_invariant_error(c, s) < 1e-9
        np.testing.assert_allclose(
            s.points["tie_inner"], c.points["tie_inner"] + np.array([0, rack, 0]), atol=1e-12
        )


def test_rack_moves_both_sides_along_same_global_axis() -> None:
    cs = good_corners()
    a, b = solve_corner(cs["FL"], 0.0, 0.01), solve_corner(cs["FR"], 0.0, 0.01)
    assert a.points["tie_inner"][1] - cs["FL"].points["tie_inner"][1] == pytest.approx(0.01)
    assert b.points["tie_inner"][1] - cs["FR"].points["tie_inner"][1] == pytest.approx(0.01)


@pytest.mark.parametrize("travel", [-0.03, -0.012, 0.007, 0.03])
def test_left_right_symmetry(travel: float) -> None:
    cs = good_corners()
    pairs: tuple[tuple[CornerId, CornerId], ...] = (("FL", "FR"), ("RL", "RR"))
    for left, right in pairs:
        a, b = solve_corner(cs[left], travel), solve_corner(cs[right], travel)
        for name, p in a.points.items():
            np.testing.assert_allclose(b.points[name], mirror_y(p), atol=1e-9, err_msg=name)
        np.testing.assert_allclose(b.spin_axis, mirror_y(a.spin_axis), atol=1e-9)
        assert b.rocker_angle == pytest.approx(a.rocker_angle, abs=1e-9)
        assert b.theta_uca == pytest.approx(a.theta_uca, abs=1e-9)
        assert b.psi == pytest.approx(a.psi, abs=1e-9)
        assert b.coilover_length == pytest.approx(a.coilover_length, abs=1e-12)


def test_bump_raises_wheel_centre_exactly_and_shortens_coilover() -> None:
    cs = good_corners()
    for cid in ALL:
        c = cs[cid]
        s0 = solve_corner(c, 0.0)
        for t in (-0.03, 0.015, 0.03):
            s = solve_corner(c, t)
            assert s.points["wheel_center"][2] - s0.points["wheel_center"][2] == pytest.approx(
                t, abs=1e-12
            )
        assert solve_corner(c, 0.02).coilover_length < s0.coilover_length
        assert solve_corner(c, -0.02).coilover_length > s0.coilover_length


def test_contact_patch_is_lowest_wheel_point() -> None:
    c = good_corners()["FL"]
    s = solve_corner(c, 0.02)
    wc, cp, n = s.points["wheel_center"], s.points["contact_patch"], s.spin_axis
    r = c.axle_cfg.wheel.tyre_radius
    assert np.linalg.norm(cp - wc) == pytest.approx(r)
    assert (cp - wc) @ n == pytest.approx(0.0, abs=1e-12)
    e1 = np.cross(n, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    for a in np.linspace(0, 2 * np.pi, 50):
        q = wc + r * (np.cos(a) * e1 + np.sin(a) * e2)
        assert q[2] >= cp[2] - 1e-12


def test_parallelogram_has_no_camber_change_and_circular_track_change() -> None:
    v = good_vehicle()
    arm = 0.27
    hp = v.front.hardpoints.model_copy(
        update={
            "uca_front": (0.1, 0.28, 0.30),
            "uca_rear": (-0.1, 0.28, 0.30),
            "uca_outer": (0.0, 0.28 + arm, 0.30),
            "lca_front": (0.1, 0.28, 0.12),
            "lca_rear": (-0.1, 0.28, 0.12),
            "lca_outer": (0.0, 0.28 + arm, 0.12),
            "tie_inner": (0.08, 0.28, 0.15),
            "tie_outer": (0.08, 0.28 + arm, 0.15),
            "wheel_center": (0.0, 0.60, 0.2032),
        }
    )
    v2 = Vehicle.model_validate(
        v.model_copy(update={"front": v.front.model_copy(update={"hardpoints": hp})}).model_dump()
    )
    c = build_corners(v2)["FL"]
    cfg = c.axle_cfg
    spin0 = wheel_spin_axis("left", cfg.wheel.static_camber, cfg.wheel.static_toe)
    for t in np.linspace(-0.03, 0.03, 13):
        s = solve_corner(c, float(t))
        np.testing.assert_allclose(s.spin_axis, spin0, atol=1e-9)
        assert s.psi == pytest.approx(0.0, abs=1e-9)
        assert s.theta_uca == pytest.approx(s.theta_lca, abs=1e-9)
        expected_y = 0.60 + arm * (np.sqrt(1.0 - (t / arm) ** 2) - 1.0)
        assert s.points["wheel_center"][1] == pytest.approx(expected_y, abs=1e-9)
        assert s.points["wheel_center"][0] == pytest.approx(0.0, abs=1e-9)


def test_rear_rack_rejected() -> None:
    c = good_corners()["RL"]
    with pytest.raises(ValueError, match="rack"):
        solve_corner(c, 0.0, 0.005)


def test_impossible_travel_raises() -> None:
    c = good_corners()["FL"]
    with pytest.raises(KinematicsError):
        solve_corner(c, 0.5)


def test_minimal_fixture_rocker_hits_toggle_in_droop() -> None:
    # The stock minimal fixture has its pushrod almost in line with the rocker lever
    # (static toggle margin ~13 deg), so the pushrod cannot follow ~3 mm of droop.
    c = build_corners(load_vehicle(FIXTURE))["FL"]
    assert solve_corner(c, 0.0).toggle_margin < 0.3
    with pytest.raises(KinematicsError):
        solve_corner(c, -0.01)


def test_guess_gives_same_solution() -> None:
    c = good_corners()["FL"]
    a = solve_corner(c, 0.02, 0.01)
    b = solve_corner(c, 0.021, 0.01, guess=a)
    d = solve_corner(c, 0.021, 0.01)
    np.testing.assert_allclose(b.points["wheel_center"], d.points["wheel_center"], atol=1e-9)
