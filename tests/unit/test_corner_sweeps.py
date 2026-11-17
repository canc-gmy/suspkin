"""Tests for corner sweeps and the result table."""

import time

import numpy as np
import pytest

from suspkin.geometry import Coilover, build_corners
from suspkin.kinematics import KinematicsError, solve_corner, sweep_corner, sweep_rack
from suspkin.sweeps import states_to_frame
from tests.unit.helpers import fixture_corners, fixture_vehicle


def test_sweep_equals_independent_solves() -> None:
    c = fixture_corners()["FL"]
    travels = np.linspace(-0.03, 0.03, 61)
    states = sweep_corner(c, travels, rack=0.005)
    assert [s.travel for s in states] == pytest.approx(list(travels))
    for t, s in zip(travels, states, strict=True):
        ref = solve_corner(c, float(t), 0.005)
        for name, p in ref.points.items():
            np.testing.assert_allclose(s.points[name], p, atol=1e-9)
        assert s.rocker_angle == pytest.approx(ref.rocker_angle, abs=1e-9)


def test_unsorted_travels_keep_input_order() -> None:
    c = fixture_corners()["RR"]
    travels = [0.02, -0.01, 0.0, 0.03, -0.03]
    states = sweep_corner(c, travels)
    assert [s.travel for s in states] == travels


def test_sweep_61_steps_is_fast() -> None:
    c = fixture_corners()["FL"]
    sweep_corner(c, [0.0, 0.001])  # warm-up
    t0 = time.perf_counter()
    sweep_corner(c, np.linspace(-0.03, 0.03, 61))
    assert time.perf_counter() - t0 < 1.0


def test_sweep_far_travel_raises() -> None:
    c = fixture_corners()["FL"]
    with pytest.raises(KinematicsError):
        sweep_corner(c, [0.0, 0.1, 0.5])


def test_sweep_rack_matches_solves() -> None:
    c = fixture_corners()["FL"]
    racks = np.linspace(-0.02, 0.02, 9)
    states = sweep_rack(c, racks, travel=0.01)
    for r, s in zip(racks, states, strict=True):
        ref = solve_corner(c, 0.01, float(r))
        np.testing.assert_allclose(s.points["wheel_center"], ref.points["wheel_center"], atol=1e-9)
        assert s.rack == pytest.approx(r)
    assert states[-1].spin_axis[0] != pytest.approx(states[0].spin_axis[0])


def test_frame_columns_and_rows() -> None:
    c = fixture_corners()["FL"]
    df = states_to_frame(sweep_corner(c, np.linspace(-0.01, 0.01, 5)))
    assert len(df) == 5
    expected = {
        "corner", "travel", "rack", "theta_uca", "theta_lca", "psi", "rocker_angle",
        "coilover_length", "toggle_margin", "max_residual", "spin_x", "spin_y", "spin_z",
        "coilover_in_range", "contact_patch_x", "contact_patch_y", "contact_patch_z",
        "wheel_center_z", "uca_outer_x", "tie_inner_y", "rocker_arb_z", "pushrod_inner_x",
    }  # fmt: skip
    assert expected <= set(df.columns)
    assert set(df["corner"]) == {"FL"}
    assert df["coilover_in_range"].dtype == bool
    assert df["coilover_in_range"].all()
    assert (df["max_residual"] < 1e-9).all()


def test_coilover_range_flag() -> None:
    v = fixture_vehicle()
    tight = Coilover(
        spring_rate=35000.0, free_length=0.2, preload=0.0, length_min=0.195, length_max=0.25
    )
    v2 = v.model_copy(update={"front": v.front.model_copy(update={"coilover": tight})})
    c = build_corners(v2)["FL"]
    df = states_to_frame(sweep_corner(c, [-0.01, 0.0, 0.02]))
    assert list(df["coilover_in_range"]) == [True, True, False]


def test_sweep_rack_with_travel_is_staged_from_static() -> None:
    c = fixture_corners()["FL"]
    racks = np.linspace(-0.025, 0.025, 11)
    states = sweep_rack(c, racks, travel=0.03)
    for r, s in zip(racks, states, strict=True):
        ref = solve_corner(c, 0.03, float(r))
        np.testing.assert_allclose(s.points["wheel_center"], ref.points["wheel_center"], atol=1e-9)
        assert s.travel == pytest.approx(0.03)
