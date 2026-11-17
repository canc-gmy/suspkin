"""Regression checks of the reference FS car (UCA pushrod, rocker above the UCA).

Motion ratio is taken from the real solver by central finite difference of the
coilover length over +-1 mm of wheel travel.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.geometry.corners import build_corners
from suspkin.kinematics import solve_corner, sweep_corner, sweep_rack

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"
TRAVELS = np.linspace(-0.030, 0.030, 61)
MIN_TOGGLE = math.radians(25.0)


@pytest.fixture(scope="module")
def corners():  # type: ignore[no-untyped-def]
    return build_corners(load_vehicle(CONFIG))


def _motion_ratio(corner, travel: float) -> float:  # type: ignore[no-untyped-def]
    h = 1e-3
    lo = solve_corner(corner, travel - h).coilover_length
    hi = solve_corner(corner, travel + h).coilover_length
    return abs(hi - lo) / (2.0 * h)


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_sweep_succeeds_and_coilover_in_range(corners, cid) -> None:  # type: ignore[no-untyped-def]
    states = sweep_corner(corners[cid], TRAVELS)
    assert len(states) == 61
    assert all(s.coilover_in_range for s in states)
    assert all(s.max_residual < 1e-8 for s in states)


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_motion_ratio_and_progression(corners, cid) -> None:  # type: ignore[no-untyped-def]
    corner = corners[cid]
    mr0 = _motion_ratio(corner, 0.0)
    assert 0.9 <= mr0 <= 1.1
    assert abs(_motion_ratio(corner, 0.025) - _motion_ratio(corner, -0.025)) / mr0 < 0.15
    # bump shortens the coilover
    assert solve_corner(corner, 0.010).coilover_length < solve_corner(corner, 0.0).coilover_length


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_toggle_margin(corners, cid) -> None:  # type: ignore[no-untyped-def]
    states = sweep_corner(corners[cid], TRAVELS)
    assert min(s.toggle_margin for s in states) >= MIN_TOGGLE


def test_front_rack_sweep(corners) -> None:  # type: ignore[no-untyped-def]
    states = sweep_rack(corners["FL"], np.linspace(-0.025, 0.025, 21), travel=0.0)
    assert len(states) == 21
    assert min(s.toggle_margin for s in states) >= MIN_TOGGLE
    assert all(s.coilover_in_range for s in states)


@pytest.mark.parametrize(("left", "right"), [("FL", "FR"), ("RL", "RR")])
def test_left_right_symmetry(corners, left, right) -> None:  # type: ignore[no-untyped-def]
    for travel in (-0.025, 0.0, 0.020):
        a = solve_corner(corners[left], travel)
        b = solve_corner(corners[right], travel)
        assert a.rocker_angle == pytest.approx(b.rocker_angle, abs=1e-9)
        assert a.coilover_length == pytest.approx(b.coilover_length, abs=1e-9)


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_reference_alignment_gradients(corners, cid) -> None:  # type: ignore[no-untyped-def]
    from suspkin.metrics import add_gradients, metrics_frame, static_summary

    states = sweep_corner(corners[cid], np.linspace(-0.01, 0.01, 21))
    s = static_summary(add_gradients(metrics_frame(states, corners[cid])))
    # bump steer below 0.01 rad/m (about 6e-4 deg/mm); actual is ~1e-3 to 5e-3 rad/m
    assert abs(s["bump_steer"]) < 1e-2
    # camber becomes more negative in bump
    assert s["camber_gain"] < 0.0
