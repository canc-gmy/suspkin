"""Tests for actuation metrics, motion ratios and wheel/ride rates."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import simpson

from suspkin.kinematics import solve_corner, sweep_corner
from suspkin.metrics import add_gradients, metrics_frame
from suspkin.metrics.actuation import (
    actuation_metrics,
    bump_stroke_remaining,
    droop_stroke_remaining,
    motion_ratios,
    pushrod_angle,
    pushrod_angle_fv,
    ride_rate,
    spring_force,
    wheel_load_spring,
    wheel_rate,
    wheel_rate_linear,
)
from tests.unit.helpers import fixture_corners

CORNERS = fixture_corners()
TRAVELS = np.linspace(-0.02, 0.02, 81)


def _spring_energy(corner, length: float) -> float:  # type: ignore[no-untyped-def]
    c = corner.axle_cfg.coilover
    x = c.free_length - length
    return c.preload * x + 0.5 * c.spring_rate * x * x


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_per_state_definitions(cid) -> None:  # type: ignore[no-untyped-def]
    c = CORNERS[cid]
    static = solve_corner(c, 0.0)
    s = solve_corner(c, 0.01)
    d = s.points["pushrod_inner"] - s.points["pushrod_outer"]
    assert pushrod_angle(s, c) == pytest.approx(math.asin(d[2] / np.linalg.norm(d)), abs=1e-12)
    assert pushrod_angle_fv(s, c) == pytest.approx(math.atan2(d[2], abs(d[1])), abs=1e-12)
    m = actuation_metrics(s, c, static)
    assert m["coilover_compression"] == pytest.approx(static.coilover_length - s.coilover_length)
    assert m["coilover_compression"] > 0.0  # bump compresses
    cv = c.axle_cfg.coilover
    assert bump_stroke_remaining(s, c) == pytest.approx(s.coilover_length - cv.length_min)
    assert droop_stroke_remaining(s, c) == pytest.approx(cv.length_max - s.coilover_length)
    assert spring_force(s, c) == pytest.approx(
        cv.preload + cv.spring_rate * (cv.free_length - s.coilover_length)
    )
    assert m["rocker_rotation"] == pytest.approx(s.rocker_angle - static.rocker_angle)
    z = actuation_metrics(static, c, static)
    assert z["coilover_compression"] == 0.0
    assert z["rocker_rotation"] == 0.0


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_motion_ratio_consistency(cid) -> None:  # type: ignore[no-untyped-def]
    c = CORNERS[cid]
    mr = motion_ratios(c, 0.0)
    assert mr.spring > 0.0
    assert mr.damper == mr.spring
    assert mr.pushrod > 0.0
    assert mr.rocker == pytest.approx(mr.spring / mr.pushrod)
    # a different step gives the same answer (finite-difference consistency)
    mr2 = motion_ratios(c, 0.0, h=1e-4)
    assert mr2.spring == pytest.approx(mr.spring, rel=1e-5)
    assert mr2.pushrod == pytest.approx(mr.pushrod, rel=1e-5)
    # progression equals the derivative of the spring MR
    h = 1e-4
    dmr = (motion_ratios(c, h).spring - motion_ratios(c, -h).spring) / (2 * h)
    assert mr.progression == pytest.approx(dmr, rel=2e-3, abs=1e-3)


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_table_mr_matches_solver_mr(cid) -> None:  # type: ignore[no-untyped-def]
    c = CORNERS[cid]
    df = add_gradients(metrics_frame(sweep_corner(c, TRAVELS), c))
    for i in range(2, len(TRAVELS) - 2, 6):
        ref = motion_ratios(c, float(TRAVELS[i])).spring
        assert df["mr_spring"].iloc[i] == pytest.approx(ref, abs=1e-4)
    for col in ("mr_progression", "wheel_rate_linear", "wheel_rate", "ride_rate",
                "wheel_load_spring", "pushrod_angle", "spring_force"):  # fmt: skip
        assert col in df.columns


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_wheel_rate_is_derivative_of_wheel_load(cid) -> None:  # type: ignore[no-untyped-def]
    c = CORNERS[cid]
    cv = c.axle_cfg.coilover
    h = 2e-4

    def load(z: float) -> float:
        s = solve_corner(c, z)
        return wheel_load_spring(spring_force(s, c), motion_ratios(c, z).spring)

    for z in (-0.01, 0.0, 0.012):
        s = solve_corner(c, z)
        mr = motion_ratios(c, z)
        wr = wheel_rate(cv.spring_rate, mr.spring, spring_force(s, c), mr.progression)
        fd = (load(z + h) - load(z - h)) / (2 * h)
        assert wr == pytest.approx(fd, rel=1e-3)


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_energy_balance(cid) -> None:  # type: ignore[no-untyped-def]
    c = CORNERS[cid]
    zs = np.linspace(-0.015, 0.015, 301)
    fw = []
    for z in zs:
        s = solve_corner(c, float(z))
        fw.append(wheel_load_spring(spring_force(s, c), motion_ratios(c, float(z)).spring))
    work = float(simpson(fw, x=zs))  # Simpson: trapezoid error ~2e-5 here
    du = _spring_energy(c, solve_corner(c, float(zs[-1])).coilover_length) - _spring_energy(
        c, solve_corner(c, float(zs[0])).coilover_length
    )
    assert work == pytest.approx(du, rel=1e-6)


def test_rate_formulas() -> None:
    assert wheel_rate_linear(30000.0, 0.9) == pytest.approx(30000.0 * 0.81)
    assert wheel_rate(30000.0, 0.9, 500.0, 2.0) == pytest.approx(30000.0 * 0.81 + 1000.0)
    assert ride_rate(20000.0, 80000.0) == pytest.approx(16000.0)
    assert ride_rate(20000.0, 80000.0) < 20000.0


@pytest.mark.parametrize(("left", "right"), [("FL", "FR"), ("RL", "RR")])
def test_left_right_equality(left, right) -> None:  # type: ignore[no-untyped-def]
    for z in (-0.01, 0.0, 0.015):
        a, b = CORNERS[left], CORNERS[right]
        ma = actuation_metrics(solve_corner(a, z), a, solve_corner(a, 0.0))
        mb = actuation_metrics(solve_corner(b, z), b, solve_corner(b, 0.0))
        for k, v in ma.items():
            assert mb[k] == pytest.approx(v, abs=1e-8), k
        ra, rb = motion_ratios(a, z), motion_ratios(b, z)
        for f in ("spring", "pushrod", "rocker", "progression"):
            assert getattr(rb, f) == pytest.approx(getattr(ra, f), abs=1e-6), f
