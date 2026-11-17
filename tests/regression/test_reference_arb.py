"""Regression: ARB of the reference car (front torsional, rear blade)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.geometry.corners import build_corners
from suspkin.kinematics import arb_stiffness, solve_arb, solve_corner
from suspkin.metrics.arb import arb_rates

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"
TRAVELS = np.linspace(-0.030, 0.030, 31)
AXLES = [("front", "FL", "FR", "torsional"), ("rear", "RL", "RR", "blade")]


@pytest.fixture(scope="module")
def vehicle():  # type: ignore[no-untyped-def]
    return load_vehicle(CONFIG)


@pytest.mark.parametrize(("axle", "left", "right", "kind"), AXLES)
def test_antisymmetric_sweep_solves(vehicle, axle, left, right, kind) -> None:  # type: ignore[no-untyped-def]
    cfg = getattr(vehicle, axle)
    assert cfg.arb.type == kind
    corners = build_corners(vehicle)
    prev = None
    elastic = []
    for z in TRAVELS:
        st = solve_arb(
            cfg,
            solve_corner(corners[left], float(z)),
            solve_corner(corners[right], -float(z)),
            guess=prev,
        )
        assert st.max_residual < 1e-9
        elastic.append(st.elastic)
        prev = st
    e = np.array(elastic)
    assert np.all(np.diff(e) > 0.0)  # monotonic in roll travel
    np.testing.assert_allclose(e, -e[::-1], atol=1e-4)  # odd in travel


@pytest.mark.parametrize(("axle", "left", "right", "kind"), AXLES)
def test_reference_rates_sane(vehicle, axle, left, right, kind) -> None:  # type: ignore[no-untyped-def]
    cfg = getattr(vehicle, axle)
    corners = build_corners(vehicle)
    rates = arb_rates(corners[left], corners[right], cfg)
    spring_wheel_rate = cfg.coilover.spring_rate  # motion ratio ~1
    assert rates.heave_coupling == pytest.approx(0.0, abs=1e-6)
    assert rates.stiffness == arb_stiffness(cfg)
    # ARB adds a meaningful but not dominating wheel rate (5 .. 60 N/mm)
    assert 5e3 < rates.wheel_rate_roll < 60e3
    assert 0.3 * spring_wheel_rate < rates.wheel_rate_roll < 1.5 * spring_wheel_rate
