"""Regression: roll sweep of the reference car."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.sweeps import sweep_vehicle, vehicle_frame

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"


@pytest.fixture(scope="module")
def vehicle():  # type: ignore[no-untyped-def]
    return load_vehicle(CONFIG)


def test_roll_sweep_speed_and_symmetry(vehicle) -> None:  # type: ignore[no-untyped-def]
    rolls = np.radians(np.linspace(-2.0, 2.0, 41))
    t0 = time.perf_counter()
    states = sweep_vehicle(vehicle, roll=rolls)
    df = vehicle_frame(states, vehicle)
    assert time.perf_counter() - t0 < 3.0
    assert all(s.max_residual < 1e-9 for s in states)
    assert np.all(np.diff(df["FL_travel"]) < 0)  # left up -> droop
    assert np.all(np.diff(df["FR_travel"]) > 0)
    np.testing.assert_allclose(df["FL_travel"], df["FR_travel"][::-1].to_numpy(), atol=1e-9)
    assert np.all(np.diff(df["arb_front_elastic"]) < 0)
    assert np.all(np.diff(df["arb_rear_elastic"]) < 0)
    # roll of 1 deg: wheel travel ~ half-track * phi
    i = int(np.argmin(np.abs(rolls - np.radians(1.0))))
    assert df["FR_travel"][i] == pytest.approx(0.61 * np.radians(1.0), rel=0.05)
