"""Vehicle summary and ground-frame sweep columns."""

from __future__ import annotations

import math

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.metrics import roll_gradient, sprung_mass, vehicle_summary
from suspkin.sweeps import sweep_vehicle, vehicle_frame
from tests.unit.helpers import FIXTURE

VEHICLE = load_vehicle(FIXTURE)
SUMMARY = vehicle_summary(VEHICLE)


def test_keys_and_units() -> None:
    d = SUMMARY.to_dict()
    for k in (
        "front",
        "rear",
        "roll_gradient",
        "tlltd_front",
        "anti_dive_front",
        "ackermann_10deg",
        "steering_ratio",
        "turning_radius",
        "pitch_centre_x",
    ):
        assert k in d
    for k in ("track", "wheel_rate", "roll_stiffness", "rc_height", "roll_camber", "rc_migration"):
        assert k in d["front"]  # type: ignore[operator]
    assert 0.5 < SUMMARY.front.track < 2.0  # metres
    assert 1e3 < SUMMARY.front.wheel_rate < 1e6  # N/m
    assert 0.0 < SUMMARY.max_steer_outer < 1.0  # rad
    assert SUMMARY.anti_lift_front_accel is None  # rwd


def test_roll_stiffness_total_is_sum() -> None:
    assert SUMMARY.roll_stiffness_total == pytest.approx(
        SUMMARY.front.roll_stiffness + SUMMARY.rear.roll_stiffness
    )


def test_roll_gradient_formula() -> None:
    expected = roll_gradient(
        SUMMARY.sprung_mass,
        SUMMARY.sprung_cg_height,
        SUMMARY.roll_axis_height_at_cg,
        SUMMARY.front.roll_stiffness,
        SUMMARY.rear.roll_stiffness,
    )
    assert SUMMARY.roll_gradient == pytest.approx(expected)
    assert SUMMARY.sprung_mass == pytest.approx(sprung_mass(VEHICLE))
    assert 0.0 < SUMMARY.tlltd_front < 1.0
    assert SUMMARY.load_transfer_front + SUMMARY.load_transfer_rear > 0.0
    assert math.isfinite(SUMMARY.turning_radius)


def test_ground_columns_at_zero_roll() -> None:
    df = vehicle_frame(sweep_vehicle(VEHICLE, roll=np.array([-0.01, 0.0, 0.01])), VEHICLE)
    r = df.iloc[1]
    for cid in ("FL", "FR", "RL", "RR"):
        assert r[f"{cid}_camber_ground"] == pytest.approx(r[f"{cid}_camber"], abs=1e-12)
    for axle in ("front", "rear"):
        assert r[f"rc_{axle}_y"] == pytest.approx(0.0, abs=1e-9)
        assert df[f"rc_{axle}_z"].iloc[0] == pytest.approx(df[f"rc_{axle}_z"].iloc[2], abs=1e-9)
        assert df[f"rc_{axle}_y"].iloc[0] == pytest.approx(-df[f"rc_{axle}_y"].iloc[2], abs=1e-9)
    # a rolled body tilts the wheels: left camber to ground differs from body camber
    assert df["FL_camber_ground"].iloc[2] != pytest.approx(df["FL_camber"].iloc[2], abs=1e-4)
