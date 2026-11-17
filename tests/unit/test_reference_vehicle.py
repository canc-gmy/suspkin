"""Static checks of the reference FS vehicle (configs/vehicles/fs_reference.yaml).

Caster/KPI/scrub/trail and roll centre height come from ``suspkin.metrics``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from suspkin.geometry import Vehicle, build_corners, load_vehicle
from suspkin.geometry.corners import Corner
from suspkin.kinematics.corner import CornerState
from suspkin.metrics import (
    caster,
    kpi,
    mechanical_trail,
    roll_centre_symmetric,
    scrub_radius,
    static_state,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"


def _static(vehicle: Vehicle, axle: str) -> tuple[CornerState, Corner]:
    corner = build_corners(vehicle)["FL" if axle == "front" else "RL"]
    return static_state(corner), corner


@pytest.fixture(scope="module")
def vehicle():  # type: ignore[no-untyped-def]
    return load_vehicle(CONFIG)


def test_wheelbase_and_tracks(vehicle) -> None:  # type: ignore[no-untyped-def]
    assert vehicle.wheelbase == pytest.approx(1.550, abs=0.005)
    assert vehicle.front_track == pytest.approx(1.220, abs=0.005)
    assert vehicle.rear_track == pytest.approx(1.180, abs=0.005)


def test_front_steering_axis(vehicle) -> None:  # type: ignore[no-untyped-def]
    s, c = _static(vehicle, "front")
    assert 4.0 <= math.degrees(caster(s, c)) <= 6.0
    assert 4.0 <= math.degrees(kpi(s, c)) <= 6.0
    assert 0.010 <= scrub_radius(s, c) <= 0.025
    assert 0.015 <= mechanical_trail(s, c) <= 0.025


def test_rear_steering_axis(vehicle) -> None:  # type: ignore[no-untyped-def]
    s, c = _static(vehicle, "rear")
    assert 0.0 <= math.degrees(caster(s, c)) <= 2.0
    assert 3.0 <= math.degrees(kpi(s, c)) <= 6.0


def test_roll_centre_heights(vehicle) -> None:  # type: ignore[no-untyped-def]
    assert 0.020 <= roll_centre_symmetric(*_static(vehicle, "front")) <= 0.040
    assert 0.030 <= roll_centre_symmetric(*_static(vehicle, "rear")) <= 0.060


def test_uca_shorter_than_lca(vehicle) -> None:  # type: ignore[no-untyped-def]
    for axle in (vehicle.front, vehicle.rear):
        h = axle.hardpoints
        uca = abs(h.uca_outer[1] - 0.5 * (h.uca_front[1] + h.uca_rear[1]))
        lca = abs(h.lca_outer[1] - 0.5 * (h.lca_front[1] + h.lca_rear[1]))
        assert uca < lca


def test_pushrod_on_uca(vehicle) -> None:  # type: ignore[no-untyped-def]
    assert vehicle.front.pushrod_mount == "uca"
    assert vehicle.rear.pushrod_mount == "uca"
