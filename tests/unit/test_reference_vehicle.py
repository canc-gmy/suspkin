"""Static checks of the reference FS vehicle (configs/vehicles/fs_reference.yaml).

Temporary: these local helpers compute caster/KPI/scrub/trail and roll centre
height directly from hardpoints. They will be replaced by the metrics module
(v0.3).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from suspkin.geometry import Axle, load_vehicle

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"

F64 = NDArray[np.float64]


def _p(v: tuple[float, float, float]) -> F64:
    return np.array(v, dtype=float)


def _steering_axis_ground_point(axle: Axle) -> F64:
    """Intersection of the ball-joint line with the ground plane (line-plane)."""
    ub = _p(axle.hardpoints.uca_outer)
    lb = _p(axle.hardpoints.lca_outer)
    u = ub - lb
    t = -lb[2] / u[2]
    return lb + t * u


def _contact_patch(axle: Axle) -> F64:
    wc = _p(axle.hardpoints.wheel_center)
    return np.array([wc[0], wc[1] + wc[2] * math.tan(-axle.wheel.static_camber), 0.0])


def _caster_kpi(axle: Axle) -> tuple[float, float]:
    u = _p(axle.hardpoints.uca_outer) - _p(axle.hardpoints.lca_outer)
    return math.atan2(-u[0], u[2]), math.atan2(-u[1], u[2])


def _scrub_trail(axle: Axle) -> tuple[float, float]:
    g = _steering_axis_ground_point(axle)
    cp = _contact_patch(axle)
    return float(cp[1] - g[1]), float(g[0] - cp[0])


def _roll_centre_height(axle: Axle) -> float:
    """Front-view roll centre height from the wishbone-axis instant centre."""
    h = axle.hardpoints
    uin = ((_p(h.uca_front) + _p(h.uca_rear)) / 2)[1:]
    lin = ((_p(h.lca_front) + _p(h.lca_rear)) / 2)[1:]
    d_u = _p(h.uca_outer)[1:] - uin
    d_l = _p(h.lca_outer)[1:] - lin
    t = np.linalg.solve(np.array([d_u, -d_l]).T, lin - uin)
    ic = uin + t[0] * d_u
    cp = _contact_patch(axle)[1:]
    s = (0.0 - cp[0]) / (ic[0] - cp[0])
    return float(cp[1] + s * (ic[1] - cp[1]))


@pytest.fixture(scope="module")
def vehicle():  # type: ignore[no-untyped-def]
    return load_vehicle(CONFIG)


def test_wheelbase_and_tracks(vehicle) -> None:  # type: ignore[no-untyped-def]
    assert vehicle.wheelbase == pytest.approx(1.550, abs=0.005)
    assert vehicle.front_track == pytest.approx(1.220, abs=0.005)
    assert vehicle.rear_track == pytest.approx(1.180, abs=0.005)


def test_front_steering_axis(vehicle) -> None:  # type: ignore[no-untyped-def]
    caster, kpi = _caster_kpi(vehicle.front)
    scrub, trail = _scrub_trail(vehicle.front)
    assert 4.0 <= math.degrees(caster) <= 6.0
    assert 4.0 <= math.degrees(kpi) <= 6.0
    assert 0.010 <= scrub <= 0.025
    assert 0.015 <= trail <= 0.025


def test_rear_steering_axis(vehicle) -> None:  # type: ignore[no-untyped-def]
    caster, kpi = _caster_kpi(vehicle.rear)
    assert 0.0 <= math.degrees(caster) <= 2.0
    assert 3.0 <= math.degrees(kpi) <= 6.0


def test_roll_centre_heights(vehicle) -> None:  # type: ignore[no-untyped-def]
    assert 0.020 <= _roll_centre_height(vehicle.front) <= 0.040
    assert 0.030 <= _roll_centre_height(vehicle.rear) <= 0.060


def test_uca_shorter_than_lca(vehicle) -> None:  # type: ignore[no-untyped-def]
    for axle in (vehicle.front, vehicle.rear):
        h = axle.hardpoints
        uca = abs(h.uca_outer[1] - 0.5 * (h.uca_front[1] + h.uca_rear[1]))
        lca = abs(h.lca_outer[1] - 0.5 * (h.lca_front[1] + h.lca_rear[1]))
        assert uca < lca


def test_pushrod_on_uca(vehicle) -> None:  # type: ignore[no-untyped-def]
    assert vehicle.front.pushrod_mount == "uca"
    assert vehicle.rear.pushrod_mount == "uca"
