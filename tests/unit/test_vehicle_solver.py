"""Vehicle-level solver: body pose -> wheel travels from ground contact."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics import KinematicsError, contact_point, solve_vehicle
from suspkin.kinematics.vehicle import _static_contact
from tests.unit.helpers import FIXTURE

VEHICLE = load_vehicle(FIXTURE)
CORNERS = build_corners(VEHICLE)
IDS = ("FL", "FR", "RL", "RR")


def _travels(s) -> dict[str, float]:  # type: ignore[no-untyped-def]
    return {k: s.corners[k].travel for k in IDS}


def test_static_pose() -> None:
    s = solve_vehicle(VEHICLE, CORNERS)
    for k in IDS:
        assert abs(s.corners[k].travel) < 1e-9
        np.testing.assert_allclose(s.contact_points[k], _static_contact(CORNERS[k]), atol=1e-9)
        np.testing.assert_allclose(
            s.contact_points[k], s.corners[k].points["contact_patch"], atol=1e-12
        )
    np.testing.assert_allclose(s.ground_normal, [0.0, 0.0, 1.0], atol=1e-15)


@pytest.mark.parametrize("h", [-0.02, 0.01, 0.03])
def test_pure_heave(h: float) -> None:
    t = _travels(solve_vehicle(VEHICLE, CORNERS, heave=h))
    assert t["FL"] == pytest.approx(t["FR"], abs=1e-9)
    assert t["RL"] == pytest.approx(t["RR"], abs=1e-9)
    for v in t.values():
        assert v == pytest.approx(-h, abs=2e-4)
        assert v * h < 0.0


def test_pure_roll() -> None:
    phi = 0.02
    p = _travels(solve_vehicle(VEHICLE, CORNERS, roll=phi))
    m = _travels(solve_vehicle(VEHICLE, CORNERS, roll=-phi))
    assert p["FL"] < 0 < p["FR"] and p["RL"] < 0 < p["RR"]
    assert p["FL"] == pytest.approx(-p["FR"], rel=0.02)
    assert p["FL"] == pytest.approx(m["FR"], abs=1e-9)  # mirror symmetry
    sp = solve_vehicle(VEHICLE, CORNERS, roll=phi)
    sm = solve_vehicle(VEHICLE, CORNERS, roll=-phi)
    sh = solve_vehicle(VEHICLE, CORNERS, heave=0.02)
    for axle in ("front", "rear"):
        if sp.arb[axle] is None:
            continue
        assert abs(sp.arb[axle].elastic) > 1e-4  # type: ignore[union-attr]
        assert sp.arb[axle].elastic == pytest.approx(-sm.arb[axle].elastic, abs=1e-6)  # type: ignore[union-attr]
        assert abs(sh.arb[axle].elastic) < 1e-6  # type: ignore[union-attr]


def test_pitch_nose_down_about_mid_wheelbase() -> None:
    t = _travels(solve_vehicle(VEHICLE, CORNERS, pitch=0.01, pivot=(-0.775, 0.0, 0.0)))
    assert t["FL"] > 0 and t["FR"] > 0
    assert t["RL"] < 0 and t["RR"] < 0


def test_rack_front_only() -> None:
    s = solve_vehicle(VEHICLE, CORNERS, rack=0.01)
    assert s.corners["FL"].rack == pytest.approx(0.01)
    assert s.corners["RL"].rack == 0.0 and s.corners["RR"].rack == 0.0


def test_out_of_travel_raises() -> None:
    with pytest.raises(KinematicsError):
        solve_vehicle(VEHICLE, CORNERS, heave=0.5)


@settings(max_examples=25, deadline=None)
@given(
    h=st.floats(-0.02, 0.02),
    phi=st.floats(-0.035, 0.035),
    th=st.floats(-0.02, 0.02),
    rack=st.floats(-0.02, 0.02),
)
def test_contact_on_ground_plane(h: float, phi: float, th: float, rack: float) -> None:
    s = solve_vehicle(VEHICLE, CORNERS, heave=h, roll=phi, pitch=th, rack=rack)
    n = s.ground_normal
    for k in IDS:
        cp = contact_point(s.corners[k], n, CORNERS[k].axle_cfg.wheel.tyre_radius)
        z_world = float(n @ cp) + h  # origin pivot
        assert z_world == pytest.approx(float(_static_contact(CORNERS[k])[2]), abs=1e-9)
        np.testing.assert_allclose(cp, s.contact_points[k], atol=1e-12)
