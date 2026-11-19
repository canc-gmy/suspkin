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


def test_pitch_nose_down_about_cg() -> None:
    t = _travels(solve_vehicle(VEHICLE, CORNERS, pitch=0.01))
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
        z_world = float(n @ (cp - s.pivot)) + s.pivot[2] + h
        assert z_world == pytest.approx(float(_static_contact(CORNERS[k])[2]), abs=1e-9)
        np.testing.assert_allclose(cp, s.contact_points[k], atol=1e-12)


def test_default_pivot_is_cg_and_overridable() -> None:
    from suspkin.kinematics import cg_pivot

    s = solve_vehicle(VEHICLE, CORNERS, pitch=0.01)
    a = VEHICLE.wheelbase * (1.0 - VEHICLE.vehicle.front_mass_ratio)
    np.testing.assert_allclose(s.pivot, [-a, 0.0, VEHICLE.vehicle.cg_height])
    np.testing.assert_allclose(s.pivot, cg_pivot(VEHICLE))
    s0 = solve_vehicle(VEHICLE, CORNERS, pitch=0.01, pivot=(0.0, 0.0, 0.0))
    assert abs(s0.corners["FL"].travel) < abs(s.corners["FL"].travel)  # origin = front axle


def _brute_lowest(state, n, radius):  # type: ignore[no-untyped-def]
    from scipy.optimize import minimize_scalar

    s = np.asarray(state.spin_axis)
    u = np.cross(s, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(s, u)
    c = np.asarray(state.points["wheel_center"])

    def pt(t: float):  # type: ignore[no-untyped-def]
        return c + radius * (np.cos(t) * u + np.sin(t) * v)

    ts = np.linspace(0.0, 2.0 * np.pi, 20001)
    h = np.array([n @ pt(t) for t in ts])
    t0 = ts[int(np.argmin(h))]
    res = minimize_scalar(lambda t: n @ pt(t), bracket=(t0 - 1e-3, t0, t0 + 1e-3), tol=1e-14)
    return pt(res.x)


@pytest.mark.parametrize(
    "pose",
    [
        {"roll": 0.03},
        {"pitch": -0.02, "heave": 0.01},
        {"roll": -0.02, "pitch": 0.015, "rack": 0.015, "heave": -0.01},
    ],
)
def test_contact_matches_brute_force(pose: dict[str, float]) -> None:
    s = solve_vehicle(VEHICLE, CORNERS, **pose)
    for k in IDS:
        r = CORNERS[k].axle_cfg.wheel.tyre_radius
        ref = _brute_lowest(s.corners[k], s.ground_normal, r)
        np.testing.assert_allclose(s.contact_points[k], ref, atol=1e-6)


@pytest.mark.parametrize(
    ("prev", "pose"),
    [
        ({"heave": 0.010}, {"heave": 0.012}),
        ({"roll": 0.010}, {"roll": 0.012}),
        ({"rack": 0.004}, {"rack": 0.006}),
        ({"heave": 0.010}, {"heave": 0.010}),  # guess at the same pose
    ],
)
def test_warm_start_matches_cold_solve(prev: dict[str, float], pose: dict[str, float]) -> None:
    """Continuation (guess + secant slope warm start) lands on the cold solution."""
    guess = solve_vehicle(VEHICLE, CORNERS, **prev)  # type: ignore[arg-type]
    assert set(guess._slopes) == set(IDS)
    warm = solve_vehicle(VEHICLE, CORNERS, guess=guess, **pose)  # type: ignore[arg-type]
    cold = solve_vehicle(VEHICLE, CORNERS, **pose)  # type: ignore[arg-type]
    for k in IDS:
        assert warm.corners[k].travel == pytest.approx(cold.corners[k].travel, abs=1e-10)
        for name, p in cold.corners[k].points.items():
            np.testing.assert_allclose(warm.corners[k].points[name], p, atol=1e-9)
        np.testing.assert_allclose(warm.contact_points[k], cold.contact_points[k], atol=1e-9)
    assert warm.max_residual < 1e-9
