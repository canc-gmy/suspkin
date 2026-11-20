"""Vehicle summary and ground-frame sweep columns."""

from __future__ import annotations

import math

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.metrics import vehicle_summary
from suspkin.metrics.vehicle import roll_gradient, sprung_mass
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


def test_pitch_metrics_use_tilted_ground_contact() -> None:
    th = 0.035  # 2 deg
    df = vehicle_frame(sweep_vehicle(VEHICLE, pitch=np.array([0.0, th])), VEHICLE)
    for cid in ("FL", "FR", "RL", "RR"):
        d_cp_x = df[f"{cid}_cp_x"].iloc[1] - df[f"{cid}_cp_x"].iloc[0]
        assert df[f"{cid}_contact_patch_dx"].iloc[1] == pytest.approx(d_cp_x, abs=1e-9)
    assert abs(df["FL_contact_patch_dx"].iloc[1]) > 1e-4  # the shift is real


def test_end_to_end_signs_roll_and_rack() -> None:
    roll = np.radians(1.0)
    df = vehicle_frame(sweep_vehicle(VEHICLE, roll=np.array([0.0, roll])), VEHICLE)
    # left up: right (outer) wheel loses negative camber relative to the ground
    assert df["FR_camber_ground"].iloc[1] > df["FR_camber_ground"].iloc[0]
    assert df["FL_camber_ground"].iloc[1] < df["FL_camber_ground"].iloc[0]
    # roll centre migration: + = toward +y (left) for left-up roll; summary agrees with table
    assert np.sign(df["rc_front_y"].iloc[1]) == np.sign(SUMMARY.front.rc_migration)
    assert df["rc_front_y"].iloc[1] / roll == pytest.approx(SUMMARY.front.rc_migration, rel=0.1)
    for axle, left, right in (("front", "FL", "FR"), ("rear", "RL", "RR")):
        mean_steer = 0.5 * (df[f"{left}_steer_ground"] + df[f"{right}_steer_ground"])
        d = float(mean_steer.iloc[1] - mean_steer.iloc[0])
        rs = getattr(SUMMARY, axle).roll_steer
        if abs(rs) > 1e-9:
            assert np.sign(d) == np.sign(rs)
        assert d / roll == pytest.approx(rs, rel=0.2, abs=1e-3)
    dr = vehicle_frame(sweep_vehicle(VEHICLE, rack=np.array([0.0, 0.01])), VEHICLE)
    assert dr["FL_steer_ground"].iloc[1] > dr["FL_steer_ground"].iloc[0]
    assert dr["FR_steer_ground"].iloc[1] > dr["FR_steer_ground"].iloc[0]


def test_sprung_cg_by_moments() -> None:
    vp = VEHICLE.vehicle
    m_s = SUMMARY.sprung_mass
    m_uf, m_ur = 2 * VEHICLE.front.unsprung_mass, 2 * VEHICLE.rear.unsprung_mass
    hu_f = vp.unsprung_cg_height or VEHICLE.front.wheel.tyre_radius
    hu_r = vp.unsprung_cg_height or VEHICLE.rear.wheel.tyre_radius
    expected = (vp.mass * vp.cg_height - m_uf * hu_f - m_ur * hu_r) / m_s
    assert SUMMARY.sprung_cg_height == pytest.approx(expected)
    assert math.isfinite(SUMMARY.front.rc_height_curvature)


def _tilted_vehicle():  # type: ignore[no-untyped-def]
    """Fixture with strongly tilted upper wishbone axes (anti geometry) on both axles."""
    fh = VEHICLE.front.hardpoints.model_copy(
        update={"uca_front": (0.10, 0.28, 0.36), "uca_rear": (-0.10, 0.28, 0.24)}
    )
    rh = VEHICLE.rear.hardpoints.model_copy(
        update={"uca_front": (-1.50, 0.26, 0.24), "uca_rear": (-1.70, 0.26, 0.36)}
    )
    return VEHICLE.model_copy(
        update={
            "front": VEHICLE.front.model_copy(update={"hardpoints": fh}),
            "rear": VEHICLE.rear.model_copy(update={"hardpoints": rh}),
        }
    )


def test_anti_geometry_hand_calculation() -> None:
    from suspkin.geometry import build_corners
    from suspkin.kinematics import solve_corner
    from suspkin.metrics.corner import side_view_ic

    veh = _tilted_vehicle()
    s = vehicle_summary(veh)
    cs = build_corners(veh)
    wb, h = veh.wheelbase, veh.vehicle.cg_height
    bias = veh.vehicle.brake_bias_front
    split = veh.vehicle.effective_drive_split_front

    def slope(cid: str, ref: str, sgn: float) -> float:
        st = solve_corner(cs[cid], 0.0)  # type: ignore[index]
        ic = side_view_ic(st, cs[cid])  # type: ignore[index]
        assert ic.is_finite
        p = st.points[ref]
        return (ic.z - p[2]) / (sgn * (p[0] - ic.h))

    assert veh.front.brakes == veh.rear.brakes == "outboard"
    dive = slope("FL", "contact_patch", 1.0) * bias * wb / h * 100.0
    lift = slope("RL", "contact_patch", -1.0) * (1.0 - bias) * wb / h * 100.0
    squat = slope("RL", "wheel_center", -1.0) * (1.0 - split) * wb / h * 100.0
    assert dive > 1.0 and lift > 1.0 and squat < -1.0  # SVIC below the wheel centre
    assert s.anti_dive_front == pytest.approx(dive, rel=1e-6)
    assert s.anti_lift_rear == pytest.approx(lift, rel=1e-6)
    assert s.anti_squat_rear == pytest.approx(squat, rel=1e-6)
    assert math.isfinite(s.pitch_centre_x) and math.isfinite(s.pitch_centre_z)


def test_rc_migration_independent_of_pivot() -> None:
    a = vehicle_summary(VEHICLE, pivot=(0.0, 0.0, 0.0))
    b = vehicle_summary(VEHICLE)
    for ax in ("front", "rear"):
        ra, rb = getattr(a, ax).rc_migration, getattr(b, ax).rc_migration
        assert ra == pytest.approx(
            rb, abs=1e-3 * 57.3 * 1e-3 * 10
        )  # 0.01 mm/deg; residual is O(roll^2) pose coupling
