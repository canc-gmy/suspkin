import math
from pathlib import Path

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.metrics import vehicle as vm

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"
G = vm.G


# ----------------------------------------------------------------- anti-features
def test_anti_hand_numbers() -> None:
    ang = math.atan(0.1)
    # tan = 0.1, bias 0.6, L 1.55, h 0.28 -> 0.1 * 0.6 * 1.55 / 0.28 * 100
    assert vm.anti_dive_front(ang, 0.28, 1.55, 0.6) == pytest.approx(33.2142857)
    assert vm.anti_lift_rear(ang, 0.28, 1.55, 0.6) == pytest.approx(22.1428571)
    assert vm.anti_squat_rear(ang, 0.28, 1.55, 0.0) == pytest.approx(55.3571429)
    assert vm.anti_squat_rear(ang, 0.28, 1.55, 0.5) == pytest.approx(27.6785714)
    assert vm.anti_lift_front_accel(ang, 0.28, 1.55, 1.0) == pytest.approx(55.3571429)
    assert vm.anti_lift_front_accel(ang, 0.28, 1.55, 0.0) == 0.0


def test_anti_zero_for_horizontal_line() -> None:
    assert vm.anti_dive_front(0.0, 0.28, 1.55, 0.6) == 0.0
    assert vm.anti_lift_rear(0.0, 0.28, 1.55, 0.6) == 0.0
    assert vm.anti_squat_rear(0.0, 0.28, 1.55, 0.0) == 0.0


def test_full_anti_dive_line_through_cg_height_at_other_axle() -> None:
    # tan = h / (bias * L) is 100 % anti-dive.
    h, big_l, bias = 0.28, 1.55, 0.6
    ang = math.atan(h / (bias * big_l))
    assert vm.anti_dive_front(ang, h, big_l, bias) == pytest.approx(100.0)
    ang_r = math.atan(h / ((1 - bias) * big_l))
    assert vm.anti_lift_rear(ang_r, h, big_l, bias) == pytest.approx(100.0)
    assert vm.anti_squat_rear(math.atan(h / big_l), h, big_l, 0.0) == pytest.approx(100.0)


def test_anti_angle_for_selection() -> None:
    v = load_vehicle(FIXTURE)
    out = v.front.model_copy(update={"brakes": "outboard"})
    inb = v.front.model_copy(update={"brakes": "inboard"})
    assert vm.anti_angle_for(out, "brake", 0.1, 0.2) == 0.1
    assert vm.anti_angle_for(inb, "brake", 0.1, 0.2) == 0.2
    assert vm.anti_angle_for(out, "drive", 0.1, 0.2) == 0.2
    assert vm.anti_angle_for(inb, "drive", 0.1, 0.2) == 0.2


# ----------------------------------------------------------------- roll
def test_roll_stiffness_components() -> None:
    assert vm.roll_stiffness_springs(30000.0, 1.2) == pytest.approx(21600.0)
    assert vm.roll_stiffness_arb(10000.0, 1.2) == pytest.approx(7200.0)
    assert vm.roll_stiffness_tyres(80000.0, 1.2) == pytest.approx(57600.0)
    # (20000) in series with (80000) = 16000
    assert vm.axle_roll_stiffness(15000.0, 5000.0, 80000.0) == pytest.approx(16000.0)


def test_roll_gradient_hand_number_and_monotonic() -> None:
    # m_s g d = 250 * 9.80665 * 0.23 = 563.882375; K = 3000 -> 563.88 / 2436.12
    g1 = vm.roll_gradient(250.0, 0.28, 0.05, 1500.0, 1500.0)
    assert g1 == pytest.approx(0.231470, rel=1e-4)
    assert math.degrees(g1) == pytest.approx(13.262, rel=1e-3)  # deg per g
    g2 = vm.roll_gradient(250.0, 0.28, 0.05, 3000.0, 3000.0)
    assert g2 < g1
    # Destabilising term: gradient exceeds the naive m g d / K.
    assert g1 > 250.0 * G * 0.23 / 3000.0


def test_roll_gradient_unstable_raises() -> None:
    with pytest.raises(ValueError):
        vm.roll_gradient(250.0, 0.28, 0.05, 200.0, 200.0)


def test_roll_axis_height_interpolation() -> None:
    assert vm.roll_axis_height_at(0.0, 0.0, 0.04, -1.5, 0.10) == pytest.approx(0.04)
    assert vm.roll_axis_height_at(-1.5, 0.0, 0.04, -1.5, 0.10) == pytest.approx(0.10)
    assert vm.roll_axis_height_at(-0.75, 0.0, 0.04, -1.5, 0.10) == pytest.approx(0.07)


def _llt(**kw: float) -> vm.LoadTransfer:
    args = {
        "ay": G,
        "m_sprung": 250.0,
        "m_unsprung_front": 22.0,
        "m_unsprung_rear": 24.0,
        "h_cg": 0.30,
        "rc_front": 0.03,
        "rc_rear": 0.07,
        "hu_front": 0.2,
        "hu_rear": 0.2,
        "k_front": 1500.0,
        "k_rear": 1800.0,
        "track_front": 1.2,
        "track_rear": 1.16,
        "wheelbase": 1.55,
        "a": 0.82,
    }
    args.update(kw)
    return vm.lateral_load_transfer(**args)


def test_llt_components_hand_numbers() -> None:
    r = _llt()
    h_ra = 0.03 + (0.82 / 1.55) * 0.04
    d = 0.30 - h_ra
    phi = 250.0 * G * d / (1500.0 + 1800.0 - 250.0 * G * d)
    assert r.front_elastic == pytest.approx(1500.0 * phi / 1.2)
    assert r.rear_elastic == pytest.approx(1800.0 * phi / 1.16)
    assert r.front_geometric == pytest.approx(250.0 * (0.73 / 1.55) * G * 0.03 / 1.2)
    assert r.rear_geometric == pytest.approx(250.0 * (0.82 / 1.55) * G * 0.07 / 1.16)
    assert r.front_unsprung == pytest.approx(22.0 * G * 0.2 / 1.2)
    assert r.front == pytest.approx(r.front_elastic + r.front_geometric + r.front_unsprung)


def test_llt_moment_balance_and_tlltd() -> None:
    r = _llt()
    assert 0.0 < r.tlltd_front < 1.0
    assert r.tlltd_front + vm.tlltd_front(r.rear, r.front) == pytest.approx(1.0)
    # sum(dW * t) = (K_f + K_r) phi + geometric moments + unsprung moments.
    tot = r.front * 1.2 + r.rear * 1.16
    h_ra = 0.03 + (0.82 / 1.55) * 0.04
    d = 0.30 - h_ra
    phi = 250.0 * G * d / (3300.0 - 250.0 * G * d)
    expect = (
        3300.0 * phi
        + 250.0 * G * (0.73 / 1.55 * 0.03 + 0.82 / 1.55 * 0.07)
        + G * 0.2 * (22.0 + 24.0)
    )
    assert tot == pytest.approx(expect)


def test_tlltd_symmetric_case_and_stiffness_bias() -> None:
    base = {
        "rc_front": 0.0,
        "rc_rear": 0.0,
        "m_unsprung_front": 0.0,
        "m_unsprung_rear": 0.0,
        "track_front": 1.2,
        "track_rear": 1.2,
    }
    sym = _llt(**base, k_front=2000.0, k_rear=2000.0)
    assert sym.tlltd_front == pytest.approx(0.5)
    stiff = _llt(**base, k_front=3000.0, k_rear=1000.0)
    assert stiff.tlltd_front == pytest.approx(0.75)
    assert math.isnan(vm.tlltd_front(0.0, 0.0))


def test_sprung_mass() -> None:
    v = load_vehicle(FIXTURE)
    assert vm.sprung_mass(v) == pytest.approx(280.0 - 2 * (11.0 + 12.0))


# ----------------------------------------------------------------- steering
def _ideal_inner(do: float, big_l: float, t: float) -> float:
    return math.atan(1.0 / (1.0 / math.tan(do) - t / big_l))


def test_ackermann_ideal_parallel_anti() -> None:
    big_l, t, do = 1.55, 1.2, 0.2
    di = _ideal_inner(do, big_l, t)
    assert vm.ackermann_percent(di, do, big_l, t) == pytest.approx(100.0)
    assert vm.ackermann_percent(do, do, big_l, t) == pytest.approx(0.0)
    assert vm.ackermann_percent(do - 0.02, do, big_l, t) < 0.0
    assert vm.ackermann_percent(0.5 * (di + do), do, big_l, t) == pytest.approx(50.0)


def test_ackermann_right_turn_mirrored_and_nan() -> None:
    big_l, t, do = 1.55, 1.2, 0.2
    di = _ideal_inner(do, big_l, t)
    assert vm.ackermann_percent(-di, -do, big_l, t) == pytest.approx(100.0)
    assert math.isnan(vm.ackermann_percent(1e-5, 1e-5, big_l, t))


def test_inner_outer() -> None:
    assert vm.inner_outer(0.3, 0.2) == (0.3, 0.2)  # left turn: left inner
    assert vm.inner_outer(-0.2, -0.3) == (-0.3, -0.2)  # right turn: right inner
    assert vm.inner_outer(0.0, 0.0) == (0.0, 0.0)


def test_steering_ratio_and_turning_radius() -> None:
    # 0.01 m / 0.06 m/rev * 2 pi = 1.047198 rad; / 0.05 = 20.944
    assert vm.steering_ratio(0.01, 0.06, 0.05) == pytest.approx(20.9440, rel=1e-5)
    assert math.isnan(vm.steering_ratio(0.0, 0.06, 0.0))
    assert vm.turning_radius(1.55, 0.12, 0.08) == pytest.approx(1.55 / math.tan(0.1))
    assert vm.turning_radius(1.55, 0.0, 0.0) == math.inf


# ----------------------------------------------------------------- pitch centre
def test_pitch_centre() -> None:
    pt = vm.pitch_centre(
        (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([2.0, 0.0]), np.array([-1.0, 1.0])),
    )
    assert pt == (pytest.approx(1.0), pytest.approx(1.0))
    par = vm.pitch_centre(
        (np.array([0.0, 0.0]), np.array([1.0, 0.0])),
        (np.array([0.0, 1.0]), np.array([2.0, 0.0])),
    )
    assert math.isnan(par[0]) and math.isnan(par[1])
