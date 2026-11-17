"""Tests of the three ARB types: models, stiffness, kinematics, energy, rates."""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from suspkin.geom_math.vectors import mirror_y
from suspkin.geometry import (
    Axle,
    BeamARB,
    BladeARB,
    TorsionalARB,
    Vehicle,
    arb_points,
    build_corners,
    load_vehicle,
    save_vehicle,
)
from suspkin.kinematics import KinematicsError, arb_energy, arb_stiffness, solve_arb, solve_corner
from suspkin.metrics.arb import arb_metrics, arb_rates

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"
TUBE = {
    "bar_length": 0.40,
    "outer_diameter": 0.016,
    "inner_diameter": 0.0,
    "shear_modulus": 7.9e10,
}
ARB_CFG: dict[str, dict[str, Any]] = {
    "torsional": {
        "type": "torsional",
        "arm_end": [-0.04, 0.20, 0.40],
        "bar_pivot": [-0.10, 0.20, 0.40],
        **TUBE,
        "arm_diameter": 0.012,
    },
    "blade": {
        "type": "blade",
        "arm_end": [-0.04, 0.20, 0.40],
        "bar_pivot": [-0.10, 0.20, 0.40],
        **TUBE,
        "blade_width": 0.025,
        "blade_thickness": 0.003,
        "blade_youngs_modulus": 2.0e11,
        "blade_angle": 0.0,
    },
    "beam": {
        "type": "beam",
        "beam_center": [-0.04, 0.0, 0.45],
        "arm_end": [-0.04, 0.2364, 0.45],
        "beam_width": 0.02,
        "beam_thickness": 0.003,
        "beam_youngs_modulus": 2.1e11,
    },
}
TYPES = ["torsional", "blade", "beam"]


def _vehicle(arb: dict[str, Any]) -> Vehicle:
    data = copy.deepcopy(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))
    data["front"]["arb"] = copy.deepcopy(arb)
    return Vehicle.model_validate(data)


def _axle(kind: str, **over: Any) -> Axle:
    return _vehicle({**ARB_CFG[kind], **over}).front


# ---------------------------------------------------------------- models


@pytest.mark.parametrize("kind", TYPES)
def test_yaml_round_trip(kind: str, tmp_path: Path) -> None:
    v = _vehicle(ARB_CFG[kind])
    assert v.front.arb is not None
    assert v.front.arb.type == kind
    save_vehicle(v, tmp_path / "v.yaml")
    assert load_vehicle(tmp_path / "v.yaml") == v


def test_no_flexural_alias() -> None:
    with pytest.raises(ValidationError):
        _vehicle({**ARB_CFG["blade"], "type": "flexural"})


def test_model_types_and_defaults() -> None:
    assert isinstance(_axle("torsional").arb, TorsionalARB)
    assert isinstance(_axle("blade").arb, BladeARB)
    assert isinstance(_axle("beam").arb, BeamARB)
    tors = _axle("torsional").arb
    assert isinstance(tors, TorsionalARB)
    assert tors.arm_youngs_modulus == 200e9
    rigid = TorsionalARB.model_validate({**ARB_CFG["torsional"], "arm_diameter": None})
    assert rigid.arm_diameter is None


@pytest.mark.parametrize(
    ("kind", "over"),
    [
        ("torsional", {"inner_diameter": 0.02}),
        ("torsional", {"bar_length": 0.5}),
        ("torsional", {"arm_end": [-0.04, 0.30, 0.40], "bar_pivot": [-0.04, 0.20, 0.40]}),
        ("torsional", {"arm_diameter": -0.01}),
        ("blade", {"inner_diameter": 0.02}),
        ("blade", {"bar_length": 0.5}),
        ("blade", {"blade_thickness": 0.0}),
        ("beam", {"beam_center": [-0.04, 0.01, 0.45]}),
        ("beam", {"arm_end": [-0.04, 0.005, 0.45]}),
        ("beam", {"beam_width": -1.0}),
        ("beam", {"beam_axis": [0.0, 0.0, 0.0]}),
    ],
)
def test_validators_reject(kind: str, over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _axle(kind, **over)


@pytest.mark.parametrize("kind", TYPES)
def test_arb_requires_rocker_arb(kind: str) -> None:
    data = copy.deepcopy(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))
    data["front"]["arb"] = ARB_CFG[kind]
    del data["front"]["hardpoints"]["rocker_arb"]
    with pytest.raises(ValidationError):
        Vehicle.model_validate(data)


@pytest.mark.parametrize("kind", TYPES)
def test_arb_points_mirror(kind: str) -> None:
    c = build_corners(_vehicle(ARB_CFG[kind]))
    left, right = arb_points(c["FL"]), arb_points(c["FR"])
    assert set(left) == {"arm_end", "beam_center" if kind == "beam" else "bar_pivot"}
    for k in left:
        np.testing.assert_allclose(right[k], mirror_y(left[k]))


# ------------------------------------------------------------- stiffness


def _j(d_out: float, d_in: float) -> float:
    return math.pi / 32.0 * (d_out**4 - d_in**4)


def test_torsional_rigid_arms_is_gj_over_l() -> None:
    k = arb_stiffness(_axle("torsional", arm_diameter=None))
    assert k == pytest.approx(7.9e10 * _j(0.016, 0.0) / 0.40, rel=1e-12)


def test_torsional_bending_arms_in_series() -> None:
    k = arb_stiffness(_axle("torsional"))
    r, dia = 0.06, 0.012
    k_tube = 7.9e10 * _j(0.016, 0.0) / 0.40
    k_arm_twist = 3.0 * 200e9 * (math.pi * dia**4 / 64.0) / r**3 * r**2  # k_tip r^2
    expected = 1.0 / (1.0 / k_tube + 2.0 / k_arm_twist)
    assert k == pytest.approx(expected, rel=1e-12)
    assert k < k_tube


def test_blade_angle_orientation() -> None:
    soft = arb_stiffness(_axle("blade", blade_angle=0.0))
    stiff = arb_stiffness(_axle("blade", blade_angle=math.pi / 2))
    assert stiff > soft
    # blade-dominated limit (very stiff tube): ratio of inertias (w/t)^2
    kw = {"shear_modulus": 1e22}
    ratio = arb_stiffness(_axle("blade", blade_angle=math.pi / 2, **kw)) / arb_stiffness(
        _axle("blade", blade_angle=0.0, **kw)
    )
    assert ratio == pytest.approx((0.025 / 0.003) ** 2, rel=1e-6)
    # blade series formula at angle 0
    r, w, t, e = 0.06, 0.025, 0.003, 2.0e11
    k_arm_twist = 3.0 * e * (w * t**3 / 12.0) / r
    k_tube = 7.9e10 * _j(0.016, 0.0) / 0.40
    assert soft == pytest.approx(1.0 / (1.0 / k_tube + 2.0 / k_arm_twist), rel=1e-12)


def test_blade_45deg_harmonic_inertia() -> None:
    w, t = 0.025, 0.003
    i_eff = 1.0 / (0.5 / (w * t**3 / 12.0) + 0.5 / (t * w**3 / 12.0))
    kw = {"shear_modulus": 1e22}
    k45 = arb_stiffness(_axle("blade", blade_angle=math.pi / 4, **kw))
    assert k45 == pytest.approx(3.0 * 2.0e11 * i_eff / (2.0 * 0.06), rel=1e-6)
    k0 = arb_stiffness(_axle("blade", blade_angle=0.0, **kw))
    assert k45 / k0 == pytest.approx(i_eff / (w * t**3 / 12.0), rel=1e-6)
    assert k45 / k0 < 2.0  # about 2x flat (not 18x as with the Voigt mix)


def test_blade_intermediate_angle_between_extremes() -> None:
    ks = [arb_stiffness(_axle("blade", blade_angle=a)) for a in (0.0, 0.5, 1.0, math.pi / 2)]
    assert ks == sorted(ks)


def test_beam_cantilever_stiffness() -> None:
    a = 0.2364
    k = arb_stiffness(_axle("beam"))
    i_flat = 0.02 * 0.003**3 / 12.0
    assert k == pytest.approx(2.0 * 3.0 * 2.1e11 * i_flat / a**3, rel=1e-12)
    k_edge = arb_stiffness(_axle("beam", beam_angle=math.pi / 2))
    assert k_edge / k == pytest.approx((0.02 / 0.003) ** 2, rel=1e-12)


def test_stiffness_requires_arb() -> None:
    bare = load_vehicle(FIXTURE).front.model_copy(update={"arb": None})
    with pytest.raises(ValueError):
        arb_stiffness(bare)


# ------------------------------------------------------------ kinematics


def _states(kind: str, zl: float, zr: float, **over: Any) -> Any:
    v = _vehicle({**ARB_CFG[kind], **over})
    c = build_corners(v)
    return v.front, solve_corner(c["FL"], zl), solve_corner(c["FR"], zr)


@pytest.mark.parametrize("kind", TYPES)
def test_static_state_is_stress_free(kind: str) -> None:
    axle, left, right = _states(kind, 0.0, 0.0)
    s = solve_arb(axle, left, right)
    assert s.elastic == pytest.approx(0.0, abs=1e-12)
    assert s.beta_left == pytest.approx(0.0, abs=1e-12)
    assert s.max_residual < 1e-12
    assert arb_energy(axle, s) == pytest.approx(0.0, abs=1e-18)


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("z", [0.03, -0.03])
def test_symmetric_heave_no_elastic(kind: str, z: float) -> None:
    axle, left, right = _states(kind, z, z)
    s = solve_arb(axle, left, right)
    assert abs(s.elastic) < 1e-9
    if kind != "beam":
        assert s.beta_left == pytest.approx(s.beta_right, abs=1e-9)
        assert abs(s.beta_left) > 1e-3
    else:
        assert abs(s.heave) > 1e-4


@pytest.mark.parametrize("kind", TYPES)
def test_antisymmetric_travel_odd_and_nonzero(kind: str) -> None:
    axle, l1, r1 = _states(kind, 0.03, -0.03)
    _, l2, r2 = _states(kind, -0.03, 0.03)
    s1, s2 = solve_arb(axle, l1, r1), solve_arb(axle, l2, r2)
    assert abs(s1.elastic) > 1e-3
    assert s1.elastic == pytest.approx(-s2.elastic, rel=1e-6)


@pytest.mark.parametrize("kind", TYPES)
def test_drop_link_length_invariant(kind: str) -> None:
    axle, left, right = _states(kind, 0.025, -0.012)
    s = solve_arb(axle, left, right)
    static = build_corners(_vehicle(ARB_CFG[kind]))
    for state, end in ((left, s.arm_end_left), (right, s.arm_end_right)):
        sc = static[state.corner_id]
        l0 = np.linalg.norm(sc.points["rocker_arb"] - arb_points(sc)["arm_end"])
        assert np.linalg.norm(state.points["rocker_arb"] - end) == pytest.approx(l0, abs=1e-9)
    assert s.max_residual < 1e-9


@pytest.mark.parametrize("kind", ["torsional", "blade"])
def test_arm_end_stays_on_circle(kind: str) -> None:
    axle, left, right = _states(kind, 0.02, -0.02)
    s = solve_arb(axle, left, right)
    arb = axle.arb
    assert isinstance(arb, TorsionalARB | BladeARB)
    pivot = np.array(arb.bar_pivot)
    r0 = np.linalg.norm(np.array(arb.arm_end) - pivot)
    assert np.linalg.norm(s.arm_end_left - pivot) == pytest.approx(r0, abs=1e-12)
    assert s.arm_end_left[1] == pytest.approx(0.20, abs=1e-12)  # stays in the plane
    assert s.arm_end_right[1] == pytest.approx(-0.20, abs=1e-12)


def test_beam_ends_move_vertically() -> None:
    axle, left, right = _states("beam", 0.02, -0.02)
    s = solve_arb(axle, left, right)
    np.testing.assert_allclose(s.arm_end_left[:2], [-0.04, 0.2364])
    np.testing.assert_allclose(s.arm_end_right[:2], [-0.04, -0.2364])
    assert s.arm_end_left[2] - 0.45 == pytest.approx(s.heave + s.elastic)
    assert s.arm_end_right[2] - 0.45 == pytest.approx(s.heave - s.elastic)


def test_guess_selects_same_branch() -> None:
    axle, left, right = _states("torsional", 0.02, -0.02)
    s0 = solve_arb(axle, left, right)
    s1 = solve_arb(axle, left, right, guess=s0)
    assert s1.elastic == pytest.approx(s0.elastic, abs=1e-12)


@pytest.mark.parametrize("kind", TYPES)
def test_unreachable_raises(kind: str) -> None:
    axle, left, right = _states(kind, 0.0, 0.0)
    far = {**left.points, "rocker_arb": np.array([5.0, 5.0, 5.0])}
    bad = dataclasses.replace(left, points=far)
    with pytest.raises(KinematicsError):
        solve_arb(axle, bad, right)


# --------------------------------------------------------------- metrics


@pytest.mark.parametrize("kind", TYPES)
def test_arb_metrics_keys(kind: str) -> None:
    axle, l0, r0 = _states(kind, 0.0, 0.0)
    static = solve_arb(axle, l0, r0)
    _, left, right = _states(kind, 0.02, -0.02)
    m = arb_metrics(axle, left, right, static)
    assert m["arb_elastic"] == pytest.approx(solve_arb(axle, left, right).elastic)
    assert m["arb_energy"] > 0.0
    for side in ("left", "right"):
        assert 0.0 <= m[f"drop_link_angle_{side}"] < math.pi / 2


@pytest.mark.parametrize("kind", TYPES)
def test_rates_relations(kind: str) -> None:
    v = _vehicle(ARB_CFG[kind])
    c = build_corners(v)
    rates = arb_rates(c["FL"], c["FR"], v.front)
    k = arb_stiffness(v.front)
    assert rates.stiffness == k
    assert rates.heave_coupling == pytest.approx(0.0, abs=1e-6)
    # roll mode moves both wheels: twice the elastic travel per unit wheel travel
    assert rates.motion_ratio_roll == pytest.approx(2.0 * rates.motion_ratio_single, rel=1e-3)
    assert rates.wheel_rate_roll == pytest.approx(0.5 * k * rates.motion_ratio_roll**2)
    assert rates.wheel_rate_single == pytest.approx(k * rates.motion_ratio_single**2)
    assert rates.wheel_rate_roll == pytest.approx(2.0 * rates.wheel_rate_single, rel=2e-3)
    assert rates.wheel_rate_roll > 0.0


@pytest.mark.parametrize("kind", TYPES)
def test_energy_consistency_with_wheel_rate(kind: str) -> None:
    v = _vehicle(ARB_CFG[kind])
    c = build_corners(v)
    rates = arb_rates(c["FL"], c["FR"], v.front)
    z = 2e-4
    s = solve_arb(v.front, solve_corner(c["FL"], z), solve_corner(c["FR"], -z))
    # two wheels, each with 1/2 k_w z^2 of work
    assert arb_energy(v.front, s) == pytest.approx(rates.wheel_rate_roll * z * z, rel=1e-3)
