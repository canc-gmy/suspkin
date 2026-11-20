"""Tests for wheel alignment and wheel position metrics."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from suspkin.geometry import build_corners, load_vehicle
from suspkin.geometry.corners import Corner
from suspkin.kinematics.corner import CornerState, solve_corner, sweep_rack
from tests.unit import _scalar_oracles as al
from tests.unit.helpers import fixture_corners

IDS = ("FL", "FR", "RL", "RR")
REF_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"
REF_CORNERS = build_corners(load_vehicle(REF_CONFIG))
FIX_CORNERS = fixture_corners()


def _hand_state(corner: Corner, lbj: Any, ubj: Any, cp: Any, wc: Any) -> CornerState:
    """Static state of ``corner`` with the given (left-frame) points, mirrored for right."""
    base = solve_corner(corner, 0.0)
    sgn = 1.0 if corner.side == "left" else -1.0
    pts = {k: np.array(v) for k, v in base.points.items()}
    for name, p in (
        ("lca_outer", lbj),
        ("uca_outer", ubj),
        ("contact_patch", cp),
        ("wheel_center", wc),
    ):
        pts[name] = np.array(p, dtype=float) * np.array([1.0, sgn, 1.0])
    return replace(base, points=pts)


def _tilted(corner: Corner, caster_a: float, kpi_a: float) -> CornerState:
    """Steering axis through a LBJ at z=0.1, length 0.2 m, tilted by the given angles."""
    length = 0.2
    # top rearward: -x; top inboard (left corner): -y
    u = np.array([-math.sin(caster_a), -math.sin(kpi_a), 0.0])
    u[2] = math.sqrt(1.0 - u[0] ** 2 - u[1] ** 2)
    lbj = np.array([0.1, 0.6, 0.1])
    return _hand_state(corner, lbj, lbj + length * u, [0.1, 0.6, 0.0], [0.1, 0.6, 0.2])


@pytest.mark.parametrize("cid", IDS)
def test_static_camber_toe_match_yaml(cid: str) -> None:
    for corners in (FIX_CORNERS, REF_CORNERS):
        c = corners[cid]  # type: ignore[index]
        s = al.static_state(c)
        assert al.camber(s, c) == pytest.approx(c.axle_cfg.wheel.static_camber, abs=1e-12)
        assert al.toe(s, c) == pytest.approx(c.axle_cfg.wheel.static_toe, abs=1e-12)


@pytest.mark.parametrize("cid", ["FL", "FR"])
def test_static_reference_matches_temporary_helpers(cid: str) -> None:
    v = load_vehicle(REF_CONFIG)
    c = REF_CORNERS[cid]  # type: ignore[index]
    s = al.static_state(c)
    h = v.front.hardpoints
    lb, ub = np.array(h.lca_outer), np.array(h.uca_outer)
    u = ub - lb
    caster_h, kpi_h = math.atan2(-u[0], u[2]), math.atan2(-u[1], u[2])
    g = lb - lb[2] / u[2] * u
    wc = np.array(h.wheel_center)
    cp = np.array([wc[0], wc[1] + wc[2] * math.tan(-v.front.wheel.static_camber), 0.0])
    scrub_h, trail_h = float(cp[1] - g[1]), float(g[0] - cp[0])
    assert al.caster(s, c) == pytest.approx(caster_h, abs=1e-6)
    assert al.kpi(s, c) == pytest.approx(kpi_h, abs=1e-6)
    # The helper's contact patch is a small-angle approximation (tan on wheel-centre
    # height); the solver uses the exact wheel_center - R * up, hence the looser bound.
    assert al.scrub_radius(s, c) == pytest.approx(scrub_h, abs=1e-5)
    assert al.mechanical_trail(s, c) == pytest.approx(trail_h, abs=1e-5)


@pytest.mark.parametrize("cid", IDS)
def test_vertical_axis_over_contact_patch_is_zero(cid: str) -> None:
    c = FIX_CORNERS[cid]  # type: ignore[index]
    s = _tilted(c, 0.0, 0.0)
    assert al.caster(s, c) == pytest.approx(0.0, abs=1e-12)
    assert al.kpi(s, c) == pytest.approx(0.0, abs=1e-12)
    assert al.scrub_radius(s, c) == pytest.approx(0.0, abs=1e-12)
    assert al.mechanical_trail(s, c) == pytest.approx(0.0, abs=1e-12)
    assert al.kingpin_offset_wc(s, c) == pytest.approx(0.0, abs=1e-12)
    assert al.spindle_length(s, c) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("cid", IDS)
def test_tilted_axis_known_angles(cid: str) -> None:
    c = FIX_CORNERS[cid]  # type: ignore[index]
    ca, ka = math.radians(5.0), math.radians(7.0)
    s = _tilted(c, ca, ka)
    u = s.points["uca_outer"] - s.points["lca_outer"]
    # Angles of the projections: tan(caster) = -ux/uz, tan(kpi) = inboard/uz.
    exp_caster = math.atan2(-u[0], u[2])
    assert al.caster(s, c) == pytest.approx(exp_caster, abs=1e-12)
    assert al.caster(s, c) > 0.0
    assert al.kpi(s, c) > 0.0
    assert al.kpi(s, c) == pytest.approx(math.atan2(math.sin(ka), u[2] / 0.2), abs=1e-12)
    # Intercept 0.1 m below the LBJ: shifts by 0.1*tan(angle).
    assert al.mechanical_trail(s, c) == pytest.approx(0.1 * math.tan(al.caster(s, c)), abs=1e-12)
    # Top inboard moves the ground intercept outboard -> negative scrub here.
    assert al.scrub_radius(s, c) == pytest.approx(-0.1 * math.tan(al.kpi(s, c)), abs=1e-12)
    # Wheel centre at z=0.2, axis there is 0.1 above LBJ: inboard by 0.1*tan(kpi).
    assert al.kingpin_offset_wc(s, c) == pytest.approx(0.1 * math.tan(al.kpi(s, c)), abs=1e-12)
    uu = u / np.linalg.norm(u)
    r = s.points["wheel_center"] - s.points["lca_outer"]
    assert al.spindle_length(s, c) == pytest.approx(float(np.linalg.norm(np.cross(r, uu))))


def test_mirrored_hand_states_are_equal() -> None:
    sl = _tilted(FIX_CORNERS["FL"], 0.08, 0.1)
    sr = _tilted(FIX_CORNERS["FR"], 0.08, 0.1)
    for f in (al.caster, al.kpi, al.scrub_radius, al.mechanical_trail, al.kingpin_offset_wc):
        assert f(sl, FIX_CORNERS["FL"]) == pytest.approx(f(sr, FIX_CORNERS["FR"]), abs=1e-12)


@pytest.mark.parametrize("corners", [FIX_CORNERS, REF_CORNERS])
@pytest.mark.parametrize("travel", [0.0, 0.02, -0.02])
def test_left_right_symmetry(corners: dict, travel: float) -> None:  # type: ignore[type-arg]
    for left, right in (("FL", "FR"), ("RL", "RR")):
        cl, cr = corners[left], corners[right]
        sl, sr = solve_corner(cl, travel), solve_corner(cr, travel)
        for f in (
            al.camber,
            al.toe,
            al.caster,
            al.kpi,
            al.scrub_radius,
            al.mechanical_trail,
            al.kingpin_offset_wc,
            al.spindle_length,
            al.half_track,
        ):
            assert f(sl, cl) == pytest.approx(f(sr, cr), abs=1e-9), f.__name__


def test_wheel_position_metrics_signs() -> None:
    c = REF_CORNERS["FL"]
    s0 = al.static_state(c)
    assert al.half_track_change(s0, s0, c) == 0.0
    assert al.contact_patch_dx(s0, s0, c) == 0.0
    s = solve_corner(c, 0.02)
    p0, p1 = s0.points, s.points
    assert al.half_track_change(s, s0, c) == pytest.approx(
        abs(p1["contact_patch"][1]) - abs(p0["contact_patch"][1])
    )
    assert al.contact_patch_dy(s, s0, c) == pytest.approx(
        p1["contact_patch"][1] - p0["contact_patch"][1]
    )
    assert al.wheel_center_recession(s, s0, c) == pytest.approx(
        p1["wheel_center"][0] - p0["wheel_center"][0]
    )
    assert al.wheelbase_change_contribution(s, s0, c) == pytest.approx(
        p1["contact_patch"][0] - p0["contact_patch"][0]
    )
    # Outboard-positive dy is mirrored on the right side.
    cr = REF_CORNERS["FR"]
    sr0, sr = al.static_state(cr), solve_corner(cr, 0.02)
    assert al.contact_patch_dy(sr, sr0, cr) == pytest.approx(
        al.contact_patch_dy(s, s0, c), abs=1e-9
    )
    assert al.half_track_change(sr, sr0, cr) == pytest.approx(
        al.half_track_change(s, s0, c), abs=1e-9
    )


@pytest.mark.parametrize("corners", [FIX_CORNERS, REF_CORNERS])
def test_rack_sweep_steer_sign(corners: dict) -> None:  # type: ignore[type-arg]
    rack = 0.01
    out = {}
    for cid in ("FL", "FR"):
        c = corners[cid]
        s0 = al.static_state(c)
        s = sweep_rack(c, [rack])[0]
        d_steer = al.steer_angle(s, c) - al.steer_angle(s0, c)
        d_toe = al.toe(s, c) - al.toe(s0, c)
        out[cid] = (d_steer, d_toe)
        # steer is -toe on the left, +toe on the right
        sign = -1.0 if c.side == "left" else 1.0
        assert al.steer_angle(s, c) == pytest.approx(sign * al.toe(s, c), abs=1e-12)
    # Rack moves along +/-y for both wheels: both wheels steer the same direction.
    assert out["FL"][0] * out["FR"][0] > 0.0
    assert abs(out["FL"][0]) > 1e-3
    # Same steer direction means opposite toe change on the two sides.
    assert out["FL"][1] * out["FR"][1] < 0.0
    # Positive rack along rack_axis: sign of steer must flip with rack sign.
    cl = corners["FL"]
    neg = sweep_rack(cl, [-rack])[0]
    assert al.steer_angle(neg, cl) * (out["FL"][0] + al.steer_angle(al.static_state(cl), cl)) < 0


def test_alignment_metrics_dict() -> None:
    c = REF_CORNERS["FL"]
    s0 = al.static_state(c)
    s = solve_corner(c, 0.01)
    m = al.alignment_metrics(s, c, s0)
    assert set(m) == {
        "camber",
        "toe",
        "steer_angle",
        "caster",
        "kpi",
        "scrub_radius",
        "mechanical_trail",
        "kingpin_offset_wc",
        "spindle_length",
        "half_track",
        "half_track_change",
        "contact_patch_dx",
        "contact_patch_dy",
        "wheel_center_recession",
        "wheelbase_change_contribution",
    }
    assert m["camber"] == al.camber(s, c)
    assert all(isinstance(v, float) for v in m.values())


def test_rear_wheelbase_contribution_sign() -> None:
    cr = REF_CORNERS["RL"]
    s0, s = al.static_state(cr), solve_corner(cr, 0.02)
    dx = al.contact_patch_dx(s, s0, cr)
    assert dx != 0.0
    assert al.wheelbase_change_contribution(s, s0, cr) == pytest.approx(-dx)
    cf = REF_CORNERS["FL"]
    f0, f = al.static_state(cf), solve_corner(cf, 0.02)
    assert al.wheelbase_change_contribution(f, f0, cf) == pytest.approx(
        al.contact_patch_dx(f, f0, cf)
    )


@pytest.mark.parametrize("corners", [FIX_CORNERS, REF_CORNERS])
@pytest.mark.parametrize("cid", ["FL", "FR"])
def test_positive_rack_steers_left(corners: dict, cid: str) -> None:  # type: ignore[type-arg]
    c = corners[cid]
    s0 = al.static_state(c)
    s = sweep_rack(c, [0.01])[0]
    assert al.steer_angle(s, c) - al.steer_angle(s0, c) > 0.0
    assert al.steer_angle(s, c) > 0.0


@pytest.mark.parametrize("cid", ["FL", "FR"])
def test_hand_built_spin_axis(cid: str) -> None:
    c = FIX_CORNERS[cid]  # type: ignore[index]
    o = 1.0 if c.side == "left" else -1.0
    cam, toe_a = math.radians(-3.0), math.radians(2.0)
    # Outboard spin axis: n_z = -sin(camber) > 0 for negative camber; n_x = sin(toe)cos(camber)
    # (toe-in: wheel front toward the centreline).
    n = np.array(
        [math.sin(toe_a) * math.cos(cam), o * math.cos(toe_a) * math.cos(cam), -math.sin(cam)]
    )
    assert n[2] > 0.0 and n[0] > 0.0
    s = replace(al.static_state(c), spin_axis=n)
    assert al.camber(s, c) == pytest.approx(cam, abs=1e-12)
    assert al.camber(s, c) < 0.0
    assert al.toe(s, c) == pytest.approx(toe_a, abs=1e-12)
    assert al.toe(s, c) > 0.0
    expected_steer = -toe_a if c.side == "left" else toe_a
    assert al.steer_angle(s, c) == pytest.approx(expected_steer, abs=1e-12)
