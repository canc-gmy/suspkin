"""Tests for instant centres and roll centre."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from suspkin.geometry import Axle, build_corners, load_vehicle
from suspkin.geometry.corners import Corner
from suspkin.kinematics import solve_corner
from suspkin.kinematics.corner import CornerState
from suspkin.metrics.corner import (
    front_view_ic,
    roll_centre_pair,
    side_view_ic,
    sv_swing_angle,
    sv_wc_angle,
)
from tests.unit._scalar_oracles import (
    centre_metrics,
    fv_swing_angle,
    fvsa_length,
    roll_centre_symmetric,
    svsa_length,
)
from tests.unit.helpers import fixture_corners

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"

Pts = dict[str, tuple[float, float, float]]


def _state(corner: Corner, pts: Pts) -> CornerState:
    return CornerState(
        corner_id=corner.id,
        travel=0.0,
        rack=0.0,
        theta_uca=0.0,
        theta_lca=0.0,
        psi=0.0,
        rocker_angle=0.0,
        points={k: np.array(v, dtype=float) for k, v in pts.items()},
        spin_axis=np.zeros(3),
        coilover_length=0.0,
        toggle_margin=0.0,
        max_residual=0.0,
    )


def _wishbones(uca_y: float, uca_z: float, lca_y: float, lca_z: float) -> Pts:
    """Left corner: chassis pivots at y = 0.2 (z 0.3 / 0.1), outer joints at x = 0."""
    return {
        "uca_front": (0.1, 0.2, 0.3),
        "uca_rear": (-0.1, 0.2, 0.3),
        "uca_outer": (0.0, uca_y, uca_z),
        "lca_front": (0.1, 0.2, 0.1),
        "lca_rear": (-0.1, 0.2, 0.1),
        "lca_outer": (0.0, lca_y, lca_z),
        "contact_patch": (0.0, 0.6, 0.0),
        "wheel_center": (0.0, 0.6, 0.2),
    }


@pytest.fixture(scope="module")
def corners() -> dict[str, Corner]:
    return dict(fixture_corners())


def test_parallel_horizontal_wishbones_infinite(corners: dict[str, Corner]) -> None:
    c = corners["FL"]
    st = _state(c, _wishbones(0.55, 0.3, 0.55, 0.1))
    ic = front_view_ic(st, c)
    assert not ic.is_finite
    assert math.isnan(ic.h) and math.isnan(ic.z)
    assert fvsa_length(st, c) == math.inf
    assert roll_centre_symmetric(st, c) == pytest.approx(0.0, abs=1e-12)
    m = centre_metrics(st, c)
    assert m["fv_swing_angle"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(m["fvic_y"])


def test_converging_wishbones_exact(corners: dict[str, Corner]) -> None:
    c = corners["FL"]
    st = _state(c, _wishbones(0.5, 0.25, 0.55, 0.15))
    # y-z lines: z = 0.3 - s/6 (UCA) and z = 0.1 + s/7 (LCA), s = y - 0.2
    s = 0.2 * 42 / 13
    ic = front_view_ic(st, c)
    assert ic.is_finite
    assert ic.h == pytest.approx(0.2 + s, abs=1e-12)
    assert ic.z == pytest.approx(0.1 + s / 7, abs=1e-12)
    assert fvsa_length(st, c) == pytest.approx(abs(0.2 + s - 0.6), abs=1e-12)
    rc = ic.z / (ic.h - 0.6) * (0.0 - 0.6)
    assert roll_centre_symmetric(st, c) == pytest.approx(rc, abs=1e-12)


def _section(
    a: np.ndarray, b: np.ndarray, o: np.ndarray, y: float
) -> tuple[np.ndarray, np.ndarray]:
    """Independent (x, z) section line of the plane through a, b, o with y = const."""
    u = b - a
    v = o - a
    t = (y - a[1]) / v[1]  # pivot axis has u_y = 0
    base = a + t * v
    return base[[0, 2]], u[[0, 2]]


def test_side_view_anti_geometry(corners: dict[str, Corner]) -> None:
    c = corners["FL"]
    pts: Pts = {
        "uca_front": (0.1, 0.2, 0.30),
        "uca_rear": (-0.1, 0.2, 0.28),
        "uca_outer": (0.0, 0.5, 0.29),
        "lca_front": (0.1, 0.2, 0.10),
        "lca_rear": (-0.1, 0.2, 0.12),
        "lca_outer": (0.0, 0.6, 0.11),
        "contact_patch": (0.0, 0.6, 0.0),
        "wheel_center": (0.0, 0.6, 0.2),
    }
    arr = {k: np.array(v) for k, v in pts.items()}
    p1, d1 = _section(arr["uca_front"], arr["uca_rear"], arr["uca_outer"], 0.6)
    p2, d2 = _section(arr["lca_front"], arr["lca_rear"], arr["lca_outer"], 0.6)
    sol = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
    ref = p1 + sol[0] * d1
    st = _state(c, pts)
    ic = side_view_ic(st, c)
    assert ic.is_finite
    assert ic.h == pytest.approx(ref[0], abs=1e-9)
    assert ic.z == pytest.approx(ref[1], abs=1e-9)
    assert svsa_length(st, c) == pytest.approx(abs(ref[0]), abs=1e-9)


def test_mirror_symmetry(corners: dict[str, Corner]) -> None:
    for fl, fr in (("FL", "FR"), ("RL", "RR")):
        cl, cr = corners[fl], corners[fr]
        for travel in (0.0, 0.02, -0.02):
            sl = solve_corner(cl, travel)
            sr = solve_corner(cr, travel)
            ml, mr = centre_metrics(sl, cl), centre_metrics(sr, cr)
            for k in ml:
                if k == "fvic_y":
                    assert mr[k] == pytest.approx(-ml[k], abs=1e-9)
                else:
                    assert mr[k] == pytest.approx(ml[k], abs=1e-9, nan_ok=True), k
            y, z = roll_centre_pair(sl, sr)
            assert y == pytest.approx(0.0, abs=1e-9)
            assert z == pytest.approx(ml["rc_height"], abs=1e-9)


def _ref_roll_centre_height(axle: Axle) -> float:
    """Independent reference: wishbone-axis (midpoint) line intersection in the y-z plane."""
    h = axle.hardpoints

    def a(v: tuple[float, float, float]) -> np.ndarray:
        return np.array(v, dtype=float)

    uin = ((a(h.uca_front) + a(h.uca_rear)) / 2)[1:]
    lin = ((a(h.lca_front) + a(h.lca_rear)) / 2)[1:]
    d_u = a(h.uca_outer)[1:] - uin
    d_l = a(h.lca_outer)[1:] - lin
    t = np.linalg.solve(np.array([d_u, -d_l]).T, lin - uin)
    ic = uin + t[0] * d_u
    wc = a(h.wheel_center)
    cp_y = wc[1] + wc[2] * math.tan(-axle.wheel.static_camber)
    return float((ic[1]) * (0.0 - cp_y) / (ic[0] - cp_y))


def test_reference_vehicle_roll_centre() -> None:
    v = load_vehicle(CONFIG)
    cs = build_corners(v)
    heights = {}
    for cid, axle in (("FL", v.front), ("RL", v.rear)):
        st = solve_corner(cs[cid], 0.0)
        heights[cid] = roll_centre_symmetric(st, cs[cid])
        assert heights[cid] == pytest.approx(_ref_roll_centre_height(axle), abs=1e-4)
    assert heights["FL"] == pytest.approx(0.0337, abs=5e-4)
    assert heights["RL"] == pytest.approx(0.0399, abs=5e-4)


def test_roll_centre_moves_with_travel(corners: dict[str, Corner]) -> None:
    c = corners["FL"]
    h0 = roll_centre_symmetric(solve_corner(c, 0.0), c)
    h1 = roll_centre_symmetric(solve_corner(c, 0.03), c)
    assert h0 != pytest.approx(h1, abs=1e-6)
    assert math.isfinite(h1)


def _mirror(pts: Pts) -> Pts:
    return {k: (x, -y, z) for k, (x, y, z) in pts.items()}


def test_right_corner_hand_built(corners: dict[str, Corner]) -> None:
    cl, cr = corners["FL"], corners["FR"]
    pts = _wishbones(0.5, 0.25, 0.55, 0.15)
    ml = centre_metrics(_state(cl, pts), cl)
    mr = centre_metrics(_state(cr, _mirror(pts)), cr)
    assert mr["fvic_y"] == pytest.approx(-ml["fvic_y"], abs=1e-12)
    for k in ml:
        if k != "fvic_y":
            assert mr[k] == pytest.approx(ml[k], abs=1e-12, nan_ok=True), k
    assert mr["fv_swing_angle"] > 0


def test_fv_swing_angle_value(corners: dict[str, Corner]) -> None:
    c = corners["FL"]
    s = 0.2 * 42 / 13
    ic_y, ic_z = 0.2 + s, 0.1 + s / 7
    st = _state(c, _wishbones(0.5, 0.25, 0.55, 0.15))
    assert fv_swing_angle(st, c) == pytest.approx(math.atan2(ic_z, 0.6 - ic_y), abs=1e-12)
    assert fv_swing_angle(st, c) > 0


def _side_pts(ic_x: float, ic_z: float) -> Pts:
    """Wishbones whose side-view section lines (y = 0.6) meet at (ic_x, ic_z)."""

    def wb(slope: float, name: str) -> Pts:
        def z_at(x: float) -> float:
            return ic_z + slope * (x - ic_x)

        return {
            f"{name}_front": (0.1, 0.2, z_at(0.1)),
            f"{name}_rear": (-0.1, 0.2, z_at(-0.1)),
            f"{name}_outer": (0.0, 0.6, z_at(0.0)),
        }

    pts: Pts = {}
    pts.update(wb(0.2, "uca"))
    pts.update(wb(-0.2, "lca"))
    pts["contact_patch"] = (0.0, 0.6, 0.0)
    pts["wheel_center"] = (0.0, 0.6, 0.2)
    return pts


@pytest.mark.parametrize("cid", ["FL", "RL", "FR", "RR"])
@pytest.mark.parametrize("ic_x", [-0.8, 0.8])
def test_side_view_angle_signs(corners: dict[str, Corner], cid: str, ic_x: float) -> None:
    c = corners[cid]
    ic_z = 0.3
    pts = _side_pts(ic_x, ic_z)
    st = _state(c, pts if c.side == "left" else _mirror(pts))
    ic = side_view_ic(st, c)
    assert ic.h == pytest.approx(ic_x, abs=1e-9)
    assert ic.z == pytest.approx(ic_z, abs=1e-9)
    s = 1.0 if c.axle == "front" else -1.0
    assert sv_swing_angle(st, c) == pytest.approx(math.atan2(ic_z, s * (0.0 - ic_x)), abs=1e-9)
    assert sv_wc_angle(st, c) == pytest.approx(math.atan2(ic_z - 0.2, s * (0.0 - ic_x)), abs=1e-9)
    anti = (c.axle == "front" and ic_x < 0) or (c.axle == "rear" and ic_x > 0)
    assert (abs(sv_swing_angle(st, c)) < math.pi / 2) == anti
    assert sv_swing_angle(st, c) > 0


@pytest.mark.parametrize("cid", ["FL", "RL"])
def test_side_view_infinite_orientation(corners: dict[str, Corner], cid: str) -> None:
    c = corners[cid]
    # parallel wishbones rising rearward with slope 0.1 (dz/d(-x))
    pts: Pts = {
        "uca_front": (0.1, 0.2, 0.29),
        "uca_rear": (-0.1, 0.2, 0.31),
        "uca_outer": (0.0, 0.6, 0.30),
        "lca_front": (0.1, 0.2, 0.09),
        "lca_rear": (-0.1, 0.2, 0.11),
        "lca_outer": (0.0, 0.6, 0.10),
        "contact_patch": (0.0, 0.6, 0.0),
        "wheel_center": (0.0, 0.6, 0.2),
    }
    st = _state(c, pts)
    assert not side_view_ic(st, c).is_finite
    # front: anti side is rearward (line rises): +atan(0.1); rear: forward (falls): -atan(0.1)
    exp = math.atan(0.1) if cid == "FL" else -math.atan(0.1)
    assert sv_swing_angle(st, c) == pytest.approx(exp, abs=1e-9)
    assert sv_wc_angle(st, c) == pytest.approx(exp, abs=1e-9)


def test_fv_infinite_orientation(corners: dict[str, Corner]) -> None:
    # parallel wishbones rising inboard (toward smaller y on the left corner), slope 0.1
    pts = _wishbones(0.6, 0.3 - 0.4 * 0.1, 0.6, 0.1 - 0.4 * 0.1)
    for cid, p in (("FL", pts), ("FR", _mirror(pts))):
        c = corners[cid]
        st = _state(c, p)
        assert not front_view_ic(st, c).is_finite
        assert fv_swing_angle(st, c) == pytest.approx(math.atan(0.1), abs=1e-9)


def test_degenerate_plane_gives_non_finite(corners: dict[str, Corner]) -> None:
    c = corners["FL"]
    pts = _wishbones(0.5, 0.25, 0.55, 0.15)
    pts["uca_outer"] = (0.0, 0.2, 0.3)  # collinear with the UCA pivot axis? no: off-axis in x
    pts["uca_outer"] = (0.05, 0.2, 0.3)  # lies on the pivot axis -> collinear
    st = _state(c, pts)
    assert not front_view_ic(st, c).is_finite
    m = centre_metrics(st, c)
    assert math.isnan(m["fvic_y"]) and math.isnan(m["rc_height"])


def test_roll_centre_pair_asymmetric(corners: dict[str, Corner]) -> None:
    cl, cr = corners["FL"], corners["FR"]
    sl = solve_corner(cl, 0.02)
    sr = solve_corner(cr, -0.02)
    y, z = roll_centre_pair(sl, sr)
    assert math.isfinite(y) and math.isfinite(z)
    for st, c in ((sl, cl), (sr, cr)):
        cp = st.points["contact_patch"]
        ic = front_view_ic(st, c)
        cross = (ic.h - cp[1]) * (z - cp[2]) - (ic.z - cp[2]) * (y - cp[1])
        assert cross == pytest.approx(0.0, abs=1e-9)
    assert abs(y) > 1e-4


def test_roll_centre_pair_infinite_fvic(corners: dict[str, Corner]) -> None:
    cl, cr = corners["FL"], corners["FR"]
    pts = _wishbones(0.55, 0.3, 0.55, 0.1)
    left, right = _state(cl, pts), _state(cr, _mirror(pts))
    assert roll_centre_pair(left, right) == (0.0, 0.0)
    sl = _wishbones(0.6, 0.26, 0.6, 0.06)
    y, z = roll_centre_pair(_state(cl, sl), _state(cr, _mirror(sl)))
    assert y == pytest.approx(0.0, abs=1e-12)
    assert z == pytest.approx(roll_centre_symmetric(_state(cl, sl), cl), abs=1e-12)
