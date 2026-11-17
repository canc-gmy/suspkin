"""Tests for instant centres and roll centre."""

from __future__ import annotations

import math

import numpy as np
import pytest

from suspkin.geometry import build_corners, load_vehicle
from suspkin.geometry.corners import Corner
from suspkin.kinematics import solve_corner
from suspkin.kinematics.corner import CornerState
from suspkin.metrics.centres import (
    centre_metrics,
    front_view_ic,
    fvsa_length,
    roll_centre_pair,
    roll_centre_symmetric,
    side_view_ic,
    svsa_length,
)
from tests.unit.helpers import fixture_corners
from tests.unit.test_reference_vehicle import CONFIG, _roll_centre_height

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


def test_reference_vehicle_roll_centre() -> None:
    v = load_vehicle(CONFIG)
    cs = build_corners(v)
    heights = {}
    for cid, axle in (("FL", v.front), ("RL", v.rear)):
        st = solve_corner(cs[cid], 0.0)
        heights[cid] = roll_centre_symmetric(st, cs[cid])
        assert heights[cid] == pytest.approx(_roll_centre_height(axle), abs=1e-4)
    assert heights["FL"] == pytest.approx(0.0337, abs=5e-4)
    assert heights["RL"] == pytest.approx(0.0399, abs=5e-4)


def test_roll_centre_moves_with_travel(corners: dict[str, Corner]) -> None:
    c = corners["FL"]
    h0 = roll_centre_symmetric(solve_corner(c, 0.0), c)
    h1 = roll_centre_symmetric(solve_corner(c, 0.03), c)
    assert h0 != pytest.approx(h1, abs=1e-6)
    assert math.isfinite(h1)
