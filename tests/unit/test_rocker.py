"""Tests for the closed-form rocker solver."""

import numpy as np
import pytest

from suspkin.geom_math import mirror_y
from suspkin.geom_math.rotations import axis_angle_matrix
from suspkin.geometry.corners import CornerId
from suspkin.kinematics import KinematicsError, solve_rocker
from tests.unit.helpers import good_corners

CORNERS = good_corners()
ALL: tuple[CornerId, ...] = ("FL", "FR", "RL", "RR")


def _moved_pushrod_outer(cid: CornerId, angle: float) -> np.ndarray:
    """A pushrod outer point reachable at the given (reported) rocker angle."""
    c = CORNERS[cid]
    sgn = 1.0 if c.side == "left" else -1.0
    piv = c.points["rocker_pivot"]
    inner = piv + axis_angle_matrix(c.rocker_axis, sgn * angle) @ (c.points["pushrod_inner"] - piv)
    return inner + (c.points["pushrod_outer"] - c.points["pushrod_inner"])


@pytest.mark.parametrize("cid", ALL)
def test_static_is_zero_angle(cid: CornerId) -> None:
    c = CORNERS[cid]
    r = solve_rocker(c, c.points["pushrod_outer"])
    assert r.angle == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(r.pushrod_inner, c.points["pushrod_inner"], atol=1e-12)
    assert r.toggle_margin == pytest.approx(np.pi / 2, abs=0.1)


@pytest.mark.parametrize("cid", ALL)
@pytest.mark.parametrize("angle", [-0.3, 0.1, 0.4])
def test_recovers_angle_and_keeps_lengths(cid: CornerId, angle: float) -> None:
    c = CORNERS[cid]
    po = _moved_pushrod_outer(cid, angle)
    r = solve_rocker(c, po, guess=angle + 0.02)
    assert r.angle == pytest.approx(angle, abs=1e-10)
    static_len = np.linalg.norm(c.points["pushrod_outer"] - c.points["pushrod_inner"])
    assert np.linalg.norm(po - r.pushrod_inner) == pytest.approx(static_len, abs=1e-12)
    piv = c.points["rocker_pivot"]
    for name, p in (("pushrod_inner", r.pushrod_inner), ("rocker_spring", r.rocker_spring)):
        assert np.linalg.norm(p - piv) == pytest.approx(
            np.linalg.norm(c.points[name] - piv), abs=1e-12
        )


def test_left_right_equal_angles_for_mirrored_input() -> None:
    po = _moved_pushrod_outer("FL", 0.25)
    a = solve_rocker(CORNERS["FL"], po)
    b = solve_rocker(CORNERS["FR"], mirror_y(po))
    assert a.angle == pytest.approx(b.angle, abs=1e-12)
    np.testing.assert_allclose(b.rocker_spring, mirror_y(a.rocker_spring), atol=1e-12)
    assert a.toggle_margin == pytest.approx(b.toggle_margin)


def test_picks_root_closest_to_guess() -> None:
    c = CORNERS["FL"]
    po = _moved_pushrod_outer("FL", 0.2)
    static_len = np.linalg.norm(c.points["pushrod_outer"] - c.points["pushrod_inner"])
    roots = set()
    for g in np.linspace(-np.pi, np.pi, 25):
        r = solve_rocker(c, po, guess=float(g))
        assert np.linalg.norm(po - r.pushrod_inner) == pytest.approx(static_len, abs=1e-12)
        roots.add(round(r.angle % (2 * np.pi), 6))
    assert len(roots) == 2  # two circle intersections
    assert solve_rocker(c, po, guess=0.19).angle == pytest.approx(0.2, abs=1e-10)


def test_unreachable_raises() -> None:
    c = CORNERS["FL"]
    with pytest.raises(KinematicsError):
        solve_rocker(c, c.points["pushrod_outer"] + np.array([0.0, 0.0, 0.5]))
