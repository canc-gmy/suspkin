"""Hexahedron map and signed distances."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from suspkin.optimize.regions import (
    _CORNERS,
    check_hexahedron,
    hex_inverse,
    hex_jacobian,
    hex_map,
    is_convex_hexahedron,
    sd_box,
    sd_cylinder,
    sd_hexahedron,
    sd_sphere,
)

_PERTURB = np.zeros((8, 3))
_PERTURB[1] = [0.1, 0.05, 0.0]
_PERTURB[2] = [0.0, 0.2, 0.05]
_PERTURB[4] = [0.0, 0.0, 0.1]
_PERTURB[5] = [0.2, 0.0, 0.0]
SKEWED = _CORNERS * np.array([1.0, 2.0, 0.5]) + np.array([0.1, 0.2, 0.3]) + _PERTURB
unit = st.floats(0.0, 1.0)


def test_map_corners_and_centre() -> None:
    assert np.allclose(hex_map(SKEWED, _CORNERS), SKEWED)
    assert np.allclose(hex_map(SKEWED, [0.5] * 3), hex_map(SKEWED, np.full(3, 0.5)))


def test_jacobian_matches_finite_difference() -> None:
    uvw = np.array([0.3, 0.6, 0.2])
    jac = hex_jacobian(SKEWED, uvw)
    for k in range(3):
        d = np.zeros(3)
        d[k] = 1e-6
        fd = (hex_map(SKEWED, uvw + d) - hex_map(SKEWED, uvw - d)) / 2e-6
        assert np.allclose(jac[:, k], fd, atol=1e-7)


@settings(deadline=None, max_examples=60)
@given(unit, unit, unit)
def test_roundtrip(u: float, v: float, w: float) -> None:
    uvw = np.array([u, v, w])
    assert np.allclose(hex_inverse(SKEWED, hex_map(SKEWED, uvw)), uvw, atol=1e-7)


def test_inverse_outside_raises() -> None:
    with pytest.raises(ValueError, match="outside"):
        hex_inverse(SKEWED, hex_map(SKEWED, [1.5, 0.5, 0.5]))


def test_valid_and_twisted() -> None:
    check_hexahedron(SKEWED)
    bad = SKEWED.copy()
    bad[[2, 3]] = bad[[3, 2]]
    with pytest.raises(ValueError, match="degenerate or twisted"):
        check_hexahedron(bad)
    flipped = SKEWED[[4, 5, 6, 7, 0, 1, 2, 3]]
    with pytest.raises(ValueError, match="degenerate or twisted"):
        check_hexahedron(flipped)


def test_box_distance() -> None:
    lo, hi = [0, 0, 0], [2, 2, 2]
    assert sd_box([1, 1, 1], lo, hi) == pytest.approx(-1.0)
    assert sd_box([3, 1, 1], lo, hi) == pytest.approx(1.0)
    assert sd_box([3, 3, 1], lo, hi) == pytest.approx(np.sqrt(2.0))
    assert sd_box([2, 1, 1], lo, hi) == pytest.approx(0.0)
    assert sd_box(np.array([[1, 1, 1], [3, 1, 1]]), lo, hi).shape == (2,)


def test_sphere_distance() -> None:
    assert sd_sphere([0, 0, 0], [0, 0, 0], 1.0) == pytest.approx(-1.0)
    assert sd_sphere([3, 0, 0], [0, 0, 0], 1.0) == pytest.approx(2.0)


def test_cylinder_distance() -> None:
    p0, p1, r = [0, 0, 0], [0, 0, 4], 1.0
    assert sd_cylinder([0, 0, 2], p0, p1, r) == pytest.approx(-1.0)
    assert sd_cylinder([3, 0, 2], p0, p1, r) == pytest.approx(2.0)
    assert sd_cylinder([0, 0, 6], p0, p1, r) == pytest.approx(2.0)
    assert sd_cylinder([2, 0, 5], p0, p1, r) == pytest.approx(np.sqrt(2.0))
    assert sd_cylinder([0.5, 0, 0.2], p0, p1, r) == pytest.approx(-0.2)


def test_hexahedron_distance_on_box_matches_box() -> None:
    v = _CORNERS * 2.0
    pts = np.array([[1, 1, 1], [3, 1, 1], [3, 3, 1], [3, 3, 3], [0.5, 1, 1], [-1, 2, 5]], float)
    assert is_convex_hexahedron(v)
    assert np.allclose(sd_hexahedron(pts, v), sd_box(pts, [0, 0, 0], [2, 2, 2]))


def test_nonconvex_detected() -> None:
    v = _CORNERS.copy()
    v[6] = [1.0, 1.0, 1.5]  # warped (non-planar) top face
    check_hexahedron(v)
    assert not is_convex_hexahedron(v)
