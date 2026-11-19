"""Tests for rotation helpers."""

import numpy as np
import pytest

from suspkin.geom_math import axis_angle_matrix, rotation_between


def test_quarter_turn_about_z() -> None:
    r = axis_angle_matrix([0, 0, 2.0], np.pi / 2)
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_matrix_is_orthonormal() -> None:
    r = axis_angle_matrix([1.0, -2.0, 0.5], 0.7)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_zero_axis_rejected() -> None:
    with pytest.raises(ValueError):
        axis_angle_matrix([0, 0, 0], 1.0)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([1, 0, 0], [0, 1, 0]),
        ([1, 2, 3], [-2, 0.5, 1]),
        ([1, 0, 0], [1, 0, 0]),
        ([0, 0, 1], [0, 0, -1]),
        ([0, 1, 0], [0, -3, 0]),
    ],
)
def test_rotation_between_maps_direction(a: list[float], b: list[float]) -> None:
    r = rotation_between(a, b)
    ua, ub = np.array(a, float), np.array(b, float)
    np.testing.assert_allclose(r @ (ua / np.linalg.norm(ua)), ub / np.linalg.norm(ub), atol=1e-12)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_rotation_between_is_minimal() -> None:
    # The component along a x b is untouched by the minimal rotation.
    r = rotation_between([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(r @ [0, 0, 1], [0, 0, 1], atol=1e-12)


def test_rotation_between_zero_rejected() -> None:
    with pytest.raises(ValueError):
        rotation_between([0, 0, 0], [1, 0, 0])
