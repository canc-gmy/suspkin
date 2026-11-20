"""Array helpers of ``geometry.vectors`` (last axis = xyz) against numpy references."""

from __future__ import annotations

import numpy as np

from suspkin.geometry.vectors import (
    axis_angle_matrix,
    cross_rows,
    dot_rows,
    norm_rows,
    rodrigues_rows,
    unit_rows,
    wrap_angle,
)

RNG = np.random.default_rng(7)
A = RNG.normal(size=(5, 3))
B = RNG.normal(size=(5, 3))


def test_dot_cross_norm_unit() -> None:
    assert np.allclose(dot_rows(A, B), np.sum(A * B, axis=-1), rtol=1e-14)
    assert dot_rows(A, B, keepdims=True).shape == (5, 1)
    assert np.allclose(cross_rows(A, B), np.cross(A, B), rtol=1e-14)
    assert np.allclose(norm_rows(A), np.linalg.norm(A, axis=-1), rtol=1e-14)
    assert np.allclose(norm_rows(unit_rows(A)), 1.0, rtol=1e-14)
    assert cross_rows(A[:, None, :], B[None]).shape == (5, 5, 3)


def test_rodrigues_matches_rotation_matrix() -> None:
    k = unit_rows(B)
    ang = RNG.uniform(-3.0, 3.0, size=5)
    got = rodrigues_rows(k, np.cos(ang)[:, None], np.sin(ang)[:, None], A)
    ref = np.stack([axis_angle_matrix(k[i], ang[i]) @ A[i] for i in range(5)])
    assert np.allclose(got, ref, atol=1e-13)


def test_wrap_angle_range_and_identity() -> None:
    a = np.array([-7.0, -np.pi + 1e-9, 0.3, 3.0, 10.0])
    w = wrap_angle(a)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.cos(w), np.cos(a)) and np.allclose(np.sin(w), np.sin(a))
