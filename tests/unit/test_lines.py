"""Tests for the line/plane helpers."""

from __future__ import annotations

import numpy as np
import pytest

from suspkin.geom_math.lines import (
    intersect_lines_2d,
    line_direction_2d,
    line_through_2d,
    plane_from_points,
    plane_section_2d,
)


def test_plane_from_points() -> None:
    p, n = plane_from_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(abs(n), [0, 0, 1])
    assert np.linalg.norm(n) == pytest.approx(1.0)
    np.testing.assert_allclose(p, 0)


def test_plane_collinear_raises() -> None:
    with pytest.raises(ValueError):
        plane_from_points([0, 0, 0], [1, 1, 1], [2, 2, 2])


def test_section_of_tilted_plane() -> None:
    # plane z = 0.5 * y, cut with x = 3 -> line z = 0.5 y in (y, z)
    p, n = plane_from_points([0, 0, 0], [1, 0, 0], [0, 2, 1])
    nrm, off = plane_section_2d(p, n, 0, 3.0)
    for y in (-1.0, 0.0, 2.0):
        assert float(nrm @ [y, 0.5 * y]) == pytest.approx(off, abs=1e-12)


def test_section_with_x_dependent_plane() -> None:
    # plane z = x + y; cut with y = 1 -> in (x, z): z = x + 1
    p, n = plane_from_points([0, 0, 0], [1, 0, 1], [0, 1, 1])
    nrm, off = plane_section_2d(p, n, 1, 1.0)
    for x in (-2.0, 0.0, 5.0):
        assert float(nrm @ [x, x + 1.0]) == pytest.approx(off, abs=1e-12)


def test_section_parallel_plane_raises() -> None:
    p, n = plane_from_points([0, 0, 0], [0, 1, 0], [0, 0, 1])  # x = 0 plane
    with pytest.raises(ValueError):
        plane_section_2d(p, n, 0, 1.0)


def test_intersect_lines_2d() -> None:
    a = line_through_2d([0, 0], [1, 1])
    b = line_through_2d([0, 2], [2, 0])
    np.testing.assert_allclose(intersect_lines_2d(a, b), [1.0, 1.0])


def test_intersect_parallel_is_none() -> None:
    a = line_through_2d([0, 0], [1, 1])
    b = line_through_2d([0, 1], [1, 2])
    assert intersect_lines_2d(a, b) is None
    assert intersect_lines_2d(a, a) is None


def test_line_direction() -> None:
    nrm, _ = line_through_2d([0, 0], [3, 4])
    d = line_direction_2d(nrm)
    np.testing.assert_allclose(abs(d), [0.6, 0.8])


def test_coincident_points_raises() -> None:
    with pytest.raises(ValueError):
        line_through_2d([1, 1], [1, 1])
