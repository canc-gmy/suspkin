import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from suspkin.geometry.vectors import distance, mirror_y, unit

_f = st.floats(-1e3, 1e3, allow_nan=False)


@given(arrays(np.float64, st.tuples(st.integers(1, 6), st.just(3)), elements=_f))
def test_mirror_y_involution(a: np.ndarray) -> None:
    np.testing.assert_array_equal(mirror_y(mirror_y(a)), a)


def test_mirror_y_single_point_and_no_alias() -> None:
    p = np.array([1.0, 2.0, 3.0])
    q = mirror_y(p)
    assert q.tolist() == [1.0, -2.0, 3.0]
    assert p.tolist() == [1.0, 2.0, 3.0]


def test_unit() -> None:
    assert np.linalg.norm(unit([3.0, 0.0, 4.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        unit([0.0, 0.0, 0.0])


def test_distance() -> None:
    assert distance([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)
