"""Property test: from_vehicle(to_vehicle(x)) == x on hexahedron, box and offset variables."""

from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from suspkin.optimize import DesignSpace
from suspkin.optimize.spec import load_study

_STUDY = Path(__file__).parents[1] / "fixtures" / "studies" / "design_study.yaml"
_study, _vehicle = load_study(_STUDY)
SPACE = DesignSpace(_study, _vehicle)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(x=st.lists(st.floats(0.0, 1.0), min_size=SPACE.n, max_size=SPACE.n))
def test_roundtrip(x: list[float]) -> None:
    xa = np.array(x)
    back = SPACE.from_vehicle(SPACE.to_vehicle(xa))
    np.testing.assert_allclose(back, xa, atol=1e-9, rtol=0)
