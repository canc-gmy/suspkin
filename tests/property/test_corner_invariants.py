"""Property tests: rigid-body invariants hold for random travel and rack."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from suspkin.geometry.corners import CornerId
from suspkin.kinematics import solve_corner
from tests.unit.helpers import fixture_corners, max_invariant_error

CORNERS = fixture_corners()


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    travel=st.floats(-0.03, 0.03),
    rack=st.floats(-0.02, 0.02),
    cid=st.sampled_from(["FL", "FR"]),
)
def test_front_invariants(travel: float, rack: float, cid: CornerId) -> None:
    c = CORNERS[cid]
    s = solve_corner(c, travel, rack)
    assert max_invariant_error(c, s) < 1e-9
    assert s.max_residual < 1e-9
    assert abs(s.points["wheel_center"][2] - c.points["wheel_center"][2] - travel) < 1e-9
    assert abs(np.linalg.norm(s.spin_axis) - 1.0) < 1e-12


@settings(max_examples=60, deadline=None)
@given(travel=st.floats(-0.03, 0.03), cid=st.sampled_from(["RL", "RR"]))
def test_rear_invariants(travel: float, cid: CornerId) -> None:
    c = CORNERS[cid]
    s = solve_corner(c, travel)
    assert max_invariant_error(c, s) < 1e-9
