"""Vehicle sweeps and the result table."""

from __future__ import annotations

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.sweeps import sweep_vehicle, vehicle_frame
from tests.unit.helpers import FIXTURE

VEHICLE = load_vehicle(FIXTURE)


def test_exactly_one_sweep_variable() -> None:
    with pytest.raises(ValueError):
        sweep_vehicle(VEHICLE, roll=[0.0], heave=[0.0])
    with pytest.raises(ValueError):
        sweep_vehicle(VEHICLE)


def test_roll_sweep_matches_pointwise_and_order() -> None:
    rolls = np.array([0.02, -0.02, 0.0, 0.01])
    states = sweep_vehicle(VEHICLE, roll=rolls)
    assert [s.roll for s in states] == list(rolls)
    for s in states:
        assert s.max_residual < 1e-9


def test_base_combines() -> None:
    states = sweep_vehicle(VEHICLE, roll=np.linspace(-0.02, 0.02, 5), base={"rack": 0.01})
    assert all(s.rack == 0.01 for s in states)
    assert all(s.corners["FL"].rack == pytest.approx(0.01) for s in states)


def test_frame_columns() -> None:
    states = sweep_vehicle(VEHICLE, heave=np.linspace(-0.01, 0.01, 3))
    df = vehicle_frame(states, VEHICLE)
    assert len(df) == 3
    for col in ("heave", "roll", "pitch", "rack", "FL_travel", "RR_camber", "FR_cp_z"):
        assert col in df.columns
    assert "arb_front_elastic" in df.columns and "arb_rear_elastic" in df.columns
    np.testing.assert_allclose(df["FL_travel"], -df["heave"], atol=2e-4)
