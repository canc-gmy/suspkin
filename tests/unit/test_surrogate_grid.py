"""Surrogate grid spec and AxleGrid."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from suspkin.optimize.spec import Study, load_study
from suspkin.surrogate.grid import axle_grids

FIX = Path(__file__).parents[1] / "fixtures" / "studies" / "surrogate_study.yaml"


def _spec(**grid: Any) -> Study:
    base = {"schema": 1, "vehicle": "v.yaml", "surrogate": {"grid": grid}}
    return Study.model_validate(base)


def test_load_with_units() -> None:
    study, _ = load_study(FIX)
    assert study.surrogate is not None
    g = study.surrogate.grid
    assert g.travel.min == pytest.approx(-0.03) and g.travel.points == 41
    assert g.rack is not None and g.rack.max == pytest.approx(0.025)
    grids = axle_grids(g)
    f, r = grids["front"], grids["rear"]
    assert f.shape == (21, 41) and r.shape == (1, 41)
    assert f.travel[0] == pytest.approx(-0.03) and f.travel[-1] == pytest.approx(0.03)
    assert np.array_equal(r.rack, [0.0])
    assert f.axle == "front" and r.axle == "rear"


def test_no_surrogate_is_none() -> None:
    assert Study.model_validate({"schema": 1, "vehicle": "v.yaml"}).surrogate is None


@pytest.mark.parametrize(
    "grid",
    [
        {"travel": {"min": "5 mm", "max": "30 mm", "points": 5}},  # 0 outside
        {"travel": {"min": "-30 mm", "max": "-5 mm", "points": 5}},
        {"travel": {"min": "-30 mm", "max": "30 mm", "points": 2}},  # too few points
        {"travel": {"min": "30 mm", "max": "-30 mm", "points": 5}},  # reversed
        {"travel": {"min": "-30 deg", "max": "30 mm", "points": 5}},  # wrong dimension
        {"travel": {"min": "-30 mm", "max": "30 mm", "points": 5}, "extra": 1},
        {
            "travel": {"min": "-30 mm", "max": "30 mm", "points": 5},
            "rack": {"min": "1 mm", "max": "2 mm", "points": 5},
        },
    ],
)
def test_validation_errors(grid: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _spec(**grid)


def test_without_rack_front_has_single_rack() -> None:
    s = _spec(travel={"min": "-10 mm", "max": "10 mm", "points": 5})
    assert s.surrogate is not None
    f = axle_grids(s.surrogate.grid)["front"]
    assert f.shape == (1, 5)


def test_rack_ignored_on_rear_and_hash_stable() -> None:
    study, _ = load_study(FIX)
    assert study.surrogate is not None
    a = axle_grids(study.surrogate.grid)
    b = axle_grids(study.surrogate.grid)
    assert a["rear"].rack.size == 1
    assert a["front"].grid_hash() == b["front"].grid_hash()
    assert a["front"].grid_hash() != a["rear"].grid_hash()
    other = _spec(travel={"min": "-30 mm", "max": "31 mm", "points": 41})
    assert other.surrogate is not None
    assert axle_grids(other.surrogate.grid)["rear"].grid_hash() != a["rear"].grid_hash()
