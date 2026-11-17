import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from suspkin.geometry import Vehicle, load_vehicle

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"


def _data() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    return copy.deepcopy(loaded)


def test_fixture_loads_and_properties() -> None:
    v = load_vehicle(FIXTURE)
    assert v.wheelbase == pytest.approx(1.55)
    assert v.front_track == pytest.approx(1.20)
    assert v.rear_track == pytest.approx(1.16)
    assert v.front.arb is not None
    assert v.front.arb.type == "torsional"
    assert v.rear.arb is not None
    assert v.rear.arb.type == "flexural"


def test_rocker_axis_normalized() -> None:
    v = load_vehicle(FIXTURE)
    ax = v.front.hardpoints.rocker_axis
    assert sum(c * c for c in ax) == pytest.approx(1.0)


def _neg_y(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["uca_outer"][1] = -0.5


def _zero_axis(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["rocker_axis"] = [0.0, 0.0, 0.0]


def _rear_steer(d: dict[str, Any]) -> None:
    d["rear"]["steering"] = {"c_factor": 0.09, "rack_travel": [-0.03, 0.03]}


def _no_front_steer(d: dict[str, Any]) -> None:
    del d["front"]["steering"]


def _arb_no_rocker(d: dict[str, Any]) -> None:
    del d["front"]["hardpoints"]["rocker_arb"]


def _bad_override(d: dict[str, Any]) -> None:
    d["front"]["right_overrides"] = {"not_a_point": [0.0, -0.5, 0.2]}


def _wc_z(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["wheel_center"][2] = 0.25


def _short_link(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["tie_inner"] = [0.08, 0.56, 0.15]


def _uca_below_lca(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["uca_outer"][2] = 0.10


def _override_left_side(d: dict[str, Any]) -> None:
    d["front"]["right_overrides"] = {"uca_outer": [0.0, 0.5, 0.3]}


def _front_behind_rear(d: dict[str, Any]) -> None:
    d["rear"]["hardpoints"]["wheel_center"][0] = 0.5


def _uca_axis_degenerate(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["uca_rear"] = [0.10, 0.28, 0.305]


def _lca_axis_degenerate(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["lca_rear"] = [0.15, 0.22, 0.125]


def _pushrod_on_rocker_axis(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["pushrod_inner"] = [0.0, 0.18, 0.45]


def _spring_on_rocker_axis(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["rocker_axis"] = [0.05, 0.0, 0.07]


def _arb_on_rocker_axis(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["rocker_arb"] = [0.0, 0.18, 0.45]


def _zero_rack_axis(d: dict[str, Any]) -> None:
    d["front"]["steering"]["rack_axis"] = [0.0, 0.0, 0.0]


def _coilover_too_long(d: dict[str, Any]) -> None:
    d["front"]["coilover"]["length_max"] = 0.20


def _coilover_too_short(d: dict[str, Any]) -> None:
    d["front"]["coilover"]["length_min"] = 0.24
    d["front"]["coilover"]["length_max"] = 0.30


def _override_outer_y_zero(d: dict[str, Any]) -> None:
    d["front"]["right_overrides"] = {"uca_outer": [0.0, 0.0, 0.3]}


def _torsional_bar_length(d: dict[str, Any]) -> None:
    d["front"]["arb"]["bar_length"] = 0.45


def _wc_z_wide(d: dict[str, Any]) -> None:
    d["front"]["hardpoints"]["wheel_center"][2] = 0.2032 + 0.025


@pytest.mark.parametrize(
    "mutate",
    [
        _uca_axis_degenerate,
        _lca_axis_degenerate,
        _pushrod_on_rocker_axis,
        _spring_on_rocker_axis,
        _arb_on_rocker_axis,
        _zero_rack_axis,
        _coilover_too_long,
        _coilover_too_short,
        _override_outer_y_zero,
        _torsional_bar_length,
        _wc_z_wide,
        _neg_y,
        _zero_axis,
        _rear_steer,
        _no_front_steer,
        _arb_no_rocker,
        _bad_override,
        _wc_z,
        _short_link,
        _uca_below_lca,
        _override_left_side,
        _front_behind_rear,
    ],
)
def test_validators_reject(mutate: Callable[[dict[str, Any]], None]) -> None:
    d = _data()
    mutate(d)
    with pytest.raises(ValidationError):
        Vehicle.model_validate(d)


def test_extra_field_forbidden() -> None:
    d = _data()
    d["front"]["bogus"] = 1
    with pytest.raises(ValidationError):
        Vehicle.model_validate(d)


def test_frozen() -> None:
    v = load_vehicle(FIXTURE)
    with pytest.raises(ValidationError):
        v.vehicle.mass = 1.0  # type: ignore[misc]
