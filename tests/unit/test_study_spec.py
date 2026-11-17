"""Study spec loading, validation and registry coverage."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from suspkin.geometry import build_corners
from suspkin.kinematics.corner import sweep_corner
from suspkin.metrics import add_gradients, vehicle_summary
from suspkin.metrics.table import metrics_frame
from suspkin.optimize import metric_registry as reg
from suspkin.optimize.spec import (
    BandConstraint,
    Study,
    StudyConfigError,
    load_study,
    sweep_range,
)
from suspkin.sweeps import states_to_frame, sweep_vehicle, vehicle_frame
from tests.unit.helpers import fixture_vehicle

FIX = Path(__file__).parents[1] / "fixtures" / "studies" / "full_study.yaml"
Mutator = Callable[[dict[str, Any]], object]


def _write(tmp_path: Path, mutate: Mutator) -> Path:
    data = yaml.safe_load(FIX.read_text(encoding="utf-8"))
    data["vehicle"] = str(FIX.parent.parent / "minimal_vehicle.yaml")
    mutate(data)
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_load_full_study() -> None:
    study, vehicle = load_study(FIX)
    assert vehicle.vehicle.name == "minimal_fs"
    assert {c.type for c in study.constraints} == {
        "band",
        "slope",
        "value",
        "reach",
        "summary",
        "inside",
        "outside",
        "distance",
        "relation",
        "joint_angle",
    }
    band = next(c for c in study.constraints if isinstance(c, BandConstraint))
    assert band.where == pytest.approx((-0.020, 0.025))
    assert band.min == pytest.approx(math.radians(-2.5))
    assert band.scale == pytest.approx(math.radians(1.0))
    by_name: dict[str, Any] = {c.name: c for c in study.constraints}
    assert by_name["camber_gain"].max == pytest.approx(math.radians(0.05) * 1000.0)
    assert by_name["camber_roll"].max == pytest.approx(0.8)
    assert by_name["rg"].max == pytest.approx(math.radians(1.2))
    assert by_name["uca_floor"].rhs == pytest.approx(0.1)
    assert by_name["uca_above"].offset == pytest.approx(0.05)
    dom: Any = study.variables[0].domain
    assert dom.vertices[2] == pytest.approx((0.112, 0.29, 0.29))
    assert sweep_range(study.sweeps["steer"], vehicle) == pytest.approx((-0.03, 0.03))
    assert study.variables[3].effective_side == "right"


def test_bad_unit_on_camber(tmp_path: Path) -> None:
    def m(d: dict[str, Any]) -> None:
        d["constraints"][0]["max"] = "5 mm"

    with pytest.raises(StudyConfigError, match=r"camber_band.*'angle'"):
        load_study(_write(tmp_path, m))


def test_unknown_metric_suggestion(tmp_path: Path) -> None:
    def m(d: dict[str, Any]) -> None:
        d["constraints"][0]["metric"] = "camer"

    with pytest.raises(StudyConfigError, match="did you mean 'camber'"):
        load_study(_write(tmp_path, m))


def test_unknown_sweep_ref(tmp_path: Path) -> None:
    def m(d: dict[str, Any]) -> None:
        d["constraints"][0]["sweep"] = "bump_x"

    with pytest.raises(StudyConfigError, match="unknown sweep 'bump_x'"):
        load_study(_write(tmp_path, m))


def test_where_outside_range(tmp_path: Path) -> None:
    def m(d: dict[str, Any]) -> None:
        d["constraints"][0]["where"] = ["-20 mm", "40 mm"]

    with pytest.raises(StudyConfigError, match="outside the sweep range"):
        load_study(_write(tmp_path, m))


def _drop_limits(d: dict[str, Any]) -> None:
    d["constraints"][0].pop("min")
    d["constraints"][0].pop("max")


def _twist(d: dict[str, Any]) -> None:
    v = d["variables"][0]["domain"]["hexahedron"]["vertices"]
    v[2], v[3] = v[3], v[2]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d["constraints"].append(dict(d["constraints"][0])), "duplicate constraint"),
        (lambda d: d["constraints"][0].update(min="1 deg", max="0 deg"), "min must be <= max"),
        (_drop_limits, "at least one"),
        (_twist, "twisted"),
        (lambda d: d["variables"][0].update(point="front.uca_frnt"), "did you mean"),
        (lambda d: d["variables"][3]["domain"]["box"].update(max=[0.02, 0.1, 0.39]), "y < 0"),
        (lambda d: d["variables"][0].update(extra=1), "extra"),
        (lambda d: d["sweeps"]["steer"].update(input="roll"), "full"),
        (lambda d: d["variables"][1].update(side="left"), "asymmetric"),
    ],
)
def test_validation_errors(tmp_path: Path, mutate: Mutator, match: str) -> None:
    with pytest.raises(StudyConfigError, match=match):
        load_study(_write(tmp_path, mutate))


def test_study_model_direct() -> None:
    s = Study.model_validate({"schema": 1, "vehicle": "v.yaml"})
    assert s.constraints == ()


def test_registry_covers_real_columns() -> None:
    v = fixture_vehicle()
    c = build_corners(v)["FL"]
    states = sweep_corner(c, [-0.01, 0.0, 0.01])
    frames = [
        add_gradients(states_to_frame(states, c, metrics=True)),
        add_gradients(metrics_frame(states, c)),
    ]
    for df in frames:
        for col in df.columns:
            if col not in ("corner", "coilover_in_range"):
                reg.corner_metric_dim(col)
    for kw in ({"roll": [-0.01, 0.0, 0.01]}, {"rack": [-0.01, 0.0, 0.01]}):
        for col in vehicle_frame(sweep_vehicle(v, **kw), v).columns:
            reg.vehicle_metric_dim(col)
    for k, val in vehicle_summary(v).to_dict().items():
        if isinstance(val, dict):
            for kk in val:
                reg.summary_metric_dim(f"{k}.{kk}")
        else:
            reg.summary_metric_dim(k)


def test_registry_errors() -> None:
    with pytest.raises(KeyError, match="did you mean"):
        reg.corner_metric_dim("camer")
    with pytest.raises(KeyError):
        reg.vehicle_metric_dim("XX_camber")
    with pytest.raises(KeyError):
        reg.summary_metric_dim("roll_gradien")
    assert reg.vehicle_metric_dim("FL_camber_ground") == "angle"
    assert reg.summary_metric_dim("front.rc_height") == "length"
    assert reg.slope_dim(reg.corner_metric_dim("camber"), "length") == "angle/length"
