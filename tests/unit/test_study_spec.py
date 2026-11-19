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


def _add_fr_sweep(d: dict[str, Any]) -> None:
    d["sweeps"]["bump_fr"] = {
        "kind": "corner",
        "corner": "FR",
        "travel": ["-30 mm", "30 mm"],
        "steps": 13,
    }


def _over_fr_left_region(d: dict[str, Any]) -> None:
    _add_fr_sweep(d)
    d["constraints"][7].update(over="bump_fr")  # keep-out sphere at y > 0


def _over_fl_right_region(d: dict[str, Any]) -> None:
    d["constraints"][7]["region"] = {"sphere": {"center": [0.1, -0.28, 0.2], "radius": "20 mm"}}


def _distance_over_wrong_axle(d: dict[str, Any]) -> None:
    d["sweeps"]["bump_rl"] = {
        "kind": "corner",
        "corner": "RL",
        "travel": ["-30 mm", "30 mm"],
        "steps": 13,
    }
    d["constraints"].append(
        {
            "name": "dist_over",
            "type": "distance",
            "a": "front.tie_outer",
            "b": "front.uca_front",
            "min": "10 mm",
            "over": "bump_rl",
        }
    )


def _offset_zero_width(d: dict[str, Any]) -> None:
    d["variables"][2]["domain"] = {"offset": {"x": [0.0, 0.0]}}


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
        (lambda d: d["constraints"][4].update(scale="1 deg"), "toggle_scale"),
        (lambda d: d["constraints"][7].update(over="roll"), "corner sweep"),
        (_over_fr_left_region, "right"),
        (_over_fl_right_region, "left"),
        (_distance_over_wrong_axle, "axle"),
        (_offset_zero_width, "min == max"),
    ],
)
def test_validation_errors(tmp_path: Path, mutate: Mutator, match: str) -> None:
    with pytest.raises(StudyConfigError, match=match):
        load_study(_write(tmp_path, mutate))


def test_study_model_direct() -> None:
    s = Study.model_validate({"schema": 1, "vehicle": "v.yaml"})
    assert s.constraints == ()


def test_unsupported_study_schema(tmp_path: Path) -> None:
    def bump(d: dict[str, Any]) -> None:
        d["schema"] = 2

    with pytest.raises(
        StudyConfigError, match="unsupported study schema 2; this suspkin supports 1"
    ):
        load_study(_write(tmp_path, bump))


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


def test_over_fr_sweep_with_right_region_loads(tmp_path: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        _add_fr_sweep(d)
        d["constraints"][7].update(over="bump_fr")
        d["constraints"][7]["region"] = {"sphere": {"center": [0.1, -0.28, 0.2], "radius": "20 mm"}}

    study, _ = load_study(_write(tmp_path, mutate))
    assert study.constraints[7].over == "bump_fr"  # type: ignore[union-attr]


# --------------------------------------------------------------------------- surrogate blocks

_GRID = {"travel": {"min": "-20 mm", "max": "20 mm", "points": 7}}


def test_surrogate_training_hpo_defaults() -> None:
    from suspkin.optimize.spec import SurrogateSpec

    s = SurrogateSpec.model_validate({"grid": _GRID})
    assert (s.training.device, s.training.max_epochs, s.training.patience) == ("auto", 2000, 100)
    assert s.training.num_threads is None and s.training.finetune_lr_factor == 0.1
    assert s.hpo.n_trials == 100 and s.hpo.timeout == 4 * 3600.0 and s.hpo.parallel == "auto"
    assert s.hpo.max_epochs == 300 and s.hpo.subsample is None


def test_surrogate_hpo_trial_budget_keys() -> None:
    from suspkin.optimize.spec import SurrogateSpec

    s = SurrogateSpec.model_validate(
        {"grid": _GRID, "hpo": {"max_epochs": None, "subsample": 5000}}
    )
    assert s.hpo.max_epochs is None and s.hpo.subsample == 5000


def test_surrogate_training_hpo_units_and_values() -> None:
    from suspkin.optimize.spec import SurrogateSpec

    s = SurrogateSpec.model_validate(
        {
            "grid": _GRID,
            "training": {"device": "cpu", "max_epochs": 5, "num_threads": 2},
            "hpo": {"n_trials": 3, "timeout": "90 min", "parallel": False},
        }
    )
    assert s.training.device == "cpu" and s.training.num_threads == 2
    assert s.hpo.timeout == 5400.0 and s.hpo.parallel is False
    none = SurrogateSpec.model_validate({"grid": _GRID, "hpo": {"timeout": None}})
    assert none.hpo.timeout is None
    assert SurrogateSpec.model_validate({"grid": _GRID, "hpo": {"timeout": 30}}).hpo.timeout == 30.0


@pytest.mark.parametrize(
    "block",
    [
        {"training": {"device": "tpu"}},
        {"training": {"max_epochs": 0}},
        {"training": {"patience": 0}},
        {"training": {"num_threads": 0}},
        {"training": {"finetune_lr_factor": 0.0}},
        {"training": {"bogus": 1}},
        {"hpo": {"n_trials": 0}},
        {"hpo": {"timeout": "2 mm"}},
        {"hpo": {"timeout": "-1 h"}},
        {"hpo": {"parallel": "yes"}},
        {"hpo": {"extra": 1}},
        {"hpo": {"max_epochs": 0}},
        {"hpo": {"subsample": 0}},
    ],
)
def test_surrogate_training_hpo_rejected(block: dict[str, Any]) -> None:
    from pydantic import ValidationError

    from suspkin.optimize.spec import SurrogateSpec

    with pytest.raises(ValidationError):
        SurrogateSpec.model_validate({"grid": _GRID, **block})


def test_active_learning_defaults_and_units() -> None:
    from suspkin.optimize.spec import SurrogateSpec

    al = SurrogateSpec.model_validate({"grid": _GRID}).active_learning
    assert (al.max_iterations, al.fom_evaluations, al.surrogate_evaluations) == (50, 2000, 20000)
    assert al.time is None and al.margin == 0.05
    assert (al.infill.top, al.infill.perturb) == (0, 4)
    ft = al.finetune
    assert (ft.epochs, ft.weight, ft.replay, ft.patience) == (60, 5.0, 2048, 30)
    assert (ft.trigger, ft.skip_after, ft.skip_for, ft.stall_override) == (1.0, 2, 2, 3.0)
    assert ft.on_cpu == "skip"
    tr = al.trust_region
    assert (tr.initial, tr.min, tr.max) == (0.2, 0.01, 0.5)
    assert (al.stop.eps_f, al.stop.patience) == (0.01, 5)
    full = SurrogateSpec.model_validate(
        {
            "grid": _GRID,
            "active_learning": {
                "time": "30 min",
                "infill": {"top": 2, "perturb": 0},
                "trust_region": {"initial": 0.1, "min": 0.05, "max": 0.3},
            },
        }
    ).active_learning
    assert full.time == 1800.0 and full.infill.top == 2 and full.trust_region.max == 0.3


@pytest.mark.parametrize(
    "block",
    [
        {"max_iterations": 0},
        {"fom_evaluations": 0},
        {"surrogate_evaluations": 0},
        {"time": "2 mm"},
        {"time": 0},
        {"margin": -0.1},
        {"infill": {"top": -1}},
        {"finetune": {"weight": 0}},
        {"finetune": {"epochs": 0}},
        {"finetune": {"replay": -1}},
        {"finetune": {"patience": 0}},
        {"finetune": {"trigger": -1}},
        {"finetune": {"skip_after": 0}},
        {"finetune": {"skip_for": -1}},
        {"finetune": {"stall_override": 0.5}},
        {"finetune": {"on_cpu": "maybe"}},
        {"trust_region": {"initial": 0.6}},
        {"trust_region": {"min": 0.3, "initial": 0.2}},
        {"stop": {"patience": 0}},
        {"stop": {"eps_f": 0}},
        {"bogus": 1},
    ],
)
def test_active_learning_rejected(block: dict[str, Any]) -> None:
    from pydantic import ValidationError

    from suspkin.optimize.spec import SurrogateSpec

    with pytest.raises(ValidationError):
        SurrogateSpec.model_validate({"grid": _GRID, "active_learning": block})
