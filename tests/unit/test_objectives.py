"""Objectives and optimizer block: spec validation, evaluation, sharing, failures, CLI."""

import importlib
import math
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from suspkin.cli import app
from suspkin.geometry import build_corners
from suspkin.kinematics import KinematicsError
from suspkin.kinematics.corner import sweep_corner
from suspkin.metrics import add_gradients, vehicle_summary
from suspkin.optimize import F_FAIL, DesignSpace, InfeasibleDesign, ObjectiveResult, evaluate
from suspkin.optimize.spec import (
    CurveObjective,
    ExtremumObjective,
    LinearityObjective,
    Optimizer,
    Study,
    StudyConfigError,
    TargetObjective,
    load_study,
)
from suspkin.sweeps import states_to_frame
from suspkin.units import parse_quantity

evaluate_mod = importlib.import_module("suspkin.optimize.evaluate")
FIX = Path(__file__).parents[1] / "fixtures" / "studies" / "objectives_study.yaml"
DEG = math.pi / 180.0
MM = 1e-3
runner = CliRunner()


@pytest.fixture(scope="module")
def space() -> DesignSpace:
    study, vehicle = load_study(FIX)
    return DesignSpace(study, vehicle)


@pytest.fixture(scope="module")
def df(space: DesignSpace) -> pd.DataFrame:
    corner = build_corners(space.baseline)["FL"]
    sw = space.study.sweeps["bump_f"]
    states = sweep_corner(corner, np.linspace(sw.travel[0], sw.travel[1], sw.steps))  # type: ignore[union-attr]
    return add_gradients(states_to_frame(states, corner, metrics=True))


def _res(ev: Any, name: str) -> ObjectiveResult:
    return next(o for o in ev.objectives if o.name == name)


def _window(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    inner = (x > lo) & (x < hi)
    return (
        np.concatenate([[lo], x[inner], [hi]]),
        np.concatenate([[np.interp(lo, x, y)], y[inner], [np.interp(hi, x, y)]]),
    )


def _only(space: DesignSpace, **upd: Any) -> DesignSpace:
    """Space whose objectives are the fixture ones with ``upd`` applied to the named one."""
    objs = []
    for o in space.study.objectives:
        objs.append(o.model_copy(update=upd.get(o.name, {})))
    study = space.study.model_copy(update={"objectives": tuple(objs)})
    return DesignSpace(study, space.baseline)


# --------------------------------------------------------------------------------------
# loading and units
# --------------------------------------------------------------------------------------


def test_load_objectives_and_optimizer(space: DesignSpace) -> None:
    objs = space.study.objectives
    assert [type(o) for o in objs] == [
        CurveObjective,
        TargetObjective,
        TargetObjective,
        ExtremumObjective,
        ExtremumObjective,
        LinearityObjective,
    ]
    curve = objs[0]
    assert isinstance(curve, CurveObjective)
    assert curve.target.x == pytest.approx((-0.03, 0.0, 0.03))
    assert curve.target.y == pytest.approx((-1.6 * DEG, -0.6 * DEG, 0.3 * DEG))
    assert curve.where == pytest.approx((-0.025, 0.025))
    assert curve.scale == pytest.approx(0.1 * DEG)
    rg = objs[2]
    assert isinstance(rg, TargetObjective)
    assert rg.target == pytest.approx(0.8 * DEG)
    opt = space.study.optimizer
    assert opt is not None
    assert (opt.mode, opt.method, opt.seed, opt.workers, opt.start) == (
        "weighted",
        "de+slsqp",
        1,
        2,
        "baseline",
    )
    assert opt.budget.evaluations == 200
    assert opt.budget.time == pytest.approx(7200.0)
    assert opt.options == {"popsize": 8}


@pytest.mark.parametrize(
    ("text", "expected"), [("2 h", 7200.0), ("90 min", 5400.0), ("30 s", 30.0), ("45", 45.0)]
)
def test_time_unit_parsing(text: str, expected: float) -> None:
    assert parse_quantity(text, "time") == pytest.approx(expected)
    with pytest.raises(ValueError, match="dimension"):
        parse_quantity("2 mm", "time")


def test_study_without_objectives_unchanged() -> None:
    study, _ = load_study(FIX.parent / "full_study.yaml")
    assert study.objectives == ()
    assert study.optimizer is None


def test_optimizer_defaults() -> None:
    assert Optimizer(mode="pareto").method == "nsga2"
    o = Optimizer()
    assert (o.mode, o.method, o.workers, o.start) == ("weighted", "de+slsqp", "auto", "baseline")
    assert o.budget.evaluations == 2000


# --------------------------------------------------------------------------------------
# hand-computed values
# --------------------------------------------------------------------------------------


def test_target_at_sweep(space: DesignSpace, df: pd.DataFrame) -> None:
    r = _res(evaluate(space), "mr_target")
    x, y = df["travel"].to_numpy(), df["mr_spring"].to_numpy()
    raw = abs(float(np.interp(0.0, x, y)) - 1.0)
    assert r.status == "ok"
    assert r.dim == "ratio"
    assert r.raw == pytest.approx(raw)
    assert r.f == pytest.approx(raw / 0.05)
    assert r.weight == 1.0


def test_target_summary(space: DesignSpace) -> None:
    r = _res(evaluate(space), "rg_target")
    rg = float(vehicle_summary(space.baseline).roll_gradient)
    raw = abs(rg - 0.8 * DEG)
    assert r.dim == "angle/gload"
    assert r.raw == pytest.approx(raw)
    assert r.f == pytest.approx(raw / (0.8 * DEG))  # no default scale: |target| is the scale


def test_minimize_max_abs_window(space: DesignSpace, df: pd.DataFrame) -> None:
    r = _res(evaluate(space), "bs_abs")
    _, ys = _window(df["travel"].to_numpy(), df["bump_steer"].to_numpy(), -0.025, 0.025)
    assert r.raw == pytest.approx(float(np.max(np.abs(ys))))
    assert r.f == pytest.approx(r.raw / (0.01 * DEG / MM))
    assert r.detail == "max_abs"


def test_maximize_sign_and_slope(space: DesignSpace, df: pd.DataFrame) -> None:
    r = _res(evaluate(space), "camber_gain_max")
    slope = float(np.polyfit(df["travel"].to_numpy(), df["camber"].to_numpy(), 1)[0])
    assert r.dim == "angle/length"
    assert r.raw == pytest.approx(slope)
    assert r.f == pytest.approx(-slope / (0.01 * DEG / MM))
    # the same objective as a minimisation has the opposite f
    sp = _only(space, camber_gain_max={})
    objs = tuple(
        ExtremumObjective(**{**o.model_dump(), "type": "minimize"})
        if o.name == "camber_gain_max" and isinstance(o, ExtremumObjective)
        else o
        for o in sp.study.objectives
    )
    sp2 = DesignSpace(sp.study.model_copy(update={"objectives": objs}), space.baseline)
    r2 = _res(evaluate(sp2), "camber_gain_max")
    assert r2.raw == pytest.approx(r.raw)
    assert r2.f == pytest.approx(-r.f)


@pytest.mark.parametrize(
    ("of", "fn"),
    [
        ("max", np.max),
        ("min", np.min),
        ("max_abs", lambda a: np.max(np.abs(a))),
    ],
)
def test_extremum_statistics(
    space: DesignSpace, df: pd.DataFrame, of: str, fn: Callable[[np.ndarray], float]
) -> None:
    sp = _only(space, bs_abs={"of": of})
    r = _res(evaluate(sp), "bs_abs")
    _, ys = _window(df["travel"].to_numpy(), df["bump_steer"].to_numpy(), -0.025, 0.025)
    assert r.raw == pytest.approx(float(fn(ys)))


def test_extremum_mean_and_value(space: DesignSpace, df: pd.DataFrame) -> None:
    x, y = df["travel"].to_numpy(), df["mr_spring"].to_numpy()
    sp = _only(space, bs_abs={"of": "mean", "metric": "mr_spring", "scale": 0.01})
    r = _res(evaluate(sp), "bs_abs")
    xs, ys = _window(x, y, -0.025, 0.025)
    assert r.raw == pytest.approx(float(np.trapezoid(ys, xs) / 0.05))
    sp = _only(
        space,
        bs_abs={"of": "value", "metric": "mr_spring", "scale": 0.01, "at": 0.01, "where": None},
    )
    r = _res(evaluate(sp), "bs_abs")
    assert r.raw == pytest.approx(float(np.interp(0.01, x, y)))


def test_curve_rms(space: DesignSpace, df: pd.DataFrame) -> None:
    r = _res(evaluate(space), "camber_curve")
    xs, ys = _window(df["travel"].to_numpy(), df["camber"].to_numpy(), -0.025, 0.025)
    tgt = np.interp(xs, [-0.03, 0.0, 0.03], [-1.6 * DEG, -0.6 * DEG, 0.3 * DEG])
    raw = float(np.sqrt(np.mean((ys - tgt) ** 2)))
    assert r.raw == pytest.approx(raw)
    assert r.f == pytest.approx(raw / (0.1 * DEG))
    assert r.weight == 2.0


def test_curve_equal_to_actual_is_zero(space: DesignSpace, df: pd.DataFrame) -> None:
    x = tuple(float(v) for v in df["travel"])
    y = tuple(float(v) for v in df["camber"])
    old = space.study.objectives[0]
    assert isinstance(old, CurveObjective)
    cur = old.model_copy(update={"target": old.target.model_copy(update={"x": x, "y": y})})
    study = space.study.model_copy(update={"objectives": (cur,)})
    r = evaluate(DesignSpace(study, space.baseline)).objectives[0]
    assert r.raw == pytest.approx(0.0, abs=1e-12)
    assert r.f == pytest.approx(0.0, abs=1e-9)
    # default window: no `where` still evaluates the overlap and gives about 0
    nowhere = cur.model_copy(update={"where": None})
    study = space.study.model_copy(update={"objectives": (nowhere,)})
    r = evaluate(DesignSpace(study, space.baseline)).objectives[0]
    assert r.raw == pytest.approx(0.0, abs=1e-12)


def test_linearity(space: DesignSpace, df: pd.DataFrame) -> None:
    r = _res(evaluate(space), "mr_lin")
    xs, ys = _window(df["travel"].to_numpy(), df["mr_spring"].to_numpy(), -0.025, 0.025)
    resid = ys - np.polyval(np.polyfit(xs, ys, 1), xs)
    assert r.raw == pytest.approx(float(np.sqrt(np.mean(resid**2))))
    assert r.f == pytest.approx(r.raw / 0.01)


def test_linearity_of_linear_data_is_zero(
    space: DesignSpace, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    lin = df.assign(mr_spring=1.0 + 3.0 * df["travel"])
    fake = SimpleNamespace(frame=lin)
    monkeypatch.setattr(evaluate_mod._Ctx, "sweep_corner_data", lambda self, name: fake)
    r = _res(evaluate(space), "mr_lin")
    assert r.status == "ok"
    assert r.raw == pytest.approx(0.0, abs=1e-12)
    assert r.f == pytest.approx(0.0, abs=1e-10)


def test_weighted_total_and_vector(space: DesignSpace) -> None:
    ev = evaluate(space)
    weights = [2.0, 1.0, 1.0, 1.0, 0.5, 1.0]
    assert [o.name for o in ev.objectives] == [o.name for o in space.study.objectives]
    assert [o.weight for o in ev.objectives] == weights
    fv = ev.f_vector()
    assert fv.shape == (6,)
    assert fv == pytest.approx([o.f for o in ev.objectives])
    assert ev.f_total == pytest.approx(float(np.dot(weights, fv)))
    # constraints are unaffected
    assert len(ev.results) == 2
    assert len(ev.frame()) == 2
    assert ev.g_vector().shape == (2,)


def test_objective_frames(space: DesignSpace) -> None:
    ev = evaluate(space)
    f = ev.objectives_frame()
    assert list(f.columns) == ["name", "type", "raw", "f", "weight", "status", "dim", "detail"]
    assert list(f["name"]) == [o.name for o in ev.objectives]
    d = ev.objectives_display_frame()
    assert "dim" not in d.columns
    assert "unit" in d.columns
    i = list(d["name"]).index("camber_curve")
    assert d["unit"].iloc[i] == "deg"
    assert d["raw"].iloc[i] == pytest.approx(f["raw"].iloc[i] / DEG)
    empty = evaluate(DesignSpace(space.study.model_copy(update={"objectives": ()}), space.baseline))
    assert empty.objectives_frame().empty
    assert empty.f_total == 0.0
    assert empty.f_vector().shape == (0,)


# --------------------------------------------------------------------------------------
# sharing and failures
# --------------------------------------------------------------------------------------


def test_sweep_shared_between_constraints_and_objectives(
    space: DesignSpace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"corner": 0, "vehicle": 0}
    real_c, real_v = evaluate_mod.sweep_corner, evaluate_mod.sweep_vehicle

    def count_c(*a: Any, **k: Any) -> Any:
        calls["corner"] += 1
        return real_c(*a, **k)

    def count_v(*a: Any, **k: Any) -> Any:
        calls["vehicle"] += 1
        return real_v(*a, **k)

    monkeypatch.setattr(evaluate_mod, "sweep_corner", count_c)
    monkeypatch.setattr(evaluate_mod, "sweep_vehicle", count_v)
    ev = evaluate(space)
    # bump_f is used by 2 constraints and 5 objectives: one solve; roll is unused
    assert calls == {"corner": 1, "vehicle": 0}
    assert all(o.status == "ok" for o in ev.objectives)


def test_unsolved_sweep_gives_f_fail(space: DesignSpace, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*a: Any, **k: Any) -> Any:
        raise KinematicsError("no solution")

    monkeypatch.setattr(evaluate_mod, "sweep_corner", fail)
    ev = evaluate(space)
    by = {o.name: o for o in ev.objectives}
    for name in ("camber_curve", "mr_target", "bs_abs", "camber_gain_max", "mr_lin"):
        assert by[name].status == "unsolved"
        assert by[name].f == F_FAIL
        assert math.isnan(by[name].raw)
    assert by["rg_target"].status == "ok"  # summary needs no sweep
    assert F_FAIL == 1e3
    assert by["camber_curve"].dim == "angle"


def test_infeasible_design_gives_f_fail(
    space: DesignSpace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bad(x: Any) -> Any:
        raise InfeasibleDesign("bad")

    monkeypatch.setattr(space, "to_vehicle", bad)
    ev = evaluate(space, np.full(space.n, 0.5))
    assert [o.status for o in ev.objectives] == ["infeasible"] * 6
    assert ev.f_vector() == pytest.approx([F_FAIL] * 6)
    assert ev.f_total == pytest.approx(F_FAIL * (2 + 1 + 1 + 1 + 0.5 + 1))


def test_summary_none_is_unsolved(space: DesignSpace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluate_mod, "vehicle_summary", lambda v: SimpleNamespace(roll_gradient=None)
    )
    r = _res(evaluate(space), "rg_target")
    assert r.status == "unsolved"
    assert r.f == F_FAIL


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------


def _write(tmp_path: Path, mutate: Callable[[dict[str, Any]], object]) -> Path:
    data = yaml.safe_load(FIX.read_text(encoding="utf-8"))
    data["vehicle"] = str(FIX.parent.parent / "minimal_vehicle.yaml")
    mutate(data)
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _obj(d: dict[str, Any], name: str) -> dict[str, Any]:
    return next(o for o in d["objectives"] if o["name"] == name)


def _pareto_wrong_method(d: dict[str, Any]) -> None:
    d["optimizer"]["mode"] = "pareto"  # method stays de+slsqp


def _pareto_one_objective(d: dict[str, Any]) -> None:
    d["optimizer"].update(mode="pareto", method="nsga2")
    d["objectives"] = d["objectives"][:1]


def _weighted_no_objective(d: dict[str, Any]) -> None:
    d["objectives"] = []


def _dup_objective(d: dict[str, Any]) -> None:
    d["objectives"][1]["name"] = d["objectives"][0]["name"]


def _clash_constraint(d: dict[str, Any]) -> None:
    d["objectives"][0]["name"] = "mr_static"


def _curve_not_increasing(d: dict[str, Any]) -> None:
    _obj(d, "camber_curve")["target"]["x"] = ["-30 mm", "0 mm", "0 mm"]


def _curve_length_mismatch(d: dict[str, Any]) -> None:
    _obj(d, "camber_curve")["target"]["y"] = ["-1.6 deg", "-0.6 deg"]


def _curve_too_short(d: dict[str, Any]) -> None:
    _obj(d, "camber_curve")["target"] = {"x": ["0 mm"], "y": ["1 deg"]}


def _curve_where_outside_target(d: dict[str, Any]) -> None:
    _obj(d, "camber_curve")["where"] = ["-29 mm", "35 mm"]


def _bad_of(d: dict[str, Any]) -> None:
    _obj(d, "bs_abs")["of"] = "median"


def _value_needs_at(d: dict[str, Any]) -> None:
    o = _obj(d, "bs_abs")
    o["of"] = "value"
    del o["where"]


def _summary_needs_value(d: dict[str, Any]) -> None:
    d["objectives"].append(
        {"name": "rg_max", "type": "minimize", "metric": "roll_gradient", "of": "max"}
    )


def _target_needs_at(d: dict[str, Any]) -> None:
    del _obj(d, "mr_target")["at"]


def _at_outside_sweep(d: dict[str, Any]) -> None:
    _obj(d, "mr_target")["at"] = "40 mm"


def _where_outside_sweep(d: dict[str, Any]) -> None:
    _obj(d, "mr_lin")["where"] = ["-40 mm", "10 mm"]


def _unknown_sweep(d: dict[str, Any]) -> None:
    _obj(d, "mr_lin")["sweep"] = "bump_g"


def _scale_required(d: dict[str, Any]) -> None:
    del _obj(d, "bs_abs")["scale"]  # angle/length has no default scale


def _bad_unit(d: dict[str, Any]) -> None:
    _obj(d, "mr_target")["at"] = "4 deg"


def _bad_time(d: dict[str, Any]) -> None:
    d["optimizer"]["budget"]["time"] = "2 mm"


def _empty_budget(d: dict[str, Any]) -> None:
    d["optimizer"]["budget"] = {}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_pareto_wrong_method, "pareto requires method nsga2"),
        (_pareto_one_objective, "at least 2 objectives"),
        (_weighted_no_objective, "at least 1 objective"),
        (_dup_objective, "duplicate objective name"),
        (_clash_constraint, "clashes with a constraint name"),
        (_curve_not_increasing, "strictly increasing"),
        (_curve_length_mismatch, "same length"),
        (_curve_too_short, "at least 2 points"),
        (_curve_where_outside_target, "inside the target x range"),
        (_bad_of, "of"),
        (_value_needs_at, "of: value needs `at`"),
        (_summary_needs_value, "needs of: value"),
        (_target_needs_at, "`at` is required"),
        (_at_outside_sweep, "outside the sweep range"),
        (_where_outside_sweep, "outside the sweep range"),
        (_unknown_sweep, "unknown sweep"),
        (_scale_required, "objective 'bs_abs': add `scale`"),
        (_bad_unit, "objective 'mr_target'"),
        (_bad_time, "optimizer budget"),
        (_empty_budget, "budget needs"),
    ],
)
def test_validation_errors(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], object], match: str
) -> None:
    with pytest.raises(StudyConfigError, match=match):
        load_study(_write(tmp_path, mutate))


def test_pareto_valid(tmp_path: Path) -> None:
    def m(d: dict[str, Any]) -> None:
        d["optimizer"] = {"mode": "pareto"}

    study, _ = load_study(_write(tmp_path, m))
    assert study.optimizer is not None
    assert study.optimizer.method == "nsga2"
    assert study.optimizer.budget.evaluations == 2000


def test_optimizer_without_objectives_only_when_weighted_fails() -> None:
    with pytest.raises(ValueError, match="at least 1 objective"):
        Study.model_validate({"schema": 1, "vehicle": "v.yaml", "optimizer": {}})
    assert Study.model_validate({"schema": 1, "vehicle": "v.yaml"}).optimizer is None


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_check_prints_objectives() -> None:
    res = runner.invoke(app, ["check", str(FIX)])
    assert res.exit_code == 0, res.output
    assert "constraints pass" in res.output
    assert "objective" in res.output
    for name in ("camber_curve", "mr_target", "rg_target", "bs_abs", "camber_gain_max", "mr_lin"):
        assert name in res.output
    assert "curve" in res.output
    assert "maximize" in res.output
    assert "f_total" in res.output
    assert "weighted" in res.output
    assert res.output.index("constraints pass") < res.output.index("camber_curve")
