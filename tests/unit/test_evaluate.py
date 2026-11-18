"""Constraint evaluation: every constraint type, g arithmetic, sweep cache, failure paths."""

import importlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from suspkin.geometry import build_corners
from suspkin.kinematics import KinematicsError
from suspkin.kinematics.corner import sweep_corner
from suspkin.metrics import joint_angles_frame, vehicle_summary
from suspkin.optimize import DesignSpace, InfeasibleDesign, evaluate
from suspkin.optimize.evaluate import G_FAIL, ConstraintResult, _scale
from suspkin.optimize.spec import (
    BoxRegion,
    Constraint,
    CylinderRegion,
    HexahedronRegion,
    SphereRegion,
    load_study,
)

evaluate_mod = importlib.import_module("suspkin.optimize.evaluate")
FIX = Path(__file__).parents[1] / "fixtures" / "studies" / "full_study.yaml"
MM = 1e-3
DEG = math.pi / 180.0


@pytest.fixture(scope="module")
def space() -> DesignSpace:
    study, vehicle = load_study(FIX)
    return DesignSpace(study, vehicle)


def _con(space: DesignSpace, name: str, **upd: Any) -> Constraint:
    c = next(c for c in space.study.constraints if c.name == name)
    return c.model_copy(update=upd)


def _one(space: DesignSpace, c: Constraint, **kw: Any) -> ConstraintResult:
    sp = DesignSpace(space.study.model_copy(update={"constraints": (c,)}), space.baseline)
    return evaluate(sp, **kw).results[0]


def _pt(space: DesignSpace, name: str) -> np.ndarray:
    return np.array(build_corners(space.baseline)["FL"].points[name], dtype=np.float64)


def _t(p: np.ndarray, *d: float) -> Any:
    return tuple(float(v) for v in p + np.array(d))


def _cube(p: np.ndarray, h: float) -> tuple[tuple[float, float, float], ...]:
    corners = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    return tuple(tuple(float(v) for v in p + h * (2 * np.array(c) - 1)) for c in corners)  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# scale and g arithmetic
# --------------------------------------------------------------------------------------


def test_scale_rules() -> None:
    assert _scale(0.5, "length", 0.0, 1.0) == 0.5
    assert _scale(None, "length", -0.01, 0.03) == pytest.approx(0.04)
    assert _scale(None, "length", None, 0.3) == pytest.approx(1 * MM)
    assert _scale(None, "angle", 0.0, 0.0) == pytest.approx(0.1 * DEG)
    assert _scale(None, "ratio", None, 1.0) == pytest.approx(0.01)
    assert _scale(None, "percent", 50.0, None) == pytest.approx(1.0)
    assert _scale(None, "force", None, 1.0) == pytest.approx(10.0)
    assert _scale(None, "stiffness", None, 1.0) == pytest.approx(1000.0)
    assert _scale(None, "angle/length", None, -2.0) == pytest.approx(2.0)
    assert _scale(None, "angle/length", None, 0.0) == pytest.approx(1.0)
    assert _scale(None, "angle/length", -3.0, -3.0) == pytest.approx(3.0)


def test_zero_limit_without_scale_rejected(tmp_path: Path) -> None:
    import yaml

    from suspkin.optimize.spec import StudyConfigError

    data = yaml.safe_load(FIX.read_text(encoding="utf-8"))
    data["vehicle"] = str(FIX.parent.parent / "minimal_vehicle.yaml")
    data["constraints"] = [
        {"name": "bs", "type": "slope", "sweep": "bump_f", "metric": "bump_steer", "max": 0}
    ]
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(StudyConfigError, match="constraint 'bs': add `scale`"):
        load_study(p)
    data["constraints"][0]["scale"] = 0.1
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    load_study(p)


def test_g_sign_and_scale(space: DesignSpace) -> None:
    # metric "travel" over the whole sweep spans [-0.03, 0.03] m exactly
    base = _con(space, "camber_band", metric="travel", where=None, scale=None, min=None, max=1.0)
    # one-sided max with the default length scale (1 mm)
    r = _one(space, base.model_copy(update={"max": 0.01}))
    assert r.status == "fail"
    assert r.value == pytest.approx(0.03)
    assert r.g == pytest.approx((0.03 - 0.01) / MM)  # length metric: default scale 1 mm
    r = _one(space, base.model_copy(update={"max": 0.05, "scale": 0.01}))
    assert r.status == "pass"
    assert r.g == pytest.approx(-2.0)
    # one-sided min with a user scale
    r = _one(space, base.model_copy(update={"min": -0.02, "max": None, "scale": 0.005}))
    assert r.value == pytest.approx(-0.03)
    assert r.g == pytest.approx(0.01 / 0.005)
    # two-sided: scale = max - min and the tighter side decides
    r = _one(space, base.model_copy(update={"min": -0.04, "max": 0.02}))
    assert r.g == pytest.approx((0.03 - 0.02) / 0.06)


# --------------------------------------------------------------------------------------
# curve constraints
# --------------------------------------------------------------------------------------


def test_band_window_interpolates_ends(space: DesignSpace) -> None:
    # metric "travel" equals the sweep input: worst point of the window is its upper end
    c = _con(space, "camber_band", metric="travel", where=(-0.0103, 0.0157), min=None, max=0.05)
    r = _one(space, c)
    assert r.value == pytest.approx(0.0157)
    assert r.g == pytest.approx((0.0157 - 0.05) / (1 * DEG))  # the fixture scale is 1 deg
    assert r.status == "pass"
    c2 = c.model_copy(update={"min": -0.0102, "max": None, "scale": 1 * MM})
    r2 = _one(space, c2)  # lower end -0.0103 is 0.1 mm below the min
    assert r2.status == "fail"
    assert r2.value == pytest.approx(-0.0103)
    assert r2.g == pytest.approx(0.1)


def test_band_pass_fail_and_default_window(space: DesignSpace) -> None:
    r = _one(space, _con(space, "camber_band"))
    assert r.status == "pass"
    assert r.dim == "angle"
    assert -2.5 * DEG <= r.value <= -0.5 * DEG
    one_sided = _one(space, _con(space, "camber_band", min=None, max=0.0))
    limit = one_sided.value - 0.1 * DEG
    tight = _one(space, _con(space, "camber_band", min=None, max=limit, scale=0.1 * DEG))
    assert tight.status == "fail"
    assert tight.g == pytest.approx(1.0)
    whole = _one(space, _con(space, "camber_band", where=None, metric="travel", max=0.01, min=None))
    assert whole.value == pytest.approx(0.03)


def test_slope_and_value_analytic(space: DesignSpace) -> None:
    s = _one(space, _con(space, "camber_gain", metric="travel", min=0.9, max=1.1))
    assert s.value == pytest.approx(1.0)
    assert s.dim == "ratio"
    assert s.status == "pass"
    s2 = _one(space, _con(space, "camber_gain", metric="travel", where=(-0.01, 0.02), max=0.5))
    assert s2.status == "fail"
    assert s2.value == pytest.approx(1.0)
    assert s2.g == pytest.approx(0.5 / 0.01)
    v = _one(space, _con(space, "mr_static", metric="travel", at=0.0123, min=0.012, max=0.0125))
    assert v.value == pytest.approx(0.0123)
    assert v.status == "pass"
    v2 = _one(space, _con(space, "mr_static", metric="travel", at=0.0123, min=0.0124, max=0.0125))
    assert v2.status == "fail"
    assert v2.g == pytest.approx(0.0001 / 0.0001)


def test_value_matches_independent_sweep(space: DesignSpace) -> None:
    from suspkin.metrics import add_gradients
    from suspkin.sweeps import states_to_frame

    corner = build_corners(space.baseline)["FL"]
    sw = space.study.sweeps["bump_f"]
    states = sweep_corner(corner, np.linspace(sw.travel[0], sw.travel[1], sw.steps))  # type: ignore[union-attr]
    df = add_gradients(states_to_frame(states, corner, metrics=True))
    expected = float(df["mr_spring"].iloc[np.argmin(np.abs(df["travel"]))])
    r = _one(space, _con(space, "mr_static"))
    assert r.value == pytest.approx(expected)
    assert r.status == ("pass" if 0.9 <= expected <= 1.1 else "fail")


def test_vehicle_sweep_slope(space: DesignSpace) -> None:
    r = _one(space, _con(space, "camber_roll"))
    assert r.status == "pass"
    assert r.dim == "ratio"
    assert np.isfinite(r.value)
    failing = _one(space, _con(space, "camber_roll", max=r.value - 0.1, scale=0.1))
    assert failing.status == "fail"
    assert failing.g == pytest.approx(1.0)


def test_reach_toggle_and_stroke(space: DesignSpace) -> None:
    r = _one(space, _con(space, "travel_f"))
    assert r.status == "pass"
    assert r.g <= 0.0
    # an impossible toggle margin requirement fails, in angle units
    bad = _one(space, _con(space, "travel_f", min_toggle_margin=179 * DEG, stroke=False))
    assert bad.status == "fail"
    assert bad.dim == "angle"
    assert bad.g == pytest.approx((179 * DEG - bad.value) / (0.1 * DEG))
    # only the solve is required
    solve_only = _one(space, _con(space, "travel_f", min_toggle_margin=None, stroke=False))
    assert solve_only.status == "pass"
    assert solve_only.g == -1.0


def test_reach_stroke_violation(space: DesignSpace) -> None:
    corner = build_corners(space.baseline)["FL"]
    states = sweep_corner(corner, np.linspace(-0.03, 0.03, 13))
    lengths = np.array([s.coilover_length for s in states])
    mid = float(0.5 * (lengths.min() + lengths.max()))
    front = space.baseline.front
    cfg = front.coilover.model_copy(update={"length_max": mid})
    veh = space.baseline.model_copy(update={"front": front.model_copy(update={"coilover": cfg})})
    c = _con(space, "travel_f", min_toggle_margin=None)
    r = _one(space, c, vehicle=veh)
    assert r.status == "fail"
    assert r.dim == "length"
    assert r.value == pytest.approx(mid - lengths.max())
    assert r.g == pytest.approx((lengths.max() - mid) / (1 * MM))


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------


def test_summary_value_and_dotted_path(space: DesignSpace) -> None:
    summ = vehicle_summary(space.baseline)
    r = _one(space, _con(space, "rg", max=summ.roll_gradient + 0.5 * DEG))
    assert r.value == pytest.approx(summ.roll_gradient)
    assert r.status == "pass"
    assert r.dim == "angle/gload"
    bad = _one(space, _con(space, "rg", max=summ.roll_gradient - 0.2 * DEG, scale=0.1 * DEG))
    assert bad.status == "fail"
    assert bad.g == pytest.approx(2.0)
    d = _one(space, _con(space, "rg", metric="front.rc_height", min=-1.0, max=1.0))
    assert d.value == pytest.approx(summ.front.rc_height)
    assert d.dim == "length"


def test_summary_none_is_unsolved(space: DesignSpace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluate_mod, "vehicle_summary", lambda v: SimpleNamespace(roll_gradient=None)
    )
    r = _one(space, _con(space, "rg"))
    assert r.status == "unsolved"
    assert r.g == G_FAIL


# --------------------------------------------------------------------------------------
# geometric constraints
# --------------------------------------------------------------------------------------


def test_outside_regions(space: DesignSpace) -> None:
    p = _pt(space, "uca_front")
    base = _con(space, "uca_keepout", over=None)
    c = base.model_copy(update={"region": SphereRegion(center=_t(p, 0.05, 0, 0), radius=0.02)})
    r = _one(space, c)
    assert r.status == "pass"
    assert r.value == pytest.approx(0.03)
    assert r.g == pytest.approx(-30.0)
    c = base.model_copy(update={"region": SphereRegion(center=_t(p, 0.05, 0, 0), radius=0.06)})
    r = _one(space, c)
    assert r.status == "fail"
    assert r.value == pytest.approx(-0.01)
    assert r.g == pytest.approx(10.0)
    cyl = lambda rad: CylinderRegion(  # noqa: E731
        p0=_t(p, -0.1, 0, -0.03),
        p1=_t(p, 0.1, 0, -0.03),
        radius=rad,
    )
    assert _one(space, base.model_copy(update={"region": cyl(0.01)})).g == pytest.approx(-20.0)
    assert _one(space, base.model_copy(update={"region": cyl(0.05)})).g == pytest.approx(20.0)
    box = BoxRegion(lo=_t(p, 0.01, -0.01, -0.01), hi=_t(p, 0.03, 0.01, 0.01))
    assert _one(space, base.model_copy(update={"region": box})).g == pytest.approx(-10.0)
    hexa = HexahedronRegion(vertices=_cube(p + np.array([0.02, 0, 0]), 0.01))
    assert _one(space, base.model_copy(update={"region": hexa})).g == pytest.approx(-10.0)


def test_inside_regions(space: DesignSpace) -> None:
    p = _pt(space, "uca_front")
    base = _con(space, "uca_zone")
    box_in = BoxRegion(lo=_t(p, -0.01, -0.01, -0.01), hi=_t(p, 0.02, 0.02, 0.02))
    r = _one(space, base.model_copy(update={"region": box_in}))
    assert r.status == "pass"
    assert r.value == pytest.approx(-0.01)
    assert r.g == pytest.approx(-10.0)
    box_out = BoxRegion(lo=_t(p, 0.005, -0.01, -0.01), hi=_t(p, 0.03, 0.01, 0.01))
    r = _one(space, base.model_copy(update={"region": box_out}))
    assert r.status == "fail"
    assert r.g == pytest.approx(5.0)
    hexa = HexahedronRegion(vertices=_cube(p, 0.01))
    assert _one(space, base.model_copy(update={"region": hexa})).g == pytest.approx(-10.0)
    sph = SphereRegion(center=_t(p, 0, 0, 0), radius=0.02)
    assert _one(space, base.model_copy(update={"region": sph})).g == pytest.approx(-20.0)
    cyl = CylinderRegion(
        p0=_t(p, -0.1, 0, 0),
        p1=_t(p, 0.1, 0, 0),
        radius=0.03,
    )
    assert _one(space, base.model_copy(update={"region": cyl})).g == pytest.approx(-30.0)


def test_region_over_sweep_takes_worst_state(space: DesignSpace) -> None:
    p = _pt(space, "tie_outer")  # moves with the upright
    box = BoxRegion(lo=_t(p, -0.001, -0.001, -0.001), hi=_t(p, 0.001, 0.001, 0.001))
    c = _con(space, "uca_zone", point="front.tie_outer", region=box)
    static = _one(space, c)
    swept = _one(space, c.model_copy(update={"over": "bump_f"}))
    assert static.status == "pass"
    assert swept.status == "fail"
    corner = build_corners(space.baseline)["FL"]
    sw = space.study.sweeps["bump_f"]
    states = sweep_corner(corner, np.linspace(sw.travel[0], sw.travel[1], sw.steps))  # type: ignore[union-attr]
    pts = np.array([s.points["tie_outer"] for s in states])
    assert swept.value == pytest.approx(float(np.max(np.abs(pts - p))) - 0.001, abs=1e-6)


def test_distance_static_and_over(space: DesignSpace) -> None:
    a, b = _pt(space, "tie_outer"), _pt(space, "uca_front")
    d0 = float(np.linalg.norm(a - b))
    c = _con(space, "tie_len", a="front.tie_outer", b="front.uca_front", min=d0 - MM, max=d0 + MM)
    r = _one(space, c)
    assert r.status == "pass"
    assert r.value == pytest.approx(d0)
    assert r.g == pytest.approx(-0.5)
    r = _one(space, c.model_copy(update={"over": "bump_f"}))
    assert r.status == "fail"
    assert r.g > 0.0
    corner = build_corners(space.baseline)["FL"]
    sw = space.study.sweeps["bump_f"]
    states = sweep_corner(corner, np.linspace(sw.travel[0], sw.travel[1], sw.steps))  # type: ignore[union-attr]
    dist = np.array([np.linalg.norm(s.points["tie_outer"] - b) for s in states])
    worst = dist[np.argmax(np.maximum(dist - (d0 + MM), (d0 - MM) - dist))]
    assert r.value == pytest.approx(worst)


def test_relation_ops_and_offset(space: DesignSpace) -> None:
    z = _pt(space, "uca_front")[2]
    c = _con(space, "uca_floor", rhs=0.1, offset=0.0)
    r = _one(space, c)
    assert r.status == "pass"
    assert r.value == pytest.approx(z)
    assert r.g == pytest.approx((0.1 - z) / MM)
    r = _one(space, c.model_copy(update={"offset": z - 0.1 + 0.002}))
    assert r.status == "fail"
    assert r.g == pytest.approx(2.0)
    r = _one(space, c.model_copy(update={"op": "<=", "offset": z - 0.1 + 0.002}))
    assert r.status == "pass"
    assert r.g == pytest.approx(-2.0)
    ref = _one(space, _con(space, "uca_above"))
    lca = _pt(space, "lca_front")[2]
    assert ref.g == pytest.approx((lca + 0.05 - z) / MM)


def test_joint_angle(space: DesignSpace) -> None:
    corner = build_corners(space.baseline)["FL"]
    sw = space.study.sweeps["bump_f"]
    states = sweep_corner(corner, np.linspace(sw.travel[0], sw.travel[1], sw.steps))  # type: ignore[union-attr]
    expected = float(joint_angles_frame(states, corner)["joint_uca_outer"].max())
    r = _one(space, _con(space, "ubj_angle"))
    assert r.status == "pass"
    assert r.dim == "angle"
    assert r.value == pytest.approx(expected)
    bad = _one(space, _con(space, "ubj_angle", max=expected - 1 * DEG, scale=1 * DEG))
    assert bad.status == "fail"
    assert bad.g == pytest.approx(1.0)
    ax = _one(space, _con(space, "ubj_angle", axis=(1.0, 0.0, 0.0)))
    assert np.isfinite(ax.value)
    assert ax.value != pytest.approx(expected)


# --------------------------------------------------------------------------------------
# cache, failures, evaluation object
# --------------------------------------------------------------------------------------


def test_sweeps_run_once(space: DesignSpace, monkeypatch: pytest.MonkeyPatch) -> None:
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
    # bump_f is shared by band, slope, value, outside and joint_angle; the reach has its own
    # travel; the steer sweep is not used by any constraint; roll is used once
    assert calls == {"corner": 2, "vehicle": 1}
    assert len(ev.results) == len(space.study.constraints)


def test_unused_sweep_does_not_run(space: DesignSpace, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("sweep must not run")

    monkeypatch.setattr(evaluate_mod, "sweep_corner", boom)
    monkeypatch.setattr(evaluate_mod, "sweep_vehicle", boom)
    monkeypatch.setattr(evaluate_mod, "vehicle_summary", boom)
    ev = evaluate(
        DesignSpace(
            space.study.model_copy(
                update={
                    "constraints": (
                        _con(space, "tie_len"),
                        _con(space, "uca_floor"),
                        _con(space, "uca_zone"),
                        _con(space, "rg", enabled=False),
                    )
                }
            ),
            space.baseline,
        )
    )
    assert [r.status for r in ev.results] == ["pass", "pass", "pass", "disabled"]


def test_unsolved_corner_sweep(space: DesignSpace, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*a: Any, **k: Any) -> Any:
        raise KinematicsError("no solution")

    monkeypatch.setattr(evaluate_mod, "sweep_corner", fail)
    ev = evaluate(space)
    by_name = {r.name: r for r in ev.results}
    for n in ("camber_band", "camber_gain", "mr_static", "travel_f", "uca_keepout", "ubj_angle"):
        assert by_name[n].status == "unsolved", n
        assert by_name[n].g == G_FAIL
        assert math.isnan(by_name[n].value)
    for n in ("camber_roll", "tie_len", "uca_floor", "uca_zone", "tube", "hexzone"):
        assert by_name[n].status in ("pass", "fail"), n
    assert not ev.feasible


def test_unsolved_vehicle_sweep(space: DesignSpace, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*a: Any, **k: Any) -> Any:
        raise KinematicsError("no solution")

    monkeypatch.setattr(evaluate_mod, "sweep_vehicle", fail)
    r = {x.name: x for x in evaluate(space).results}
    assert r["camber_roll"].status == "unsolved"
    assert r["camber_roll"].g == G_FAIL
    assert r["camber_band"].status == "pass"


def test_infeasible_design(space: DesignSpace, monkeypatch: pytest.MonkeyPatch) -> None:
    def bad(x: Any) -> Any:
        raise InfeasibleDesign("bad")

    monkeypatch.setattr(space, "to_vehicle", bad)
    ev = evaluate(space, np.full(space.n, 0.5))
    enabled = [r for r in ev.results if r.status != "disabled"]
    assert all(r.status == "infeasible" and r.g == G_FAIL for r in enabled)
    assert not ev.feasible
    assert ev.max_g == G_FAIL
    assert any(r.status == "disabled" for r in ev.results)


def test_inputs(space: DesignSpace) -> None:
    base = evaluate(space)
    via_vehicle = evaluate(space, vehicle=space.baseline)
    assert [r.g for r in base.results if r.status != "disabled"] == pytest.approx(
        [r.g for r in via_vehicle.results if r.status != "disabled"]
    )
    x = np.full(space.n, 0.5)
    ev = evaluate(space, x)
    assert len(ev.results) == len(base.results)
    assert ev.g_vector().shape == base.g_vector().shape
    with pytest.raises(ValueError, match="either x or vehicle"):
        evaluate(space, x, vehicle=space.baseline)
    with pytest.raises(ValueError):
        evaluate(space, np.full(space.n + 1, 0.5))


def test_g_vector_order_and_frames(space: DesignSpace) -> None:
    ev = evaluate(space)
    enabled = [c.name for c in space.study.constraints if c.enabled]
    gv = ev.g_vector()
    assert gv.shape == (len(enabled),)
    assert gv == pytest.approx([r.g for r in ev.results if r.status != "disabled"])
    assert [r.name for r in ev.results if r.status != "disabled"] == enabled
    assert ev.max_g == pytest.approx(float(gv.max()))
    assert ev.feasible == bool(np.all(gv <= 0.0))
    df = ev.frame()
    assert list(df.columns) == [
        "name",
        "type",
        "value",
        "min",
        "max",
        "g",
        "status",
        "dim",
        "detail",
    ]
    assert list(df["name"]) == [c.name for c in space.study.constraints]
    row = df[df["name"] == "camber_band"].iloc[0]
    assert row["min"] == pytest.approx(-2.5 * DEG)
    assert row["max"] == pytest.approx(-0.5 * DEG)


def test_display_frame_units(space: DesignSpace) -> None:
    ev = evaluate(space)
    si = ev.frame().set_index("name")
    disp = ev.display_frame().set_index("name")
    assert "dim" not in disp.columns
    assert disp.loc["camber_band", "unit"] == "deg"
    assert disp.loc["camber_band", "min"] == pytest.approx(-2.5)
    assert disp.loc["camber_band", "max"] == pytest.approx(-0.5)
    assert disp.loc["camber_band", "value"] == pytest.approx(si.loc["camber_band", "value"] / DEG)
    assert disp.loc["camber_gain", "unit"] == "deg/mm"
    assert disp.loc["camber_gain", "max"] == pytest.approx(0.05)
    assert disp.loc["tie_len", "unit"] == "mm"
    assert disp.loc["tie_len", "min"] == pytest.approx(300.0)
    assert disp.loc["rg", "unit"] == "deg/g"
    assert disp.loc["rg", "max"] == pytest.approx(1.2)
    assert disp.loc["mr_static", "unit"] == ""
    assert disp.loc["ack", "unit"] == "%"
    assert disp.loc["ack", "status"] == "disabled"
    assert disp.loc["camber_band", "g"] == pytest.approx(si.loc["camber_band", "g"])


def test_reach_scales_and_detail(space: DesignSpace) -> None:
    c = _con(space, "travel_f", min_toggle_margin=179 * DEG, stroke=False)
    r = _one(space, c)
    assert r.detail == "toggle margin"
    r2 = _one(space, c.model_copy(update={"toggle_scale": 1.0 * DEG}))
    assert r2.g == pytest.approx((179 * DEG - r2.value) / (1.0 * DEG))
    stroke_only = _one(space, _con(space, "travel_f", min_toggle_margin=None, stroke=True))
    assert stroke_only.detail.endswith("stroke remaining")
    assert stroke_only.dim == "length"
    assert stroke_only.g == pytest.approx(-stroke_only.value / MM)
    scaled = _one(
        space, _con(space, "travel_f", min_toggle_margin=None, stroke=True, stroke_scale=5 * MM)
    )
    assert scaled.g == pytest.approx(-scaled.value / (5 * MM))


def _with_fr_sweep(space: DesignSpace, c: Constraint) -> DesignSpace:
    sw = space.study.sweeps["bump_f"].model_copy(update={"corner": "FR"})
    study = space.study.model_copy(
        update={"constraints": (c,), "sweeps": {**space.study.sweeps, "bump_fr": sw}}
    )
    return DesignSpace(study, space.baseline)


def test_region_over_fr_sweep_with_right_region(space: DesignSpace) -> None:
    corner = build_corners(space.baseline)["FR"]
    p = np.array(corner.points["tie_outer"], dtype=np.float64)
    assert p[1] < 0.0
    box = BoxRegion(lo=_t(p, -0.001, -0.001, -0.001), hi=_t(p, 0.001, 0.001, 0.001))
    c = _con(space, "uca_zone", point="front.tie_outer", region=box, over="bump_fr")
    r = evaluate(_with_fr_sweep(space, c)).results[0]
    assert r.status == "fail"
    sw = space.study.sweeps["bump_f"]
    states = sweep_corner(corner, np.linspace(sw.travel[0], sw.travel[1], sw.steps))  # type: ignore[union-attr]
    pts = np.array([s.points["tie_outer"] for s in states])
    assert r.value == pytest.approx(float(np.max(np.abs(pts - p))) - 0.001, abs=1e-6)


def test_missing_point_in_states_is_unsolved(
    space: DesignSpace, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _con(space, "uca_zone", point="front.tie_outer", over="bump_f")
    fake = SimpleNamespace(states=[SimpleNamespace(points={})])
    monkeypatch.setattr(evaluate_mod._Ctx, "sweep_corner_data", lambda self, name: fake)
    assert _one(space, c).status == "unsolved"


def test_fit_line_needs_two_distinct_x() -> None:
    import numpy as np

    from suspkin.optimize.evaluate import _fit_line, _Unsolved

    with pytest.raises(_Unsolved, match="fewer than 2 distinct"):
        _fit_line(np.array([0.01, 0.01]), np.array([1.0, 2.0]))
    assert _fit_line(np.array([0.0, 1.0]), np.array([1.0, 3.0]))[0] == pytest.approx(2.0)
