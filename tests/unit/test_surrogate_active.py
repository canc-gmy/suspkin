"""Active-learning optimizer: pure rules, box adapter and a tiny end-to-end run (CPU)."""

# ruff: noqa: N806, E501
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from suspkin.optimize.drivers import RunError
from suspkin.optimize.history import cache_key, read_history
from suspkin.optimize.spec import (
    ActiveLearningSpec,
    Optimizer,
    StudyConfigError,
    TrustRegionSpec,
    load_study,
)
from suspkin.plotting.surrogate import plot_al_progress, plot_al_radius, plot_al_residual
from suspkin.surrogate.active import (
    AL_COLUMNS,
    BoxProblem,
    StopState,
    advance_stop,
    box_bounds,
    crowding_distance,
    curve_residual,
    hits_boundary,
    select_infill,
    stop_reason,
    trust_ratio,
    update_radius,
)
from tests.unit.test_surrogate_sampling import REAR_VAR, STUDY

TR = TrustRegionSpec(initial=0.2, min=0.01, max=0.5)


# --------------------------------------------------------------------------- pure rules


@pytest.mark.parametrize(
    ("r", "rho", "hit", "expected"),
    [
        (0.2, 0.9, True, 0.4),  # good and at the boundary: expand
        (0.4, 0.9, True, 0.5),  # capped at max
        (0.2, 0.9, False, 0.2),  # good but interior: keep
        (0.2, 0.5, True, 0.2),  # middling: keep
        (0.2, 0.75, True, 0.2),  # boundary of the rule is exclusive
        (0.2, 0.1, True, 0.1),  # poor: shrink
        (0.2, 0.25, True, 0.2),  # exactly 0.25 keeps
        (0.015, 0.0, False, 0.01),  # floored at min
        (0.2, -3.0, True, 0.1),  # worse than before
        (0.2, math.nan, True, 0.1),  # no predicted improvement
        (0.2, math.inf, True, 0.4),
    ],
)
def test_update_radius_table(r: float, rho: float, hit: bool, expected: float) -> None:
    assert update_radius(r, rho, hit, TR) == pytest.approx(expected)


def test_trust_ratio_cases() -> None:
    kw = {"feasible_best": True, "feasible_fom": True}
    assert trust_ratio(10.0, 6.0, 7.0, **kw) == pytest.approx(0.75)  # 3 of 4 realised
    assert trust_ratio(10.0, 6.0, 10.0, **kw) == pytest.approx(0.0)
    assert trust_ratio(10.0, 6.0, 12.0, **kw) == pytest.approx(-0.5)
    assert math.isnan(trust_ratio(10.0, 10.0, 8.0, **kw))  # nothing predicted
    assert math.isnan(trust_ratio(10.0, 11.0, 8.0, **kw))
    assert trust_ratio(10.0, 6.0, 1.0, feasible_best=True, feasible_fom=False) == 0.0
    # no feasible verified design yet: feasibility decides
    nb = {"feasible_best": False, "max_g_best": 0.5}
    assert trust_ratio(10.0, 20.0, 30.0, feasible_fom=True, **nb) == 1.0
    assert trust_ratio(10.0, 20.0, 30.0, feasible_fom=False, max_g_fom=0.2, **nb) == 0.5
    assert trust_ratio(10.0, 20.0, 30.0, feasible_fom=False, max_g_fom=0.9, **nb) == 0.0


def test_box_and_boundary() -> None:
    lo, hi = box_bounds([0.1, 0.5, 0.95], 0.2)
    assert np.allclose(lo, [0.0, 0.3, 0.75]) and np.allclose(hi, [0.3, 0.7, 1.0])
    c = np.array([0.5, 0.5])
    assert hits_boundary([0.7, 0.5], c, 0.2)
    assert hits_boundary([0.5, 0.3], c, 0.2)
    assert not hits_boundary([0.6, 0.55], c, 0.2)
    assert not hits_boundary([1.0, 0.5], np.array([0.9, 0.5]), 0.2)  # clipped by the cube


def _spec(**stop: object) -> ActiveLearningSpec:
    return ActiveLearningSpec.model_validate(
        {"max_iterations": 10, "fom_evaluations": 100, "stop": {"patience": 3, **stop}}
    )


def test_advance_stop_counters() -> None:
    s = StopState()
    s = advance_stop(s, agree=True, improved=False, at_min=False)
    s = advance_stop(s, agree=True, improved=False, at_min=True)
    assert (s.agree, s.stale, s.at_min) == (2, 2, 1)
    s = advance_stop(s, agree=False, improved=True, at_min=True)
    assert (s.agree, s.stale, s.at_min) == (0, 0, 0)  # an improvement resets all but agree rule
    s = advance_stop(s, agree=False, improved=False, at_min=True)
    assert (s.agree, s.stale, s.at_min) == (0, 1, 1)


@pytest.mark.parametrize(
    ("state", "done", "n_fom", "elapsed", "kw", "expected"),
    [
        (StopState(), 0, 5, 0.0, {}, None),
        (StopState(agree=3), 4, 5, 0.0, {}, "agreement"),
        (StopState(agree=2), 4, 5, 0.0, {}, None),
        (StopState(stale=3), 4, 5, 0.0, {}, "stagnation"),
        (StopState(at_min=3), 4, 5, 0.0, {}, "radius_min"),
        (StopState(), 10, 5, 0.0, {}, "max_iterations"),
        (StopState(), 4, 100, 0.0, {}, "fom_budget"),
        (StopState(), 4, 40, 0.0, {"fom_budget": 40}, "fom_budget"),
        (StopState(), 4, 100, 0.0, {"fom_budget": 500}, None),
        (StopState(), 4, 5, 50.0, {"time_budget": 40.0}, "time_budget"),
        (StopState(), 4, 5, 50.0, {}, None),
        (StopState(agree=3, stale=3), 10, 100, 1e9, {"time_budget": 1.0}, "agreement"),
    ],
)
def test_stop_reason_table(
    state: StopState,
    done: int,
    n_fom: int,
    elapsed: float,
    kw: dict[str, float],
    expected: str | None,
) -> None:
    got = stop_reason(
        state,
        iterations_done=done,
        n_fom=n_fom,
        elapsed=elapsed,
        al=_spec(),
        **kw,  # type: ignore[arg-type]
    )
    assert got == expected


def test_crowding_distance() -> None:
    pts = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [4.0, 0.0]])
    d = crowding_distance(pts)
    assert np.isinf(d[0]) and np.isinf(d[3])
    assert d[1] == pytest.approx((2 - 0) / 4 + (4 - 2) / 4)
    assert d[2] == pytest.approx((4 - 1) / 4 + (3 - 0) / 4)
    assert np.isinf(crowding_distance(pts[:2])).all()


def test_curve_residual() -> None:
    y = np.zeros((2, 2, 3))
    p = y.copy()
    p[0, :, :] = 0.002  # channel 0 off by 0.002
    mask = np.ones((2, 3), dtype=bool)
    tol0 = 0.002
    from suspkin.surrogate.channels import channels_for

    names = tuple(c.name for c in channels_for("rear"))[:2]
    from suspkin.surrogate.active import _TOL

    r = curve_residual(p, y, mask, names)
    assert r == pytest.approx(0.5 * tol0 / _TOL[names[0]])
    assert math.isnan(curve_residual(p, y, np.zeros((2, 3), dtype=bool), names))
    bad = p.copy()
    bad[:, 0, 0] = 99.0
    mask2 = mask.copy()
    mask2[0, 0] = False  # the wild node is masked out
    assert curve_residual(bad, y, mask2, names) == pytest.approx(r)


def _history(x: np.ndarray, ft: np.ndarray, feas: np.ndarray) -> pd.DataFrame:
    n = x.shape[1]
    df = pd.DataFrame(x, columns=[f"x_v{i}" for i in range(n)])
    df["f_a"] = ft
    df["f_b"] = ft[::-1]
    df["f_total"] = ft
    df["max_g"] = np.where(feas, -0.1, 0.5 + ft)
    df["feasible"] = feas
    df["status"] = "ok"
    return df


def test_select_infill_weighted() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((12, 3))
    ft = np.arange(12, dtype=float)[::-1]  # row 11 is the best
    feas = np.ones(12, dtype=bool)
    feas[11] = False  # the best is infeasible: skip it
    study, _ = _tiny_study_objects()
    df = _history(x, ft, feas)
    lo, hi = np.zeros(3), np.ones(3)
    known = {cache_key(x[9])}
    x_star, X = select_infill(
        df,
        study,
        "weighted",
        top=2,
        perturb=3,
        lo=lo,
        hi=hi,
        r=0.2,
        rng=np.random.default_rng(1),
        known=known,
    )
    assert np.allclose(x_star, x[10])  # best feasible
    assert np.allclose(X[0], x_star)
    assert np.allclose(X[1], x[8]) and np.allclose(X[2], x[7])  # next best, known row 9 skipped
    assert X.shape == (6, 3)
    pert = X[3:]
    assert np.all(np.abs(pert - x_star) <= 0.05 + 1e-12)
    again = select_infill(
        df,
        study,
        "weighted",
        top=2,
        perturb=3,
        lo=lo,
        hi=hi,
        r=0.2,
        rng=np.random.default_rng(1),
        known=known,
    )[1]
    assert np.array_equal(X, again)  # seeded
    # perturbations stay in the box
    _, Xb = select_infill(
        df,
        study,
        "weighted",
        top=0,
        perturb=20,
        lo=x_star - 0.01,
        hi=x_star + 0.01,
        r=0.4,
        rng=np.random.default_rng(2),
    )
    assert np.all(Xb >= x_star - 0.01 - 1e-12) and np.all(Xb <= x_star + 0.01 + 1e-12)


def test_select_infill_pareto_and_fallback() -> None:
    study, _ = _tiny_study_objects()
    cols = [f"f_{o.name}" for o in study.objectives]
    assert len(cols) >= 2
    n = 6
    x = np.linspace(0.1, 0.9, n)[:, None] * np.ones((1, 2))
    df = pd.DataFrame(x, columns=["x_a", "x_b"])
    for j, c in enumerate(cols):
        df[c] = np.linspace(0, 1, n) if j == 0 else np.linspace(1, 0, n)
    for c in cols[2:]:
        df[c] = 0.0
    df["f_total"] = df[cols].sum(axis=1)
    df["max_g"] = -1.0
    df["feasible"] = True
    df["status"] = "ok"
    lo, hi = np.zeros(2), np.ones(2)
    xs, X = select_infill(
        df, study, "pareto", top=2, perturb=0, lo=lo, hi=hi, r=0.2, rng=np.random.default_rng(0)
    )
    assert X.shape[0] == 3
    # the extremes of the front have infinite crowding distance and follow the knee
    firsts = {tuple(np.round(x[0], 6)), tuple(np.round(x[-1], 6))}
    assert {tuple(np.round(r, 6)) for r in X[1:]} == firsts
    # without a feasible point it falls back to the least infeasible design
    df2 = df.copy()
    df2["feasible"] = False
    df2["max_g"] = np.arange(n, dtype=float)[::-1] + 1.0
    xs2, _ = select_infill(
        df2, study, "pareto", top=0, perturb=0, lo=lo, hi=hi, r=0.2, rng=np.random.default_rng(0)
    )
    assert np.allclose(xs2, x[-1])
    assert xs.shape == (2,)


# --------------------------------------------------------------------------- tiny study

TINY_BLOCKS = """sweeps:
  bump_f: {kind: corner, corner: FL, travel: [-15 mm, 15 mm], steps: 7}
constraints:
  - {name: camber_band_f, type: band, sweep: bump_f, metric: camber, where: [-15 mm, 15 mm], min: -4 deg, max: 1 deg, scale: 1 deg}
  - {name: toe_band_f, type: band, sweep: bump_f, metric: toe, where: [-15 mm, 15 mm], min: -1 deg, max: 1 deg, scale: 0.1 deg}
  - {name: tie_len, type: distance, a: front.tie_inner, b: front.tie_outer, min: 300 mm}
objectives:
  - name: camber_curve
    type: curve
    sweep: bump_f
    metric: camber
    target: {x: [-15 mm, 0 mm, 15 mm], y: [-1.2 deg, -0.6 deg, 0 deg]}
    where: [-15 mm, 15 mm]
    weight: 1.0
    scale: 0.1 deg
  - {name: bump_steer_f, type: minimize, sweep: bump_f, metric: bump_steer, of: max_abs, where: [-15 mm, 15 mm], scale: 0.01 deg/mm, weight: 1.0}
optimizer:
  mode: weighted
  method: de
  seed: 1
  workers: 1
  budget: {evaluations: 100}
"""

AL_BLOCK = """  training: {device: cpu, max_epochs: 4, patience: 2}
  hpo: {n_trials: 1, timeout: 5 min}
  active_learning:
    max_iterations: %d
    fom_evaluations: %d
    surrogate_evaluations: 300
    infill: {top: 1, perturb: 2}
    finetune: {epochs: 3, weight: 5.0, trigger: 0, skip_for: 0, on_cpu: run}
    margin: 0.05
    trust_region: {initial: 0.2, min: 0.01, max: 0.5}
    stop: {eps_f: 0.01, patience: 5}
"""


def tiny_text(max_iterations: int = 2, fom: int = 60, n: int = 40, rear: bool = False) -> str:
    base = (STUDY if rear else STUDY.replace(REAR_VAR, "")).replace(
        "n_initial: 12", f"n_initial: {n}"
    )
    return base.rstrip("\n") + "\n" + AL_BLOCK % (max_iterations, fom) + TINY_BLOCKS


def _tiny_study_objects():  # type: ignore[no-untyped-def]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.yaml"
        p.write_text(tiny_text(), encoding="utf-8")
        return load_study(p)


@pytest.fixture(scope="module")
def pristine(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Study file and a surrogate root with the sampled dataset and a trained tiny model."""
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    from suspkin.surrogate.dataset import sample_axle
    from suspkin.surrogate.workflow import train_surrogate

    base = tmp_path_factory.mktemp("al")
    study = base / "study.yaml"
    study.write_text(tiny_text(), encoding="utf-8")
    root = base / "root"
    sample_axle(study, root, "front")
    train_surrogate(study, root, axles=["front"], n_trials=1)
    return study, root


@pytest.fixture
def work(pristine: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path, Path]:
    study, root = pristine
    new_root = tmp_path / "root"
    shutil.copytree(root, new_root)
    s = tmp_path / "study.yaml"
    shutil.copyfile(study, s)
    return s, new_root, tmp_path / "run"


# --------------------------------------------------------------------------- box adapter


def test_box_problem_maps_and_drivers_run(tmp_path: Path) -> None:
    from suspkin.optimize.drivers import _run_de
    from suspkin.surrogate import axle_grids
    from suspkin.surrogate.evaluate import OraclePredictor
    from suspkin.surrogate.problem import SurrogateProblem

    p = tmp_path / "s.yaml"
    p.write_text(tiny_text(), encoding="utf-8")
    study, veh = load_study(p)
    from suspkin.optimize.design import DesignSpace

    space = DesignSpace(study, veh)
    grids = axle_grids(study.surrogate.grid)  # type: ignore[union-attr]
    preds = {a: OraclePredictor(space, a, grids[a]) for a in ("front", "rear")}
    with SurrogateProblem(p, None, tmp_path / "run", predictors=preds, evaluations=30) as sp:  # type: ignore[arg-type]
        center = space.from_vehicle(clip=True)
        lo, hi = box_bounds(center, 0.2)
        box = BoxProblem(sp, lo, hi, center)
        assert box.n == space.n
        z = np.array([[0.0] * space.n, [1.0] * space.n, [0.5] * space.n])
        assert np.allclose(box.to_x(z), [lo, hi, (lo + hi) / 2])
        assert np.allclose(box.to_z(box.to_x(z)), z)
        assert np.allclose(box.space.from_vehicle(clip=True), box.to_z(center))
        assert np.allclose(box.to_x(box.space.from_vehicle(clip=True)), center)
        box.evaluate_batch(z[:1], "x")
        assert cache_key(lo) in sp.cache  # the problem caches x, not z
        _run_de(box, Optimizer(seed=1), False, evals=30)  # type: ignore[arg-type]
        df = read_history(sp.history_path)
        zdf = read_history(box.history_path)
    xcols = [f"x_{n}" for n in space.names]
    x = df[xcols].to_numpy()
    assert len(df) >= 30
    assert np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)
    assert np.allclose(zdf[xcols].to_numpy(), box.to_z(x))
    assert zdf[xcols].to_numpy().min() >= 0.0 and zdf[xcols].to_numpy().max() <= 1.0


def test_box_requires_positive_span(tmp_path: Path) -> None:
    class Fake:
        study = None
        space = None

    with pytest.raises(ValueError, match="hi > lo"):
        BoxProblem(Fake(), np.zeros(2), np.zeros(2), np.zeros(2))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- end to end


def _n_rows(root: Path, axle: str = "front") -> tuple[int, pd.Series]:
    from suspkin.surrogate.dataset import load_dataset

    ds = load_dataset(root, axle)  # type: ignore[arg-type]
    return ds.n, pd.Series(ds.source)


def test_end_to_end_and_resume(work: tuple[Path, Path, Path]) -> None:
    from suspkin.surrogate.active import run_active_learning

    study, root, run = work
    n0, _ = _n_rows(root)
    msgs: list[str] = []
    res = run_active_learning(
        study, root, run, workers=1, device="cpu", report=False, echo=msgs.append
    )
    assert res.new_iterations == 2 and res.iterations == 2 and res.stop_reason == "max_iterations"
    assert any("iter 0" in m for m in msgs)

    hist = read_history(run / "history.csv")
    assert hist["source"].iloc[0] == "baseline"
    al = pd.read_csv(run / "al.csv")
    assert list(al.columns) == list(AL_COLUMNS) and list(al["k"]) == [0, 1]
    assert (al["radius"] > 0).all() and (al["n_fom"].diff().dropna() >= 0).all()
    for k in (0, 1):
        n_k = int((hist["source"] == f"al_{k}").sum())
        assert n_k == int(al["n_new"][k]) and 1 <= n_k <= 4
        assert (run / "al" / f"iter_{k}" / "surrogate_history.csv").is_file()
        assert (root / "models" / "front" / f"al_{k}" / "model.pt").is_file()
    assert (root / "models" / "front" / "pre_al" / "model.pt").is_file()
    assert len(hist) == al["n_fom"].iloc[-1]

    # dataset grew by exactly the new FOM points, weighted 5
    from suspkin.surrogate.dataset import load_dataset

    ds = load_dataset(root, "front")
    new = ds.source != "lhs"
    assert ds.n == n0 + int(al["n_new"].sum()) and int(new.sum()) == int(al["n_new"].sum())
    assert np.all(ds.weight[new] == 5.0) and np.all(ds.weight[~new] == 1.0)
    assert set(ds.source[new]) == {"al_0", "al_1"}

    # outputs: best.yaml is a FOM-verified design
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    bx = result["best"]["x"]
    xcols = [f"x_{n}" for n in bx]
    match = np.all(np.isclose(hist[xcols].to_numpy(), [bx[n] for n in bx], atol=1e-9), axis=1)
    assert match.any()
    if hist["feasible"].any():
        assert result["best"]["feasible"] and bool(hist.loc[match, "feasible"].iloc[0])
        best_f = hist.loc[hist["feasible"], "f_total"].min()
        assert result["best"]["f_total"] == pytest.approx(best_f, abs=1e-6)
    assert (run / "best.yaml").is_file() and (run / "study.yaml").is_file()
    for fig in ("al_progress", "al_residual", "al_radius", "convergence"):
        assert (run / "figures" / f"{fig}.png").stat().st_size > 0
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["surrogate"]["device"] == "cpu" and meta["surrogate"]["models"]["front"]
    assert meta["surrogate"]["models_pre_al"]["front"]
    assert meta["active_learning"]["stop_reason"] == "max_iterations"

    # resume with a larger iteration budget continues the counter
    study.write_text(tiny_text(max_iterations=3), encoding="utf-8")
    res2 = run_active_learning(study, root, run, resume=True, workers=1, device="cpu", report=False)
    assert res2.new_iterations == 1 and res2.iterations == 3
    al2 = pd.read_csv(run / "al.csv")
    assert list(al2["k"]) == [0, 1, 2]
    assert list(al2["radius"].iloc[:2]) == list(al["radius"])
    assert al2["radius"].iloc[2] == pytest.approx(al["radius_next"].iloc[1])
    assert (read_history(run / "history.csv")["source"] == "al_2").any()
    # nothing left to do: stops without a new iteration
    res3 = run_active_learning(study, root, run, resume=True, workers=1, device="cpu", report=False)
    assert res3.new_iterations == 0 and res3.stop_reason == "max_iterations"


def test_fom_budget_stops(work: tuple[Path, Path, Path]) -> None:
    from suspkin.surrogate.active import run_active_learning

    study, root, run = work
    res = run_active_learning(
        study, root, run, workers=1, device="cpu", report=False, evaluations=4
    )
    hist = read_history(run / "history.csv")
    assert len(hist) <= 4 and res.stop_reason == "fom_budget"
    assert res.new_iterations >= 1


def test_errors(work: tuple[Path, Path, Path], tmp_path: Path) -> None:
    from suspkin.surrogate.active import run_active_learning
    from suspkin.surrogate.dataset import DatasetError

    study, root, run = work
    with pytest.raises(DatasetError):
        run_active_learning(study, tmp_path / "nowhere", run, workers=1, device="cpu")
    no_model = tmp_path / "nm"
    shutil.copytree(root / "dataset", no_model / "dataset")
    with pytest.raises(DatasetError, match="no trained model"):
        run_active_learning(study, no_model, run, workers=1, device="cpu")
    with pytest.raises(RunError, match="--resume needs"):
        run_active_learning(study, root, None, resume=True, workers=1, device="cpu")
    with pytest.raises(RunError, match="no history"):
        run_active_learning(study, root, run, resume=True, workers=1, device="cpu")
    with pytest.raises(RunError, match="unknown method"):
        run_active_learning(study, root, run, method="bogus", workers=1, device="cpu")
    plain = tmp_path / "plain.yaml"
    plain.write_text(STUDY.split("surrogate:")[0], encoding="utf-8")
    with pytest.raises(StudyConfigError):
        run_active_learning(plain, root, run)
    with pytest.raises(ValueError, match="device"):
        run_active_learning(study, root, run, device="tpu")


# --------------------------------------------------------------------------- figures


def _al_table() -> pd.DataFrame:
    k = np.arange(5)
    return pd.DataFrame(
        {
            "k": k,
            "radius": [0.2, 0.2, 0.1, 0.05, 0.05],
            "f_nn": [3.0, 2.5, 2.0, 1.9, 1.8],
            "f_fom": [5.0, 3.0, 2.2, 1.95, 1.85],
            "best_f": [5.0, 3.0, 2.2, 1.95, 1.85],
            "resid_front": [8.0, 3.0, 1.5, 1.0, 0.8],
            "resid_rear": [np.nan] * 5,
        }
    )


def test_al_figures_from_table() -> None:
    al = _al_table()
    assert len(plot_al_progress(al).axes[0].lines) == 3
    f = plot_al_residual(al)
    assert len(f.axes[0].lines) == 2  # front curve + tolerance line (rear is all nan)
    assert len(plot_al_radius(al).axes[0].lines) >= 1
    with pytest.raises(ValueError):
        plot_al_progress(al.drop(columns="f_nn"))


def test_loop_with_oracle_predictors_improves_and_agrees(tmp_path: Path) -> None:
    """Perfect surrogate (oracle curves, no fine-tuning): the loop must improve and agree."""
    from suspkin.optimize.design import DesignSpace
    from suspkin.surrogate import axle_grids
    from suspkin.surrogate.active import run_active_learning
    from suspkin.surrogate.evaluate import OraclePredictor

    text = (
        tiny_text(max_iterations=6)
        .replace("surrogate_evaluations: 300", "surrogate_evaluations: 150")
        .replace("margin: 0.05", "margin: 0.0")
        .replace("patience: 5", "patience: 2")
    )
    study = tmp_path / "s.yaml"
    study.write_text(text, encoding="utf-8")
    st, veh = load_study(study)
    space = DesignSpace(st, veh)
    grids = axle_grids(st.surrogate.grid)  # type: ignore[union-attr]
    preds = {a: OraclePredictor(space, a, grids[a]) for a in ("front", "rear")}
    res = run_active_learning(
        study,
        tmp_path / "no_root",
        tmp_path / "run",
        workers=1,
        report=False,
        finetune=False,
        predictors=preds,  # type: ignore[arg-type]
    )
    al = pd.read_csv(tmp_path / "run" / "al.csv")
    assert res.stop_reason == "agreement" and res.iterations < 6
    assert res.run.f_total < res.run.baseline.f_total - 1e-3
    assert res.run.feasible
    # interpolation of the grid curves (vehicle layer) leaves ~0.1 % difference
    assert np.allclose(al["f_nn"], al["f_fom"], rtol=5e-3)
    assert (al["resid_front"] < 1e-6).all()
    assert not (tmp_path / "no_root").exists()
    print(
        f"oracle loop: baseline {res.run.baseline.f_total:.4f} best {res.run.f_total:.4f} "
        f"iterations {res.iterations} stop {res.stop_reason}"
    )


# --------------------------------------------------------------------------- oracle loops


class Biased:
    """Oracle predictor with a smooth, design-dependent bias on one channel."""

    def __init__(self, inner, channel: str, amp: float) -> None:  # type: ignore[no-untyped-def]
        self.inner, self.channel, self.amp = inner, channel, amp
        self.channels, self.travel, self.rack = inner.channels, inner.travel, inner.rack

    def predict(self, x):  # type: ignore[no-untyped-def]
        y, p = self.inner.predict(x)
        y = y.copy()
        i = list(self.channels).index(self.channel)
        y[:, i] += self.amp * np.sin(3.0 * np.atleast_2d(x).sum(axis=1))[:, None, None]
        return y, p


def fast(text: str, evals: int = 60) -> str:
    return text.replace("surrogate_evaluations: 300", f"surrogate_evaluations: {evals}").replace(
        "margin: 0.05", "margin: 0.0"
    )


def oracle_preds(study: Path, bias: float = 0.0):  # type: ignore[no-untyped-def]
    from suspkin.optimize.design import DesignSpace
    from suspkin.surrogate import axle_grids
    from suspkin.surrogate.evaluate import OraclePredictor

    st, veh = load_study(study)
    space = DesignSpace(st, veh)
    grids = axle_grids(st.surrogate.grid)  # type: ignore[union-attr]
    preds = {a: OraclePredictor(space, a, grids[a]) for a in ("front", "rear")}
    if bias:
        preds["front"] = Biased(preds["front"], "camber", bias)  # type: ignore[assignment]
    return preds


def oracle_run(tmp: Path, text: str, name: str = "run", bias: float = 0.0, **kw):  # type: ignore[no-untyped-def]
    from suspkin.surrogate.active import run_active_learning

    study = tmp / f"{name}.yaml"
    study.write_text(text, encoding="utf-8")
    res = run_active_learning(
        study,
        tmp / "no_root",
        tmp / name,
        workers=1,
        report=False,
        finetune=False,
        predictors=oracle_preds(study, bias),
        **kw,
    )
    return res, study


def test_biased_oracle_runs_rho_paths(tmp_path: Path) -> None:
    text = fast(tiny_text(max_iterations=3, rear=True))
    res, _ = oracle_run(tmp_path, text, bias=2e-3)
    al = pd.read_csv(tmp_path / "run" / "al.csv")
    assert res.iterations == 3 and res.stop_reason == "max_iterations"
    assert (np.diff(al["best_f"]) <= 1e-12).all()  # the verified best never gets worse
    assert (al["resid_front"] > 0.02).all()  # the bias is seen in the curve residual
    assert np.isfinite(al["resid_rear"]).all() and (al["resid_rear"] < 1e-6).all()
    assert (np.abs(al["f_nn"] - al["f_fom"]) > 1e-4).any()
    rho = al["rho"].dropna()
    assert len(rho) >= 1 and (rho != 1.0).any()  # rho is a genuine ratio, not a flag
    assert (al["radius"] >= 0.01 - 1e-12).all() and (al["radius"] <= 0.5 + 1e-12).all()
    assert list(al["radius"].iloc[1:]) == pytest.approx(list(al["radius_next"].iloc[:-1]))
    print(f"biased oracle (7 vars): {al['sec_total'].mean():.1f} s per iteration")


def test_oracle_budget_resume_and_changed_study(tmp_path: Path) -> None:
    from suspkin.surrogate.active import run_active_learning

    text = fast(tiny_text(max_iterations=10))
    res, study = oracle_run(tmp_path, text, evaluations=5)
    run = tmp_path / "run"
    n1 = len(read_history(run / "history.csv"))
    assert res.stop_reason == "fom_budget" and n1 <= 5
    preds = oracle_preds(study)
    root = tmp_path / "no_root"

    def go(path: Path, **kw):  # type: ignore[no-untyped-def]
        return run_active_learning(
            path, root, run, resume=True, workers=1, report=False, finetune=False,
            predictors=preds, **kw,
        )  # fmt: skip

    again = go(study, evaluations=5)
    assert again.new_iterations == 0 and again.stop_reason == "fom_budget"
    assert len(read_history(run / "history.csv")) == n1  # budget counts the resumed rows
    more = go(study, evaluations=n1 + 4)
    n2 = len(read_history(run / "history.csv"))
    assert more.new_iterations >= 1 and n1 < n2 <= n1 + 4
    before = (run / "study.yaml").read_text(encoding="utf-8")
    changed = tmp_path / "changed.yaml"
    changed.write_text(text.replace("weight: 1.0", "weight: 2.0", 1), encoding="utf-8")
    with pytest.raises(RunError, match="changed"):
        go(changed)
    assert (run / "study.yaml").read_text(encoding="utf-8") == before


def test_no_candidates_stops_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import suspkin.surrogate.active as act

    monkeypatch.setattr(act, "_run_inner", lambda box, opt: None)  # nothing evaluated
    res, _ = oracle_run(tmp_path, fast(tiny_text(max_iterations=3)))
    assert res.stop_reason == "no_candidates" and res.iterations == 0
    assert res.run.feasible  # the baseline is feasible: the CLI would exit 0
    assert (tmp_path / "run" / "best.yaml").is_file()


def test_pareto_mode_oracle(tmp_path: Path) -> None:
    text = fast(tiny_text(max_iterations=2), evals=40).replace(
        "mode: weighted\n  method: de", "mode: pareto\n  method: nsga2"
    )
    assert "pareto" in text
    res, _ = oracle_run(tmp_path, text)
    al = pd.read_csv(tmp_path / "run" / "al.csv")
    assert res.iterations == 2 and al["rho"].isin([0.0, 1.0]).all()
    assert res.run.n_pareto >= 1 and (tmp_path / "run" / "pareto").is_dir()
    hist = read_history(tmp_path / "run" / "history.csv")
    pareto = pd.read_csv(tmp_path / "run" / "pareto.csv")
    assert set(pareto["eval_id"]) <= set(hist["eval_id"])  # FOM-verified points only
    print(f"pareto oracle: {al['sec_total'].mean():.1f} s per iteration")


def test_dominated_helper() -> None:
    from suspkin.surrogate.active import _dominated

    front = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    assert _dominated([2.5, 2.5], front)
    assert not _dominated([2.0, 2.0], front)  # equal is not dominated
    assert not _dominated([0.5, 5.0], front)
    assert not _dominated([1.0, 1.0], front[:0])


# --------------------------------------------------------------------------- dataset consistency


def test_failed_infill_point_is_appended_all_invalid(
    work: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from suspkin.optimize.design import DesignSpace, InfeasibleDesign
    from suspkin.surrogate import axle_grids
    from suspkin.surrogate.active import _axle_sample
    from suspkin.surrogate.dataset import append_rows, load_dataset
    from suspkin.surrogate.sampling import AxleSpace

    study, root, _ = work
    st, veh = load_study(study)
    space = DesignSpace(st, veh)
    asp = AxleSpace(space, "front")
    grid = axle_grids(st.surrogate.grid)["front"]  # type: ignore[union-attr]

    def bad(self, x, *a, **k):  # type: ignore[no-untyped-def]
        raise InfeasibleDesign("out of domain")

    with monkeypatch.context() as m:
        m.setattr(DesignSpace, "to_vehicle", bad)
        smp = _axle_sample(space, asp, grid, asp.baseline())
    assert not smp.mask.any() and not smp.feasible and np.isnan(smp.Y).all()
    n0 = load_dataset(root, "front").n
    append_rows(root, "front", asp.baseline()[None, :], [smp], "al_9", weight=5.0)
    ds = load_dataset(root, "front")
    assert ds.n == n0 + 1 and not ds.mask[-1].any()
    assert ds.weight[-1] == 5.0 and ds.source[-1] == "al_9"


def test_resume_recovers_interrupted_append(
    work: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import suspkin.surrogate.active as act
    from suspkin.surrogate.dataset import load_dataset

    study, root, run = work
    n0 = load_dataset(root, "front").n
    real = act.append_rows

    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("disk full")

    monkeypatch.setattr(act, "append_rows", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        act.run_active_learning(study, root, run, workers=1, device="cpu", report=False)
    monkeypatch.setattr(act, "append_rows", real)
    hist = read_history(run / "history.csv")
    n_al = int(hist["source"].astype(str).str.startswith("al_").sum())
    assert n_al >= 1 and load_dataset(root, "front").n == n0  # FOM rows without dataset rows
    msgs: list[str] = []
    study.write_text(tiny_text(max_iterations=2), encoding="utf-8")
    act.run_active_learning(
        study, root, run, resume=True, workers=1, device="cpu", report=False, echo=msgs.append
    )
    assert any("appended" in m for m in msgs)
    ds = load_dataset(root, "front")
    al_x = ds.X[np.char.startswith(ds.source.astype(str), "al_")]
    keys = [cache_key(x) for x in al_x]
    assert len(set(keys)) == len(keys) and len(keys) >= n_al  # recovered, no duplicates


# --------------------------------------------------------------------------- shared models


def test_no_dataset_models_only(work: tuple[Path, Path, Path]) -> None:
    from suspkin.surrogate.active import run_active_learning
    from suspkin.surrogate.dataset import DatasetError

    study, root, run = work
    shutil.rmtree(root / "dataset")
    msgs: list[str] = []
    res = run_active_learning(
        study, root, run, workers=1, device="cpu", report=False, echo=msgs.append
    )
    assert res.new_iterations == 2 and not (root / "dataset").exists()
    al = pd.read_csv(run / "al.csv", keep_default_na=False)
    assert set(al["skip_reason_front"]) == {"no_dataset"} and al["skipped_front"].all()
    assert sum("dataset not found" in m and "fine-tuning disabled" in m for m in msgs) == 1

    # resume works too (no dataset recovery attempted)
    study.write_text(tiny_text(max_iterations=3), encoding="utf-8")
    res2 = run_active_learning(study, root, run, resume=True, workers=1, device="cpu", report=False)
    assert res2.new_iterations == 1

    # a model of another study is still refused
    mj = root / "models" / "front" / "model" / "model.json"
    info = json.loads(mj.read_text(encoding="utf-8"))
    info["meta"]["grid_hash"] = "bogus"
    mj.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(DatasetError, match="does not match the study"):
        run_active_learning(study, root, run.with_name("run2"), workers=1, device="cpu")


def test_with_dataset_not_flagged_no_dataset(work: tuple[Path, Path, Path]) -> None:
    from suspkin.surrogate.active import run_active_learning

    study, root, run = work
    msgs: list[str] = []
    run_active_learning(
        study, root, run, workers=1, device="cpu", report=False, evaluations=4, echo=msgs.append
    )
    al = pd.read_csv(run / "al.csv", keep_default_na=False)
    assert "no_dataset" not in set(al["skip_reason_front"])
    assert not any("dataset not found" in m for m in msgs)
