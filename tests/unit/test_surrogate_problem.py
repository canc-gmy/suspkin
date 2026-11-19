"""SurrogateProblem and the surrogate evaluator (oracle predictors; torch only where noted)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from suspkin.geometry import build_corners
from suspkin.kinematics.batch import corner_columns, make_static, rebuild
from suspkin.kinematics.corner import sweep_corner
from suspkin.metrics import add_gradients
from suspkin.metrics.joints import JOINTS, joint_angle, joint_angles_frame
from suspkin.optimize.design import DesignSpace
from suspkin.optimize.evaluate import G_FAIL, evaluate
from suspkin.optimize.history import read_history
from suspkin.optimize.problem import BudgetExhausted
from suspkin.optimize.spec import StudyConfigError, load_study
from suspkin.surrogate import axle_grids
from suspkin.surrogate.dataset import DatasetError, _current_hashes
from suspkin.surrogate.evaluate import (
    BaselineAxle,
    OraclePredictor,
    SurrogateEvaluator,
    _joint_angles,
    hermite,
)
from suspkin.surrogate.extract import extract_corner
from suspkin.surrogate.problem import SurrogateProblem
from suspkin.sweeps import states_to_frame

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "configs" / "studies" / "fs_reference_optimize.yaml"
VEHICLE = ROOT / "configs" / "vehicles" / "fs_reference.yaml"


def write_study(path: Path, travel_pts: int = 13, rack_pts: int = 5, base: Path = STUDY) -> Path:
    """Copy of a study with an absolute vehicle path and a ``surrogate.grid`` block."""
    txt = base.read_text(encoding="utf-8").replace(
        "vehicle: ../vehicles/fs_reference.yaml", f"vehicle: {VEHICLE.as_posix()}"
    )
    txt += (
        "\nsurrogate:\n  grid:\n"
        f"    travel: {{min: -30 mm, max: 30 mm, points: {travel_pts}}}\n"
        f"    rack: {{min: -30 mm, max: 30 mm, points: {rack_pts}}}\n"
    )
    path.write_text(txt, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    path = write_study(tmp_path_factory.mktemp("study") / "study.yaml")
    study, veh = load_study(path)
    space = DesignSpace(study, veh)
    grids = axle_grids(study.surrogate.grid)  # type: ignore[union-attr]
    preds = {a: OraclePredictor(space, a, grids[a]) for a in ("front", "rear")}
    return path, space, grids, preds


def random_designs(space: DesignSpace, n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([space.from_vehicle(clip=True), rng.random((n, space.n))])


# ------------------------------------------------------------------------------ interpolation


def test_hermite_nodes_smooth_and_outside() -> None:
    x = np.linspace(-0.03, 0.03, 13)
    y = np.stack([np.sin(30 * x), x**2])
    assert np.allclose(hermite(y, x, x), y, atol=1e-12)
    q = np.linspace(-0.03, 0.03, 50)
    ref = np.stack([np.sin(30 * q), q**2])
    assert np.max(np.abs(hermite(y, x, q) - ref)) < 5e-3
    assert np.isnan(hermite(y, x, np.array([0.031]))).all()
    y2 = y.copy()
    y2[0, 6] = np.nan  # a masked node spoils its neighbourhood only
    out = hermite(y2, x, q)
    assert np.isnan(out[0]).any()
    assert np.isfinite(out[0, q < -0.02]).all()
    assert np.isfinite(out[1]).all()


# ------------------------------------------------------------------------------ rebuild vs FOM


@pytest.mark.parametrize("cid", ["FL", "RL", "FR", "RR"])
def test_rebuild_matches_solver_states(setup, cid: str) -> None:  # type: ignore[no-untyped-def]
    _, space, _, _ = setup
    vehs = [space.to_vehicle(x) for x in random_designs(space, 2)]
    corners = [build_corners(v)[cid] for v in vehs]  # type: ignore[index]
    travel = np.linspace(-0.03, 0.03, 13)
    states = [sweep_corner(c, travel) for c in corners]
    st = make_static(corners)
    th = {
        n: np.array([[getattr(s, n) for s in ss] for ss in states])
        for n in ("theta_uca", "theta_lca", "psi")
    }
    guess = np.array([[s.rocker_angle for s in ss] for ss in states])
    rb = rebuild(st, th["theta_uca"], th["theta_lca"], th["psi"], guess)
    for b, ss in enumerate(states):
        for k, s in enumerate(ss):
            for name, p in s.points.items():
                assert np.allclose(rb.pt(name)[b, k], p, atol=1e-10), (cid, name)
            assert rb.coil[b, k] == pytest.approx(s.coilover_length, abs=1e-10)
            assert rb.toggle[b, k] == pytest.approx(s.toggle_margin, abs=1e-9)
            assert rb.rocker[b, k] == pytest.approx(s.rocker_angle, abs=1e-10)
            assert np.allclose(rb.spin[b, k], s.spin_axis, atol=1e-12)


@pytest.mark.parametrize("cid", ["FL", "RL", "FR"])
def test_columns_match_fom_tables(setup, cid: str) -> None:  # type: ignore[no-untyped-def]
    _, space, _, _ = setup
    vehs = [space.to_vehicle(x) for x in random_designs(space, 2)]
    corners = [build_corners(v)[cid] for v in vehs]  # type: ignore[index]
    travel = np.linspace(-0.03, 0.03, 13)
    states = [sweep_corner(c, travel) for c in corners]
    th = {
        n: np.array([[getattr(s, n) for s in ss] for ss in states])
        for n in ("theta_uca", "theta_lca", "psi")
    }
    guess = np.array([[s.rocker_angle for s in ss] for ss in states])
    rb = rebuild(make_static(corners), th["theta_uca"], th["theta_lca"], th["psi"], guess)
    cols = corner_columns(rb, travel)
    for b, c in enumerate(corners):
        df = add_gradients(states_to_frame(states[b], c, metrics=True))
        for name in df.columns:
            if df[name].dtype.kind != "f" or name in ("max_residual",):
                continue
            ref = df[name].to_numpy()
            scale = max(1.0, float(np.nanmax(np.abs(ref[np.isfinite(ref)]), initial=0.0)))
            got = cols[name][b] if name in cols else None
            if got is None:
                got = np.stack([rb.pt(name[:-2])[b, :, "xyz".index(name[-1])]])[0]
            assert np.allclose(got, ref, rtol=1e-8, atol=1e-8 * scale, equal_nan=True), (cid, name)


def test_joint_angles_match_fom(setup) -> None:  # type: ignore[no-untyped-def]
    _, space, _, _ = setup
    veh = space.to_vehicle(random_designs(space, 1)[1])
    corner = build_corners(veh)["FL"]
    travel = np.linspace(-0.03, 0.03, 13)
    states = sweep_corner(corner, travel)
    th = [np.array([[getattr(s, n) for s in states]]) for n in ("theta_uca", "theta_lca", "psi")]
    guess = np.array([[s.rocker_angle for s in states]])
    rb = rebuild(make_static([corner]), *th, guess)
    ref = joint_angles_frame(states, corner)
    for j in JOINTS:
        assert np.allclose(_joint_angles(rb, j, None)[0], ref[f"joint_{j}"], atol=1e-8), j
    static = states[6]
    axis = (0.2, 0.9, 0.3)
    for j in ("uca_outer", "tie_outer", "pushrod_inner"):
        want = [joint_angle(s, static, corner, j, axis) for s in states]
        assert np.allclose(_joint_angles(rb, j, axis)[0], want, atol=1e-8), j


# ------------------------------------------------------------------------------ evaluator


def assert_same(fom, sur, tol: float = 1e-3) -> None:  # type: ignore[no-untyped-def]
    assert [r.status for r in fom.results] == [r.status for r in sur.results]
    assert [o.status for o in fom.objectives] == [o.status for o in sur.objectives]
    assert np.allclose(fom.g_vector(), sur.g_vector(), atol=tol)
    assert np.allclose(fom.f_vector(), sur.f_vector(), atol=tol)


def test_evaluator_matches_fom_with_oracle(setup) -> None:  # type: ignore[no-untyped-def]
    _, space, _, preds = setup
    ev = SurrogateEvaluator(space, preds)
    xs = random_designs(space, 2)
    for x, got in zip(xs, ev.evaluate(xs), strict=True):
        assert_same(evaluate(space, x), got)
    assert set(ev.timers) >= {"vehicles", "predict", "layer", "evaluator", "rebuild"}


def test_masked_node_makes_dependent_constraints_unsolved(setup) -> None:  # type: ignore[no-untyped-def]
    _, space, _, preds = setup

    class Masked:
        """Oracle with the front travel node at +10 mm invalid."""

        def __init__(self, inner: OraclePredictor) -> None:
            self.inner = inner
            self.channels, self.travel, self.rack = inner.channels, inner.travel, inner.rack

        def predict(self, x):  # type: ignore[no-untyped-def]
            y, p = self.inner.predict(x)
            p = p.copy()
            p[:, :, int(np.argmin(np.abs(self.travel - 0.01)))] = 0.0
            return y, p

    ev = SurrogateEvaluator(space, {"front": Masked(preds["front"]), "rear": preds["rear"]})  # type: ignore[arg-type]
    out = ev.evaluate(space.from_vehicle(clip=True))[0]
    by = {r.name: r for r in out.results}
    for name in ("camber_band_f", "toe_band_f", "mr_front", "travel_f", "ubj_angle"):
        assert by[name].status == "unsolved" and by[name].g == G_FAIL, name
    for name in ("bump_steer_r", "mr_rear", "uca_front_zone", "tie_len"):
        assert by[name].status in ("pass", "fail"), name
    assert {o.name: o.status for o in out.objectives}["camber_curve"] == "unsolved"
    assert {o.name: o.status for o in out.objectives}["rc_f"] in ("ok", "unsolved")


def test_unsupported_vehicle_column_is_unsolved(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, _, _, _ = setup
    txt = STUDY.read_text(encoding="utf-8").replace(
        "vehicle: ../vehicles/fs_reference.yaml", f"vehicle: {VEHICLE.as_posix()}"
    )
    txt = txt.replace(
        "constraints:\n",
        "constraints:\n  - {name: arb_roll, type: band, sweep: roll, metric: arb_front_elastic, "
        "max: 1}\n",
        1,
    )
    path = tmp_path / "s.yaml"
    path.write_text(txt, encoding="utf-8")
    write_study(path, base=path)
    study, veh = load_study(path)
    space = DesignSpace(study, veh)
    grids = axle_grids(study.surrogate.grid)  # type: ignore[union-attr]
    preds = {a: OraclePredictor(space, a, grids[a]) for a in ("front", "rear")}
    out = SurrogateEvaluator(space, preds).evaluate(space.from_vehicle(clip=True))[0]
    r = out.results[0]
    assert r.name == "arb_roll" and r.status == "unsolved" and r.g == G_FAIL
    assert "surrogate vehicle layer" in r.detail
    assert all(x.status != "unsolved" for x in out.results[1:])


def test_grid_and_range_checks(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, space, grids, preds = setup
    short = {a: replace_travel(preds[a], 0.02) for a in preds}
    with pytest.raises(StudyConfigError, match="grid"):
        SurrogateEvaluator(space, short)  # model grid differs from the study block
    study, veh = load_study(write_study(tmp_path / "s.yaml"))
    study = study.model_copy(update={"surrogate": None})
    narrow = DesignSpace(study, veh)
    with pytest.raises(StudyConfigError, match="exceeds the surrogate grid"):
        SurrogateEvaluator(narrow, {a: replace_travel(preds[a], 0.02) for a in preds})
    assert grids["front"].shape == (5, 13)


def replace_travel(p: OraclePredictor, half: float):  # type: ignore[no-untyped-def]
    class P:
        channels = p.channels
        travel = np.linspace(-half, half, p.travel.size)
        rack = p.rack

        def predict(self, x):  # type: ignore[no-untyped-def]
            return p.predict(x)

    return P()


def test_baseline_axle_when_no_variables(tmp_path: Path) -> None:
    txt = STUDY.read_text(encoding="utf-8")
    # drop the rear variable: the rear axle then uses its baseline curves
    start = txt.index("  # Rear pushrod outer end")
    end = txt.index("sweeps:")
    path = tmp_path / "front_only.yaml"
    path.write_text(txt[:start] + "\n" + txt[end:], encoding="utf-8")
    write_study(path, base=path)
    study, veh = load_study(path)
    space = DesignSpace(study, veh)
    grids = axle_grids(study.surrogate.grid)  # type: ignore[union-attr]
    preds = {
        "front": OraclePredictor(space, "front", grids["front"]),
        "rear": BaselineAxle(space, "rear", grids["rear"]),
    }
    xs = random_designs(space, 1)
    got = SurrogateEvaluator(space, preds).evaluate(xs)
    for x, g in zip(xs, got, strict=True):
        assert_same(evaluate(space, x), g)


# ------------------------------------------------------------------------------ problem


def make_problem(setup, run_dir: Path, **kw):  # type: ignore[no-untyped-def]
    path, _, _, preds = setup
    return SurrogateProblem(path, None, run_dir, predictors=preds, **kw)


def test_margin_shifts_g_exactly_and_keeps_failures(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    xs = random_designs(setup[1], 2)
    with (
        make_problem(setup, tmp_path / "a") as p0,
        make_problem(setup, tmp_path / "b", margin=0.05) as p1,
    ):
        f0, g0, ft0, _, _ = p0.evaluate_batch(xs)
        f1, g1, ft1, mg1, fe1 = p1.evaluate_batch(xs)
        assert np.allclose(g1, g0 + 0.05, atol=1e-12)
        assert np.allclose(f1, f0) and np.allclose(ft1, ft0)
        assert np.allclose(mg1, g0.max(axis=1) + 0.05)
        assert np.array_equal(fe1, g1.max(axis=1) <= 0.0)
        raw = np.array([p1.raw_g[k] for k in p1.raw_g])
        assert np.allclose(raw, g0)


def test_margin_leaves_failed_constraints_at_g_fail(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path, space, _, preds = setup

    class Invalid:
        def __init__(self, inner: OraclePredictor) -> None:
            self.inner = inner
            self.channels, self.travel, self.rack = inner.channels, inner.travel, inner.rack

        def predict(self, x):  # type: ignore[no-untyped-def]
            y, p = self.inner.predict(x)
            return y, np.zeros_like(p)

    bad = {a: Invalid(preds[a]) for a in preds}
    with SurrogateProblem(path, None, tmp_path / "r", predictors=bad, margin=0.1) as p:  # type: ignore[arg-type]
        _, g, f_total, _, feasible = p.evaluate_batch(space.from_vehicle(clip=True))
    names = [c.name for c in space.study.constraints if c.enabled]
    curve = {"camber_band_f", "toe_band_f", "bump_steer_r", "mr_front", "mr_rear", "travel_f"}
    for n, gv in zip(names, g[0], strict=True):
        if n in curve:
            assert gv == G_FAIL, n
        if n in ("uca_front_zone", "tie_len", "uca_above_lca"):
            assert gv < 1.0  # static constraints are evaluated (and shifted)
    assert not feasible[0] and f_total[0] > 100.0


def test_budget_cache_and_history(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    xs = random_designs(setup[1], 3)
    with make_problem(setup, tmp_path / "run", evaluations=2) as p:
        assert p.n == setup[1].n and p.workers == 1
        p.evaluate_batch(xs[:2])
        assert p.n_evals == 2 and p.exhausted()
        p.evaluate_batch(xs[:2])  # fully cached: free
        assert p.n_evals == 2
        with pytest.raises(BudgetExhausted):
            p.evaluate_batch(xs[2:3])
        f, g, _, mg, _ = p.evaluate_one(xs[0])
        assert f.shape == (4,) and np.isclose(mg, g.max())
        assert p.n_evals == 2
    df = read_history(tmp_path / "run" / "history.csv")
    assert len(df) == 2 and set(df["source"]) == {"surrogate"}
    assert not math.isnan(df["f_total"].iloc[0])


def test_de_driver_runs_on_the_surrogate_problem(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from suspkin.optimize.drivers import _run_de
    from suspkin.optimize.spec import Optimizer

    with make_problem(setup, tmp_path / "run", evaluations=20) as p:
        _run_de(p, Optimizer(seed=1), False, evals=20)
        n = p.n_evals
    df = read_history(tmp_path / "run" / "history.csv")
    assert len(df) == n >= 20
    assert (df["source"] == "surrogate").all()
    assert isinstance(df, pd.DataFrame)


def test_missing_model_folder_is_reported(tmp_path: Path) -> None:
    path = write_study(tmp_path / "s.yaml")
    with pytest.raises(FileNotFoundError, match="no trained model"):
        SurrogateProblem(path, tmp_path / "models", tmp_path / "run")


def test_extract_channels_contain_solver_coordinates(tmp_path: Path) -> None:
    study, veh = load_study(write_study(tmp_path / "s.yaml"))
    grid = axle_grids(study.surrogate.grid)["rear"]  # type: ignore[union-attr]
    s = extract_corner(veh, "rear", grid)
    assert {"theta_uca", "theta_lca", "psi"} <= set(s.channels)
    assert s.feasible


# ------------------------------------------------------------------------------ real networks


def synthetic_models(
    space: DesignSpace, grids, root: Path, study_path: Path, names_override=None, meta_patch=None
) -> None:  # type: ignore[no-untyped-def]
    """Save tiny networks (baseline curves + smooth x-dependent noise) as ``root/<axle>/model``."""
    from suspkin.surrogate.channels import channels_for
    from suspkin.surrogate.data_types import SPLIT_TEST, SPLIT_VAL, AxleDataset
    from suspkin.surrogate.model import MLPConfig
    from suspkin.surrogate.sampling import AxleSpace
    from suspkin.surrogate.train import TrainConfig, train

    for axle in ("front", "rear"):
        asp = AxleSpace(space, axle)  # type: ignore[arg-type]
        g = grids[axle]
        base = extract_corner(space.baseline, axle, g)  # type: ignore[arg-type]
        ch = channels_for(axle)  # type: ignore[arg-type]
        rng = np.random.default_rng(0)
        n = 120
        x = rng.random((n, asp.n))
        pert = np.sin(x @ rng.standard_normal((asp.n, len(ch))))
        tol = np.array([c.tol for c in ch])
        y = base.Y[None] + 3 * tol[None, :, None, None] * pert[:, :, None, None]
        mask = np.broadcast_to(base.mask, (n, *base.mask.shape)).copy()
        split = np.zeros(n, dtype=np.uint8)
        split[96:108], split[108:] = SPLIT_VAL, SPLIT_TEST
        ds = AxleDataset(
            axle,  # type: ignore[arg-type]
            x,
            y.astype(np.float32),
            mask,
            mask.all(axis=(1, 2)),
            split,
            np.array(["lhs"] * n),
            np.ones(n),
            tuple(c.name for c in ch),
            (names_override or {}).get(axle, asp.names),
            g.travel,
            g.rack,
            {
                **_current_hashes(study_path, axle),
                "channels": [c.name for c in ch],
                **(meta_patch or {}),
            },
        )
        model = train(
            ds,
            MLPConfig(n_layers=2, width=32),
            TrainConfig(max_epochs=8, patience=8, batch_size=32, device="cpu"),
        )
        model.save(root / axle / "model")


def test_problem_loads_models_and_runs_de(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    from suspkin.optimize.drivers import _run_de
    from suspkin.optimize.spec import Optimizer

    path, space, grids, _ = setup
    synthetic_models(space, grids, tmp_path / "models", path)
    with SurrogateProblem(
        path, tmp_path / "models", tmp_path / "run", device="cpu", evaluations=40
    ) as p:
        f, g, ft, _, _ = p.evaluate_batch(random_designs(space, 3))
        assert f.shape == (4, 4) and g.shape == (4, 12) and np.isfinite(ft).all()
        _run_de(p, Optimizer(seed=1), False, evals=40)
    df = read_history(tmp_path / "run" / "history.csv")
    assert len(df) >= 40 and (df["source"] == "surrogate").all()


def test_model_parameter_mismatch_is_reported(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    from suspkin.surrogate.sampling import AxleSpace

    path, space, grids, _ = setup
    n_front = AxleSpace(space, "front").n  # type: ignore[arg-type]
    wrong = {"front": tuple(f"q{i}" for i in range(n_front))}
    synthetic_models(space, grids, tmp_path / "models", path, names_override=wrong)
    with pytest.raises(StudyConfigError, match="model parameters"):
        SurrogateProblem(path, tmp_path / "models", tmp_path / "run", device="cpu")


# ------------------------------------------------------------------------------ pushrod mounts


def _check_rebuild(corners) -> None:  # type: ignore[no-untyped-def]
    travel = np.linspace(-0.03, 0.03, 13)
    states = [sweep_corner(c, travel) for c in corners]
    th = [
        np.array([[getattr(s, n) for s in ss] for ss in states])
        for n in ("theta_uca", "theta_lca", "psi")
    ]
    guess = np.array([[s.rocker_angle for s in ss] for ss in states])
    rb = rebuild(make_static(corners), *th, guess)
    cols = corner_columns(rb, travel)
    for b, ss in enumerate(states):
        for k, s in enumerate(ss):
            for name, p in s.points.items():
                assert np.allclose(rb.pt(name)[b, k], p, atol=1e-10), name
            assert rb.coil[b, k] == pytest.approx(s.coilover_length, abs=1e-10)
            assert rb.toggle[b, k] == pytest.approx(s.toggle_margin, abs=1e-9)
        ref = add_gradients(states_to_frame(ss, corners[b], metrics=True))
        for name in ("camber", "toe", "pushrod_angle", "mr_spring", "rc_height"):
            assert np.allclose(cols[name][b], ref[name].to_numpy(), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("mount", ["lca", "upright"])
def test_rebuild_pushrod_mount_variants(setup, tmp_path: Path, mount: str) -> None:  # type: ignore[no-untyped-def]
    from suspkin.geometry import load_vehicle

    _, space, _, _ = setup
    vpath = tmp_path / "v.yaml"
    vpath.write_text(
        VEHICLE.read_text(encoding="utf-8").replace(
            "pushrod_mount: uca", f"pushrod_mount: {mount}"
        ),
        encoding="utf-8",
    )
    base = load_vehicle(vpath)
    assert base.front.pushrod_mount == mount and base.rear.pushrod_mount == mount
    study = space.study.model_copy(update={"surrogate": None})
    sp = DesignSpace(study, base)
    for cid in ("FL", "FR", "RR"):
        vehs = [sp.to_vehicle(x) for x in random_designs(sp, 1)]
        _check_rebuild([build_corners(v)[cid] for v in vehs])  # type: ignore[index]


# ------------------------------------------------------------------------------ per-design range


def test_full_rack_range_follows_each_designs_vehicle(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from suspkin.surrogate.vehicle_layer import AxleBatch

    txt = STUDY.read_text(encoding="utf-8").replace(
        "constraints:\n",
        "constraints:\n  - {name: steer_b, type: band, sweep: steer, metric: rack, max: 1}\n",
        1,
    )
    path = tmp_path / "s.yaml"
    path.write_text(txt, encoding="utf-8")
    write_study(path, base=path)
    study, veh = load_study(path)
    space = DesignSpace(study, veh)
    grids = axle_grids(study.surrogate.grid)  # type: ignore[union-attr]
    preds = {a: OraclePredictor(space, a, grids[a]) for a in ("front", "rear")}
    ev = SurrogateEvaluator(space, preds)
    d = veh.model_dump()
    d["front"]["steering"]["rack_travel"] = (-0.02, 0.02)
    narrow = type(veh).model_validate(d)
    xs = random_designs(space, 1)  # two designs; the second gets the narrow vehicle
    batches = {}
    for a in ("front", "rear"):
        y, p = preds[a].predict(np.ascontiguousarray(xs[:, ev.asp[a].idx]))
        batches[a] = AxleBatch(y, p >= 0.5, tuple(preds[a].channels), ev.grids[a])
    _, tables = ev._run_layer([veh, narrow], batches)
    rack = tables["steer"]["rack"]
    assert rack[0, 0] == pytest.approx(-0.03) and rack[0, -1] == pytest.approx(0.03)
    assert rack[1, 0] == pytest.approx(-0.02) and rack[1, -1] == pytest.approx(0.02)


def test_model_with_other_vehicle_hash_is_rejected(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    path, space, grids, _ = setup
    synthetic_models(space, grids, tmp_path / "models", path, meta_patch={"vehicle_hash": "0" * 64})
    with pytest.raises(DatasetError, match="vehicle_hash differs"):
        SurrogateProblem(path, tmp_path / "models", tmp_path / "run", device="cpu")
