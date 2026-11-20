"""Hardpoint grid: snapping, study resolution, optimiser outputs, surrogate, active learning."""
# ruff: noqa: N803, N806

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import yaml

from suspkin.optimize import read_history, run_optimization
from suspkin.optimize.design import (
    DesignSpace,
    has_grid_node,
    node_range,
    snap_hexahedron,
    snap_interval,
)
from suspkin.optimize.problem import Problem
from suspkin.optimize.regions import hex_contains, hex_locate, hex_map
from suspkin.optimize.spec import DEFAULT_RESOLUTION, Study, StudyConfigError, load_study
from tests.unit.test_surrogate_sampling import STUDY, VEHICLE

RES = 5e-4
FIX_DIR = Path(__file__).parents[1] / "fixtures"
TINY = FIX_DIR / "studies" / "tiny_optimize.yaml"

_CUBE = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=float,
)


def on_grid(v: Any, res: float = RES) -> bool:
    k = np.asarray(v, dtype=float) / res
    return bool(np.all(np.abs(k - np.rint(k)) < 1e-6))


def _space(tmp_path: Path, text: str) -> DesignSpace:
    p = tmp_path / "s.yaml"
    p.write_text(text, encoding="utf-8")
    study, veh = load_study(p)
    return DesignSpace(study, veh)


# --------------------------------------------------------------------------- functions


def test_snap_interval_absolute_grid_and_bounds() -> None:
    lo, hi = np.array([0.1003]), np.array([0.1097])
    p = np.array([[0.10031], [0.1049], [0.10501], [0.1096]])
    q = snap_interval(p, lo, hi, RES)
    assert on_grid(q)
    # bounds are off-grid: the nearest node inside is taken (ceil / floor)
    assert q[0, 0] == pytest.approx(0.1005) and q[3, 0] == pytest.approx(0.1095)
    assert q[1, 0] == pytest.approx(0.105) and q[2, 0] == pytest.approx(0.105)
    assert np.all((q >= lo) & (q <= hi))
    # bounds on nodes are kept
    assert snap_interval([0.1], [0.1], [0.11], RES)[0] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="no grid node"):
        snap_interval([0.1], [0.1001], [0.1004], RES)
    kmin, kmax = node_range([0.1003], [0.1097], RES)
    assert (kmin[0], kmax[0]) == (201.0, 219.0)


def test_snap_interval_vectorised_matches_rows() -> None:
    rng = np.random.default_rng(1)
    lo, hi = np.array([0.0, 0.2, -0.1]), np.array([0.05, 0.27, -0.05])
    p = lo + rng.random((50, 3)) * (hi - lo)
    q = snap_interval(p, lo, hi, RES)
    assert q.shape == (50, 3) and on_grid(q)
    assert np.allclose(q, np.vstack([snap_interval(r, lo, hi, RES) for r in p]))
    assert np.all(np.abs(q - p) <= RES / 2 + 1e-12)


def test_snap_hexahedron_inside_and_snap_to_neighbour() -> None:
    # a thin slab 0.3 mm thick in z, at z in [0.1, 0.1003]: rounding in z often leaves it
    verts = _CUBE * np.array([0.0105, 0.0105, 0.0003]) + np.array([0.2, 0.1, 0.1])
    assert has_grid_node(verts, RES)
    rng = np.random.default_rng(2)
    p = np.asarray(hex_map(verts, rng.random((300, 3))))
    q = snap_hexahedron(p, verts, RES)
    assert on_grid(q) and np.all(hex_contains(verts, q))
    assert np.allclose(q[:, 2], 0.1)  # the only z node of the slab
    # a point whose rounded node is inside is just rounded
    inner = _CUBE * 0.01 + 0.2
    pt = np.array([0.2051, 0.2049, 0.2052])
    assert np.allclose(snap_hexahedron(pt, inner, RES), [0.205, 0.205, 0.205])


def test_has_grid_node_empty_region() -> None:
    tiny = _CUBE * 0.0002 + np.array([0.2001, 0.1001, 0.1001])
    assert not has_grid_node(tiny, RES)
    assert has_grid_node(_CUBE * 0.0002 + np.array([0.1999, 0.0999, 0.0999]), RES)


def test_hex_locate_roundtrip_and_outside() -> None:
    verts = _CUBE * 0.02 + 0.1
    u = np.random.default_rng(3).random((20, 3))
    p = np.asarray(hex_map(verts, u))
    uu, ok = hex_locate(verts, p)
    assert ok.all() and np.allclose(uu, u, atol=1e-9)
    assert not hex_contains(verts, np.array([[0.5, 0.5, 0.5], [0.09, 0.1, 0.1]])).any()


# --------------------------------------------------------------------------- spec


def _study(extra: str = "", var_extra: str = "", rest: str = "") -> str:
    return (
        f"schema: 1\nvehicle: {VEHICLE}\n{extra}"
        "variables:\n  - point: front.tie_inner\n"
        f"    domain: {{offset: {{x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}}}}\n{var_extra}{rest}"
    )


def test_spec_resolution_default_units_null_and_override(tmp_path: Path) -> None:
    def load(text: str) -> Study:
        p = tmp_path / "s.yaml"
        p.write_text(text, encoding="utf-8")
        return load_study(p)[0]

    s = load(_study())
    assert s.resolution == DEFAULT_RESOLUTION == RES
    assert load(_study("resolution: 1 mm\n")).resolution == pytest.approx(1e-3)
    assert load(_study("resolution: 0.001\n")).resolution == pytest.approx(1e-3)  # SI
    assert load(_study("resolution: null\n")).resolution is None
    v = load(_study("resolution: 1 mm\n", "    resolution: 0.25 mm\n")).variables[0]
    assert v.resolution == pytest.approx(2.5e-4)
    v = load(_study("", "    resolution: null\n")).variables[0]
    assert "resolution" in v.model_fields_set and v.resolution is None
    for bad in ("resolution: 0 mm\n", "resolution: -1 mm\n"):
        with pytest.raises(StudyConfigError):
            load(_study(bad))


def test_spec_empty_region_is_a_clear_error(tmp_path: Path) -> None:
    text = (
        f"schema: 1\nvehicle: {VEHICLE}\nvariables:\n  - point: front.tie_inner\n"
        "    domain: {offset: {x: [-0.0004 mm, 0.0004 mm]}}\n"
    )
    # the offset of +-0.4 um around the baseline contains no node unless the baseline is on one
    p = tmp_path / "s.yaml"
    p.write_text(text.replace("-0.0004 mm, 0.0004 mm", "0.1 mm, 0.3 mm"), encoding="utf-8")
    with pytest.raises(StudyConfigError, match=r"no node of the 0\.5 mm grid"):
        load_study(p)
    p.write_text(text.replace("-0.0004 mm, 0.0004 mm", "0.1 mm, 0.3 mm") + "resolution: null\n")
    load_study(p)  # continuous: fine
    box = (
        f"schema: 1\nvehicle: {VEHICLE}\nvariables:\n  - point: front.tie_inner\n"
        "    domain: {box: {min: [0.0801, 0.1501, 0.1401], max: [0.0804, 0.1504, 0.1404]}}\n"
    )
    p.write_text(box, encoding="utf-8")
    with pytest.raises(StudyConfigError, match="no node"):
        load_study(p)


def test_spec_empty_hexahedron_error(tmp_path: Path) -> None:
    v = (_CUBE * 0.0002 + np.array([0.2001, 0.1001, 0.1001])).tolist()
    text = (
        f"schema: 1\nvehicle: {VEHICLE}\nvariables:\n  - point: front.tie_inner\n"
        f"    domain: {{hexahedron: {{vertices: {v}}}}}\n"
    )
    p = tmp_path / "s.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(StudyConfigError, match="no node"):
        load_study(p)


# --------------------------------------------------------------------------- design space

REF_STUDY = Path("configs/studies/fs_reference_optimize.yaml")


def test_reference_study_designs_are_on_the_grid() -> None:
    study, veh = load_study(REF_STUDY)
    space = DesignSpace(study, veh)
    assert space.n > 0
    rng = np.random.default_rng(0)
    X = rng.random((200, space.n))
    Q = space.quantise_x(X)
    assert Q.shape == X.shape
    assert np.allclose(space.quantise_x(Q), Q, atol=1e-12)  # idempotent
    for x in X[:25]:
        for lab, p in space.points(x).items():
            assert on_grid(p), (lab, p)
        assert np.allclose(
            [c for p in space.points(x).values() for c in p],
            [c for p in space.points(space.quantise_x(x)).values() for c in p],
            atol=1e-9,
        )
    # single row in, single row out
    assert space.quantise_x(X[0]).shape == (space.n,)
    # the baseline x is not moved to the grid by from_vehicle (reference evaluated as-is)
    xb = space.from_vehicle(clip=True)
    assert np.allclose(
        space.points(xb, quantise=False)["rear.pushrod_outer"],
        [
            *veh.rear.hardpoints.pushrod_outer  # type: ignore[misc]
        ],
    )


def test_quantise_false_and_null_resolution_are_continuous(tmp_path: Path) -> None:
    cont = _space(tmp_path, STUDY + "resolution: null\n")
    grid = _space(tmp_path, STUDY)
    x = np.random.default_rng(4).random(grid.n)
    assert np.array_equal(cont.quantise_x(x), x)
    assert not on_grid([c for p in cont.points(x).values() for c in p])
    a = grid.points(x, quantise=False)
    b = cont.points(x)
    assert all(np.allclose(a[k], b[k]) for k in a)
    assert grid.to_vehicle(x, quantise=False) == cont.to_vehicle(x)
    assert np.all(cont.grid_step_z() == 0.0)


def test_offset_domain_snaps_to_absolute_grid_inside_the_bounds(tmp_path: Path) -> None:
    sp = _space(tmp_path, STUDY)
    base = np.array(sp.baseline.rear.hardpoints.pushrod_outer)
    assert not on_grid(base)  # the reference pushrod is off the grid
    lo = base + np.array([-0.010, 0.0, -0.008])
    hi = base + np.array([0.010, 0.0, 0.008])
    rng = np.random.default_rng(5)
    for _ in range(100):
        p = sp.points(rng.random(sp.n))["rear.pushrod_outer"]
        assert on_grid(p[[0, 2]])
        assert p[1] == pytest.approx(base[1])  # fixed axis untouched
        assert p[0] >= lo[0] - 1e-12 and p[0] <= hi[0] + 1e-12
        assert p[2] >= lo[2] - 1e-12 and p[2] <= hi[2] + 1e-12


def test_per_variable_resolution(tmp_path: Path) -> None:
    text = STUDY.replace(
        "domain: {offset: {x: [-4 mm, 4 mm], y: [-4 mm, 4 mm], z: [-4 mm, 4 mm]}}",
        "domain: {offset: {x: [-4 mm, 4 mm], y: [-4 mm, 4 mm], z: [-4 mm, 4 mm]}}\n"
        "    resolution: 2 mm",
    )
    sp = _space(tmp_path, text)
    x = np.random.default_rng(6).random(sp.n)
    pts = sp.points(x)
    assert on_grid(pts["front.uca_outer"], 2e-3)
    assert on_grid(pts["front.tie_inner"][[0, 2]])
    steps = sp.grid_step_z()
    assert steps.min() > 0 and steps.max() == pytest.approx(2e-3 / 8e-3)


def test_problem_alias_and_fd_step(tmp_path: Path) -> None:
    study = tmp_path / "s.yaml"
    study.write_text(
        TINY.read_text(encoding="utf-8").replace(
            "vehicle: ../minimal_vehicle.yaml",
            f"vehicle: {(FIX_DIR / 'minimal_vehicle.yaml').as_posix()}",
        ),
        encoding="utf-8",
    )
    with Problem(study, tmp_path / "run", workers=1) as pb:
        x = np.array([0.4137, 0.6021])
        F1, *_ = pb.evaluate_batch(x[None, :])
        xq = pb.space.quantise_x(x)
        assert not np.allclose(x, xq)
        n0 = pb.n_evals
        F2, *_ = pb.evaluate_batch(xq[None, :])  # the snapped design is cached
        assert pb.n_evals == n0 and np.array_equal(F1, F2)
        hist = read_history(pb.history_path)
        assert len(hist) == 1 and np.allclose(hist[[f"x_{n}" for n in pb.space.names]], xq)
        # finite differences step at least one grid step
        grad, *_ = pb.fd_jacobian(x)
        steps = pb.space.grid_step_z()
        assert np.all(steps > 1e-3) and np.all(np.isfinite(grad))
        assert len(read_history(pb.history_path)) >= 3


def test_designs_snapping_to_one_node_count_once_against_the_budget(tmp_path: Path) -> None:
    study = tmp_path / "s.yaml"
    study.write_text(
        TINY.read_text(encoding="utf-8").replace(
            "vehicle: ../minimal_vehicle.yaml",
            f"vehicle: {(FIX_DIR / 'minimal_vehicle.yaml').as_posix()}",
        ),
        encoding="utf-8",
    )
    with Problem(study, tmp_path / "run", workers=1) as pb:
        pb.max_evals = 1
        x = np.array([0.4137, 0.6021])
        xq = pb.space.quantise_x(x)
        x2 = xq + 0.1 * pb.space.grid_step_z() * np.array([1.0, -1.0])  # same node, other raw x
        assert np.array_equal(pb.space.quantise_x(x2), xq) and not np.allclose(x, x2)
        pb.evaluate_batch(x[None, :])
        assert pb.n_evals == 1 and pb.exhausted()
        # Budget is spent, but every alias of the evaluated node is a cache hit: no error,
        # no new evaluation, no new history row. n_evals counts evaluations, not cache keys.
        pb.evaluate_batch(np.vstack([x, x2, xq]))
        assert pb.n_evals == 1
        assert len(pb.cache) > pb.n_evals  # aliases live in the cache
        assert len(read_history(pb.history_path)) == pb.n_evals


# --------------------------------------------------------------------------- classic optimiser


def test_classic_optimise_reports_grid_designs(tmp_path: Path) -> None:
    study = tmp_path / "tiny.yaml"
    study.write_text(
        TINY.read_text(encoding="utf-8").replace(
            "vehicle: ../minimal_vehicle.yaml",
            f"vehicle: {(FIX_DIR / 'minimal_vehicle.yaml').as_posix()}",
        ),
        encoding="utf-8",
    )
    res = run_optimization(
        study, tmp_path / "run", method="de+slsqp", workers=1, evaluations=40, report=False
    )
    df = read_history(res.out_dir / "history.csv")
    pcols = [c for c in df.columns if c.startswith("p_front.tie_inner_")]
    assert len(pcols) == 3
    ok = df["status"] != "infeasible"
    assert ok.any()
    # free axes x and z are on the grid; y is fixed at the baseline
    assert on_grid(df.loc[ok, "p_front.tie_inner_x"].to_numpy())
    assert on_grid(df.loc[ok, "p_front.tie_inner_z"].to_numpy())
    best = yaml.safe_load((res.out_dir / "best.yaml").read_text(encoding="utf-8"))
    pt = best["front"]["hardpoints"]["tie_inner"]
    assert on_grid([pt[0], pt[2]])
    # the reported best is the evaluated, snapped design
    assert res.feasible and res.f_total <= res.baseline.f_total + 1e-12


def test_classic_optimise_continuous_when_null(tmp_path: Path) -> None:
    study = tmp_path / "tiny.yaml"
    txt = TINY.read_text(encoding="utf-8").replace(
        "vehicle: ../minimal_vehicle.yaml",
        f"vehicle: {(FIX_DIR / 'minimal_vehicle.yaml').as_posix()}",
    )
    study.write_text(txt.replace("variables:", "resolution: null\nvariables:", 1), encoding="utf-8")
    res = run_optimization(
        study, tmp_path / "run", method="de", workers=1, evaluations=16, report=False
    )
    df = read_history(res.out_dir / "history.csv")
    assert not on_grid(df["p_front.tie_inner_x"].to_numpy())


# --------------------------------------------------------------------------- surrogate


def test_surrogate_problem_feeds_quantised_reencoded_inputs(tmp_path: Path) -> None:
    from suspkin.surrogate import axle_grids
    from suspkin.surrogate.evaluate import OraclePredictor
    from suspkin.surrogate.problem import SurrogateProblem
    from suspkin.surrogate.sampling import AxleSpace

    path = tmp_path / "s.yaml"
    path.write_text(STUDY.replace("n_initial: 12", "n_initial: 6"), encoding="utf-8")
    study, veh = load_study(path)
    space = DesignSpace(study, veh)
    grids = axle_grids(study.surrogate.grid)  # type: ignore[union-attr]

    class Recorder:
        def __init__(self, inner: OraclePredictor) -> None:
            self.inner = inner
            self.seen: list[np.ndarray] = []
            self.channels, self.travel, self.rack = inner.channels, inner.travel, inner.rack

        def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            self.seen.append(np.array(X))
            return self.inner.predict(X)

    rec = {a: Recorder(OraclePredictor(space, a, grids[a])) for a in ("front", "rear")}
    x = np.random.default_rng(7).random((3, space.n))
    with SurrogateProblem(path, None, tmp_path / "run", predictors=rec) as sp:  # type: ignore[arg-type]
        F1, *_ = sp.evaluate_batch(x)
        xq = space.quantise_x(x)
        assert not np.allclose(x, xq)
        for a in ("front", "rear"):
            seen = np.vstack(rec[a].seen)
            assert np.allclose(seen, xq[:, AxleSpace(space, a).idx], atol=1e-12)
        n0 = sp.n_evals
        F2, *_ = sp.evaluate_batch(xq)  # the snapped designs are already cached
        assert sp.n_evals == n0 and np.allclose(F1, F2)
        hist = read_history(sp.history_path)
        assert np.allclose(hist[[f"x_{n}" for n in space.names]], xq)
        assert on_grid(hist["p_front.tie_inner_x"].to_numpy())


def test_dataset_sampling_is_continuous_and_hashes_ignore_resolution(tmp_path: Path) -> None:
    from suspkin.surrogate.dataset import _current_hashes, load_dataset, sample_axle
    from suspkin.surrogate.sampling import lhs

    text = STUDY.replace("n_initial: 12", "n_initial: 6")
    runs = {}
    for name, txt in {
        "default": text,
        "null": text + "resolution: null\n",
        "coarse": text + "resolution: 2 mm\n",
    }.items():
        p = tmp_path / f"{name}.yaml"
        p.write_text(txt, encoding="utf-8")
        sample_axle(p, tmp_path / name, "rear", workers=1)
        runs[name] = (p, load_dataset(tmp_path / name, "rear"))
    base = runs["null"][1]
    for name in ("default", "coarse"):
        assert np.array_equal(runs[name][1].X, base.X), name
        assert np.array_equal(runs[name][1].Y, base.Y, equal_nan=True), name
        assert _current_hashes(runs[name][0], "rear") == _current_hashes(runs["null"][0], "rear")
    # the sampled X is the seeded LHS (baseline first), never snapped
    study, veh = load_study(runs["default"][0])
    space = DesignSpace(study, veh)
    from suspkin.surrogate.sampling import AxleSpace

    asp = AxleSpace(space, "rear")
    ref = lhs(6, asp.n, seed=3)
    assert np.allclose(base.X[1:], ref[1:])
    assert np.allclose(base.X[0], asp.baseline())
    assert not np.allclose(space.quantise_x(asp.to_full(base.X[1]))[asp.idx], base.X[1])


# --------------------------------------------------------------------------- active learning


def test_select_infill_dedupes_designs_that_round_together() -> None:
    import pandas as pd

    from suspkin.surrogate.active import select_infill
    from suspkin.surrogate.channels import CHANNELS  # noqa: F401  (import check)
    from tests.unit.test_surrogate_active import _tiny_study_objects

    study, _ = _tiny_study_objects()
    x = np.array([[0.30, 0.30], [0.31, 0.31], [0.70, 0.70]])
    df = pd.DataFrame(x, columns=["x_a", "x_b"])
    df["f_total"] = [1.0, 2.0, 3.0]
    df["max_g"] = -1.0
    df["feasible"] = True
    df["status"] = "ok"

    def coarse(a: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(a) * 4.0) / 4.0  # grid of 0.25 in the unit cube

    _, X = select_infill(
        df, study, "weighted", top=2, perturb=6, lo=np.zeros(2), hi=np.ones(2), r=0.2,
        rng=np.random.default_rng(0), quantise=coarse,
    )  # fmt: skip
    keys = {tuple(np.round(r, 9)) for r in X}
    assert len(keys) == len(X)  # no two rows round to the same design
    assert np.allclose(X, coarse(X))
    # x_star and its neighbour round to the same node: the second candidate is dropped
    _, X2 = select_infill(
        df, study, "weighted", top=2, perturb=0, lo=np.zeros(2), hi=np.ones(2), r=0.2,
        rng=np.random.default_rng(0), quantise=coarse,
    )  # fmt: skip
    assert len(X2) == 2 and np.allclose(X2[1], [0.75, 0.75])


def test_hits_boundary_slack() -> None:
    from suspkin.surrogate.active import hits_boundary

    c = np.array([0.5, 0.5])
    x = np.array([0.5, 0.5 + 0.19])  # snapped 0.01 inside the 0.2 box
    assert not hits_boundary(x, c, 0.2)
    assert hits_boundary(x, c, 0.2, slack=0.5 * np.array([0.02, 0.02]))


def test_al_radius_never_below_one_grid_step_and_no_candidates(tmp_path: Path) -> None:
    import pandas as pd

    from tests.unit.test_surrogate_active import fast, oracle_run, tiny_text

    # coarse grid (6 mm on +-15 mm offsets: step 0.2 in the unit cube) and a tiny min radius
    text = fast(tiny_text(max_iterations=3)).replace("min: 0.01", "min: 0.001")
    text = text.replace("variables:", "resolution: 6 mm\nvariables:", 1)
    res, study_path = oracle_run(tmp_path, text)
    study, veh = load_study(study_path)
    step = min(DesignSpace(study, veh).grid_step_z().max(), 0.5)  # clamped to trust_region.max
    assert step > 0.01
    al = pd.read_csv(res.al_path)
    assert len(al) >= 1
    assert (al["radius"] >= step - 1e-12).all() and (al["radius_next"] >= step - 1e-12).all()
    hist = read_history(res.run.out_dir / "history.csv")
    ok = hist["status"] != "infeasible"
    assert on_grid(hist.loc[ok, "p_front.tie_inner_x"].to_numpy(), 2e-3)
    # FOM rows are snapped designs: unique x
    xs = hist[[c for c in hist.columns if c.startswith("x_")]].to_numpy()
    assert len(np.unique(np.round(xs, 9), axis=0)) == len(xs)


def test_al_stops_cleanly_when_no_new_grid_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import suspkin.surrogate.active as act
    from tests.unit.test_surrogate_active import fast, oracle_run, tiny_text

    def known_only(sur_history: Any, study: Any, mode: str, **kw: Any) -> tuple[Any, Any]:
        # the network search only finds the design that is already FOM-verified
        base = kw["quantise"](np.array(next(iter(kw["known"])), dtype=float))
        return base, base[None, :]

    monkeypatch.setattr(act, "select_infill", known_only)
    res, _ = oracle_run(tmp_path, fast(tiny_text(max_iterations=3)))
    assert res.stop_reason == "no_candidates" and res.iterations == 0
    assert res.run.feasible and (tmp_path / "run" / "best.yaml").is_file()
