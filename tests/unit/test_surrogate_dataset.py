"""Sharded dataset: split, sampling, resume, append, load."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import suspkin.surrogate.dataset as mod
from suspkin.kinematics.rocker import KinematicsError
from suspkin.optimize.design import DesignSpace, InfeasibleDesign
from suspkin.optimize.spec import StudyConfigError, load_study
from suspkin.surrogate import AxleGrid
from suspkin.surrogate.data_types import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, AxleDataset
from suspkin.surrogate.dataset import (
    DatasetError,
    append_rows,
    assign_split,
    bytes_per_sample,
    load_dataset,
    sample_axle,
)
from suspkin.surrogate.extract import CornerSample
from suspkin.surrogate.sampling import AxleSpace, NoAxleVariablesError
from tests.unit.test_surrogate_sampling import REAR_VAR, STUDY


@pytest.fixture
def study_path(tmp_path: Path) -> Path:
    p = tmp_path / "study.yaml"
    p.write_text(STUDY.replace("n_initial: 12", "n_initial: 10"), encoding="utf-8")
    return p


def _same(a: AxleDataset, b: AxleDataset) -> None:
    assert a.channels == b.channels and a.param_names == b.param_names
    for k in ("X", "mask", "feasible", "split", "source", "weight", "travel", "rack"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert np.array_equal(a.Y, b.Y, equal_nan=True)


def test_split_fractions_and_determinism() -> None:
    rows = np.arange(200)
    s = assign_split(1, rows)
    assert np.array_equal(s, assign_split(1, rows))
    assert np.array_equal(s[:50], assign_split(1, rows[:50]))  # independent of dataset size
    assert not np.array_equal(s, assign_split(2, rows))
    frac = [np.mean(s == c) for c in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)]
    assert frac[0] == pytest.approx(0.8, abs=0.08)
    assert frac[1] == pytest.approx(0.1, abs=0.06)
    assert frac[2] == pytest.approx(0.1, abs=0.06)


def test_bytes_per_sample() -> None:
    g = AxleGrid("rear", np.linspace(-0.01, 0.01, 5), np.array([0.0]))
    assert bytes_per_sample(2, 3, g) == 3 * 5 * 4 + 5 + 16 + 16 + 2 + 12


def test_sample_load_and_meta(study_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    msgs: list[str] = []
    rep = sample_axle(study_path, out, "rear", workers=1, echo=msgs.append)
    assert rep.n_samples == 10 and rep.n_new == 10 and rep.workers == 1
    assert rep.feasible_fraction > 0.9 and rep.warning is None
    assert "bytes" in msgs[0]
    adir = out / "dataset" / "rear"
    names = sorted(p.name for p in adir.glob("shard_*.npz"))
    assert names == [f"shard_{k:05d}.npz" for k in range(3)]
    meta = json.loads((adir / "meta.json").read_text())
    for k in ("grid_hash", "design_space_hash", "vehicle_hash", "package_version", "created"):
        assert meta[k]
    assert meta["seed"] == 3 and meta["n_initial"] == 10 and meta["shard_size"] == 4

    ds = load_dataset(out, "rear")
    assert ds.n == 10 and ds.X.shape == (10, 2) and ds.Y.dtype == np.float32
    assert ds.Y.shape[2:] == (1, 7) and ds.mask.shape == (10, 1, 7)
    assert set(ds.source) == {"lhs"} and np.all(ds.weight == 1.0)
    assert np.array_equal(ds.split, assign_split(3, np.arange(10)))
    study, veh = load_study(study_path)
    assert np.allclose(ds.X[0], AxleSpace(DesignSpace(study, veh), "rear").baseline())
    assert ds.feasible[0] and ds.mask[0].all()
    with pytest.raises(DatasetError, match="not empty"):
        sample_axle(study_path, out, "rear")
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "nothing", "rear")


def test_workers_identical(study_path: Path, tmp_path: Path) -> None:
    sample_axle(study_path, tmp_path / "a", "rear", n=8, workers=1)
    sample_axle(study_path, tmp_path / "b", "rear", n=8, workers=2)
    _same(load_dataset(tmp_path / "a", "rear"), load_dataset(tmp_path / "b", "rear"))


def test_resume(study_path: Path, tmp_path: Path) -> None:
    full, part = tmp_path / "full", tmp_path / "part"
    sample_axle(study_path, full, "rear", workers=1)
    shutil.copytree(full, part)
    for name in ("shard_00001.npz", "shard_00002.npz"):  # simulate an interruption
        (part / "dataset" / "rear" / name).unlink()
    rep = sample_axle(study_path, part, "rear", workers=1, resume=True)
    assert rep.n_new == 6 and rep.n_samples == 10
    _same(load_dataset(full, "rear"), load_dataset(part, "rear"))
    assert sample_axle(study_path, part, "rear", workers=1, resume=True).n_new == 0

    changed = tmp_path / "changed.yaml"
    changed.write_text(study_path.read_text().replace("points: 7", "points: 9"))
    with pytest.raises(DatasetError, match="does not match"):
        sample_axle(changed, part, "rear", resume=True)
    with pytest.raises(DatasetError, match="does not match"):
        sample_axle(study_path, part, "rear", n=12, resume=True)


def test_errors(study_path: Path, tmp_path: Path) -> None:
    only_front = tmp_path / "f.yaml"
    only_front.write_text(study_path.read_text().replace(REAR_VAR, ""))
    with pytest.raises(NoAxleVariablesError):
        sample_axle(only_front, tmp_path / "o", "rear")
    nosur = tmp_path / "n.yaml"
    nosur.write_text(study_path.read_text().split("surrogate:")[0])
    with pytest.raises(StudyConfigError, match="surrogate"):
        sample_axle(nosur, tmp_path / "o2", "front")


def test_infeasible_row_kept(study_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study, veh = load_study(study_path)
    aspace = AxleSpace(DesignSpace(study, veh), "rear")
    grid = AxleGrid("rear", np.linspace(-0.01, 0.01, 5), np.array([0.0]))

    class Bad:
        def to_vehicle(self, x: Any) -> Any:
            raise InfeasibleDesign("invalid")

    y, mask, feasible = mod._eval_row(Bad(), aspace, grid, aspace.baseline())  # type: ignore[arg-type]
    assert not feasible and not mask.any() and np.isnan(y).all() and y.dtype == np.float32

    class Fine:
        def to_vehicle(self, x: Any) -> Any:
            return veh

    def boom(*a: Any, **k: Any) -> Any:
        raise KinematicsError("does not assemble")

    monkeypatch.setattr(mod, "extract_corner", boom)
    y, mask, feasible = mod._eval_row(Fine(), aspace, grid, aspace.baseline())  # type: ignore[arg-type]
    assert not feasible and not mask.any() and np.isnan(y).all()


def test_append_rows(study_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    sample_axle(study_path, out, "rear", n=4, workers=1)
    ds = load_dataset(out, "rear")
    c, r, t = ds.Y.shape[1:]
    smp = [
        CornerSample(np.full((c, r, t), 0.5), np.ones((r, t), dtype=bool), True, ds.channels)
        for _ in range(2)
    ]
    path = append_rows(out, "rear", np.full((2, 2), 0.25), smp, "al_1", weight=5.0)
    assert path.name == "shard_00001.npz"
    ds2 = load_dataset(out, "rear")
    assert ds2.n == 6
    assert list(ds2.source[4:]) == ["al_1", "al_1"]
    assert np.all(ds2.weight[4:] == 5.0) and np.all(ds2.split[4:] == SPLIT_TRAIN)
    assert ds2.Y.dtype == np.float32 and np.allclose(ds2.Y[4:], 0.5)
    with pytest.raises(DatasetError):
        append_rows(out, "rear", np.zeros((2, 3)), smp, "al_2")
    with pytest.raises(DatasetError):
        append_rows(out, "rear", np.zeros((2, 2)), smp, "lhs")


def test_lhs_stored_once_and_setting_checked(
    study_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from suspkin.surrogate.sampling import resolve_lhs_optimization

    assert resolve_lhs_optimization("auto", 5000) == "random-cd"
    assert resolve_lhs_optimization("auto", 5001) == "none"
    out = tmp_path / "out"
    calls: list[int] = []
    real = mod.lhs

    def spy(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(mod, "lhs", spy)
    sample_axle(study_path, out, "rear", workers=1)
    adir = out / "dataset" / "rear"
    assert len(calls) == 1 and (adir / "lhs.npy").is_file()
    assert json.loads((adir / "meta.json").read_text())["lhs_optimization"] == "random-cd"
    (adir / "shard_00002.npz").unlink()
    sample_axle(study_path, out, "rear", workers=1, resume=True)
    assert len(calls) == 1  # loaded, not recomputed
    changed = tmp_path / "c.yaml"
    changed.write_text(
        study_path.read_text().replace("seed: 3,", "seed: 3, lhs_optimization: none,")
    )
    with pytest.raises(DatasetError, match="does not match"):
        sample_axle(changed, out, "rear", resume=True)


def test_dataset_without_solver_channels_is_rejected(study_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    sample_axle(study_path, out, "rear", workers=1)
    mod.check_dataset_matches(study_path, out, "rear")
    mp = out / "dataset" / "rear" / "meta.json"
    meta = json.loads(mp.read_text())
    meta["channels"] = [c for c in meta["channels"] if c not in ("theta_uca", "theta_lca", "psi")]
    mp.write_text(json.dumps(meta))
    with pytest.raises(DatasetError, match=r"theta_uca.*sample again"):
        mod.check_dataset_matches(study_path, out, "rear")
