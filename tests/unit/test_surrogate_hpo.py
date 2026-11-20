"""Optuna search tests."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("optuna")

from suspkin.surrogate.hpo import search, search_axles
from suspkin.surrogate.train import SurrogatePredictor, TrainConfig
from tests.unit.test_surrogate_model import make_ds


def test_search_runs_resumes_and_saves(tmp_path: Path) -> None:
    ds = make_ds(n=200)
    base = TrainConfig(max_epochs=4, patience=4, warmup_epochs=1)
    out = tmp_path / "front"
    res = search(ds, out, n_trials=3, timeout_s=None, seed=0, device="cpu", base=base)
    assert (out / "optuna.db").exists()
    assert (out / "trials.csv").exists()
    assert res.n_trials == 3
    assert (out / "model" / "model.pt").exists()
    res2 = search(ds, out, n_trials=2, timeout_s=None, seed=0, device="cpu", base=base)
    assert res2.n_trials == 5
    y, _ = SurrogatePredictor.load(res2.model_dir, device="cpu").predict(ds.X[:3])
    assert y.shape == (3, 3, 2, 7)


def test_search_axles_parallel_spawn(tmp_path: Path) -> None:
    from dataclasses import replace

    front = make_ds(n=120)
    rear = replace(front, axle="rear")
    base = TrainConfig(max_epochs=2, patience=2, warmup_epochs=1)
    res = search_axles(
        {"front": front, "rear": rear},
        tmp_path,
        n_trials=1,
        device="cpu",
        base=base,
        parallel=True,
    )
    assert set(res) == {"front", "rear"}
    assert (tmp_path / "rear" / "model" / "model.pt").exists()
