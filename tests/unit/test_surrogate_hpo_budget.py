"""Trial budget, resume robustness and the search report."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
import pytest

pytest.importorskip("torch")

from suspkin.surrogate import hpo
from suspkin.surrogate.data import SPLIT_TRAIN, SPLIT_VAL, AxleDataset
from suspkin.surrogate.dataset import DatasetError
from suspkin.surrogate.hpo import search, subsample_train, trial_budget, write_hpo_report
from suspkin.surrogate.train import TrainConfig
from tests.unit.test_surrogate_model import make_ds

BASE = TrainConfig(max_epochs=6, patience=20, warmup_epochs=1)


def test_trial_budget_rules() -> None:
    b = trial_budget(TrainConfig(max_epochs=2000, patience=100), 300, 20000, 5000)
    assert b == {"max_epochs": 300, "patience": 30, "subsample": 5000}
    # None cap -> training cap; hpo cap above training cap -> training cap.
    assert (
        trial_budget(TrainConfig(max_epochs=50, patience=100), None, 100, None)["max_epochs"] == 50
    )
    assert (
        trial_budget(TrainConfig(max_epochs=50, patience=100), 300, 100, None)["max_epochs"] == 50
    )
    # Patience floor of 10 and never above training patience.
    assert trial_budget(TrainConfig(max_epochs=2000, patience=5), 40, 100, None)["patience"] == 5
    assert trial_budget(TrainConfig(max_epochs=2000, patience=100), 40, 100, None)["patience"] == 10
    # Subsample >= train size means all rows.
    assert trial_budget(BASE, None, 100, 100)["subsample"] is None


def test_subsample_train_keeps_val_and_is_deterministic() -> None:
    ds = make_ds(n=200)
    n_train = int((ds.split == SPLIT_TRAIN).sum())
    sub = subsample_train(ds, 30, seed=3)
    assert int((sub.split == SPLIT_TRAIN).sum()) == 30
    assert int((sub.split == SPLIT_VAL).sum()) == int((ds.split == SPLIT_VAL).sum())
    assert sub.n == ds.n - n_train + 30
    again = subsample_train(ds, 30, seed=3)
    assert (again.X == sub.X).all()
    assert subsample_train(ds, None, seed=3) is ds
    assert subsample_train(ds, n_train + 5, seed=3) is ds


def test_search_caps_trial_epochs_and_subsamples_but_not_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_ds(n=200)
    calls: list[tuple[int, int, int, int]] = []
    real_train = hpo.train

    def spy(d: AxleDataset, mcfg: Any, tcfg: TrainConfig, **kw: Any) -> Any:
        calls.append(
            (
                int((d.split == SPLIT_TRAIN).sum()),
                int((d.split == SPLIT_VAL).sum()),
                tcfg.max_epochs,
                tcfg.patience,
            )
        )
        return real_train(d, mcfg, tcfg, **kw)

    monkeypatch.setattr(hpo, "train", spy)
    pruners: list[Any] = []

    class Spy(optuna.pruners.HyperbandPruner):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            pruners.append(self)

    monkeypatch.setattr(optuna.pruners, "HyperbandPruner", Spy)
    base = TrainConfig(max_epochs=8, patience=50, warmup_epochs=1)
    n_val = int((ds.split == SPLIT_VAL).sum())
    n_train = int((ds.split == SPLIT_TRAIN).sum())
    res = search(
        ds,
        tmp_path / "front",
        n_trials=1,
        device="cpu",
        base=base,
        trial_max_epochs=5,
        subsample=40,
    )
    assert pruners[0]._max_resource == 5
    trial_call, final_call = calls[0], calls[-1]
    assert trial_call == (40, n_val, 5, 10)
    assert final_call == (n_train, n_val, 8, 50)
    assert res.n_trials == 1
    info = json.loads((tmp_path / "front" / "hpo_result.json").read_text(encoding="utf-8"))
    assert info["subsample"] == 40 and info["n_train"] == n_train
    assert info["trial_budget"]["max_epochs"] == 5
    assert (tmp_path / "hpo_report.md").exists() is False  # only search_axles writes it


def test_resume_with_other_budget_fails_and_running_marked_fail(tmp_path: Path) -> None:
    ds = make_ds(n=120)
    out = tmp_path / "front"
    search(ds, out, n_trials=1, device="cpu", base=BASE, trial_max_epochs=3)
    with pytest.raises(DatasetError, match="--root"):
        search(ds, out, n_trials=1, device="cpu", base=BASE, trial_max_epochs=4)
    with pytest.raises(DatasetError, match="trial budget"):
        search(ds, out, n_trials=1, device="cpu", base=BASE, trial_max_epochs=3, subsample=20)
    # A trial left RUNNING by a killed process.
    study = optuna.load_study(study_name="suspkin_front", storage=f"sqlite:///{out / 'optuna.db'}")
    study.ask()
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.RUNNING in states
    search(ds, out, n_trials=1, device="cpu", base=BASE, trial_max_epochs=3)
    study = optuna.load_study(study_name="suspkin_front", storage=f"sqlite:///{out / 'optuna.db'}")
    final = [t.state for t in study.trials]
    assert optuna.trial.TrialState.RUNNING not in final
    assert optuna.trial.TrialState.FAIL in final


def _tiny_study(adir: Path, *, finished: bool) -> None:
    adir.mkdir(parents=True)
    study = optuna.create_study(
        study_name="suspkin_front", storage=f"sqlite:///{adir / 'optuna.db'}", direction="minimize"
    )
    study.set_user_attr("trial_budget", {"max_epochs": 30, "patience": 10, "subsample": 50})
    for k in range(3):
        t = study.ask()
        t.suggest_int("n_layers", 2, 8)
        t.suggest_categorical("activation", ["silu", "gelu"])
        t.suggest_float("lr", 1e-4, 1e-2, log=True)
        for step in range(4):
            t.report(1.0 + k - 0.1 * step, step)
        if k == 2:
            study.tell(t, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(t, 0.5 + k)
    t = study.ask()  # stays RUNNING
    t.suggest_int("n_layers", 2, 8)
    t.report(2.0, 0)
    if finished:
        (adir / "model").mkdir()
        (adir / "model" / "model.json").write_text(
            json.dumps({"val_score": 0.42, "history": [{"epoch": 0.0}] * 7}), encoding="utf-8"
        )


def test_report_sections_without_result_json(tmp_path: Path) -> None:
    _tiny_study(tmp_path / "front", finished=False)
    path = write_hpo_report(tmp_path, ["front", "rear"])
    assert path == tmp_path / "hpo_report.md"
    text = path.read_text(encoding="utf-8")
    for needle in (
        "## Front axle",
        "search in progress",
        "COMPLETE: 2",
        "PRUNED: 1",
        "RUNNING: 1",
        "### Best trial",
        "Trial 0, score 0.5000",
        "Top 2 completed trials",
        "#2@4",
        "Not available yet",
        "mean over channels",
        "## Rear axle",
        "No `optuna.db`",
    ):
        assert needle in text, needle
    assert "| 0 | 0.5000 | 4 |" in text


def test_report_with_final_model_and_custom_out(tmp_path: Path) -> None:
    _tiny_study(tmp_path / "front", finished=True)
    (tmp_path / "front" / "hpo_result.json").write_text(
        json.dumps(
            {
                "final_score": 0.4,
                "final_epochs": 9,
                "final_seconds": 65.0,
                "device": "cpu",
                "n_train": 100,
                "n_val": 20,
                "n_test": 20,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "sub" / "r.md"
    text = write_hpo_report(tmp_path, ["front"], out=out).read_text(encoding="utf-8")
    assert "search in progress" not in text
    assert "Epochs trained: 9" in text and "1m 05s" in text
    assert "100 / 20 / 20" in text


def test_hpo_result_json_helpers_roundtrip(tmp_path: Path) -> None:
    ds = replace(make_ds(n=120), axle="front")
    search(ds, tmp_path / "front", n_trials=1, device="cpu", base=BASE)
    text = write_hpo_report(tmp_path, ["front"]).read_text(encoding="utf-8")
    assert "Epochs trained" in text and "COMPLETE: 1" in text


def test_retrain_best_without_trials(tmp_path: Path) -> None:
    ds = replace(make_ds(n=120), axle="front")
    out = tmp_path / "front"
    search(ds, out, n_trials=1, device="cpu", base=BASE)
    storage = f"sqlite:///{out / 'optuna.db'}"
    optuna.load_study(study_name="suspkin_front", storage=storage).ask()  # left RUNNING
    (out / "model" / "model.json").unlink()
    res = hpo.retrain_best(ds, out, device="cpu", base=BASE)
    assert (out / "model" / "model.json").exists() and res.n_complete == 1
    states = [
        t.state for t in optuna.load_study(study_name="suspkin_front", storage=storage).trials
    ]
    assert optuna.trial.TrialState.RUNNING in states  # retrain_best leaves live trials alone
    assert optuna.trial.TrialState.FAIL not in states
    info = json.loads((out / "hpo_result.json").read_text(encoding="utf-8"))
    assert info["retrain_only"] is True and info["search_seconds"] is None
    assert "trial_budget" in info  # carried over from the original search
    assert "retrain only" in write_hpo_report(tmp_path, ["front"]).read_text(encoding="utf-8")


def test_search_result_is_not_retrain_only(tmp_path: Path) -> None:
    ds = replace(make_ds(n=120), axle="front")
    search(ds, tmp_path / "front", n_trials=1, device="cpu", base=BASE)
    info = json.loads((tmp_path / "front" / "hpo_result.json").read_text(encoding="utf-8"))
    assert info["retrain_only"] is False and info["search_seconds"] > 0
    assert (tmp_path / "front" / "trials.csv").exists()


def test_retrain_best_errors(tmp_path: Path) -> None:
    ds = replace(make_ds(n=120), axle="front")
    with pytest.raises(DatasetError, match=r"optuna.db"):
        hpo.retrain_best(ds, tmp_path / "front", device="cpu", base=BASE)
    storage = f"sqlite:///{tmp_path / 'other' / 'optuna.db'}"
    (tmp_path / "other").mkdir()
    optuna.create_study(study_name="suspkin_rear", storage=storage)
    with pytest.raises(DatasetError, match="no study"):
        hpo.retrain_best(ds, tmp_path / "other", device="cpu", base=BASE)
    (tmp_path / "empty").mkdir()
    st = optuna.create_study(
        study_name="suspkin_front", storage=f"sqlite:///{tmp_path / 'empty' / 'optuna.db'}"
    )
    st.ask()
    with pytest.raises(DatasetError, match="no completed trial"):
        hpo.retrain_best(ds, tmp_path / "empty", device="cpu", base=BASE)
