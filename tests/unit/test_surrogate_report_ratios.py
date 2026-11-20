"""HPO report: per-channel ratio summary of the final model and per-trial worst channel."""

import json
from pathlib import Path

import optuna

from suspkin.surrogate.hpo import TRIAL_RATIO_ATTR, write_hpo_report

RATIOS = {"a": 1.0, "b": 5.0, "c": 3.0, "d": 100.0, "e": 2.0, "f": 7.0, "g": 4.0}


def _study(adir: Path, *, trial_ratios: bool) -> None:
    adir.mkdir(parents=True)
    study = optuna.create_study(
        study_name="s", storage=f"sqlite:///{adir / 'optuna.db'}", direction="minimize"
    )
    t = study.ask()
    t.suggest_int("n_layers", 2, 8)
    if trial_ratios:
        t.set_user_attr(TRIAL_RATIO_ATTR, RATIOS)
    study.tell(t, 17.0)
    (adir / "model").mkdir()
    info = {"val_score": 17.0, "history": [{}], "meta": {"val_channel_ratio": RATIOS}}
    (adir / "model" / "model.json").write_text(json.dumps(info), encoding="utf-8")


def test_final_model_ratio_summary(tmp_path: Path) -> None:
    _study(tmp_path / "front", trial_ratios=False)
    text = write_hpo_report(tmp_path, ["front"]).read_text(encoding="utf-8")
    assert "median 4" in text and "max 100" in text
    assert "Worst 5 channels" in text
    names = [ln.split("|")[1].strip() for ln in text.splitlines() if ln.startswith("| ")]
    assert [n for n in names if n in RATIOS] == ["d", "f", "b", "g", "c"]
    assert "| worst channel |" not in text


def test_trial_worst_channel_column(tmp_path: Path) -> None:
    _study(tmp_path / "front", trial_ratios=True)
    text = write_hpo_report(tmp_path, ["front"]).read_text(encoding="utf-8")
    assert "| worst channel |" in text and "d (100)" in text


def test_report_without_ratios_still_works(tmp_path: Path) -> None:
    _study(tmp_path / "front", trial_ratios=False)
    (tmp_path / "front" / "model" / "model.json").write_text(
        json.dumps({"val_score": 1.0, "history": []}), encoding="utf-8"
    )
    text = write_hpo_report(tmp_path, ["front"]).read_text(encoding="utf-8")
    assert "Validation score" in text and "Worst 5" not in text
