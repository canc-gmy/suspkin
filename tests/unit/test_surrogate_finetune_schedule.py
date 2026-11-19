"""Active-learning fine-tune scheduling: trigger, stall rule, skipped axles, resume (CPU)."""

# ruff: noqa: F401, F811
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from suspkin.optimize.spec import FinetuneSpec
from suspkin.surrogate.active import (
    AL_COLUMNS,
    FinetuneState,
    _replay_finetune_state,
    advance_finetune,
    finetune_decision,
)
from tests.unit.test_surrogate_active import _n_rows, pristine, tiny_text, work

FT = FinetuneSpec(trigger=1.0, skip_after=2, skip_for=2, stall_override=3.0)


def test_finetune_decision_table() -> None:
    idle, stalled = FinetuneState(), FinetuneState(kept_run=0, skip_left=2)
    assert finetune_decision(0.4, idle, FT) == (False, "trigger")
    assert finetune_decision(1.0, idle, FT) == (False, "trigger")  # must exceed the trigger
    assert finetune_decision(1.5, idle, FT) == (True, "train")
    assert finetune_decision(math.nan, idle, FT) == (True, "train")
    assert finetune_decision(1.5, stalled, FT) == (False, "stall")
    assert finetune_decision(3.0, stalled, FT) == (False, "stall")
    assert finetune_decision(3.1, stalled, FT) == (True, "train")  # override
    assert finetune_decision(0.5, stalled, FT) == (False, "trigger")
    always = FinetuneSpec(trigger=0.0, skip_for=0)
    assert finetune_decision(1e-9, idle, always) == (True, "train")


def test_finetune_decision_cpu_rule() -> None:
    idle = FinetuneState()
    skip, run = FinetuneSpec(trigger=1.0), FinetuneSpec(trigger=1.0, on_cpu="run")
    assert FinetuneSpec().on_cpu == "skip"
    assert finetune_decision(50.0, idle, skip, device_type="cpu") == (False, "cpu")
    assert finetune_decision(math.nan, idle, skip, device_type="cpu") == (False, "cpu")
    assert finetune_decision(50.0, idle, skip, device_type="cuda") == (True, "train")
    assert finetune_decision(0.4, idle, skip, device_type="cuda") == (False, "trigger")
    assert finetune_decision(50.0, idle, run, device_type="cpu") == (True, "train")
    assert finetune_decision(0.4, idle, run, device_type="cpu") == (False, "trigger")


def test_replay_ignores_cpu_skips() -> None:
    rows: list[dict[str, object]] = [
        {"val_front": math.nan, "skipped_front": True, "skip_reason_front": "cpu"},
    ] * 3
    assert _replay_finetune_state(rows, "front", FT) == FinetuneState()


def test_cpu_default_skips_finetuning_and_records_reason(work: tuple[Path, Path, Path]) -> None:
    from suspkin.surrogate.active import run_active_learning

    study, root, run = work
    text = tiny_text(max_iterations=2).replace(", on_cpu: run", "")
    study.write_text(text, encoding="utf-8")
    msgs: list[str] = []
    run_active_learning(study, root, run, workers=1, device="cpu", report=False, echo=msgs.append)
    al = pd.read_csv(run / "al.csv")
    assert al["skipped_front"].all() and al["val_front"].isna().all()
    assert (al["skip_reason_front"] == "cpu").all()
    assert sum("disabled on CPU" in m and "on_cpu: run" in m for m in msgs) == 1
    assert not (root / "models" / "front" / "al_0").exists()
    assert (root / "models" / "front" / "pre_al" / "model.pt").is_file()


def test_finetune_state_machine() -> None:
    s = advance_finetune(FinetuneState(), skipped=False, kept=True, ft=FT)
    assert s == FinetuneState(1, 0)
    s = advance_finetune(s, skipped=False, kept=False, ft=FT)  # an improvement resets the run
    assert s == FinetuneState(0, 0)
    s = advance_finetune(s, skipped=False, kept=True, ft=FT)
    s = advance_finetune(s, skipped=False, kept=True, ft=FT)  # skip_after reached
    assert s == FinetuneState(0, 2)
    s = advance_finetune(s, skipped=True, kept=False, ft=FT)
    assert s == FinetuneState(0, 1)
    s = advance_finetune(s, skipped=True, kept=False, ft=FT)
    assert s == FinetuneState(0, 0)
    assert advance_finetune(s, skipped=True, kept=False, ft=FT).skip_left == 0
    off = FinetuneSpec(skip_after=1, skip_for=0)  # the stall window needs skip_for > 0
    assert advance_finetune(FinetuneState(), skipped=False, kept=True, ft=off) == FinetuneState(
        1, 0
    )
    # trained through the window (override): the window is dropped
    assert (
        advance_finetune(FinetuneState(0, 2), skipped=False, kept=False, ft=FT) == FinetuneState()
    )


def test_replay_finetune_state_matches_live_updates() -> None:
    rows: list[dict[str, object]] = [
        {"val_front": 0.5, "kept_init_front": True, "skipped_front": False},
        {"val_front": 0.5, "kept_init_front": True, "skipped_front": False},
        {"val_front": math.nan, "kept_init_front": False, "skipped_front": True},
        {"val_front": math.nan, "kept_init_front": False},  # old table, nothing happened
    ]
    live = FinetuneState()
    for kept, skipped in ((True, False), (True, False), (False, True)):
        live = advance_finetune(live, skipped=skipped, kept=kept, ft=FT)
    assert live == FinetuneState(0, 1)
    assert _replay_finetune_state(rows[:3], "front", FT) == live
    assert _replay_finetune_state(rows, "front", FT) == live
    assert _replay_finetune_state([], "rear", FT) == FinetuneState()


def test_skipped_finetune_logs_keeps_dataset_and_resumes(work: tuple[Path, Path, Path]) -> None:
    from suspkin.surrogate.active import run_active_learning
    from suspkin.surrogate.dataset import load_dataset

    study, root, run = work
    text = tiny_text(max_iterations=2).replace("trigger: 0, skip_for: 0", "trigger: 1e6")
    study.write_text(text, encoding="utf-8")
    n0, _ = _n_rows(root)
    msgs: list[str] = []
    res = run_active_learning(
        study, root, run, workers=1, device="cpu", report=False, echo=msgs.append
    )
    assert res.new_iterations == 2
    al = pd.read_csv(run / "al.csv")
    assert list(al.columns) == list(AL_COLUMNS)
    assert al["skipped_front"].all() and not al["kept_init_front"].any()
    assert al["val_front"].isna().all() and not al["skipped_rear"].any()
    assert any("skipped (trigger" in m for m in msgs)
    assert (al["skip_reason_front"] == "trigger").all()
    assert not (root / "models" / "front" / "al_0").exists()  # nothing fine-tuned
    assert (root / "models" / "front" / "pre_al" / "model.pt").is_file()
    assert load_dataset(root, "front").n == n0 + int(al["n_new"].sum())  # rows still appended

    # resume from a table written before the skipped_<axle> columns existed
    al.drop(columns=["skipped_front", "skipped_rear"]).to_csv(run / "al.csv", index=False)
    study.write_text(text.replace("max_iterations: 2", "max_iterations: 3"), encoding="utf-8")
    run_active_learning(study, root, run, resume=True, workers=1, device="cpu", report=False)
    al2 = pd.read_csv(run / "al.csv")
    assert list(al2.columns) == list(AL_COLUMNS) and list(al2["k"]) == [0, 1, 2]
    assert not al2["skipped_front"].iloc[:2].any()  # unknown for old rows: default False
    assert bool(al2["skipped_front"].iloc[2])


def test_finetuned_run_keeps_cache_path_consistent(work: tuple[Path, Path, Path]) -> None:
    """Two fine-tuned iterations (second one uses the cache extended by add_rows)."""
    from suspkin.surrogate.active import run_active_learning
    from suspkin.surrogate.train import TrainedModel

    study, root, run = work
    study.write_text(tiny_text(max_iterations=2), encoding="utf-8")
    run_active_learning(study, root, run, workers=1, device="cpu", report=False)
    al = pd.read_csv(run / "al.csv")
    assert not al["skipped_front"].any() and al["val_front"].notna().all()
    for k in (0, 1):
        m = TrainedModel.load(root / "models" / "front" / f"al_{k}", device="cpu")
        assert m.meta["al_iteration"] == k
