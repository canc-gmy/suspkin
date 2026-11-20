"""Optuna architecture search per axle and its Markdown report (needs torch and optuna).

The report (``write_hpo_report``) reads ``optuna.db`` read-only: it is safe to call while
another process still writes the study, and issues only plain SELECTs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

from suspkin.surrogate.data import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, AxleDataset
from suspkin.surrogate.dataset import DatasetError
from suspkin.surrogate.model import MLPConfig
from suspkin.surrogate.train import TrainConfig, TrainedModel, resolve_device, train

_ACTS = ("silu", "gelu", "relu", "tanh")


@dataclass(frozen=True)
class HpoResult:
    """Outcome of one axle search."""

    axle: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    n_complete: int
    out_dir: Path
    model_dir: Path
    trials_csv: Path
    final_score: float


def configs_from_params(p: dict[str, Any], base: TrainConfig) -> tuple[MLPConfig, TrainConfig]:
    """Build the configs of a trial from its parameter dict."""
    pod = p["output"] == "pod"
    mcfg = MLPConfig(
        n_layers=int(p["n_layers"]),
        width=int(p["width"]),
        activation=p["activation"],
        residual=bool(p["residual"]),
        layernorm=bool(p["layernorm"]),
        dropout=float(p["dropout"]),
        output="pod" if pod else "raw",
        pod_k=int(p["pod_k"]) if pod else 32,
    )
    tcfg = replace(
        base,
        lr=float(p["lr"]),
        weight_decay=float(p["weight_decay"]),
        batch_size=int(p["batch_size"]),
        lam_feas=float(p["lam_feas"]),
    )
    return mcfg, tcfg


def suggest(trial: optuna.Trial) -> dict[str, Any]:
    """Sample the search space of plan section 13.8."""
    p: dict[str, Any] = {
        "n_layers": trial.suggest_int("n_layers", 2, 8),
        "width": trial.suggest_int("width", 64, 1024, log=True),
        "activation": trial.suggest_categorical("activation", list(_ACTS)),
        "residual": trial.suggest_categorical("residual", [True, False]),
        "layernorm": trial.suggest_categorical("layernorm", [True, False]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_int("batch_size", 128, 2048, log=True),
        "output": trial.suggest_categorical("output", ["raw", "pod"]),
        "lam_feas": trial.suggest_float("lam_feas", 0.01, 1.0, log=True),
    }
    p["pod_k"] = trial.suggest_int("pod_k", 8, 128) if p["output"] == "pod" else 32
    return p


def trial_budget(
    base: TrainConfig, trial_max_epochs: int | None, n_train: int, subsample: int | None
) -> dict[str, int | None]:
    """Epoch cap, patience and train-row count of a search trial (not of the final retrain).

    The cap is ``min(trial_max_epochs, base.max_epochs)`` (``None``: ``base.max_epochs``), the
    patience ``min(base.patience, max(10, cap // 10))``; ``subsample`` is ``None`` when it
    is unset or not smaller than ``n_train``.
    """
    cap = base.max_epochs if trial_max_epochs is None else min(trial_max_epochs, base.max_epochs)
    cap = max(int(cap), 1)
    sub = None if subsample is None or subsample >= n_train else int(subsample)
    return {
        "max_epochs": cap,
        "patience": min(base.patience, max(10, cap // 10)),
        "subsample": sub,
    }


def subsample_train(ds: AxleDataset, n: int | None, seed: int) -> AxleDataset:
    """Keep ``n`` random train rows (seeded, same rows every call); val/test stay complete."""
    is_train = ds.split == SPLIT_TRAIN
    if n is None or n >= int(is_train.sum()):
        return ds
    idx = np.flatnonzero(is_train)
    keep = ~is_train
    keep[np.sort(np.random.default_rng(seed).choice(idx, size=n, replace=False))] = True
    return ds.subset(keep)


def _check_budget(study: optuna.Study, budget: dict[str, int | None], base: TrainConfig) -> None:
    """Refuse to resume a study whose trials ran under a different budget."""
    stored = study.user_attrs.get("trial_budget")
    if stored is None:
        if not study.trials:
            study.set_user_attr("trial_budget", budget)
            return
        # Study from before budgets were recorded: full epochs, full data.
        stored = {"max_epochs": base.max_epochs, "patience": base.patience, "subsample": None}
    if dict(stored) != budget:
        raise DatasetError(
            f"existing study in this folder used trial budget {dict(stored)} but the current "
            f"settings give {budget}: trial scores would not be comparable. Use a new --root "
            "or delete models/<axle>/ to restart the search."
        )


_LOG = logging.getLogger(__name__)


def _safe_report(root: Path, axles: list[str]) -> None:
    """Write the HPO report without masking an exception already propagating."""
    try:
        write_hpo_report(root, axles)
    except Exception:
        _LOG.warning("could not write the HPO report in %s", root, exc_info=True)


def _fail_running(study: optuna.Study) -> None:
    """Mark trials left RUNNING by a killed process as FAIL."""
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL, skip_if_finished=True)


def search(
    ds: AxleDataset,
    out_dir: str | Path,
    *,
    n_trials: int,
    timeout_s: float | None = None,
    seed: int = 0,
    device: str = "auto",
    base: TrainConfig | None = None,
    trial_max_epochs: int | None = None,
    subsample: int | None = None,
) -> HpoResult:
    """Run (or resume) the Optuna study, then retrain the best config into ``out_dir/model``.

    The study lives in ``out_dir/optuna.db`` (SQLite); rerunning with the same directory
    adds trials to the same study. ``trials.csv`` lists all trials and ``hpo_result.json``
    the outcome and timings.

    Search trials use a reduced budget (see :func:`trial_budget`): at most
    ``trial_max_epochs`` epochs and ``subsample`` train rows (validation split complete).
    The best configuration is then retrained on the full train split with ``base``.
    Resuming with a different budget raises :class:`DatasetError`; trials left RUNNING by
    a killed process are marked FAIL.
    """
    t_start = time.monotonic()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolve_device(device)  # fail early
    base = replace(base or TrainConfig(), device=device)  # type: ignore[arg-type]
    storage = f"sqlite:///{(out / 'optuna.db').as_posix()}"
    name = f"suspkin_{ds.axle}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_train = int((ds.split == SPLIT_TRAIN).sum())
    budget = trial_budget(base, trial_max_epochs, n_train, subsample)
    cap, sub, pat = budget["max_epochs"], budget["subsample"], budget["patience"]
    assert cap is not None and pat is not None
    trial_base = replace(base, max_epochs=cap, patience=pat)
    trial_ds = subsample_train(ds, sub, base.seed)
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=max(cap, 4))
    try:
        n_prev = len(optuna.load_study(study_name=name, storage=storage).trials)
    except KeyError:
        n_prev = 0
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed + n_prev),
        pruner=pruner,
    )
    _check_budget(study, budget, base)
    _fail_running(study)

    def objective(trial: optuna.Trial) -> float:
        mcfg, tcfg = configs_from_params(suggest(trial), trial_base)

        def report(epoch: int, score: float) -> None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            model = train(trial_ds, mcfg, tcfg, report=report)
            ratios = model.meta.get("val_channel_ratio")
            if ratios:
                trial.set_user_attr("channel_ratio", dict(ratios))
            return model.val_score
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned() from exc
        except FloatingPointError as exc:  # nan loss: diverged configuration
            raise optuna.TrialPruned() from exc

    t_search = time.monotonic()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_s,
        gc_after_trial=True,
        catch=(RuntimeError, ValueError, FloatingPointError),
    )
    search_s = time.monotonic() - t_search
    return retrain_best(
        ds,
        out,
        device=device,
        base=base,
        extra={
            "search_seconds": search_s,
            "n_trials_requested": n_trials,
            "timeout_s": timeout_s,
            "subsample": sub,
            "trial_budget": budget,
        },
        t_start=t_start,
    )


def retrain_best(
    ds: AxleDataset,
    out_dir: str | Path,
    *,
    device: str = "auto",
    base: TrainConfig | None = None,
    extra: dict[str, Any] | None = None,
    t_start: float | None = None,
) -> HpoResult:
    """Retrain the best completed trial of the study in ``out_dir`` and save the model.

    No trial is run and the trial budget is not checked. Trials left RUNNING by a killed
    process are left untouched (the search may still be live; only :func:`search` fails them
    on resume). Writes ``model/``, ``trials.csv`` and ``hpo_result.json``
    (``retrain_only`` is true and ``search_seconds`` null unless ``extra`` is given, which
    :func:`search` does).

    Raises
    ------
    DatasetError
        No ``optuna.db`` / study in ``out_dir``, or no completed trial.
    """
    t_start = time.monotonic() if t_start is None else t_start
    out = Path(out_dir)
    db = out / "optuna.db"
    if not db.exists():
        raise DatasetError(f"no optuna.db in {out}: run the search first")
    dev = resolve_device(device)
    base = replace(base or TrainConfig(), device=device)  # type: ignore[arg-type]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.load_study(
            study_name=f"suspkin_{ds.axle}", storage=f"sqlite:///{db.as_posix()}"
        )
    except KeyError as exc:
        raise DatasetError(f"no study 'suspkin_{ds.axle}' in {db}") from exc
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        msg = f"no completed trial in {db}; cannot select a best configuration"
        raise DatasetError(msg) if extra is None else RuntimeError(msg)
    best = study.best_trial
    mcfg, tcfg = configs_from_params(best.params, base)
    n_train = int((ds.split == SPLIT_TRAIN).sum())
    t_final = time.monotonic()
    final: TrainedModel = train(ds, mcfg, tcfg)
    final.save(out / "model")
    result = HpoResult(
        axle=ds.axle,
        best_params=dict(best.params),
        best_value=float(best.value) if best.value is not None else float("nan"),
        n_trials=len(study.trials),
        n_complete=len(done),
        out_dir=out,
        model_dir=out / "model",
        trials_csv=out / "trials.csv",
        final_score=final.val_score,
    )
    info: dict[str, Any] = {
        "axle": result.axle,
        "best_params": result.best_params,
        "best_value": result.best_value,
        "n_trials": result.n_trials,
        "n_complete": result.n_complete,
        "final_score": result.final_score,
        "final_epochs": len(final.history),
        "final_seconds": time.monotonic() - t_final,
        "search_seconds": None,
        "total_seconds": time.monotonic() - t_start,
        "device": str(dev),
        "n_train": n_train,
        "n_val": int((ds.split == SPLIT_VAL).sum()),
        "n_test": int((ds.split == SPLIT_TEST).sum()),
        "retrain_only": extra is None,
    }
    if extra is None:  # keep the search settings recorded by the original run
        try:
            old = json.loads((out / "hpo_result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        keep = ("n_trials_requested", "timeout_s", "subsample", "trial_budget")
        info.update({k: old[k] for k in keep if isinstance(old, dict) and k in old})
    info.update(extra or {})
    (out / "hpo_result.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return result


def _search_one(ds: AxleDataset, out_dir: str, kwargs: dict[str, Any]) -> HpoResult:
    return search(ds, out_dir, **kwargs)


def search_axles(
    datasets: dict[str, AxleDataset],
    root: str | Path,
    *,
    n_trials: int,
    timeout_s: float | None = None,
    seed: int = 0,
    device: str = "auto",
    base: TrainConfig | None = None,
    parallel: bool | None = None,
    trial_max_epochs: int | None = None,
    subsample: int | None = None,
) -> dict[str, HpoResult]:
    """Search every axle in ``root/<axle>``.

    ``parallel=None`` runs the axles in parallel spawn processes on CUDA and sequentially
    on the CPU (two processes would compete for the same cores); a bool forces the mode.
    """
    if parallel is None:
        parallel = resolve_device(device).type == "cuda"
    kwargs: dict[str, Any] = {
        "n_trials": n_trials,
        "timeout_s": timeout_s,
        "seed": seed,
        "device": device,
        "base": base,
        "trial_max_epochs": trial_max_epochs,
        "subsample": subsample,
    }
    root = Path(root)
    # The report covers every axle searched under root, not only this call's (``--axle front``).
    axles = [a for a in ("front", "rear") if a in datasets or (root / a / "optuna.db").exists()]
    if not parallel or len(datasets) < 2:
        try:
            return {a: search(ds, root / a, **kwargs) for a, ds in datasets.items()}
        finally:
            _safe_report(root, axles)
    ex = ProcessPoolExecutor(max_workers=len(datasets), mp_context=get_context("spawn"))
    try:
        futs = {a: ex.submit(_search_one, ds, str(root / a), kwargs) for a, ds in datasets.items()}
        wait(list(futs.values()), return_when=FIRST_EXCEPTION)
        for a, f in futs.items():
            if f.done() and f.exception() is not None:
                for other in futs.values():
                    other.cancel()
                for proc in list(getattr(ex, "_processes", {}).values()):
                    proc.terminate()
                exc = f.exception()
                if isinstance(exc, DatasetError):
                    raise exc
                raise RuntimeError(f"hyperparameter search failed for axle {a!r}") from exc
        return {a: f.result() for a, f in futs.items()}
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        _safe_report(root, axles)


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

#: What the validation score of the search and of the final model means (see ``train.py``).
SCORE_DEFINITION = (
    "Score = mean over channels of (masked RMSE in physical units / channel tolerance) on the "
    "complete validation split; lower is better, 1.0 means every channel is at its tolerance. "
    "The mean can hide a single bad channel: see the median / max channel ratio and the worst "
    "channels of the final model. Search-trial scores (subsampled train rows, trial epoch cap) "
    "are not directly comparable with the final-model score (full train split, full schedule)."
)

#: Trial user attribute holding ``{channel: ratio}`` (optional, written by the search).
TRIAL_RATIO_ATTR = "channel_ratio"
_N_WORST = 5

_MAIN_PARAMS = (
    "n_layers",
    "width",
    "activation",
    "residual",
    "layernorm",
    "dropout",
    "lr",
    "batch_size",
    "pod_k",
)
_TOP_N = 10


def _connect_ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True, timeout=30.0)


def _parse_dt(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _param_value(raw: float, dist_json: str) -> Any:
    """Decode a stored parameter (categorical values are stored as choice indices)."""
    dist = json.loads(dist_json)
    attrs = dist.get("attributes", {})
    if dist.get("name") == "CategoricalDistribution":
        return attrs["choices"][int(raw)]
    if dist.get("name") == "IntDistribution":
        return int(raw)
    return float(raw)


def _read_study(db: Path) -> dict[str, Any]:
    """Trials (state, times, value, params, intermediate steps) and user attrs of the study."""
    con = _connect_ro(db)
    try:
        trials: dict[int, dict[str, Any]] = {}
        for tid, num, state, t0, t1 in con.execute(
            "SELECT trial_id, number, state, datetime_start, datetime_complete FROM trials "
            "ORDER BY number"
        ):
            trials[tid] = {
                "number": int(num),
                "state": str(state),
                "start": _parse_dt(t0),
                "end": _parse_dt(t1),
                "value": None,
                "params": {},
                "last_step": None,
                "ratios": None,
            }
        for tid, value in con.execute("SELECT trial_id, value FROM trial_values WHERE objective=0"):
            if tid in trials:
                trials[tid]["value"] = None if value is None else float(value)
        for tid, name, raw, dist in con.execute(
            "SELECT trial_id, param_name, param_value, distribution_json FROM trial_params"
        ):
            if tid in trials:
                trials[tid]["params"][name] = _param_value(raw, dist)
        for tid, last in con.execute(
            "SELECT trial_id, MAX(step) FROM trial_intermediate_values GROUP BY trial_id"
        ):
            if tid in trials:
                trials[tid]["last_step"] = int(last)
        try:
            for tid, key, val in con.execute(
                "SELECT trial_id, key, value_json FROM trial_user_attributes"
            ):
                if tid in trials and key == TRIAL_RATIO_ATTR:
                    ratios = json.loads(val)
                    trials[tid]["ratios"] = ratios if isinstance(ratios, dict) else None
        except sqlite3.Error:
            pass  # older database layout: no per-trial ratios
        attrs: dict[str, Any] = {}
        for key, val in con.execute("SELECT key, value_json FROM study_user_attributes"):
            attrs[str(key)] = json.loads(val)
    finally:
        con.close()
    return {"trials": sorted(trials.values(), key=lambda t: t["number"]), "attrs": attrs}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _fmt_s(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    s = round(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def _fmt_num(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _duration(t: dict[str, Any]) -> float | None:
    if t["start"] is None or t["end"] is None:
        return None
    return float((t["end"] - t["start"]).total_seconds())


def _epochs(t: dict[str, Any]) -> str:
    return "n/a" if t["last_step"] is None else str(t["last_step"] + 1)


def _worst_channel(ratios: dict[str, Any] | None) -> str:
    """``name (ratio)`` of the largest channel ratio, ``n/a`` without data."""
    if not ratios:
        return "n/a"
    name, val = max(ratios.items(), key=lambda kv: float(kv[1]))
    return f"{name} ({float(val):.3g})"


def _ratio_lines(ratios: dict[str, Any]) -> list[str]:
    """Median, max and worst-channel lines of a ``{channel: ratio}`` mapping."""
    items = sorted(((str(k), float(v)) for k, v in ratios.items()), key=lambda kv: -kv[1])
    vals = [v for _, v in items]
    lines = [
        f"- Channel ratios (RMSE / tolerance): mean {statistics.fmean(vals):.4g}, "
        f"median {statistics.median(vals):.4g}, max {max(vals):.4g}.",
        "",
        f"Worst {min(_N_WORST, len(items))} channels:",
        "",
        "| channel | ratio |",
        "|---|---|",
        *[f"| {k} | {v:.4g} |" for k, v in items[:_N_WORST]],
    ]
    return lines


def _axle_section(axle: str, adir: Path) -> list[str]:
    lines = [f"## {axle.capitalize()} axle", ""]
    db = adir / "optuna.db"
    if not db.exists():
        return [*lines, f"No `optuna.db` found in `{adir}`.", ""]
    try:
        study = _read_study(db)
    except sqlite3.Error as exc:
        return [*lines, f"Could not read `{db}`: {exc}", ""]
    trials: list[dict[str, Any]] = study["trials"]
    budget: dict[str, Any] = study["attrs"].get("trial_budget") or {}
    res = _read_json(adir / "hpo_result.json") or {}
    model_info = _read_json(adir / "model" / "model.json")
    counts = Counter(t["state"] for t in trials)
    running = counts.get("RUNNING", 0)
    has_final = model_info is not None or "final_score" in res
    if running and not has_final:
        lines += ["**Status: search in progress.**", ""]

    # Dataset and budget.
    lines += ["### Dataset and budget", ""]
    n_tr, n_va, n_te = res.get("n_train"), res.get("n_val"), res.get("n_test")
    sub = budget.get("subsample", res.get("subsample"))
    rows = [
        ("Rows train / val / test", f"{n_tr} / {n_va} / {n_te}" if n_tr is not None else "n/a"),
        ("Rows per search trial (train)", "all" if sub is None else f"{sub}"),
        ("Device", res.get("device", "n/a")),
        ("Trials requested", res.get("n_trials_requested", "n/a")),
        ("Timeout", _fmt_s(res.get("timeout_s")) if res.get("timeout_s") else "none / n/a"),
        ("Trial epoch cap", budget.get("max_epochs", "n/a")),
        ("Trial patience", budget.get("patience", "n/a")),
    ]
    lines += ["| item | value |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows], ""]
    lines += [SCORE_DEFINITION, ""]

    # Trial counts and wall time.
    states = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no trials"
    starts = [t["start"] for t in trials if t["start"] is not None]
    ends = [t["end"] for t in trials if t["end"] is not None]
    span = (max(ends) - min(starts)).total_seconds() if starts and ends else None
    lines += [
        "### Trials",
        "",
        f"- Trial counts by state: {states} (total {len(trials)}).",
        f"- Search wall time (first start to last finish in the database): {_fmt_s(span)}.",
    ]
    if res.get("retrain_only"):
        lines.append("- Final model: **retrain only** (best trial retrained, no new trials run).")
    elif res.get("search_seconds") is not None:
        lines.append(f"- Last `search` call took {_fmt_s(res['search_seconds'])}.")
    lines.append("")

    done = sorted(
        (t for t in trials if t["state"] == "COMPLETE" and t["value"] is not None),
        key=lambda t: t["value"],
    )
    if done:
        b = done[0]
        with_ratios = any(t["ratios"] for t in done[:_TOP_N])
        lines += [
            "### Best trial",
            "",
            f"Trial {b['number']}, score {b['value']:.4f}.",
            "",
            "| param | value |",
            "|---|---|",
            *[f"| {k} | {_fmt_num(v)} |" for k, v in sorted(b["params"].items())],
            "",
            f"### Top {min(_TOP_N, len(done))} completed trials",
            "",
            "| trial | score | epochs | duration | "
            + ("worst channel | " if with_ratios else "")
            + " | ".join(_MAIN_PARAMS)
            + " |",
            "|" + "---|" * (4 + int(with_ratios) + len(_MAIN_PARAMS)),
        ]
        for t in done[:_TOP_N]:
            cells = [_fmt_num(t["params"].get(p, "")) for p in _MAIN_PARAMS]
            worst = f"{_worst_channel(t['ratios'])} | " if with_ratios else ""
            lines.append(
                f"| {t['number']} | {t['value']:.4f} | {_epochs(t)} | "
                f"{_fmt_s(_duration(t))} | {worst}" + " | ".join(cells) + " |"
            )
        lines.append("")
    else:
        lines += ["### Best trial", "", "No completed trial yet.", ""]

    pruned = [t for t in trials if t["state"] == "PRUNED"]
    lines += ["### Pruning", ""]
    if pruned:
        at = ", ".join(f"#{t['number']}@{_epochs(t)}" for t in pruned)
        lines.append(f"{len(pruned)} pruned trial(s) (number@epochs reached): {at}.")
    else:
        lines.append("No pruned trial.")
    lines.append("")

    lines += ["### Final model", ""]
    if model_info is not None or "final_score" in res:
        final_score = res.get("final_score", (model_info or {}).get("val_score"))
        hist = (model_info or {}).get("history") or []
        epochs = res.get("final_epochs", len(hist) if hist else None)
        lines += [
            f"- Validation score: {_fmt_num(final_score)}.",
            f"- Epochs trained: {epochs if epochs is not None else 'n/a'}.",
            f"- Training time: {_fmt_s(res.get('final_seconds'))}.",
            "- Trained on the full train split with the full epoch cap and patience.",
        ]
        ratios = ((model_info or {}).get("meta") or {}).get("val_channel_ratio")
        if isinstance(ratios, dict) and ratios:
            lines += _ratio_lines(ratios)
    else:
        lines.append("Not available yet (search not finished).")
    lines.append("")
    return lines


def write_hpo_report(
    models_dir: str | Path,
    axles: Sequence[str] = ("front", "rear"),
    *,
    out: str | Path | None = None,
) -> Path:
    """Write the search report (one section per axle) and return its path.

    Reads ``models_dir/<axle>/optuna.db`` read-only (safe while the search still runs) and,
    when present, ``hpo_result.json`` and ``model/model.json`` of the axle. The default
    output is ``models_dir/hpo_report.md``; axles without a database are noted, not fatal.
    """
    mdir = Path(models_dir)
    target = Path(out) if out is not None else mdir / "hpo_report.md"
    lines = ["# Surrogate architecture search report", ""]
    for ax in axles:
        lines += _axle_section(ax, mdir / ax)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target
