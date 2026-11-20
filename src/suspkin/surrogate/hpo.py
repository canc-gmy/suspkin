"""Optuna architecture search per axle (experimental, needs torch and optuna)."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

from suspkin.surrogate.data import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, AxleDataset
from suspkin.surrogate.dataset import DatasetError
from suspkin.surrogate.model import MLPConfig
from suspkin.surrogate.report import write_hpo_report
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
    """Build the configs of a trial from its parameter dict.

    Args:
        p: Hyper-parameters of a trial (see :func:`suggest`).
        base: Training settings that the trial does not vary.

    Returns:
        ``(MLPConfig, TrainConfig)`` of the trial.
    """
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
    """Sample one hyper-parameter set from the Optuna search space.

    See ``docs/concepts/surrogate.md`` for the ranges.
    """
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

    Args:
        base: Full training settings.
        trial_max_epochs: Epoch cap of a trial; ``None`` keeps ``base.max_epochs``.
        n_train: Number of train rows available.
        subsample: Train rows per trial; ``None`` for all.

    Returns:
        Dict with ``max_epochs``, ``patience`` and ``n_train`` (``None`` for all rows).
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
    """Keep ``n`` random train rows (seeded, same rows every call); val/test stay complete.

    Args:
        ds: Axle dataset.
        n: Train rows to keep; ``None`` keeps all.
        seed: Random seed of the row choice.

    Returns:
        The reduced dataset.
    """
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

    Args:
        ds: Training dataset of the axle.
        out_dir: Axle folder holding ``optuna.db`` and the output ``model/``.
        n_trials: Number of Optuna trials to run in total.
        timeout_s: Wall-time limit of the search [s]; ``None`` for none.
        seed: Sampler seed.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        base: Training settings that trials do not vary; ``None`` for defaults.
        trial_max_epochs: Epoch cap of a trial; ``None`` keeps ``base.max_epochs``.
        subsample: Train rows per trial; ``None`` for all.

    Returns:
        Outcome of the search and the final retrain.
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

    Args:
        ds: Training dataset of the axle.
        out_dir: Axle folder holding ``optuna.db``.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        base: Training settings; ``None`` for defaults.
        extra: Extra fields stored in ``hpo_result.json``.
        t_start: ``time.perf_counter()`` value at the start of the search, for the timings.

    Returns:
        Outcome of the retrain.

    Raises:
        DatasetError: No ``optuna.db`` / study in ``out_dir``, or no completed trial.
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

    Args:
        datasets: Dataset per axle name.
        root: Models folder; each axle uses ``root/<axle>``.
        n_trials: Number of Optuna trials per axle.
        timeout_s: Wall-time limit per axle [s]; ``None`` for none.
        seed: Sampler seed.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        base: Training settings that trials do not vary.
        parallel: Force parallel or sequential axles; see above.
        trial_max_epochs: Epoch cap of a trial.
        subsample: Train rows per trial.

    Returns:
        Outcome per axle name.
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
