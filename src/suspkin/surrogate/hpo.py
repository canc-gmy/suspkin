"""Optuna architecture search per axle (experimental, needs torch and optuna)."""

from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import optuna
import torch

from suspkin.surrogate.data_types import AxleDataset
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


def search(
    ds: AxleDataset,
    out_dir: str | Path,
    *,
    n_trials: int,
    timeout_s: float | None = None,
    seed: int = 0,
    device: str = "auto",
    base: TrainConfig | None = None,
) -> HpoResult:
    """Run (or resume) the Optuna study, then retrain the best config into ``out_dir/model``.

    The study lives in ``out_dir/optuna.db`` (SQLite); rerunning with the same directory
    adds trials to the same study. ``trials.csv`` lists all trials.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolve_device(device)  # fail early / log the choice
    base = replace(base or TrainConfig(), device=device)  # type: ignore[arg-type]
    storage = f"sqlite:///{(out / 'optuna.db').as_posix()}"
    name = f"suspkin_{ds.axle}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=max(base.max_epochs, 4))
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

    def objective(trial: optuna.Trial) -> float:
        mcfg, tcfg = configs_from_params(suggest(trial), base)

        def report(epoch: int, score: float) -> None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            return train(ds, mcfg, tcfg, report=report).val_score
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned() from exc
        except FloatingPointError as exc:  # nan loss: diverged configuration
            raise optuna.TrialPruned() from exc

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_s,
        gc_after_trial=True,
        catch=(RuntimeError, ValueError, FloatingPointError),
    )
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise RuntimeError("no completed trial; cannot select a best configuration")
    best = study.best_trial
    mcfg, tcfg = configs_from_params(best.params, base)
    final: TrainedModel = train(ds, mcfg, tcfg)
    final.save(out / "model")
    return HpoResult(
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
    }
    root = Path(root)
    if not parallel or len(datasets) < 2:
        return {a: search(ds, root / a, **kwargs) for a, ds in datasets.items()}
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
                raise RuntimeError(
                    f"hyperparameter search failed for axle {a!r}"
                ) from f.exception()
        return {a: f.result() for a, f in futs.items()}
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
