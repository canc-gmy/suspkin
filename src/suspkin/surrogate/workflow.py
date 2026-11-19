"""Training and validation of the per-axle surrogate from a study and a surrogate root folder.

Experimental, needs torch and optuna. Folder layout under the root ``DIR``::

    DIR/dataset/<axle>/      written by ``surrogate sample``
    DIR/models/<axle>/       optuna.db, trials.csv, model/
    DIR/validation/<axle>/   validation.csv, worst.csv, parity.csv, worst_curves.csv,
                             summary.json, figures/
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from suspkin.optimize.spec import StudyConfigError, SurrogateSpec, load_study
from suspkin.surrogate.data_types import SPLIT_TEST, Axle, AxleDataset
from suspkin.surrogate.dataset import DatasetError, check_dataset_matches, load_dataset
from suspkin.surrogate.hpo import HpoResult, retrain_best, search_axles
from suspkin.surrogate.report import write_hpo_report
from suspkin.surrogate.train import (
    EvalReport,
    TrainConfig,
    TrainedModel,
    evaluate,
    predict_physical,
    resolve_device,
)

__all__ = [
    "display_unit",
    "retrain_surrogate",
    "train_surrogate",
    "validate_surrogate",
    "validation_table",
]

_RAD2DEG = 180.0 / math.pi
_MM_CHANNELS = frozenset(
    {
        "scrub_radius",
        "mechanical_trail",
        "half_track_change",
        "contact_patch_dx",
        "contact_patch_dy",
        "wheel_center_recession",
        "coilover_length",
    }
)
_KEY_VALUE_CHANNELS = ("camber", "toe", "coilover_length")
_KEY_ANGLES = ("fv_swing_angle", "sv_swing_angle", "sv_wc_angle")
_MAX_PARITY_POINTS = 4000
_N_WORST_CURVES = 3

_HASH_KEYS = ("grid_hash", "design_space_hash", "vehicle_hash")


def display_unit(channel: str) -> tuple[str, float]:
    """``(unit label, factor from SI)`` of a channel: mm or deg."""
    if channel in _MM_CHANNELS:
        return "mm", 1e3
    return "deg", _RAD2DEG


def _load(study_path: str | Path) -> SurrogateSpec:
    study, _vehicle = load_study(str(study_path))
    if study.surrogate is None:
        raise StudyConfigError(f"{study_path}: study has no 'surrogate' block")
    return study.surrogate


def _load_checked(
    study_path: str | Path, root: Path, axles: Sequence[Axle]
) -> tuple[SurrogateSpec, dict[Axle, AxleDataset]]:
    spec = _load(study_path)
    out: dict[Axle, AxleDataset] = {}
    for ax in axles:
        check_dataset_matches(study_path, root, ax)
        out[ax] = load_dataset(root, ax)
    return spec, out


# --------------------------------------------------------------------------- training


def _train_config(spec: SurrogateSpec) -> TrainConfig:
    return TrainConfig(
        max_epochs=spec.training.max_epochs,
        patience=spec.training.patience,
        num_threads=spec.training.num_threads,
        finetune_lr_factor=spec.training.finetune_lr_factor,
        seed=spec.dataset.seed,
    )


def retrain_surrogate(
    study_path: str | Path,
    root: str | Path,
    *,
    axle: Axle,
    device: str | None = None,
) -> HpoResult:
    """Retrain the best configuration of an existing search of ``axle`` (no new trials).

    Uses the same dataset check and training settings as :func:`train_surrogate`, then
    refreshes ``models/hpo_report.md`` for every axle that has a study.
    """
    root = Path(root)
    spec, datasets = _load_checked(study_path, root, [axle])
    res = retrain_best(
        datasets[axle],
        root / "models" / str(axle),
        device=device or spec.training.device,
        base=_train_config(spec),
    )
    models = root / "models"
    write_hpo_report(models, [a for a in ("front", "rear") if (models / a / "optuna.db").exists()])
    return res


def train_surrogate(
    study_path: str | Path,
    root: str | Path,
    *,
    axles: Sequence[Axle],
    n_trials: int | None = None,
    timeout: float | None = None,
    device: str | None = None,
    echo: Callable[[str], None] | None = None,
) -> dict[str, HpoResult]:
    """Run the architecture search of every axle and save the best network.

    Settings come from ``surrogate.training`` / ``surrogate.hpo`` of the study; ``n_trials``
    (count), ``timeout`` (seconds) and ``device`` override them. Models land in
    ``root/models/<axle>/model``.

    Raises
    ------
    StudyConfigError
        Study without ``surrogate`` block.
    DatasetError
        Missing dataset, or one that does not match the study (grid, design space, vehicle).
    RuntimeError
        ``device="cuda"`` without CUDA, or no trial completed.
    """
    say = echo or (lambda _m: None)
    root = Path(root)
    spec, datasets = _load_checked(study_path, root, axles)
    dev_name = device or spec.training.device
    dev = resolve_device(dev_name)
    say(f"device: {dev} (requested {dev_name})")
    base = _train_config(spec)
    parallel = None if spec.hpo.parallel == "auto" else bool(spec.hpo.parallel)
    trials = spec.hpo.n_trials if n_trials is None else int(n_trials)
    tmo = spec.hpo.timeout if timeout is None else float(timeout)
    if trials < 1:
        raise ValueError("n_trials must be >= 1")
    if tmo is not None and tmo <= 0:
        raise ValueError("timeout must be positive")
    say(
        f"search: {trials} trials, timeout "
        + ("none" if tmo is None else f"{tmo:.0f} s")
        + f", axles {', '.join(datasets)}"
    )
    results = search_axles(
        {str(k): v for k, v in datasets.items()},
        root / "models",
        n_trials=trials,
        timeout_s=tmo,
        seed=spec.dataset.seed,
        device=dev_name,
        base=base,
        parallel=parallel,
        trial_max_epochs=spec.hpo.max_epochs,
        subsample=spec.hpo.subsample,
    )
    for ax, res in results.items():
        say(
            f"{ax}: {res.n_complete}/{res.n_trials} trials complete, best val score "
            f"{res.best_value:.3f} (retrained {res.final_score:.3f}), model {res.model_dir}"
        )
    return results


# --------------------------------------------------------------------------- validation


def _sample_points(n: int, rng: np.random.Generator) -> NDArray[np.intp]:
    if n <= _MAX_PARITY_POINTS:
        return np.arange(n)
    return np.sort(rng.choice(n, size=_MAX_PARITY_POINTS, replace=False))


def _parity_table(model: TrainedModel, sub: AxleDataset, y: NDArray[np.float64]) -> pd.DataFrame:
    """Valid test grid points of the key channels, in display units (angles from cos/sin)."""
    names = list(model.channels)
    truth = np.nan_to_num(sub.Y.astype(np.float64))
    rng = np.random.default_rng(0)
    frames: list[pd.DataFrame] = []

    def add(name: str, fom: NDArray[Any], pred: NDArray[Any], unit: str) -> None:
        sel = _sample_points(fom.size, rng)
        frames.append(
            pd.DataFrame(
                {"channel": name, "unit": unit, "fom": fom[sel], "pred": pred[sel]},
            )
        )

    for name in _KEY_VALUE_CHANNELS:
        if name not in names:
            continue
        i = names.index(name)
        unit, f = display_unit(name)
        m = sub.mask
        add(name, truth[:, i][m] * f, y[:, i][m] * f, unit)
    for base in _KEY_ANGLES:
        if base not in names:
            continue
        i = names.index(base)
        m = sub.mask
        fom = truth[:, i][m] * _RAD2DEG
        pred = y[:, i][m] * _RAD2DEG
        add(base, fom, pred, "deg")
    if not frames:
        return pd.DataFrame({"channel": [], "unit": [], "fom": [], "pred": []})
    return pd.concat(frames, ignore_index=True)


def _worst_curves(
    model: TrainedModel,
    sub: AxleDataset,
    y: NDArray[np.float64],
    channel: str,
    n: int = _N_WORST_CURVES,
) -> pd.DataFrame:
    """Predicted vs FOM travel curves of the ``n`` worst test samples on ``channel``."""
    i = list(model.channels).index(channel)
    unit, f = display_unit(channel)
    truth = np.nan_to_num(sub.Y.astype(np.float64))[:, i]  # (N, R, T)
    pred = y[:, i]
    se = np.where(sub.mask, (pred - truth) ** 2, 0.0)
    cnt = np.maximum(sub.mask.sum(axis=(1, 2)), 1)
    score = np.sqrt(se.sum(axis=(1, 2)) / cnt)
    score = np.where(sub.mask.any(axis=(1, 2)), score, -np.inf)
    rows: list[pd.DataFrame] = []
    travel_mm = sub.travel * 1e3
    for rank, s in enumerate(np.argsort(-score)[:n]):
        if not np.isfinite(score[s]):
            continue
        r = int(np.argmax(se[s].sum(axis=1)))
        ok = sub.mask[s, r]
        rows.append(
            pd.DataFrame(
                {
                    "rank": rank,
                    "sample": int(s),
                    "channel": channel,
                    "unit": unit,
                    "rack_mm": float(sub.rack[r] * 1e3),
                    "travel_mm": travel_mm[ok],
                    "fom": truth[s, r][ok] * f,
                    "pred": pred[s, r][ok] * f,
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            {k: [] for k in ("rank", "sample", "channel", "unit", "travel_mm", "fom", "pred")}
        )
    return pd.concat(rows, ignore_index=True)


def validation_table(report: EvalReport) -> pd.DataFrame:
    """Per-channel test error in display units: ``channel, unit, rmse, tol, ratio, passed``."""
    units = [display_unit(c) for c in report.channels]
    fac = np.array([f for _, f in units])
    return pd.DataFrame(
        {
            "channel": list(report.channels),
            "unit": [u for u, _ in units],
            "rmse": report.per_channel_rmse * fac,
            "tol": report.tol * fac,
            "ratio": report.per_channel_ratio,
            "passed": report.per_channel_ratio <= 1.0,
        }
    )


def _write_figures(
    table: pd.DataFrame, parity: pd.DataFrame, curves: pd.DataFrame, fig_dir: Path, axle: str
) -> None:
    from suspkin.plotting.figures import save_figure
    from suspkin.plotting.surrogate import plot_parity, plot_rmse_ratio, plot_worst_curves

    save_figure(
        plot_rmse_ratio(table, f"{axle}: test error per channel"),
        fig_dir / "rmse_ratio",
        ("png",),
        150,
    )
    if len(parity):
        save_figure(
            plot_parity(parity, title=f"{axle}: parity (test split)"),
            fig_dir / "parity",
            ("png",),
            150,
        )
    if len(curves):
        save_figure(plot_worst_curves(curves), fig_dir / "worst_curves", ("png",), 150)


def validate_surrogate(
    study_path: str | Path,
    root: str | Path,
    *,
    axles: Sequence[Axle],
    device: str | None = None,
    echo: Callable[[str], None] | None = None,
) -> dict[str, EvalReport]:
    """Evaluate the saved networks on the TEST split and write tables and figures.

    Writes ``root/validation/<axle>/`` (see the module docstring). The RMSE table is in
    display units (mm, deg) against the tolerance of the channel registry.

    Raises
    ------
    StudyConfigError
        Study without ``surrogate`` block.
    DatasetError
        Dataset or model missing, mismatching the study, or an empty test split.
    RuntimeError
        ``device="cuda"`` without CUDA.
    """
    say = echo or (lambda _m: None)
    root = Path(root)
    spec, datasets = _load_checked(study_path, root, axles)
    dev_name = device or spec.training.device
    dev = resolve_device(dev_name)
    say(f"device: {dev} (requested {dev_name})")
    out: dict[str, EvalReport] = {}
    for ax, ds in datasets.items():
        mdir = root / "models" / ax / "model"
        if not (mdir / "model.json").is_file():
            raise DatasetError(f"{ax}: no trained model in {mdir}; run `surrogate train` first")
        model = TrainedModel.load(mdir, device=dev.type)
        for key in _HASH_KEYS:
            if model.meta.get(key) != ds.meta.get(key):
                raise DatasetError(f"{ax}: model was trained on another dataset ({key} differs)")
        sub = ds.subset(SPLIT_TEST)
        if sub.n == 0:
            raise DatasetError(f"{ax}: the test split is empty; sample more points")
        report = evaluate(model, ds, SPLIT_TEST, device=dev)
        y, _p = predict_physical(model, sub.X, device=dev)

        vdir = root / "validation" / ax
        vdir.mkdir(parents=True, exist_ok=True)
        table = validation_table(report)
        table.to_csv(vdir / "validation.csv", index=False)
        pd.DataFrame(
            [{"rank": k, "test_index": i, "score": s} for k, (i, s) in enumerate(report.worst)],
            columns=["rank", "test_index", "score"],
        ).to_csv(vdir / "worst.csv", index=False)
        parity = _parity_table(model, sub, y)
        parity.to_csv(vdir / "parity.csv", index=False)
        worst_channel = str(table["channel"].iloc[int(np.argmax(table["ratio"].to_numpy()))])
        curves = _worst_curves(model, sub, y, worst_channel)
        curves.to_csv(vdir / "worst_curves.csv", index=False)
        n_over = int((~table["passed"]).sum())
        (vdir / "summary.json").write_text(
            json.dumps(
                {
                    "axle": ax,
                    "n_test": report.n,
                    "score": report.score,
                    "worst_channel": worst_channel,
                    "channels_over_tolerance": n_over,
                    "validity_accuracy": report.validity_accuracy,
                    "validity_auc": report.validity_auc,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_figures(table, parity, curves, vdir / "figures", ax)
        out[ax] = report
    return out
