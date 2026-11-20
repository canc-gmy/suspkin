"""Active-learning optimizer: surrogate search in a trust region, FOM verification, fine-tuning.

Experimental (see ``docs/guides/stability.md``); background in ``docs/concepts/surrogate.md``.
Each iteration ``k``:

1. runs the study's driver on a :class:`~suspkin.surrogate.problem.SurrogateProblem` restricted
   to the box ``[x_b - r, x_b + r]`` around the best FOM-verified design ``x_b``;
2. picks the infill points (predicted optimum, extra candidates, perturbations);
3. evaluates them on the FOM (source ``al_<k>``) and appends their curves to the datasets;
4. logs the curve and objective residuals of the predicted optimum;
5. fine-tunes the networks on the grown datasets;
6. updates the trust-region radius from the ratio of actual to predicted improvement.

The numerical rules (radius update, stopping, infill selection) are small pure functions.
Torch is imported only when a network is fine-tuned or loaded.
"""

# ruff: noqa: N806  (X, F, G follow the Problem interface)
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time as _time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from suspkin.kinematics.rocker import KinematicsError
from suspkin.optimize.design import DesignSpace, InfeasibleDesign
from suspkin.optimize.drivers import (
    RunError,
    RunResult,
    _finish,
    _run_de,
    _run_de_slsqp,
    _run_local,
    _run_nsga2,
    _start_point,
)
from suspkin.optimize.evaluate import evaluate
from suspkin.optimize.history import cache_key, read_history
from suspkin.optimize.outputs import (
    _ok_feasible,
    best_row,
    copy_inputs,
    pareto_rows,
    read_meta,
    study_hash,
)
from suspkin.optimize.pool import resolve_workers
from suspkin.optimize.problem import BudgetExhausted, Problem
from suspkin.optimize.spec import Optimizer, Study, StudyConfigError, load_study
from suspkin.optimize.spec.surrogate import FinetuneSpec
from suspkin.plotting.optimization import knee_index
from suspkin.plotting.surrogate import write_surrogate_figures
from suspkin.surrogate.active_steps import (
    BoxProblem,
    StopState,
    advance_finetune,
    advance_stop,
    box_bounds,
    curve_residual,
    dominated,
    finetune_decision,
    hits_boundary,
    is_improved,
    replay_finetune_state,
    select_infill,
    stop_reason,
    trust_ratio,
    update_radius,
)
from suspkin.surrogate.channels import CornerSample, channels_for, extract_corner
from suspkin.surrogate.data import (
    SPLIT_TRAIN,
    SPLIT_VAL,
    Axle,
    AxleDataset,
    AxleGrid,
    AxleSpace,
    axle_grids,
)
from suspkin.surrogate.dataset import (
    DatasetError,
    append_rows,
    check_dataset_matches,
    check_model_matches,
    load_dataset,
)
from suspkin.surrogate.evaluate import AxlePredictor
from suspkin.surrogate.problem import SurrogateProblem

F64 = NDArray[np.float64]
Echo = Callable[[str], None]

__all__ = [
    "AL_COLUMNS",
    "ActiveResult",
    "run_active_learning",
]

_AXLES: tuple[Axle, ...] = ("front", "rear")
AL_COLUMNS = (
    "k",
    "radius",
    "radius_next",
    "f_nn",
    "f_fom",
    "feasible_nn",
    "feasible_fom",
    "rho",
    "best_f",
    "best_feasible",
    "n_fom",
    "n_new",
    "resid_front",
    "resid_rear",
    "val_front",
    "val_rear",
    "kept_init_front",
    "kept_init_rear",
    "skipped_front",
    "skipped_rear",
    "skip_reason_front",
    "skip_reason_rear",
    "agree",
    "improved",
    "at_min",
    "sec_surrogate",
    "sec_fom",
    "sec_extract",
    "sec_finetune",
    "sec_total",
)


def _axle_sample(space: DesignSpace, asp: AxleSpace, grid: AxleGrid, x_axle: F64) -> CornerSample:
    """FOM curves of one axle at one design (all-invalid when the design cannot be built)."""
    try:
        return extract_corner(space.to_vehicle(asp.to_full(x_axle)), asp.axle, grid)
    except (InfeasibleDesign, KinematicsError):
        n_r, n_t = grid.shape
        chans = tuple(c.name for c in channels_for(asp.axle))
        return CornerSample(
            np.full((len(chans), n_r, n_t), np.nan), np.zeros((n_r, n_t), dtype=bool), False, chans
        )


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    """Best row of a history: ``x``, ``f_total``, ``feasible`` (status ok) and ``max_g``."""
    i = best_row(df)
    xcols = [c for c in df.columns if c.startswith("x_")]
    return {
        "x": df[xcols].to_numpy(dtype=np.float64)[i],
        "f": float(df["f_total"].iloc[i]),
        "feasible": bool(_ok_feasible(df)[i]),
        "max_g": float(df["max_g"].iloc[i]),
    }


def _front(df: pd.DataFrame, study: Study) -> tuple[F64, F64]:
    """Objective values and design vectors of the feasible non-dominated rows of a history."""
    idx = pareto_rows(df, study)
    ocols = [f"f_{o.name}" for o in study.objectives]
    xcols = [c for c in df.columns if c.startswith("x_")]
    return (
        df[ocols].to_numpy(dtype=np.float64)[idx],
        df[xcols].to_numpy(dtype=np.float64)[idx],
    )


def _shard_rows(path: Path, template: AxleDataset) -> AxleDataset:
    """The rows of one appended shard as a dataset shaped like ``template``."""
    with np.load(path) as z:
        return AxleDataset(
            axle=template.axle,
            X=z["X"].astype(np.float64),
            Y=z["Y"].astype(np.float32),
            mask=z["mask"].astype(bool),
            feasible=z["feasible"].astype(bool),
            split=z["split"].astype(np.uint8),
            source=z["source"],
            weight=z["weight"].astype(np.float64),
            channels=template.channels,
            param_names=template.param_names,
            travel=template.travel,
            rack=template.rack,
            meta=dict(template.meta),
        )


def _finetune(
    root: Path,
    axle: Axle,
    *,
    ft: FinetuneSpec,
    lr_factor: float,
    device: str,
    seed: int,
    k: int,
    caches: dict[str, Any],
) -> tuple[float, bool]:
    """Fine-tune the current model of ``axle``; returns ``(score, kept_init_weights)``.

    Trains on all active-learning rows plus ``ft.replay`` earlier train rows (drawn once from
    ``seed``, normalised and moved to the device on the first call, then reused through
    ``caches``); the validation split is unchanged. The early-stopping score is the mean of
    the validation score and the score on all active-learning rows; training stops after
    ``ft.patience`` epochs without a new best.
    """
    from dataclasses import replace

    from suspkin.surrogate.train import FinetuneCache, TrainedModel, train

    mdir = root / "models" / axle / "model"
    pre = root / "models" / axle / "pre_al"
    if not pre.exists():
        shutil.copytree(mdir, pre)
    model = TrainedModel.load(mdir, device=device)
    cache: FinetuneCache | None = caches.get(axle)
    if cache is None:
        full = load_dataset(root, axle)
        is_al = np.char.startswith(full.source.astype(str), "al_")
        pool = np.flatnonzero((full.split == SPLIT_TRAIN) & ~is_al)
        rng = np.random.default_rng([seed, 4096])
        pick = (
            rng.choice(pool, size=min(ft.replay, pool.size), replace=False) if pool.size else pool
        )
        keep = full.split == SPLIT_VAL
        keep[pick] = True
        cache = FinetuneCache(model, full.subset(keep), device)
        if is_al.any():
            cache.add_rows(full.subset(is_al))
        caches[axle] = cache
        del full
    tcfg = replace(
        model.tcfg,
        max_epochs=ft.epochs,
        patience=ft.patience,
        device=cast(Any, device),
        finetune_lr_factor=lr_factor,
        seed=seed + k,
    )
    new = train(cache.template, model.mcfg, tcfg, init=model, cache=cache)
    new.meta["al_iteration"] = k
    new.save(root / "models" / axle / f"al_{k}")
    new.save(mdir)
    return float(new.val_score), bool(new.meta.get("kept_init_weights", False))


def _run_inner(box: BoxProblem, opt: Optimizer) -> None:
    """The study's driver on the box problem; a spent budget just ends it."""
    p = cast(Problem, box)
    m = opt.method
    try:
        if m == "nsga2":
            _run_nsga2(p, opt, False)
        elif m == "de":
            _run_de(p, opt, False, evals=p.max_evals)
        elif m == "de+slsqp":
            _run_de_slsqp(p, opt, False)
        else:
            _run_local(p, opt, m, _start_point(p, opt))
    except BudgetExhausted:
        pass


def _recover_dataset(
    root: Path,
    axle: Axle,
    asp: AxleSpace,
    grid: AxleGrid,
    space: DesignSpace,
    hist: pd.DataFrame,
    weight: float,
) -> int:
    """Append the ``al_*`` FOM rows of ``hist`` that the axle dataset lacks (an interrupted run).

    Rows are matched by the axle part of ``x``. Returns the number of rows appended.
    """
    ds = load_dataset(root, axle)
    have = {cache_key(x) for x in ds.X}
    src = hist["source"].to_numpy(dtype=str)
    xcols = [f"x_{n}" for n in space.names]
    xs = hist[xcols].to_numpy(dtype=np.float64)
    groups: dict[str, list[F64]] = {}
    for i in np.flatnonzero(np.char.startswith(src, "al_")):
        xa = xs[i][asp.idx]
        key = cache_key(xa)
        if key in have:
            continue
        have.add(key)
        groups.setdefault(str(src[i]), []).append(xa)
    n = 0
    for source, pts in groups.items():
        samples = [_axle_sample(space, asp, grid, x) for x in pts]
        append_rows(root, axle, np.array(pts), samples, source, weight=weight)
        n += len(pts)
    return n


@dataclass
class ActiveResult:
    """Outcome of :func:`run_active_learning`."""

    run: RunResult
    iterations: int  # iterations in al.csv (including earlier runs)
    new_iterations: int
    stop_reason: str
    n_fom: int  # rows in the FOM history
    al_path: Path
    surrogate_root: Path
    device: str


def run_active_learning(
    study_path: str | Path,
    surrogate_root: str | Path,
    out_dir: str | Path | None = None,
    *,
    resume: bool = False,
    workers: int | str | None = None,
    evaluations: int | None = None,
    time: float | None = None,
    method: str | None = None,
    device: str | None = None,
    report: bool = True,
    echo: Echo | None = None,
    finetune: bool = True,
    predictors: Mapping[str, AxlePredictor] | None = None,
) -> ActiveResult:
    """Run the active-learning optimizer of a study and write the run folder.

    ``surrogate_root`` holds ``dataset/<axle>/`` and ``models/<axle>/model`` (from
    ``surrogate sample`` and ``train``); datasets and models there are updated in place.
    ``evaluations`` is the budget of FOM evaluations counted over the whole run folder
    (resumed rows included); ``time`` (seconds) counts this invocation. ``method`` and
    ``workers`` override the study's optimizer block; ``device`` the training device.

    ``finetune=False`` leaves datasets and models untouched (the loop then only searches and
    verifies). ``predictors`` injects axle predictors (for example oracles, a test hook) in
    place of the saved networks; it needs ``finetune=False`` and no surrogate root content.

    Budgets: the FOM budget counts every row of the run folder's history (baseline and resumed
    rows included, unlike the v0.8 resume); a resume refuses a study whose constraints or
    objectives changed, instead of dropping the cache.

    Args:
        study_path: Study YAML with objectives, an ``optimizer`` block and ``surrogate`` settings.
        surrogate_root: Folder with the datasets and models.
        out_dir: Run folder; ``None`` creates a timestamped one.
        resume: Continue an existing run folder.
        workers: Overrides the study's worker count.
        evaluations: Overrides the FOM evaluation budget.
        time: Overrides the time budget [s].
        method: Overrides the optimizer method.
        device: Training device.
        report: Also write the before/after report.
        echo: Optional line printer.
        finetune: Fine-tune the networks; see above.
        predictors: Injected axle predictors; see above.

    Returns:
        Summary of the run.

    Raises:
        StudyConfigError, RunError, DatasetError: Invalid study or arguments, missing or
            mismatching dataset / models, or a changed study on resume.
    """
    say: Echo = echo or (lambda _m: None)
    study_path = Path(study_path)
    root = Path(surrogate_root)
    started = datetime.now().astimezone()
    study, vehicle = load_study(study_path)
    if study.surrogate is None:
        raise StudyConfigError(f"{study_path}: study has no 'surrogate' block")
    if not study.objectives:
        raise StudyConfigError(f"{study_path}: the study has no objectives")
    spec = study.surrogate
    al = spec.active_learning
    opt = study.optimizer or Optimizer()
    if method is not None:
        if method not in ("nsga2", "de", "slsqp", "trust-constr", "de+slsqp"):
            raise RunError(f"unknown method {method!r}")
        opt = opt.model_copy(update={"method": method})
    if opt.mode == "pareto" and opt.method != "nsga2":
        raise RunError("pareto mode requires method nsga2")
    if resume:
        if out_dir is None:
            raise RunError("--resume needs an explicit run folder (--out)")
        if not (Path(out_dir) / "history.csv").is_file():
            raise RunError(f"no history.csv in {out_dir}")

    # surrogate root: datasets and models of every axle with variables
    if predictors is not None and finetune:
        raise ValueError("injected predictors need finetune=False")
    dev_name = device or spec.training.device
    if dev_name not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda, got {dev_name!r}")
    dev_used = "injected"
    dev_type = "injected"
    if predictors is None:
        from suspkin.surrogate.train import resolve_device

        dev_resolved = resolve_device(dev_name)
        dev_used, dev_type = str(dev_resolved), dev_resolved.type
    full_space = DesignSpace(study, vehicle)
    asps = {a: AxleSpace(full_space, a) for a in _AXLES}
    active_axles = [a for a in _AXLES if asps[a].n > 0]
    if not active_axles:
        raise StudyConfigError(f"{study_path}: the study has no design variable")
    # Shared models: without ROOT/dataset/<axle> the model metadata alone is checked and the
    # axle is never fine-tuned (reason "no_dataset").
    no_ds = {
        a: predictors is None and not (root / "dataset" / a / "meta.json").is_file()
        for a in active_axles
    }
    for ax in active_axles if predictors is None else []:
        if not no_ds[ax]:
            check_dataset_matches(study_path, root, ax)
        if not (root / "models" / ax / "model" / "model.json").is_file():
            raise DatasetError(
                f"{ax}: no trained model in {root / 'models' / ax / 'model'}; "
                "run `surrogate train` first"
            )
        check_model_matches(study_path, root / "models" / ax / "model", ax)
    grids = axle_grids(spec.grid)

    run_dir = (
        Path(out_dir)
        if out_dir is not None
        else Path("results") / "opt" / f"{study_path.stem}_{started.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    shash = study_hash(study)
    old_meta = read_meta(run_dir) if resume else {}
    if old_meta.get("study_hash") not in (None, shash):
        raise RunError("the constraints/objectives changed since the last run; start a new run")
    copy_inputs(study_path, study_path.parent / study.vehicle, run_dir)

    fom_budget = evaluations if evaluations is not None else al.fom_evaluations
    time_budget = time if time is not None else al.time
    n_infill = 1 + al.infill.top + al.infill.perturb
    al_path = run_dir / "al.csv"
    rows: list[dict[str, Any]] = []
    if resume and al_path.is_file():
        rows = [
            {str(a): b for a, b in rw.items()} for rw in pd.read_csv(al_path).to_dict("records")
        ]
        for rw in rows:  # tables of earlier versions lack the skipped_<axle> columns
            rw.setdefault("skipped_front", False)
            rw.setdefault("skipped_rear", False)
            rw.setdefault("skip_reason_front", "")
            rw.setdefault("skip_reason_rear", "")
    state = StopState()
    for rw in rows:
        state = advance_stop(
            state,
            agree=bool(rw["agree"]),
            improved=bool(rw["improved"]),
            at_min=bool(rw["at_min"]),
        )
    k0 = int(rows[-1]["k"]) + 1 if rows else 0
    ft_state = {a: replay_finetune_state(rows, a, al.finetune) for a in _AXLES}
    ft_caches: dict[str, Any] = {}
    # On a hardpoint grid the trust region cannot be smaller than one grid step per variable.
    step_z = full_space.grid_step_z()
    tr = al.trust_region
    tr_min = min(max(tr.min, float(np.max(step_z, initial=0.0))), tr.max)
    tr = tr.model_copy(update={"min": tr_min, "initial": max(tr.initial, tr_min)})
    r = max(float(rows[-1]["radius_next"]) if rows else tr.initial, tr_min)

    t_start = _time.perf_counter()
    reason = "max_iterations"
    error: BaseException | None = None
    n_workers = min(resolve_workers(opt.workers if workers is None else workers), n_infill)
    new_iters = 0
    with Problem(
        study_path, run_dir, resume=resume, workers=n_workers, evaluations=None, time=None
    ) as fom:
        fom.max_evals = None  # budgets are handled here, per iteration
        fom.max_time = None
        space = fom.space
        base_x = space.from_vehicle(clip=True)
        baseline = evaluate(space)
        fom.evaluate_batch(base_x[None, :], "baseline")
        f_scale = max(abs(baseline.f_total), 1.0)
        if resume and finetune:
            hist0 = read_history(fom.history_path)
            for a in (a for a in active_axles if not no_ds[a]):
                n_rec = _recover_dataset(
                    root, a, asps[a], grids[a], space, hist0, al.finetune.weight
                )
                if n_rec:
                    say(f"resume: appended {n_rec} missing active-learning row(s), {a} dataset")
        if finetune and predictors is None:
            for a in active_axles:  # keep the starting networks (also if an axle is never tuned)
                pre = root / "models" / a / "pre_al"
                if not pre.exists():
                    shutil.copytree(root / "models" / a / "model", pre)
        if finetune and any(no_ds.values()):
            miss = ", ".join(a for a in active_axles if no_ds[a])
            say(
                f"dataset not found for {miss} ({root / 'dataset'}): fine-tuning disabled, "
                "the networks stay as trained"
            )
        if finetune and dev_type == "cpu" and al.finetune.on_cpu == "skip":
            say(
                "fine-tuning disabled on CPU (it takes minutes per iteration): the networks stay "
                "as trained; set `active_learning.finetune.on_cpu: run` to force it"
            )
        say(
            f"active learning: {len(active_axles)} axle network(s) on {dev_used}, "
            f"inner method {opt.method} / {opt.mode}, FOM budget {fom_budget}, "
            f"radius {r:.3g}, margin {al.margin:g}"
        )
        try:
            for k in count(k0):
                df = read_history(fom.history_path)
                why = stop_reason(
                    state,
                    iterations_done=k,
                    n_fom=len(df),
                    elapsed=_time.perf_counter() - t_start,
                    al=al,
                    fom_budget=fom_budget,
                    time_budget=time_budget,
                )
                if why is not None:
                    reason = why
                    break
                t_iter = _time.perf_counter()
                best0 = _summary(df)
                x_b = best0["x"]
                front_obj, front_x = _front(df, study) if opt.mode == "pareto" else (None, None)
                if front_obj is not None and front_x is not None and len(front_obj):
                    x_b = front_x[knee_index(front_obj)]
                rng = np.random.default_rng([opt.seed, k])

                # 1. optimise on the networks inside the trust region
                lo, hi = box_bounds(x_b, r)
                it_dir = run_dir / "al" / f"iter_{k}"
                it_dir.mkdir(parents=True, exist_ok=True)
                t0 = _time.perf_counter()
                with SurrogateProblem(
                    study_path,
                    root / "models",
                    it_dir,
                    device=dev_name,
                    margin=al.margin,
                    evaluations=al.surrogate_evaluations,
                    predictors=predictors,
                ) as sp:
                    sp.max_time = (
                        None
                        if time_budget is None
                        else max(time_budget - (_time.perf_counter() - t_start), 1.0)
                    )
                    box = BoxProblem(sp, lo, hi, x_b)
                    _run_inner(box, opt.model_copy(update={"seed": opt.seed + k}))
                    sdf = read_history(sp.history_path)
                    if len(sdf) == 0:
                        reason = "no_candidates"
                        break
                    # 2. infill set (x_star first), deduplicated against the FOM history
                    known = fom.cache.keys()
                    x_star, X = select_infill(
                        sdf,
                        study,
                        opt.mode,
                        top=al.infill.top,
                        perturb=al.infill.perturb,
                        lo=lo,
                        hi=hi,
                        r=r,
                        rng=rng,
                        known=known,
                        quantise=space.quantise_x,
                    )
                    if all(cache_key(x) in known for x in X):
                        reason = "no_candidates"  # no new design on the grid
                        break
                    ent = sp.cache[cache_key(x_star)]
                    f_nn, feas_nn = float(ent.f_total), bool(ent.feasible)
                    preds = cast(SurrogateProblem, sp).evaluator.predictors
                    for a in active_axles:
                        if tuple(preds[a].channels) != tuple(c.name for c in channels_for(a)):
                            raise DatasetError(f"{a}: model channels differ from the registry")
                    pred_y = {
                        a: preds[a].predict(x_star[asps[a].idx][None, :])[0][0]
                        for a in active_axles
                    }
                shutil.move(str(it_dir / "history.csv"), str(it_dir / "surrogate_history.csv"))
                (it_dir / "history_z.csv").unlink(missing_ok=True)
                t_sur = _time.perf_counter() - t0

                # 3. FOM verification (x_star first); respect the FOM budget
                is_new = [cache_key(x) not in fom.cache for x in X]
                room = max(fom_budget - len(df), 0)
                keep: list[int] = []
                for i in range(len(X)):
                    if is_new[i]:
                        if room <= 0:
                            continue
                        room -= 1
                    keep.append(i)
                if 0 not in keep:
                    reason = "fom_budget"
                    break
                X = X[keep]
                is_new = [is_new[i] for i in keep]
                t0 = _time.perf_counter()
                fo, _, ft, mg, fe = fom.evaluate_batch(X, f"al_{k}")
                t_fom = _time.perf_counter() - t0
                f_fom, feas_fom, mg_fom = float(ft[0]), bool(fe[0]), float(mg[0])

                # 4. FOM curves, residuals and dataset growth
                t0 = _time.perf_counter()
                resid: dict[str, float] = {}
                new_idx = [i for i, nw in enumerate(is_new) if nw]
                all_samples: dict[str, dict[int, CornerSample]] = {}
                for a in active_axles:  # extract every axle first: the appends below go together
                    asp, grid = asps[a], grids[a]
                    all_samples[a] = {
                        i: _axle_sample(space, asp, grid, X[i][asp.idx])
                        for i in sorted({0, *new_idx})
                    }
                    s0 = all_samples[a][0]
                    resid[a] = curve_residual(pred_y[a], s0.Y, s0.mask, s0.channels)
                if new_idx and finetune:
                    for a in (a for a in active_axles if not no_ds[a]):
                        shard = append_rows(
                            root,
                            a,
                            X[new_idx][:, asps[a].idx],
                            [all_samples[a][i] for i in new_idx],
                            f"al_{k}",
                            weight=al.finetune.weight,
                        )
                        if a in ft_caches:  # keep the device tensors in step with the dataset
                            ft_caches[a].add_rows(_shard_rows(shard, ft_caches[a].template))
                t_ext = _time.perf_counter() - t0

                # 5. fine-tune on the grown datasets
                t0 = _time.perf_counter()
                val: dict[str, float] = {}
                kept: dict[str, bool] = {}
                skipped: dict[str, bool] = {}
                skip_why: dict[str, str] = {}
                if new_idx and finetune:
                    for a in active_axles:
                        if no_ds[a]:
                            skipped[a] = True
                            skip_why[a] = "no_dataset"
                            continue
                        run_ft, why = finetune_decision(
                            resid[a], ft_state[a], al.finetune, device_type=dev_type
                        )
                        if not run_ft:
                            skipped[a] = True
                            skip_why[a] = why
                            if why != "cpu":
                                ft_state[a] = advance_finetune(
                                    ft_state[a], skipped=True, kept=False, ft=al.finetune
                                )
                                say(
                                    f"  iter {k}: {a} fine-tuning skipped ({why}, "
                                    f"residual {resid[a]:.2f})"
                                )
                            continue
                        val[a], kept[a] = _finetune(
                            root,
                            a,
                            ft=al.finetune,
                            lr_factor=spec.training.finetune_lr_factor,
                            device=dev_name,
                            seed=spec.dataset.seed,
                            k=k,
                            caches=ft_caches,
                        )
                        ft_state[a] = advance_finetune(
                            ft_state[a], skipped=False, kept=kept[a], ft=al.finetune
                        )
                        if kept[a]:
                            say(f"  warning: iter {k}: {a} fine-tuning kept the previous weights")
                t_ft = _time.perf_counter() - t0

                # 6. trust region and bookkeeping
                df2 = read_history(fom.history_path)
                best1 = _summary(df2)
                if front_obj is not None:
                    # Pareto mode: rho 1 if the new x* extends the verified front, else 0
                    rho = 1.0 if is_new[0] and feas_fom and not dominated(fo[0], front_obj) else 0.0
                    improved = any(
                        is_new[i] and bool(fe[i]) and not dominated(fo[i], front_obj)
                        for i in range(len(X))
                    )
                else:
                    rho = trust_ratio(
                        best0["f"],
                        f_nn,
                        f_fom,
                        feasible_fom=feas_fom,
                        feasible_best=best0["feasible"],
                        max_g_fom=mg_fom,
                        max_g_best=best0["max_g"],
                    )
                    improved = is_improved(best0, best1, 0.1 * al.stop.eps_f * f_scale)
                hit = hits_boundary(x_star, x_b, r, slack=0.5 * step_z)
                r_next = update_radius(r, rho, hit, tr)
                agree = (
                    math.isfinite(f_fom)
                    and abs(f_nn - f_fom) / f_scale < al.stop.eps_f
                    and feas_nn == feas_fom
                )
                at_min = r_next <= tr.min * (1.0 + 1e-9)
                state = advance_stop(state, agree=agree, improved=improved, at_min=at_min)
                rows.append(
                    {
                        "k": k,
                        "radius": r,
                        "radius_next": r_next,
                        "f_nn": f_nn,
                        "f_fom": f_fom,
                        "feasible_nn": feas_nn,
                        "feasible_fom": feas_fom,
                        "rho": rho,
                        "best_f": best1["f"],
                        "best_feasible": best1["feasible"],
                        "n_fom": len(df2),
                        "n_new": len(new_idx),
                        "resid_front": resid.get("front", math.nan),
                        "resid_rear": resid.get("rear", math.nan),
                        "val_front": val.get("front", math.nan),
                        "val_rear": val.get("rear", math.nan),
                        "kept_init_front": kept.get("front", False),
                        "kept_init_rear": kept.get("rear", False),
                        "skipped_front": skipped.get("front", False),
                        "skipped_rear": skipped.get("rear", False),
                        "skip_reason_front": skip_why.get("front", ""),
                        "skip_reason_rear": skip_why.get("rear", ""),
                        "agree": agree,
                        "improved": improved,
                        "at_min": at_min,
                        "sec_surrogate": t_sur,
                        "sec_fom": t_fom,
                        "sec_extract": t_ext,
                        "sec_finetune": t_ft,
                        "sec_total": _time.perf_counter() - t_iter,
                    }
                )
                pd.DataFrame(rows, columns=list(AL_COLUMNS)).to_csv(al_path, index=False)
                new_iters += 1
                say(
                    f"  iter {k}: r {r:.3g}, f_nn {f_nn:.4g}, f_fom {f_fom:.4g} "
                    f"({'feasible' if feas_fom else 'infeasible'}), rho {rho:.2f}, best "
                    f"{best1['f']:.4g}, {len(df2)} FOM evals, {t_sur + t_fom + t_ext + t_ft:.1f} s"
                )
                r = r_next
        except BaseException as exc:  # incl. KeyboardInterrupt: still write the run folder
            error = exc
            reason = "error"
        n_new = fom.n_evals
        workers_used = fom.workers
    seconds = _time.perf_counter() - t_start
    ended = datetime.now().astimezone()

    result: RunResult | None = None
    try:
        result = _finish(
            run_dir=run_dir,
            study=study,
            study_path=study_path,
            space=space,
            opt=opt,
            baseline=baseline,
            baseline_x=base_x,
            valid_from=0,
            shash=shash,
            started=started,
            ended=ended,
            seconds=seconds,
            n_new=n_new,
            workers_used=workers_used,
            resume=resume,
            report=report and error is None,
            budget=(fom_budget, time_budget),
        )
    except Exception:
        if error is None:
            raise
    if error is not None:
        raise error
    assert result is not None

    al_df = pd.DataFrame(rows, columns=list(AL_COLUMNS))
    meta = read_meta(run_dir)
    meta["surrogate"] = {
        "root": str(root.resolve()),
        "device": dev_used,
        "models": {a: _sha256(root / "models" / a / "model" / "model.pt") for a in active_axles},
        "models_pre_al": {
            a: _sha256(root / "models" / a / "pre_al" / "model.pt") for a in active_axles
        },
    }
    meta["active_learning"] = {
        "iterations": len(al_df),
        "new_iterations": new_iters,
        "stop_reason": reason,
        "fom_budget": fom_budget,
        "time_budget": time_budget,
        "config": al.model_dump(mode="json"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    write_surrogate_figures(run_dir / "figures", al=al_df)
    return ActiveResult(
        run=result,
        iterations=len(al_df),
        new_iterations=new_iters,
        stop_reason=reason,
        n_fom=result.n_total,
        al_path=al_path,
        surrogate_root=root,
        device=dev_used,
    )
