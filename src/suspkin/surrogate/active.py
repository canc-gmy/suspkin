"""Active-learning optimizer: surrogate search in a trust region, FOM verification, fine-tuning.

Experimental (see ``docs/guides/stability.md``), design in ``docs/DESIGN_PLAN.md`` section 13.9.
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

# ruff: noqa: N803, N806  (X, F, G follow the Problem interface)
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time as _time
from collections.abc import Callable, Container, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

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
from suspkin.optimize.history import CacheEntry, cache_key, read_history
from suspkin.optimize.outputs import (
    _ok_feasible,
    best_row,
    copy_inputs,
    pareto_rows,
    ranked_rows,
    read_meta,
    study_hash,
)
from suspkin.optimize.problem import BudgetExhausted, Problem, resolve_workers
from suspkin.optimize.spec import (
    ActiveLearningSpec,
    Optimizer,
    Study,
    StudyConfigError,
    TrustRegionSpec,
    load_study,
)
from suspkin.plotting.optimization import knee_index
from suspkin.surrogate.channels import CHANNELS, channels_for
from suspkin.surrogate.data_types import SPLIT_TRAIN, SPLIT_VAL, Axle
from suspkin.surrogate.dataset import (
    DatasetError,
    append_rows,
    check_dataset_matches,
    check_model_matches,
    load_dataset,
)
from suspkin.surrogate.evaluate import AxlePredictor
from suspkin.surrogate.extract import CornerSample, extract_corner
from suspkin.surrogate.grid import AxleGrid, axle_grids
from suspkin.surrogate.problem import SurrogateProblem
from suspkin.surrogate.sampling import AxleSpace

F64 = NDArray[np.float64]
Echo = Callable[[str], None]

__all__ = [
    "ActiveResult",
    "BoxProblem",
    "StopState",
    "advance_stop",
    "box_bounds",
    "crowding_distance",
    "curve_residual",
    "hits_boundary",
    "run_active_learning",
    "select_infill",
    "stop_reason",
    "trust_ratio",
    "update_radius",
]

_AXLES: tuple[Axle, ...] = ("front", "rear")
_TOL = {c.name: c.tol for c in CHANNELS}
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
    "agree",
    "improved",
    "at_min",
    "sec_surrogate",
    "sec_fom",
    "sec_extract",
    "sec_finetune",
    "sec_total",
)


# --------------------------------------------------------------------------- pure rules


def box_bounds(center: ArrayLike, r: float) -> tuple[F64, F64]:
    """Trust-region box ``[c - r, c + r]`` intersected with the unit cube."""
    c = np.asarray(center, dtype=np.float64)
    return np.clip(c - r, 0.0, 1.0), np.clip(c + r, 0.0, 1.0)


def hits_boundary(x: ArrayLike, center: ArrayLike, r: float, rtol: float = 1e-3) -> bool:
    """True if ``x`` reaches the trust-region boundary (``|x_i - c_i| = r``) in any coordinate.

    A side clipped by the unit cube is not a trust-region boundary.
    """
    d = np.abs(np.asarray(x, dtype=np.float64) - np.asarray(center, dtype=np.float64))
    return bool(np.any(d >= r * (1.0 - rtol)))


def trust_ratio(
    f_best: float,
    f_nn: float,
    f_fom: float,
    *,
    feasible_fom: bool,
    feasible_best: bool,
    max_g_fom: float = math.inf,
    max_g_best: float = math.inf,
) -> float:
    """Ratio ``rho`` of the actual to the predicted improvement of the infill optimum.

    With a feasible verified best: ``(f_b - f_FOM) / (f_b - f_NN)``; ``nan`` when no
    improvement was predicted (the caller shrinks), ``0`` when the FOM finds the point
    infeasible. Without a feasible verified design the ``f`` comparison is meaningless: a
    feasible FOM result gives 1, a smaller ``max_g`` gives 0.5, anything else 0.
    """
    if not feasible_best:
        if feasible_fom:
            return 1.0
        return 0.5 if max_g_fom < max_g_best else 0.0
    pred = f_best - f_nn
    if not math.isfinite(pred) or pred <= 0.0:
        return math.nan
    if not feasible_fom:
        return 0.0
    return (f_best - f_fom) / pred


def update_radius(r: float, rho: float, hit_boundary: bool, tr: TrustRegionSpec) -> float:
    """Trust-region radius after an iteration.

    ``rho > 0.75`` and the step reached the boundary: double (up to ``max``); ``rho < 0.25``
    or undefined: halve (down to ``min``); otherwise keep.
    """
    if math.isnan(rho) or rho < 0.25:
        return max(r / 2.0, tr.min)
    if rho > 0.75 and hit_boundary:
        return min(2.0 * r, tr.max)
    return r


@dataclass(frozen=True)
class StopState:
    """Consecutive-iteration counters of the stopping rules."""

    agree: int = 0
    stale: int = 0
    at_min: int = 0


def advance_stop(state: StopState, *, agree: bool, improved: bool, at_min: bool) -> StopState:
    """Counters after one iteration (``at_min``: radius at its minimum without improvement)."""
    return StopState(
        agree=state.agree + 1 if agree else 0,
        stale=0 if improved else state.stale + 1,
        at_min=state.at_min + 1 if (at_min and not improved) else 0,
    )


def stop_reason(
    state: StopState,
    *,
    iterations_done: int,
    n_fom: int,
    elapsed: float,
    al: ActiveLearningSpec,
    fom_budget: int | None = None,
    time_budget: float | None = None,
) -> str | None:
    """Name of the first stopping rule that fires, or ``None``.

    Order: ``agreement``, ``stagnation``, ``radius_min``, ``max_iterations``, ``fom_budget``,
    ``time_budget``. Budget overrides replace the ``al`` values.
    """
    p = al.stop.patience
    if state.agree >= p:
        return "agreement"
    if state.stale >= p:
        return "stagnation"
    if state.at_min >= p:
        return "radius_min"
    if iterations_done >= al.max_iterations:
        return "max_iterations"
    if n_fom >= (al.fom_evaluations if fom_budget is None else fom_budget):
        return "fom_budget"
    tb = al.time if time_budget is None else time_budget
    if tb is not None and elapsed >= tb:
        return "time_budget"
    return None


def crowding_distance(objs: ArrayLike) -> F64:
    """NSGA-II crowding distance of the rows of ``objs`` (boundary points get ``inf``)."""
    pts = np.asarray(objs, dtype=np.float64)
    n, m = pts.shape
    out = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(pts[:, j], kind="stable")
        span = pts[order[-1], j] - pts[order[0], j]
        out[order[0]] = out[order[-1]] = np.inf
        if span <= 0.0:
            continue
        out[order[1:-1]] += (pts[order[2:], j] - pts[order[:-2], j]) / span
    return out


def select_infill(
    sur_history: pd.DataFrame,
    study: Study,
    mode: str,
    *,
    top: int,
    perturb: int,
    lo: ArrayLike,
    hi: ArrayLike,
    r: float,
    rng: np.random.Generator,
    known: Container[tuple[float, ...]] = (),
) -> tuple[F64, F64]:
    """Infill set from the surrogate history of an iteration.

    Returns ``(x_star, X)``; ``X[0]`` is ``x_star`` (always kept), then ``top`` further
    candidates and ``perturb`` points uniform in a box of radius ``r / 4`` around ``x_star``
    (clipped to ``[lo, hi]``). Points already in ``known`` and duplicates are dropped (not
    ``x_star``).

    Weighted mode: ``x_star`` is the best surrogate-feasible ``f_total`` (else the least
    infeasible) and the candidates the next best distinct designs. Pareto mode: ``x_star`` is
    the knee of the predicted front and the candidates the points with the largest crowding
    distance; without a feasible front point it falls back to the weighted rule.
    """
    xcols = [c for c in sur_history.columns if c.startswith("x_")]
    xs = sur_history[xcols].to_numpy(dtype=np.float64)
    if len(xs) == 0:
        raise ValueError("empty surrogate history")
    cands: list[F64] = []
    front = pareto_rows(sur_history, study) if mode == "pareto" else np.array([], dtype=np.intp)
    if front.size:
        fcols = [f"f_{o.name}" for o in study.objectives]
        objs = sur_history[fcols].to_numpy(dtype=np.float64)[front]
        knee = knee_index(objs)
        crowd = crowding_distance(objs)
        others = [int(i) for i in np.argsort(-crowd, kind="stable") if int(i) != knee]
        order = [knee, *others]
        cands = [xs[front[i]] for i in order]
    else:
        cands = [xs[i] for i in ranked_rows(sur_history)]
    x_star = cands[0].copy()
    out = [x_star]
    seen = {cache_key(x_star)}
    for x in cands[1:]:
        if len(out) > top:
            break
        k = cache_key(x)
        if k in seen or k in known:
            continue
        seen.add(k)
        out.append(x.copy())
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    for _ in range(perturb):
        x = np.clip(x_star + rng.uniform(-r / 4.0, r / 4.0, size=x_star.size), lo_a, hi_a)
        k = cache_key(x)
        if k in seen or k in known:
            continue
        seen.add(k)
        out.append(x)
    return x_star, np.array(out)


def curve_residual(
    y_pred: NDArray[np.float64],
    y_fom: NDArray[np.float64],
    mask: NDArray[np.bool_],
    channels: tuple[str, ...],
) -> float:
    """Mean over channels of ``RMSE / tol`` on the FOM-valid grid nodes (``nan`` if none).

    ``y_pred`` and ``y_fom`` are ``(C, R, T)`` in physical units, ``mask`` is ``(R, T)``.
    """
    if not mask.any():
        return math.nan
    tol = np.array([_TOL[c] for c in channels], dtype=np.float64)
    d = (np.asarray(y_pred, dtype=np.float64) - np.asarray(y_fom, dtype=np.float64))[:, mask]
    rmse = np.sqrt(np.mean(d**2, axis=1))
    return float(np.mean(rmse / tol))


# --------------------------------------------------------------------------- box adapter


class _SpaceView:
    """The design space as seen by a driver on the box: the baseline is the box centre."""

    def __init__(self, space: DesignSpace, z_center: F64) -> None:
        self._space = space
        self._zc = z_center

    def from_vehicle(self, v: Any = None, *, clip: bool = False) -> F64:
        """The box centre in box coordinates (the driver's warm start)."""
        out: F64 = self._zc.copy()
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._space, name)


class _CacheView:
    """Read-only view of the inner cache keyed by box coordinates (as the drivers look up)."""

    def __init__(self, inner: Problem) -> None:
        self._inner = inner
        self._xkey: dict[tuple[float, ...], tuple[float, ...]] = {}

    def record(self, z: F64, x: F64) -> None:
        for zi, xi in zip(z, x, strict=True):
            self._xkey[cache_key(zi)] = cache_key(xi)

    def __contains__(self, zkey: tuple[float, ...]) -> bool:
        xk = self._xkey.get(zkey)
        return xk is not None and xk in self._inner.cache

    def __getitem__(self, zkey: tuple[float, ...]) -> CacheEntry:
        return self._inner.cache[self._xkey[zkey]]


class BoxProblem:
    """Adapter that lets the unchanged drivers search a box of the unit cube.

    The drivers work in ``z in [0, 1]^n``; the wrapped problem is evaluated at
    ``x = lo + z * (hi - lo)``, so its cache and history keep storing ``x``. The ``history_path``
    seen by the drivers is a copy of the inner history converted to ``z``.
    """

    def __init__(self, inner: Problem, lo: ArrayLike, hi: ArrayLike, center: ArrayLike) -> None:
        self.inner = inner
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self.span = self.hi - self.lo
        if np.any(self.span <= 0.0):
            raise ValueError("box needs hi > lo in every coordinate")
        self.study = inner.study
        self.valid_from = 0
        self.space = _SpaceView(inner.space, self.to_z(center))
        self.cache = _CacheView(inner)
        self._z_rows = -1  # inner.next_id when history_z.csv was last written

    # ---- coordinates
    def to_x(self, z: ArrayLike) -> F64:
        """Box coordinates to design vectors (clipped to the unit cube)."""
        zc = np.clip(np.asarray(z, dtype=np.float64), 0.0, 1.0)
        out: F64 = np.clip(self.lo + zc * self.span, 0.0, 1.0)
        return out

    def to_z(self, x: ArrayLike) -> F64:
        """Design vectors to box coordinates (clipped to ``[0, 1]``)."""
        out: F64 = np.clip((np.asarray(x, dtype=np.float64) - self.lo) / self.span, 0.0, 1.0)
        return out

    # ---- Problem surface used by the drivers
    @property
    def n(self) -> int:
        """Number of design parameters."""
        return self.inner.n

    @property
    def max_evals(self) -> int | None:
        """Evaluation budget of the wrapped problem."""
        return self.inner.max_evals

    @max_evals.setter
    def max_evals(self, v: int | None) -> None:
        self.inner.max_evals = v

    @property
    def max_time(self) -> float | None:
        """Wall-time budget of the wrapped problem (seconds)."""
        return self.inner.max_time

    @max_time.setter
    def max_time(self, v: float | None) -> None:
        self.inner.max_time = v

    @property
    def n_evals(self) -> int:
        """New evaluations of the wrapped problem."""
        return self.inner.n_evals

    @property
    def elapsed(self) -> float:
        """Wall seconds since the wrapped problem was created."""
        return self.inner.elapsed

    @property
    def workers(self) -> int:
        """Worker processes of the wrapped problem."""
        return self.inner.workers

    def exhausted(self) -> bool:
        """True if the budget of the wrapped problem is used up."""
        return self.inner.exhausted()

    @property
    def history_path(self) -> Path:
        """Inner history with the ``x_*`` columns in box coordinates.

        ``history_z.csv`` is rewritten only when the inner history has grown since the last
        access (the drivers read it a few times per run, not per batch).
        """
        out = self.inner.history_path.with_name("history_z.csv")
        if self._z_rows != self.inner.next_id or not out.is_file():
            df = read_history(self.inner.history_path)
            cols = [f"x_{n}" for n in self.inner.space.names]
            if len(df):
                df[cols] = self.to_z(df[cols].to_numpy(dtype=np.float64))
            df.to_csv(out, index=False)
            self._z_rows = self.inner.next_id
        return out

    def evaluate_batch(
        self, X: ArrayLike, source: str = "eval"
    ) -> tuple[F64, F64, F64, F64, NDArray[np.bool_]]:
        """:meth:`Problem.evaluate_batch` on box coordinates."""
        zm = np.atleast_2d(np.asarray(X, dtype=np.float64))
        xm = self.to_x(zm)
        self.cache.record(zm, xm)
        return self.inner.evaluate_batch(xm, source)

    def evaluate_one(
        self, x: ArrayLike, source: str = "eval"
    ) -> tuple[F64, F64, float, float, bool]:
        """Single-point wrapper of :meth:`evaluate_batch`."""
        F, G, ft, mg, fe = self.evaluate_batch(np.asarray(x, dtype=np.float64)[None, :], source)
        return F[0], G[0], float(ft[0]), float(mg[0]), bool(fe[0])

    def fd_jacobian(
        self, x: ArrayLike, source: str = "fd", step: float | None = None
    ) -> tuple[F64, F64, F64, tuple[F64, F64, float]]:
        """Forward-difference Jacobians in box coordinates (as :meth:`Problem.fd_jacobian`)."""
        return Problem.fd_jacobian(cast(Problem, self), x, source, step)


# --------------------------------------------------------------------------- helpers


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


def _improved(old: dict[str, Any], new: dict[str, Any], min_gain: float) -> bool:
    if new["feasible"] and not old["feasible"]:
        return True
    if new["feasible"] and old["feasible"]:
        return bool(old["f"] - new["f"] > min_gain)
    if not new["feasible"] and not old["feasible"]:
        return bool(old["max_g"] - new["max_g"] > 1e-9)
    return False


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _front(df: pd.DataFrame, study: Study) -> tuple[F64, F64]:
    """Objective values and design vectors of the feasible non-dominated rows of a history."""
    idx = pareto_rows(df, study)
    ocols = [f"f_{o.name}" for o in study.objectives]
    xcols = [c for c in df.columns if c.startswith("x_")]
    return (
        df[ocols].to_numpy(dtype=np.float64)[idx],
        df[xcols].to_numpy(dtype=np.float64)[idx],
    )


def _dominated(obj: ArrayLike, front: F64) -> bool:
    """True if some row of ``front`` is <= ``obj`` everywhere and < somewhere."""
    o = np.asarray(obj, dtype=np.float64)
    if len(front) == 0:
        return False
    return bool(np.any(np.all(front <= o, axis=1) & np.any(front < o, axis=1)))


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


def _finetune(
    root: Path,
    axle: Axle,
    *,
    epochs: int,
    replay: int,
    lr_factor: float,
    device: str,
    seed: int,
    k: int,
) -> tuple[float, bool]:
    """Fine-tune the current model of ``axle``; returns ``(score, kept_init_weights)``.

    Trains on all active-learning rows plus ``replay`` random earlier train rows (drawn
    afresh from ``seed`` and ``k``); the validation split is unchanged. The early-stopping
    score is the mean of the validation score and the score on all active-learning rows.
    """
    from dataclasses import replace

    from suspkin.surrogate.train import TrainedModel, train

    mdir = root / "models" / axle / "model"
    pre = root / "models" / axle / "pre_al"
    if not pre.exists():
        shutil.copytree(mdir, pre)
    model = TrainedModel.load(mdir, device=device)
    full = load_dataset(root, axle)
    is_al = np.char.startswith(full.source.astype(str), "al_")
    pool = np.flatnonzero((full.split == SPLIT_TRAIN) & ~is_al)
    rng = np.random.default_rng([seed, k, 4096])
    pick = rng.choice(pool, size=min(replay, pool.size), replace=False) if pool.size else pool
    keep = (full.split == SPLIT_VAL) | is_al
    keep[pick] = True
    extra = full.subset(is_al) if is_al.any() else None
    ds = full.subset(keep)
    tcfg = replace(
        model.tcfg,
        max_epochs=epochs,
        device=cast(Any, device),
        finetune_lr_factor=lr_factor,
        seed=seed + k,
    )
    new = train(ds, model.mcfg, tcfg, init=model, extra_val=extra)
    new.meta["al_iteration"] = k
    new.save(root / "models" / axle / f"al_{k}")
    new.save(mdir)
    return float(new.val_score), bool(new.meta.get("kept_init_weights", False))


# --------------------------------------------------------------------------- driver


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

    Raises
    ------
    StudyConfigError, RunError, DatasetError
        Invalid study or arguments, missing or mismatching dataset / models, changed study
        on resume.
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
    if predictors is None:
        from suspkin.surrogate.train import resolve_device

        dev_used = str(resolve_device(dev_name))
    full_space = DesignSpace(study, vehicle)
    asps = {a: AxleSpace(full_space, a) for a in _AXLES}
    active_axles = [a for a in _AXLES if asps[a].n > 0]
    if not active_axles:
        raise StudyConfigError(f"{study_path}: the study has no design variable")
    for ax in active_axles if predictors is None else []:
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
    state = StopState()
    for rw in rows:
        state = advance_stop(
            state,
            agree=bool(rw["agree"]),
            improved=bool(rw["improved"]),
            at_min=bool(rw["at_min"]),
        )
    k0 = int(rows[-1]["k"]) + 1 if rows else 0
    r = float(rows[-1]["radius_next"]) if rows else al.trust_region.initial

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
            for a in active_axles:
                n_rec = _recover_dataset(
                    root, a, asps[a], grids[a], space, hist0, al.finetune.weight
                )
                if n_rec:
                    say(f"resume: appended {n_rec} missing active-learning row(s), {a} dataset")
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
                    )
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
                    for a in active_axles:
                        append_rows(
                            root,
                            a,
                            X[new_idx][:, asps[a].idx],
                            [all_samples[a][i] for i in new_idx],
                            f"al_{k}",
                            weight=al.finetune.weight,
                        )
                t_ext = _time.perf_counter() - t0

                # 5. fine-tune on the grown datasets
                t0 = _time.perf_counter()
                val: dict[str, float] = {}
                kept: dict[str, bool] = {}
                if new_idx and finetune:
                    for a in active_axles:
                        val[a], kept[a] = _finetune(
                            root,
                            a,
                            epochs=al.finetune.epochs,
                            replay=al.finetune.replay,
                            lr_factor=spec.training.finetune_lr_factor,
                            device=dev_name,
                            seed=spec.dataset.seed,
                            k=k,
                        )
                        if kept[a]:
                            say(f"  warning: iter {k}: {a} fine-tuning kept the previous weights")
                t_ft = _time.perf_counter() - t0

                # 6. trust region and bookkeeping
                df2 = read_history(fom.history_path)
                best1 = _summary(df2)
                if front_obj is not None:
                    # Pareto mode: rho 1 if the new x* extends the verified front, else 0
                    rho = (
                        1.0 if is_new[0] and feas_fom and not _dominated(fo[0], front_obj) else 0.0
                    )
                    improved = any(
                        is_new[i] and bool(fe[i]) and not _dominated(fo[i], front_obj)
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
                    improved = _improved(best0, best1, 0.1 * al.stop.eps_f * f_scale)
                r_next = update_radius(r, rho, hits_boundary(x_star, x_b, r), al.trust_region)
                agree = (
                    math.isfinite(f_fom)
                    and abs(f_nn - f_fom) / f_scale < al.stop.eps_f
                    and feas_nn == feas_fom
                )
                at_min = r_next <= al.trust_region.min * (1.0 + 1e-9)
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
    _write_figures(run_dir, al_df)
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


def _write_figures(run_dir: Path, al: pd.DataFrame) -> None:
    if len(al) == 0:
        return
    from suspkin.plotting import save_figure
    from suspkin.plotting.surrogate import plot_al_progress, plot_al_radius, plot_al_residual

    fdir = run_dir / "figures"
    save_figure(plot_al_progress(al), fdir / "al_progress", ("png",))
    save_figure(plot_al_residual(al), fdir / "al_residual", ("png",))
    save_figure(plot_al_radius(al), fdir / "al_radius", ("png",))
