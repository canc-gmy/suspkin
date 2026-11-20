"""Active-learning building blocks: trust-region, stopping, infill and fine-tuning rules.

Experimental (see ``docs/guides/stability.md``). The numerical rules (box, radius update,
stopping, infill selection, fine-tune schedule) are small pure functions; ``BoxProblem``
restricts a surrogate problem to the trust-region box. The driver, history helpers and the
network fine-tuning are in :mod:`suspkin.surrogate.active`.
"""

# ruff: noqa: N803, N806  (X, F, G follow the Problem interface)
from __future__ import annotations

import math
from collections.abc import Callable, Container
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from suspkin.optimize.design import DesignSpace
from suspkin.optimize.history import CacheEntry, cache_key, cache_keys, read_history
from suspkin.optimize.outputs import (
    pareto_rows,
    ranked_rows,
)
from suspkin.optimize.problem import Problem
from suspkin.optimize.spec import Study
from suspkin.optimize.spec.surrogate import ActiveLearningSpec, FinetuneSpec, TrustRegionSpec
from suspkin.plotting.optimization import knee_index
from suspkin.surrogate.channels import CHANNELS
from suspkin.surrogate.data import (
    Axle,
)

F64 = NDArray[np.float64]


# --------------------------------------------------------------------------------------
# pure rules
# --------------------------------------------------------------------------------------


_TOL = {c.name: c.tol for c in CHANNELS}


def box_bounds(center: ArrayLike, r: float) -> tuple[F64, F64]:
    """Trust-region box ``[c - r, c + r]`` intersected with the unit cube."""
    c = np.asarray(center, dtype=np.float64)
    return np.clip(c - r, 0.0, 1.0), np.clip(c + r, 0.0, 1.0)


def hits_boundary(
    x: ArrayLike,
    center: ArrayLike,
    r: float,
    rtol: float = 1e-3,
    slack: ArrayLike = 0.0,
) -> bool:
    """True if ``x`` reaches the trust-region boundary (``|x_i - c_i| = r``) in any coordinate.

    A side clipped by the unit cube is not a trust-region boundary. ``slack`` (scalar or per
    coordinate) shortens the radius, e.g. by half a grid step when ``x`` was snapped to a grid.
    """
    d = np.abs(np.asarray(x, dtype=np.float64) - np.asarray(center, dtype=np.float64))
    return bool(np.any(d >= (r - np.asarray(slack, dtype=np.float64)) * (1.0 - rtol)))


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
    quantise: Callable[[F64], F64] | None = None,
) -> tuple[F64, F64]:
    """Infill set from the surrogate history of an iteration.

    Returns ``(x_star, X)``; ``X[0]`` is ``x_star`` (always kept), then ``top`` further
    candidates and ``perturb`` points uniform in a box of radius ``r / 4`` around ``x_star``
    (clipped to ``[lo, hi]``). With ``quantise`` (``DesignSpace.quantise_x``) every point is
    snapped to the hardpoint grid first, so points that round to the same design, or to one
    in ``known``, are dropped. Points already in ``known`` and duplicates are dropped (not
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
    q: Callable[[F64], F64] = quantise if quantise is not None else (lambda a: a)
    x_star = q(cands[0].copy())
    out = [x_star]
    seen = {cache_key(x_star)}
    for x in cands[1:]:
        if len(out) > top:
            break
        x = q(x.copy())
        k = cache_key(x)
        if k in seen or k in known:
            continue
        seen.add(k)
        out.append(x)
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    for _ in range(perturb):
        x = q(np.clip(x_star + rng.uniform(-r / 4.0, r / 4.0, size=x_star.size), lo_a, hi_a))
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


# --------------------------------------------------------------------------------------
# box adapter
# --------------------------------------------------------------------------------------


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
        self._xkey.update(zip(cache_keys(z), cache_keys(x), strict=True))

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

    def _fd_min_step(self) -> F64:
        """One grid step in box coordinates (the drivers' finite-difference floor)."""
        out: F64 = self.inner.space.grid_step_z() / self.span
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


# --------------------------------------------------------------------------------------
# history helpers
# --------------------------------------------------------------------------------------


def is_improved(old: dict[str, Any], new: dict[str, Any], min_gain: float) -> bool:
    if new["feasible"] and not old["feasible"]:
        return True
    if new["feasible"] and old["feasible"]:
        return bool(old["f"] - new["f"] > min_gain)
    if not new["feasible"] and not old["feasible"]:
        return bool(old["max_g"] - new["max_g"] > 1e-9)
    return False


def dominated(obj: ArrayLike, front: F64) -> bool:
    """True if some row of ``front`` is <= ``obj`` everywhere and < somewhere."""
    o = np.asarray(obj, dtype=np.float64)
    if len(front) == 0:
        return False
    return bool(np.any(np.all(front <= o, axis=1) & np.any(front < o, axis=1)))


# --------------------------------------------------------------------------------------
# fine-tuning
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FinetuneState:
    """Per-axle fine-tuning history: consecutive kept-init fine-tunes and remaining skips."""

    kept_run: int = 0
    skip_left: int = 0


def finetune_decision(
    resid: float, state: FinetuneState, ft: FinetuneSpec, *, device_type: str = "cuda"
) -> tuple[bool, str]:
    """Whether to fine-tune an axle this iteration: ``(run, reason)``.

    ``reason`` is ``"train"``, ``"cpu"`` (networks on the CPU and ``ft.on_cpu == "skip"``,
    whatever the residual), ``"trigger"`` (residual of the new point within ``ft.trigger``,
    tolerance units) or ``"stall"`` (skip window after repeated kept-init fine-tunes, unless
    the residual exceeds ``ft.stall_override * ft.trigger``). A non-finite residual trains.
    """
    if device_type == "cpu" and ft.on_cpu == "skip":
        return False, "cpu"
    if not math.isfinite(resid):
        return True, "train"
    if resid <= ft.trigger:
        return False, "trigger"
    if state.skip_left > 0 and resid <= ft.stall_override * ft.trigger:
        return False, "stall"
    return True, "train"


def advance_finetune(
    state: FinetuneState, *, skipped: bool, kept: bool, ft: FinetuneSpec
) -> FinetuneState:
    """Next state after an iteration.

    A skip uses up one skip iteration. A fine-tune that kept the init weights counts towards
    ``ft.skip_after`` (then ``ft.skip_for`` skips start); an improving one resets the count.
    """
    if skipped:
        return FinetuneState(state.kept_run, max(state.skip_left - 1, 0))
    run = state.kept_run + 1 if kept else 0
    if run >= ft.skip_after and ft.skip_for > 0:
        return FinetuneState(0, ft.skip_for)
    return FinetuneState(run, 0)


def replay_finetune_state(
    rows: list[dict[str, Any]], axle: Axle, ft: FinetuneSpec
) -> FinetuneState:
    """Rebuild the fine-tuning state of ``axle`` from the rows of an earlier al.csv."""
    st = FinetuneState()
    for rw in rows:
        skipped = bool(rw.get(f"skipped_{axle}", False))
        if skipped and rw.get(f"skip_reason_{axle}") in ("cpu", "no_dataset"):
            continue  # a CPU / no-dataset skip leaves the schedule untouched
        val = rw.get(f"val_{axle}", math.nan)
        trained = isinstance(val, int | float) and math.isfinite(val)
        if skipped or trained:
            st = advance_finetune(
                st, skipped=skipped, kept=bool(rw.get(f"kept_init_{axle}", False)), ft=ft
            )
    return st
