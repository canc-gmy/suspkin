"""Optimisation problem: cached, budgeted, optionally parallel batch evaluation of designs."""
# ruff: noqa: N803, N806

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin.optimize.design import DesignSpace
from suspkin.optimize.evaluate import Evaluation, evaluate, failed_evaluation
from suspkin.optimize.history import (
    CacheEntry,
    EvalCache,
    HistoryWriter,
    cache_key,
    read_history,
)
from suspkin.optimize.spec import Optimizer, load_study

F64 = NDArray[np.float64]
FD_STEP = 1e-3

__all__ = ["BudgetExhausted", "Problem", "resolve_workers"]


class BudgetExhausted(RuntimeError):  # noqa: N818
    """The evaluation or wall-time budget is used up."""


def _now() -> float:
    return time.perf_counter()


def resolve_workers(workers: int | str) -> int:
    """``auto`` -> ``max(1, cpu_count - 1)``; an int is used as given."""
    if workers == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return int(workers)


# --------------------------------------------------------------------------------------
# worker side (top-level, picklable; one study load per process)
# --------------------------------------------------------------------------------------

_WORKER_SPACE: DesignSpace | None = None


def _init_worker(study_path: str) -> None:
    global _WORKER_SPACE
    study, vehicle = load_study(study_path)
    _WORKER_SPACE = DesignSpace(study, vehicle)


def _eval_with(space: DesignSpace, x: F64) -> tuple[Evaluation, float]:
    t0 = _now()
    try:
        ev = evaluate(space, x)
    except Exception as exc:  # a crashing design must not abort the run
        ev = failed_evaluation(space, f"{type(exc).__name__}: {exc}")
    return ev, _now() - t0


def _eval_worker(x: F64) -> tuple[Evaluation, float]:
    """Evaluate ``x`` in a worker process; returns the (picklable) Evaluation and seconds."""
    if _WORKER_SPACE is None:
        raise RuntimeError("worker not initialised")
    return _eval_with(_WORKER_SPACE, x)


# --------------------------------------------------------------------------------------


class Problem:
    """Evaluate batches of designs with caching, history logging and budget accounting.

    Parameters
    ----------
    study_path
        Study YAML.
    run_dir
        Run folder; the history goes to ``run_dir/history.csv``.
    resume
        Load an existing history into the cache and continue ``eval_id`` numbering.
        The budget counts only new evaluations.
    workers, evaluations, time
        Overrides of the study's ``optimizer.workers`` / ``budget.evaluations`` /
        ``budget.time`` (seconds).
    """

    def __init__(
        self,
        study_path: str | Path,
        run_dir: str | Path,
        *,
        resume: bool = False,
        workers: int | str | None = None,
        evaluations: int | None = None,
        time: float | None = None,
    ) -> None:
        self.study_path = str(Path(study_path))
        study, vehicle = load_study(self.study_path)
        self.study = study
        self.space = DesignSpace(study, vehicle)
        opt = study.optimizer or Optimizer()
        self.workers = resolve_workers(opt.workers if workers is None else workers)
        self.max_evals = evaluations if evaluations is not None else opt.budget.evaluations
        self.max_time = time if time is not None else opt.budget.time
        self.run_dir = Path(run_dir)
        self.history_path = self.run_dir / "history.csv"
        self.cache = EvalCache()
        self._next_id = 0
        self.valid_from = 0  # history rows with a smaller eval_id are stale (changed study)
        if resume and self.history_path.exists() and self.history_path.stat().st_size > 0:
            df = read_history(self.history_path)
            self.cache = EvalCache.from_history(df, self.space, study)
            if len(df):
                self._next_id = int(df["eval_id"].max()) + 1
        self._writer = HistoryWriter(self.history_path, self.space, study, resume=resume)
        self.n_evals = 0
        self._t0 = _now()
        self._pool: ProcessPoolExecutor | None = None

    # ---- bookkeeping --------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Wall seconds since the problem was created."""
        return _now() - self._t0

    @property
    def n(self) -> int:
        """Number of design parameters."""
        return self.space.n

    @property
    def next_id(self) -> int:
        """The ``eval_id`` the next new evaluation gets."""
        return self._next_id

    def exhausted(self) -> bool:
        """True if the evaluation or time budget is used up."""
        if self.max_evals is not None and self.n_evals >= self.max_evals:
            return True
        return self.max_time is not None and self.elapsed >= self.max_time

    def close(self) -> None:
        """Shut the pool down and close the history file."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._writer.close()

    def __enter__(self) -> Problem:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- evaluation ---------------------------------------------------------------

    def _run(self, xs: list[F64]) -> list[tuple[Evaluation, float]]:
        if self.workers == 1 or not xs:
            return [_eval_with(self.space, x) for x in xs]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.study_path,),
            )
        return list(self._pool.map(_eval_worker, xs, chunksize=1))

    def evaluate_batch(
        self, X: ArrayLike, source: str = "eval"
    ) -> tuple[F64, F64, F64, F64, NDArray[np.bool_]]:
        """Evaluate rows of ``X``; returns ``(F, G, f_total, max_g, feasible)`` arrays.

        Cached points are free; duplicates inside the batch are evaluated once. New points
        are appended to the history in input order. Raises :class:`BudgetExhausted` if some
        points need evaluating and the budget is already used up when the batch starts (a
        fully cached batch never raises); a batch crossing the limit completes.
        """
        xm = np.atleast_2d(np.asarray(X, dtype=np.float64))
        keys = [cache_key(x) for x in xm]
        misses: list[int] = []
        seen: set[tuple[float, ...]] = set()
        for i, k in enumerate(keys):
            if k not in self.cache and k not in seen:
                seen.add(k)
                misses.append(i)
        if misses and self.exhausted():
            raise BudgetExhausted(
                f"budget exhausted ({self.n_evals} evaluations, {self.elapsed:.1f} s)"
            )
        outs = self._run([xm[i] for i in misses])
        for i, (ev, secs) in zip(misses, outs, strict=True):
            self._writer.append(self._next_id, xm[i], ev, secs, source)
            self._next_id += 1
            self.n_evals += 1
            self.cache[keys[i]] = CacheEntry(
                ev.f_vector(), ev.g_vector(), ev.f_total, ev.max_g, ev.feasible
            )
        ents = [self.cache[k] for k in keys]
        nf = len(self.study.objectives)
        ng = sum(1 for c in self.study.constraints if c.enabled)
        F = np.array([e.f_vector for e in ents], dtype=np.float64).reshape(len(ents), nf)
        G = np.array([e.g_vector for e in ents], dtype=np.float64).reshape(len(ents), ng)
        ft = np.array([e.f_total for e in ents], dtype=np.float64)
        mg = np.array([e.max_g for e in ents], dtype=np.float64)
        fe = np.array([e.feasible for e in ents], dtype=bool)
        return F, G, ft, mg, fe

    def evaluate_one(
        self, x: ArrayLike, source: str = "eval"
    ) -> tuple[F64, F64, float, float, bool]:
        """Single-point wrapper of :meth:`evaluate_batch`."""
        F, G, ft, mg, fe = self.evaluate_batch(np.asarray(x, dtype=np.float64)[None, :], source)
        return F[0], G[0], float(ft[0]), float(mg[0]), bool(fe[0])

    def fd_jacobian(
        self, x: ArrayLike, source: str = "fd", step: float | None = None
    ) -> tuple[F64, F64, F64, tuple[F64, F64, float]]:
        """Forward-difference Jacobians in the unit cube (backward step where ``x + h > 1``).

        The base point and the ``n`` perturbed points form one parallel batch.

        Returns
        -------
        f_total_grad (n,), G_jac (m, n), F_jac (k, n), (F0, G0, f_total0)
        """
        h = FD_STEP if step is None else float(step)
        x0 = np.asarray(x, dtype=np.float64)
        n = x0.size
        pts = [x0]
        hs = np.empty(n)
        for i in range(n):
            hi = h if x0[i] + h <= 1.0 else -h
            xi = x0.copy()
            xi[i] += hi
            pts.append(xi)
            hs[i] = hi
        F, G, ft, _, _ = self.evaluate_batch(np.array(pts), source)
        grad = (ft[1:] - ft[0]) / hs
        g_jac = (G[1:] - G[0]).T / hs
        f_jac = (F[1:] - F[0]).T / hs
        return grad, g_jac, f_jac, (F[0], G[0], float(ft[0]))
