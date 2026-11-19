"""Optimizer drivers: NSGA-II (pymoo), DE and SLSQP / trust-constr (scipy), DE + SLSQP.

Every method works in the unit cube ``[0, 1]^n`` through a :class:`Problem` (cache, history,
budget). A method that runs into :class:`BudgetExhausted` just stops; the result is always
taken from the history (best feasible ``f_total``, else least ``max_g``).

``start`` (``baseline`` | ``best``) applies only to the ``slsqp`` and ``trust-constr`` methods;
``de+slsqp`` always polishes from the best point in the history.

Caution: the finite-difference gradients are unreliable near the failure cliff (``F_FAIL`` /
``G_FAIL`` plateaus around unsolved or infeasible designs), so SLSQP / trust-constr work best
as a polish from a feasible start.
"""
# ruff: noqa: N803, N806

from __future__ import annotations

import contextlib
import math
import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, NonlinearConstraint, differential_evolution, minimize
from scipy.stats import qmc

from suspkin.optimize.design import DesignSpace
from suspkin.optimize.evaluate import Evaluation, evaluate
from suspkin.optimize.history import CacheEntry, EvalCache, cache_key, cache_keys, read_history
from suspkin.optimize.outputs import (
    best_row,
    copy_inputs,
    make_meta,
    pareto_rows,
    ranked_rows,
    read_meta,
    study_hash,
    write_outputs,
)
from suspkin.optimize.problem import BudgetExhausted, Problem
from suspkin.optimize.spec import Optimizer, Study, StudyConfigError, load_study

F64 = NDArray[np.float64]
Echo = Callable[[str], None]

__all__ = ["RunResult", "run_optimization"]

DE_SHARE = 0.8
_INF = float("inf")


class RunError(ValueError):
    """The run cannot start (bad arguments, missing history)."""


@dataclass
class RunResult:
    """Outcome of :func:`run_optimization`."""

    out_dir: Path
    study: Study
    method: str
    mode: str
    workers: int
    n_evals: int  # new evaluations of this run
    n_total: int  # rows in history.csv
    seconds: float
    resumed: bool
    best_x: F64
    best_eval_id: int | None
    feasible: bool
    f_total: float
    baseline: Evaluation
    best: Evaluation
    n_pareto: int
    report_path: Path | None
    budget_evaluations: int | None
    budget_time: float | None


# --------------------------------------------------------------------------- helpers


def _history(problem: Problem) -> pd.DataFrame:
    """History rows valid for this run (stale rows of a changed study are dropped)."""
    df = read_history(problem.history_path)
    if problem.valid_from > 0:
        df = df[df["eval_id"] >= problem.valid_from].reset_index(drop=True)
    return df


def _x_of(df: pd.DataFrame, space: DesignSpace) -> F64:
    return df[[f"x_{n}" for n in space.names]].to_numpy(dtype=np.float64)


def _best_x(problem: Problem, fallback: F64) -> F64:
    df = _history(problem)
    if len(df) == 0:
        return fallback
    row = best_row(df)
    out: F64 = _x_of(df, problem.space)[row]
    return out


def _top_rows(problem: Problem, k: int) -> F64:
    """Best ``k`` history rows (feasible first, by ``f_total``) as an ``(m, n)`` array."""
    df = _history(problem)
    if len(df) == 0:
        return np.empty((0, problem.n))
    xs = _x_of(df, problem.space)[ranked_rows(df)]
    _, first = np.unique(np.round(xs, 12), axis=0, return_index=True)
    xs = xs[np.sort(first)]
    out: F64 = xs[:k]
    return out


def _initial_population(problem: Problem, size: int, seed: int, resume: bool) -> F64:
    """Warm start on resume (history), else Latin hypercube with the baseline as first row."""
    pop = np.empty((0, problem.n))
    if resume:
        pop = _top_rows(problem, size)
        if len(pop) >= size:
            return pop
    lhs = qmc.LatinHypercube(d=problem.n, seed=seed).random(size)
    if len(pop) == 0:
        lhs[0] = problem.space.from_vehicle(clip=True)
        return np.asarray(lhs, dtype=np.float64)
    out: F64 = np.vstack([pop, lhs[: size - len(pop)]])
    return out


def _clip(x: ArrayLike) -> F64:
    out: F64 = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return out


def _lookup(problem: Problem, X: F64, source: str) -> list[CacheEntry]:
    """Cache entries of rows of ``X`` (evaluating the rows that are missing)."""
    keys = cache_keys(X)
    if any(k not in problem.cache for k in keys):
        problem.evaluate_batch(X, source)
    return [problem.cache[k] for k in keys]


# --------------------------------------------------------------------------- NSGA-II


def _run_nsga2(problem: Problem, opt: Optimizer, resume: bool) -> None:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination

    n = problem.n
    nf = len(problem.study.objectives)
    ng = sum(1 for c in problem.study.constraints if c.enabled)
    pareto = opt.mode == "pareto"
    pop_size = int(opt.options.get("pop_size", max(20, 4 * n)))

    class _P(PymooProblem):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(
                n_var=n,
                n_obj=nf if pareto else 1,
                n_ieq_constr=ng,
                xl=np.zeros(n),
                xu=np.ones(n),
            )

        def _evaluate(self, X: F64, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            F, G, ft, _, _ = problem.evaluate_batch(_clip(X), "nsga2")
            out["F"] = F if pareto else ft[:, None]
            if ng:
                out["G"] = G

    n_gen = max(1, problem.max_evals // pop_size) if problem.max_evals else 10_000
    algo = NSGA2(
        pop_size=pop_size,
        sampling=_initial_population(problem, pop_size, opt.seed, resume),
        eliminate_duplicates=True,
    )
    with contextlib.suppress(BudgetExhausted):
        pymoo_minimize(
            _P(),
            algo,
            get_termination("n_gen", n_gen),
            seed=opt.seed,
            verbose=False,
        )


# --------------------------------------------------------------------------- DE


def _run_de(
    problem: Problem,
    opt: Optimizer,
    resume: bool,
    *,
    evals: int | None,
    source: str = "de",
) -> None:
    n = problem.n
    o = opt.options
    if "popsize" in o:
        popsize = max(1, int(o["popsize"]))
    elif evals is not None:
        popsize = int(min(15, max(2, math.ceil(evals / 8 / n))))
    else:
        popsize = 15
    pop = max(8, popsize * n)
    maxiter = (
        int(o["maxiter"])
        if "maxiter" in o
        else (max(1, evals // pop - 1) if evals is not None else 1000)
    )

    def fun(XT: F64) -> F64:
        X = _clip(np.asarray(XT).T)
        _, _, ft, _, _ = problem.evaluate_batch(X, source)
        return ft

    def gfun(XT: F64) -> F64:
        # scipy probes the constraint once with a single 1-D point, then with (n, S) blocks
        single = np.ndim(XT) == 1
        X = _clip(np.asarray(XT).reshape(n, -1).T)
        ents = _lookup(problem, X, source)
        g = np.array([e.g_vector for e in ents], dtype=np.float64).T
        return g[:, 0] if single else g

    ng = sum(1 for c in problem.study.constraints if c.enabled)
    kw: dict[str, Any] = {
        k: o[k] for k in ("mutation", "recombination", "strategy", "tol", "atol") if k in o
    }
    with contextlib.suppress(BudgetExhausted):
        differential_evolution(
            fun,
            Bounds(np.zeros(n), np.ones(n)),
            constraints=NonlinearConstraint(gfun, -_INF, 0.0) if ng else (),
            init=_initial_population(problem, pop, opt.seed, resume),
            maxiter=maxiter,
            seed=opt.seed,
            polish=False,
            vectorized=True,
            updating="deferred",
            **kw,
        )


# --------------------------------------------------------------------------- local


class _FdCache:
    """Memoises the value and forward-difference Jacobians of the last point."""

    def __init__(self, problem: Problem, source: str, step: float | None) -> None:
        self.problem = problem
        self.source = source
        self.step = step
        self._key: tuple[float, ...] | None = None
        self._val: tuple[Any, ...] | None = None

    def value(self, x: F64) -> tuple[F64, F64, float, float, bool]:
        return self.problem.evaluate_one(_clip(x), self.source)

    def jac(self, x: F64) -> tuple[F64, F64]:
        xc = _clip(x)
        k = cache_key(xc)
        if k != self._key or self._val is None:
            grad, g_jac, _, _ = self.problem.fd_jacobian(xc, self.source, self.step)
            self._key, self._val = k, (grad, g_jac)
        grad, g_jac = self._val
        return grad, g_jac


def _run_local(
    problem: Problem, opt: Optimizer, method: str, x0: F64, *, source: str | None = None
) -> None:
    o = opt.options
    n = problem.n
    src = source or method
    fd = _FdCache(problem, src, float(o["fd_step"]) if "fd_step" in o else None)
    ng = sum(1 for c in problem.study.constraints if c.enabled)
    maxiter = int(o.get("maxiter", 100))
    x0 = _clip(x0)

    def fun(x: F64) -> float:
        return float(fd.value(x)[2])

    def jac(x: F64) -> F64:
        return fd.jac(x)[0]

    try:
        if method == "slsqp":
            cons: list[dict[str, Any]] = []
            if ng:
                cons.append(
                    {
                        "type": "ineq",
                        "fun": lambda x: -fd.value(x)[1],
                        "jac": lambda x: -fd.jac(x)[1],
                    }
                )
            sl_opts: dict[str, Any] = {"maxiter": maxiter}
            if "ftol" in o:
                sl_opts["ftol"] = float(o["ftol"])
            minimize(
                fun,
                x0,
                jac=jac,
                bounds=Bounds(np.zeros(n), np.ones(n)),
                constraints=cons,
                method="SLSQP",
                options=sl_opts,
            )
        else:
            from scipy.optimize import BFGS

            tc_cons = (
                [
                    NonlinearConstraint(
                        lambda x: fd.value(x)[1],
                        -_INF,
                        0.0,
                        jac=lambda x: fd.jac(x)[1],
                    )
                ]
                if ng
                else []
            )
            tc_opts: dict[str, Any] = {"maxiter": maxiter}
            for key in ("xtol", "gtol"):
                if key in o:
                    tc_opts[key] = float(o[key])
            minimize(
                fun,
                x0,
                jac=jac,
                hess=BFGS(),
                bounds=Bounds(np.zeros(n), np.ones(n)),
                constraints=tc_cons,
                method="trust-constr",
                options=tc_opts,
            )
    except BudgetExhausted:
        pass


def _start_point(problem: Problem, opt: Optimizer) -> F64:
    base = problem.space.from_vehicle(clip=True)
    if opt.start == "best":
        return _best_x(problem, base)
    return base


def _run_de_slsqp(problem: Problem, opt: Optimizer, resume: bool) -> None:
    max_e, max_t = problem.max_evals, problem.max_time
    de_evals: int | None = None
    if max_e is not None:
        de_evals = max(1, int(DE_SHARE * (max_e - problem.n_evals)))
        problem.max_evals = problem.n_evals + de_evals
    if max_t is not None:
        problem.max_time = problem.elapsed + DE_SHARE * (max_t - problem.elapsed)
    try:
        _run_de(problem, opt, resume, evals=de_evals, source="de")
    finally:
        problem.max_evals, problem.max_time = max_e, max_t
    base = problem.space.from_vehicle(clip=True)
    _run_local(problem, opt, "slsqp", _best_x(problem, base), source="slsqp")


# --------------------------------------------------------------------------- progress


def _install_progress(problem: Problem, echo: Echo, every: float = 5.0) -> None:
    """Wrap ``problem.evaluate_batch`` to echo the best feasible f_total every few seconds."""
    inner = problem.evaluate_batch
    state = {"best": _INF, "last": _time.perf_counter()}

    def wrapped(X: ArrayLike, source: str = "eval") -> Any:
        out = inner(X, source)
        _, _, ft, _, fe = out
        if fe.any():
            state["best"] = min(state["best"], float(np.min(ft[fe])))
        now = _time.perf_counter()
        if now - state["last"] >= every:
            state["last"] = now
            best = "-" if math.isinf(state["best"]) else f"{state['best']:.4g}"
            echo(
                f"  {problem.n_evals} evals, {problem.elapsed:.0f} s, best feasible f_total {best}"
            )
        return out

    problem.evaluate_batch = wrapped  # type: ignore[method-assign]


# --------------------------------------------------------------------------- driver


def run_optimization(
    study_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    resume: bool = False,
    workers: int | str | None = None,
    evaluations: int | None = None,
    time: float | None = None,
    method: str | None = None,
    report: bool = True,
    echo: Echo | None = None,
) -> RunResult:
    """Run the optimisation of a study and write the run folder.

    Parameters
    ----------
    study_path
        Study YAML with objectives and an ``optimizer`` block.
    out_dir
        Run folder; default ``results/opt/<study stem>_<YYYYmmdd-HHMMSS>``. ``resume=True``
        needs an explicit folder that contains a ``history.csv``.
    workers, evaluations, time, method
        Overrides of the study's optimizer block (``time`` in seconds).
    report
        Also write the before/after report.
    echo
        Optional line printer for the start line and periodic progress.

    Raises
    ------
    StudyConfigError, RunError
        Invalid study, or invalid arguments (e.g. resume without history, method override
        incompatible with the mode).
    """
    study_path = Path(study_path)
    started = datetime.now().astimezone()
    study, _ = load_study(study_path)
    opt = study.optimizer or Optimizer()
    if not study.objectives:
        raise StudyConfigError(f"{study_path}: the study has no objectives")
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
    run_dir = (
        Path(out_dir)
        if out_dir is not None
        else Path("results") / "opt" / f"{study_path.stem}_{started.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    vehicle_path = study_path.parent / study.vehicle
    copy_inputs(study_path, vehicle_path, run_dir)

    old_meta = read_meta(run_dir) if resume else {}
    t0 = _time.perf_counter()
    error: BaseException | None = None
    with Problem(
        study_path, run_dir, resume=resume, workers=workers, evaluations=evaluations, time=time
    ) as problem:
        space = problem.space
        shash = study_hash(study)
        valid_from = 0
        if resume:
            valid_from = int(old_meta.get("valid_from_eval_id", 0) or 0)
            if old_meta.get("study_hash") not in (None, shash):
                valid_from = problem.next_id
                if echo is not None:
                    echo(
                        "warning: constraints/objectives changed since the last run; "
                        "cached results are dropped and old points are re-evaluated on demand"
                    )
            if valid_from > 0:
                problem.valid_from = valid_from
                hist = read_history(problem.history_path)
                problem.cache = EvalCache.from_history(
                    hist[hist["eval_id"] >= valid_from].reset_index(drop=True), space, study
                )
        if echo is not None:
            ng = sum(1 for c in study.constraints if c.enabled)
            bud = []
            if problem.max_evals is not None:
                bud.append(f"{problem.max_evals} evaluations")
            if problem.max_time is not None:
                bud.append(f"{problem.max_time:.0f} s")
            echo(
                f"optimize: method {opt.method}, mode {opt.mode}, {space.n} variables, "
                f"{ng} constraints, {len(study.objectives)} objectives, "
                f"{problem.workers} workers, budget {' / '.join(bud)}"
            )
            _install_progress(problem, echo)
        baseline_x = space.from_vehicle(clip=True)
        baseline = evaluate(space)
        m = opt.method
        try:
            if m == "nsga2":
                _run_nsga2(problem, opt, resume)
            elif m == "de":
                _run_de(problem, opt, resume, evals=problem.max_evals)
            elif m == "de+slsqp":
                _run_de_slsqp(problem, opt, resume)
            else:
                _run_local(problem, opt, m, _start_point(problem, opt))
        except BudgetExhausted:
            pass
        except BaseException as exc:  # e.g. BrokenProcessPool, KeyboardInterrupt
            error = exc
        n_new = problem.n_evals
        workers_used = problem.workers
        budget_e, budget_t = problem.max_evals, problem.max_time
    seconds = _time.perf_counter() - t0
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
            baseline_x=baseline_x,
            valid_from=valid_from,
            shash=shash,
            started=started,
            ended=ended,
            seconds=seconds,
            n_new=n_new,
            workers_used=workers_used,
            resume=resume,
            report=report and error is None,
            budget=(budget_e, budget_t),
        )
    except Exception:
        if error is None:
            raise
    if error is not None:
        raise error
    assert result is not None
    return result


def _finish(
    *,
    run_dir: Path,
    study: Study,
    study_path: Path,
    space: DesignSpace,
    opt: Optimizer,
    baseline: Evaluation,
    baseline_x: F64,
    valid_from: int,
    shash: str,
    started: datetime,
    ended: datetime,
    seconds: float,
    n_new: int,
    workers_used: int,
    resume: bool,
    report: bool,
    budget: tuple[int | None, float | None],
) -> RunResult:
    """Select the best design from the history and write the run folder."""
    full = read_history(run_dir / "history.csv")
    df = full[full["eval_id"] >= valid_from].reset_index(drop=True) if valid_from else full
    pareto_idx: np.ndarray[Any, np.dtype[np.intp]] | None = None
    best_id: int | None = None
    if len(df):
        if opt.mode == "pareto":
            pareto_idx = pareto_rows(df, study)
        if pareto_idx is not None and pareto_idx.size:
            from suspkin.plotting.optimization import knee_index

            cols = [f"f_{o.name}" for o in study.objectives]
            sub = df.iloc[pareto_idx]
            row = int(pareto_idx[knee_index(sub[cols].to_numpy(dtype=float))])
        else:
            row = best_row(df)
        best_x = _x_of(df, space)[row]
        best_id = int(df["eval_id"].iloc[row])
    else:
        best_x = baseline_x
    best_eval = evaluate(space, best_x)
    meta = make_meta(
        study_path=study_path,
        seed=opt.seed,
        method=opt.method,
        mode=opt.mode,
        started=started,
        ended=ended,
        n_evals=n_new,
        n_total=len(full),
        workers=workers_used,
        resumed=resume,
    )
    meta["study_hash"] = shash
    meta["valid_from_eval_id"] = valid_from
    report_path = write_outputs(
        run_dir,
        study=study,
        space=space,
        history=df,
        baseline=baseline,
        best_x=best_x,
        best_eval=best_eval,
        best_eval_id=best_id,
        pareto_idx=pareto_idx,
        meta=meta,
        report=report,
    )
    return RunResult(
        out_dir=run_dir,
        study=study,
        method=opt.method,
        mode=opt.mode,
        workers=workers_used,
        n_evals=n_new,
        n_total=len(full),
        seconds=seconds,
        resumed=resume,
        best_x=best_x,
        best_eval_id=best_id,
        feasible=best_eval.feasible and all(o.status == "ok" for o in best_eval.objectives),
        f_total=best_eval.f_total,
        baseline=baseline,
        best=best_eval,
        n_pareto=0 if pareto_idx is None else int(pareto_idx.size),
        report_path=report_path,
        budget_evaluations=budget[0],
        budget_time=budget[1],
    )
