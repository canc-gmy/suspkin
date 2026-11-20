"""``SurrogateProblem``: the v0.8 optimizer ``Problem`` evaluated on the networks.

Experimental (see ``docs/guides/stability.md``). Same public surface as
:class:`suspkin.optimize.problem.Problem` (cache, history, budget), so the optimizer drivers
run on it unchanged. All cache misses of a batch are predicted and evaluated together
(:class:`~suspkin.surrogate.evaluate.SurrogateEvaluator`); there is no process pool.

``margin`` (delta) tightens the constraints: every constraint that was evaluated reports
``g + delta``, so a surrogate-feasible design has a predicted ``g <= -delta``. Unsolved and
infeasible constraints keep ``G_FAIL``. The raw values are kept in :attr:`raw_g`.
"""

# ruff: noqa: N803  (X follows the dataset contract)
from __future__ import annotations

import time as _time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin.optimize.evaluate import ConstraintResult, Evaluation
from suspkin.optimize.history import cache_keys
from suspkin.optimize.problem import Problem
from suspkin.optimize.spec import StudyConfigError
from suspkin.surrogate.data import AxleSpace, axle_grids
from suspkin.surrogate.evaluate import (
    AxlePredictor,
    BaselineAxle,
    NetworkPredictor,
    SurrogateEvaluator,
)

F64 = NDArray[np.float64]

__all__ = ["SURROGATE_SOURCE", "SurrogateProblem"]

SURROGATE_SOURCE = "surrogate"
_AXLES: tuple[Literal["front", "rear"], ...] = ("front", "rear")


def _shifted(ev: Evaluation, margin: float) -> Evaluation:
    """``ev`` with ``g + margin`` on every evaluated (``pass`` / ``fail``) constraint."""
    if margin == 0.0:
        return ev
    res: list[ConstraintResult] = []
    for r in ev.results:
        if r.status in ("pass", "fail"):
            g = r.g + margin
            res.append(
                ConstraintResult(
                    r.name,
                    r.type,
                    r.value,
                    r.min,
                    r.max,
                    g,
                    "pass" if g <= 0.0 else "fail",
                    r.dim,
                    r.detail,
                )
            )
        else:
            res.append(r)
    return Evaluation(tuple(res), ev.objectives)


class SurrogateProblem(Problem):
    """Evaluate designs on the trained axle networks with the :class:`Problem` interface.

    Parameters
    ----------
    study_path
        Study YAML (its ``surrogate.grid`` must match the models; symmetric variables only).
    model_root
        Folder with ``<axle>/model/`` of each axle that has design variables (what
        ``suspkin surrogate train`` saves). May be ``None`` when ``predictors`` is given.
    run_dir
        Run folder; the history goes to ``run_dir/history.csv`` with source ``surrogate``.
    device
        ``auto``, ``cpu`` or ``cuda`` for the networks.
    margin
        Constraint margin delta on the normalised ``g`` (see the module docstring).
    evaluations, time
        Budget overrides (as :class:`Problem`); ``workers`` is not accepted (no pool).
    resume
        Continue an existing history (as :class:`Problem`).
    predictors
        Injected axle predictors (``front`` / ``rear``), e.g. an
        :class:`~suspkin.surrogate.evaluate.OraclePredictor`; replaces loading the models.
    """

    def __init__(
        self,
        study_path: str | Path,
        model_root: str | Path | None,
        run_dir: str | Path,
        *,
        device: str = "auto",
        margin: float = 0.0,
        evaluations: int | None = None,
        time: float | None = None,
        resume: bool = False,
        predictors: Mapping[str, AxlePredictor] | None = None,
    ) -> None:
        super().__init__(
            study_path, run_dir, resume=resume, workers=1, evaluations=evaluations, time=time
        )
        self.margin = float(margin)
        self.raw_g: dict[tuple[float, ...], F64] = {}
        preds = dict(predictors) if predictors is not None else {}
        for axle in _AXLES:
            if axle in preds:
                continue
            asp = AxleSpace(self.space, axle)
            if asp.n == 0:
                spec = self.study.surrogate
                if spec is None:
                    raise StudyConfigError(
                        f"no design variable on the {axle} axle and no surrogate.grid in the "
                        f"study: cannot build the baseline curves of that axle"
                    )
                preds[axle] = BaselineAxle(self.space, axle, axle_grids(spec.grid)[axle])
                continue
            if model_root is None:
                raise ValueError(f"no model_root and no predictor for the {axle} axle")
            preds[axle] = self._load(
                Path(model_root) / axle / "model", device, asp, self.study_path
            )
        self.evaluator = SurrogateEvaluator(self.space, preds)

    @staticmethod
    def _load(model_dir: Path, device: str, asp: AxleSpace, study_path: str) -> AxlePredictor:
        if not (model_dir / "model.json").is_file():
            raise FileNotFoundError(f"no trained model in {model_dir}")
        from suspkin.surrogate.dataset import check_model_matches
        from suspkin.surrogate.train import SurrogatePredictor  # imports torch

        check_model_matches(study_path, model_dir, asp.axle)

        pred = SurrogatePredictor.load(model_dir, device=device)
        if tuple(pred.param_names) != asp.names:
            raise StudyConfigError(
                f"{model_dir}: the model parameters {list(pred.param_names)} differ from the "
                f"study's {asp.axle} variables {list(asp.names)}"
            )
        return NetworkPredictor(pred)

    # ---- evaluation ---------------------------------------------------------------

    def _run(self, xs: list[F64]) -> list[tuple[Evaluation, float]]:
        if not xs:
            return []
        t0 = _time.perf_counter()
        evs = self.evaluator.evaluate(np.array(xs))
        per = (_time.perf_counter() - t0) / len(xs)
        out: list[tuple[Evaluation, float]] = []
        for k, ev in zip(cache_keys(np.array(xs)), evs, strict=True):
            self.raw_g[k] = ev.g_vector()
            out.append((_shifted(ev, self.margin), per))
        return out

    def evaluate_batch(
        self, X: ArrayLike, source: str = SURROGATE_SOURCE
    ) -> tuple[F64, F64, F64, F64, NDArray[np.bool_]]:
        """As :meth:`Problem.evaluate_batch`; the history source is always ``surrogate``."""
        return super().evaluate_batch(X, SURROGATE_SOURCE)
