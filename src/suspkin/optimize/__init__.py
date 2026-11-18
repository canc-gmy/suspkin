"""Optimization study spec: regions, metric registry, study loader, evaluator, objectives."""

from suspkin.optimize.design import DesignSpace, InfeasibleDesign
from suspkin.optimize.drivers import RunResult, run_optimization
from suspkin.optimize.evaluate import (
    F_FAIL,
    G_FAIL,
    ConstraintResult,
    Evaluation,
    ObjectiveResult,
    evaluate,
)
from suspkin.optimize.history import EvalCache, HistoryWriter, read_history
from suspkin.optimize.problem import BudgetExhausted, Problem

__all__ = [
    "F_FAIL",
    "G_FAIL",
    "BudgetExhausted",
    "ConstraintResult",
    "DesignSpace",
    "EvalCache",
    "Evaluation",
    "HistoryWriter",
    "InfeasibleDesign",
    "ObjectiveResult",
    "Problem",
    "RunResult",
    "evaluate",
    "read_history",
    "run_optimization",
]
