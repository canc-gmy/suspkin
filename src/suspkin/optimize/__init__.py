"""Optimization study spec: regions, metric registry, study loader, evaluator, objectives."""

from suspkin.optimize.design import DesignSpace, InfeasibleDesign
from suspkin.optimize.evaluate import (
    F_FAIL,
    G_FAIL,
    ConstraintResult,
    Evaluation,
    ObjectiveResult,
    evaluate,
)

__all__ = [
    "F_FAIL",
    "G_FAIL",
    "ConstraintResult",
    "DesignSpace",
    "Evaluation",
    "InfeasibleDesign",
    "ObjectiveResult",
    "evaluate",
]
