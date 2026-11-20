"""Study objectives and optimizer settings (schema 1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    model_validator,
)

from suspkin._spec_base import _Frozen, _scalar
from suspkin.optimize.registry import (
    MetricNameError,
    corner_metric_dim,
    summary_metric_dim,
    vehicle_metric_dim,
)
from suspkin.optimize.spec.constraints import _raw_sweep
from suspkin.optimize.spec.variables import (
    _RANGE_TOL,
    CornerSweep,
    Pair,
    VehicleSweep,
    _pair,
    default_scale,
    sweep_input_dim,
    sweep_metric_dim,
)
from suspkin.units import Dim, slope_dim

# --------------------------------------------------------------------------------------
# objectives and optimizer
# --------------------------------------------------------------------------------------


class CurveTarget(_Frozen):
    """Target curve ``y(x)`` (linear interpolation); x strictly increasing, SI."""

    x: tuple[float, ...]
    y: tuple[float, ...]

    @model_validator(mode="after")
    def _check(self) -> CurveTarget:
        if len(self.x) != len(self.y):
            raise ValueError(
                f"target x and y must have the same length, got {len(self.x)}, {len(self.y)}"
            )
        if len(self.x) < 2:
            raise ValueError("target needs at least 2 points")
        if any(b <= a for a, b in zip(self.x, self.x[1:], strict=False)):
            raise ValueError("target x must be strictly increasing")
        return self


class _Objective(_Frozen):
    name: str = Field(min_length=1)
    weight: float = Field(default=1.0, ge=0)
    scale: float | None = Field(default=None, gt=0)
    note: str | None = None


def _check_where(name: str, where: Pair | None) -> None:
    if where is not None and where[0] >= where[1]:
        raise ValueError(f"objective {name!r}: where must be [lo, hi] with lo < hi")


class CurveObjective(_Objective):
    """RMS error of a sweep metric to a target curve (default window: overlap with the sweep)."""

    type: Literal["curve"]
    sweep: str
    metric: str
    target: CurveTarget
    where: Pair | None = None

    @model_validator(mode="after")
    def _check(self) -> CurveObjective:
        _check_where(self.name, self.where)
        if self.where is not None and not (
            self.target.x[0] - _RANGE_TOL <= self.where[0]
            and self.where[1] <= self.target.x[-1] + _RANGE_TOL
        ):
            raise ValueError(f"objective {self.name!r}: where must lie inside the target x range")
        return self


class TargetObjective(_Objective):
    """``|value - target|``: metric at ``at`` of a sweep, or a summary metric (no sweep)."""

    type: Literal["target"]
    sweep: str | None = None
    metric: str
    at: float | None = None
    target: float

    @model_validator(mode="after")
    def _check(self) -> TargetObjective:
        if self.sweep is None and self.at is not None:
            raise ValueError(f"objective {self.name!r}: `at` needs a sweep")
        if self.sweep is not None and self.at is None:
            raise ValueError(f"objective {self.name!r}: `at` is required with a sweep")
        return self


class ExtremumObjective(_Objective):
    """Minimise or maximise a metric statistic (``of``) over a sweep window or a summary."""

    type: Literal["minimize", "maximize"]
    sweep: str | None = None
    metric: str
    of: Literal["value", "slope", "max", "min", "max_abs", "mean"] = "value"
    at: float | None = None
    where: Pair | None = None

    @model_validator(mode="after")
    def _check(self) -> ExtremumObjective:
        n = self.name
        _check_where(n, self.where)
        if self.sweep is None:
            if self.of != "value":
                raise ValueError(f"objective {n!r}: a summary metric (no sweep) needs of: value")
            if self.at is not None or self.where is not None:
                raise ValueError(f"objective {n!r}: `at`/`where` need a sweep")
        elif self.of == "value":
            if self.at is None:
                raise ValueError(f"objective {n!r}: of: value needs `at`")
            if self.where is not None:
                raise ValueError(f"objective {n!r}: of: value uses `at`, not `where`")
        elif self.at is not None:
            raise ValueError(f"objective {n!r}: `at` is only for of: value (use `where`)")
        return self


class LinearityObjective(_Objective):
    """RMS residual of a least-squares line through the metric over ``where``."""

    type: Literal["linearity"]
    sweep: str
    metric: str
    where: Pair | None = None

    @model_validator(mode="after")
    def _check(self) -> LinearityObjective:
        _check_where(self.name, self.where)
        return self


Objective = Annotated[
    CurveObjective | TargetObjective | ExtremumObjective | LinearityObjective,
    Field(discriminator="type"),
]


class Budget(_Frozen):
    """Run budget: number of evaluations and/or wall time in seconds (SI)."""

    evaluations: int | None = Field(default=None, gt=0)
    time: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> Budget:
        if self.evaluations is None and self.time is None:
            raise ValueError("budget needs `evaluations` and/or `time`")
        return self


class Optimizer(_Frozen):
    """Optimizer block (the drivers live in a later slice)."""

    mode: Literal["weighted", "pareto"] = "weighted"
    method: Literal["nsga2", "de", "slsqp", "trust-constr", "de+slsqp"] = "de+slsqp"
    seed: int = 0
    workers: Annotated[int, Field(ge=1)] | Literal["auto"] = "auto"
    budget: Budget = Field(default_factory=lambda: Budget(evaluations=2000))
    start: Literal["baseline", "best"] = "baseline"
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _default_method(cls, raw: Any) -> Any:
        if isinstance(raw, Mapping) and "method" not in raw and raw.get("mode") == "pareto":
            return {**raw, "method": "nsga2"}
        return raw

    @model_validator(mode="after")
    def _check(self) -> Optimizer:
        if self.mode == "pareto" and self.method != "nsga2":
            raise ValueError("optimizer: mode pareto requires method nsga2")
        return self


def _prep_optimizer(raw: Any) -> Any:
    """Parse the unit-carrying ``budget.time`` of a raw optimizer block into seconds."""
    if not isinstance(raw, Mapping) or not isinstance(raw.get("budget"), Mapping):
        return raw
    budget = dict(raw["budget"])
    if budget.get("time") is not None:
        try:
            budget["time"] = _scalar(budget["time"], "time")
        except ValueError as exc:
            raise ValueError(f"optimizer budget, field 'time': {exc}") from exc
    return {**raw, "budget": budget}


def objective_dim(sweeps: Mapping[str, CornerSweep | VehicleSweep], o: Any) -> Dim:
    """Dimension of the raw value of an objective (SI); the sweep must exist."""
    if o.sweep is None:
        return summary_metric_dim(o.metric)
    sw = sweeps[o.sweep]
    mdim = sweep_metric_dim(sw, o.metric)
    if isinstance(o, ExtremumObjective) and o.of == "slope":
        return slope_dim(mdim, sweep_input_dim(sw))
    return mdim


def objective_scale(o: Any, dim: Dim) -> float | None:
    """Normalisation scale (SI): user ``scale``, else the dim default, else ``|target|``."""
    if o.scale is not None:
        return float(o.scale)
    d = default_scale(dim)
    if d is not None:
        return d
    if isinstance(o, TargetObjective) and o.target != 0.0:
        return abs(float(o.target))
    return None


def _seq_any(v: Any, what: str) -> list[Any]:
    if not isinstance(v, list | tuple):
        raise ValueError(f"{what} must be a list, got {v!r}")
    return list(v)


def _prep_objective(raw: Any, sweeps: Mapping[str, Any]) -> Any:
    """Parse the unit-carrying fields of a raw objective into SI (needs the sweeps)."""
    if not isinstance(raw, Mapping):
        return raw
    out = dict(raw)
    t = raw.get("type")
    name = raw.get("name", "?")
    if t not in ("curve", "target", "minimize", "maximize", "linearity"):
        return raw

    def conv(key: str, dim: Dim, pair: bool = False) -> None:
        if out.get(key) is None:
            return
        try:
            out[key] = _pair(out[key], dim) if pair else _scalar(out[key], dim)
        except ValueError as exc:
            raise ValueError(f"objective {name!r}, field {key!r}: {exc}") from exc

    try:
        metric = raw.get("metric")
        if not isinstance(metric, str):
            raise ValueError("metric must be a string")
        in_dim: Dim | None = None
        if raw.get("sweep") is None:
            if t in ("curve", "linearity"):
                raise ValueError("`sweep` is required")
            mdim = summary_metric_dim(metric)
        else:
            kind, in_dim, _ = _raw_sweep(sweeps, raw["sweep"])
            mdim = corner_metric_dim(metric) if kind == "corner" else vehicle_metric_dim(metric)
        odim = mdim
        if t in ("minimize", "maximize") and raw.get("of") == "slope" and in_dim is not None:
            odim = slope_dim(mdim, in_dim)
        if in_dim is not None:
            conv("at", in_dim)
            conv("where", in_dim, pair=True)
        conv("scale", odim)
        if t == "target":
            conv("target", odim)
        if t == "curve":
            tgt = raw.get("target")
            if not isinstance(tgt, Mapping) or set(tgt) != {"x", "y"}:
                raise ValueError("target must be a mapping with keys x and y")
            assert in_dim is not None
            out["target"] = {
                "x": [_scalar(v, in_dim) for v in _seq_any(tgt["x"], "target x")],
                "y": [_scalar(v, mdim) for v in _seq_any(tgt["y"], "target y")],
            }
    except MetricNameError as exc:
        raise ValueError(f"objective {name!r}: {exc}") from exc
    except ValueError as exc:
        if f"objective {name!r}" in str(exc):
            raise
        raise ValueError(f"objective {name!r}: {exc}") from exc
    return out
