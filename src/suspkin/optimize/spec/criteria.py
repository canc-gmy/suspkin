"""Study constraints, objectives and optimizer settings (schema 1)."""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from suspkin.optimize.regions import is_convex_hexahedron
from suspkin.optimize.registry import (
    MetricNameError,
    corner_metric_dim,
    summary_metric_dim,
    vehicle_metric_dim,
)
from suspkin.optimize.spec.variables import (
    COORD_RE,
    RANGE_TOL,
    VEHICLE_INPUT_DIM,
    CornerSweep,
    Frozen,
    HexahedronRegion,
    Pair,
    Region,
    Vec3,
    VehicleSweep,
    check_point_ref,
    default_scale,
    parse_coord_ref,
    parse_pair,
    parse_scalar,
    sweep_input_dim,
    sweep_metric_dim,
)
from suspkin.units import Dim, slope_dim

# --------------------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------------------


class _Constraint(Frozen):
    name: str = Field(min_length=1)
    enabled: bool = True
    scale: float | None = Field(default=None, gt=0)
    note: str | None = None


class _Limits(_Constraint):
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_limits(self) -> _Limits:
        if self.min is None and self.max is None:
            raise ValueError(f"constraint {self.name!r}: at least one of min, max is required")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"constraint {self.name!r}: min must be <= max")
        return self


class BandConstraint(_Limits):
    """Metric stays within ``[min, max]`` over ``where`` of a sweep (default: whole sweep)."""

    type: Literal["band"]
    sweep: str
    metric: str
    where: Pair | None = None


class SlopeConstraint(_Limits):
    """Least-squares slope d(metric)/d(sweep input) over ``where`` within ``[min, max]``."""

    type: Literal["slope"]
    sweep: str
    metric: str
    where: Pair | None = None


class ValueConstraint(_Limits):
    """Metric at the sweep input ``at`` within ``[min, max]``."""

    type: Literal["value"]
    sweep: str
    metric: str
    at: float


class ReachConstraint(_Constraint):
    """Corner solves over ``travel`` (``stroke``: coilover in range; toggle margin minimum)."""

    type: Literal["reach"]
    corner: Literal["FL", "FR", "RL", "RR"]
    travel: Pair
    min_toggle_margin: float | None = None
    stroke: bool = True
    toggle_scale: float | None = Field(default=None, gt=0)  # rad
    stroke_scale: float | None = Field(default=None, gt=0)  # m

    @model_validator(mode="after")
    def _check(self) -> ReachConstraint:
        if not self.travel[0] < self.travel[1]:
            raise ValueError("travel must be [lo, hi] with lo < hi")
        return self


class SummaryConstraint(_Limits):
    """Scalar of the vehicle summary (``roll_gradient``, ``front.rc_height``, ...)."""

    type: Literal["summary"]
    metric: str


class _RegionConstraint(_Constraint):
    point: str
    region: Region
    over: str | None = None

    @field_validator("point")
    @classmethod
    def _point(cls, v: str) -> str:
        return check_point_ref(v)

    @model_validator(mode="after")
    def _convex(self) -> _RegionConstraint:
        r = self.region
        if isinstance(r, HexahedronRegion) and not is_convex_hexahedron(r.vertices):
            raise ValueError(f"constraint {self.name!r}: hexahedron region must be convex")
        return self


class InsideConstraint(_RegionConstraint):
    """Point stays inside ``region``."""

    type: Literal["inside"]


class OutsideConstraint(_RegionConstraint):
    """Point stays outside ``region`` (keep-out)."""

    type: Literal["outside"]


class DistanceConstraint(_Limits):
    """Distance between two points within ``[min, max]``."""

    type: Literal["distance"]
    a: str
    b: str
    over: str | None = None

    @field_validator("a", "b")
    @classmethod
    def _point(cls, v: str) -> str:
        return check_point_ref(v)


class RelationConstraint(_Constraint):
    """``lhs op rhs + offset`` between coordinates (rhs: coordinate ref or length)."""

    type: Literal["relation"]
    lhs: str
    op: Literal[">=", "<="]
    rhs: str | float
    offset: float = 0.0

    @field_validator("lhs")
    @classmethod
    def _lhs(cls, v: str) -> str:
        parse_coord_ref(v)
        return v

    @field_validator("rhs")
    @classmethod
    def _rhs(cls, v: str | float) -> str | float:
        if isinstance(v, str):
            parse_coord_ref(v)
        return v


class JointAngleConstraint(_Limits):
    """Joint angle (rotation of the joint axis between its two bodies) over a corner sweep."""

    type: Literal["joint_angle"]
    corner: Literal["FL", "FR", "RL", "RR"]
    joint: Literal[
        "uca_outer",
        "lca_outer",
        "tie_inner",
        "tie_outer",
        "pushrod_inner",
        "pushrod_outer",
        "spring_chassis",
        "rocker_spring",
    ]
    sweep: str
    axis: Vec3 | None = None

    @field_validator("axis")
    @classmethod
    def _axis(cls, v: Vec3 | None) -> Vec3 | None:
        if v is not None and sum(c * c for c in v) <= 0.0:
            raise ValueError("axis must be non-zero")
        return v


Constraint = Annotated[
    BandConstraint
    | SlopeConstraint
    | ValueConstraint
    | ReachConstraint
    | SummaryConstraint
    | InsideConstraint
    | OutsideConstraint
    | DistanceConstraint
    | RelationConstraint
    | JointAngleConstraint,
    Field(discriminator="type"),
]


def raw_sweep(sweeps: Mapping[str, Any], name: Any) -> tuple[str, Dim, Mapping[str, Any]]:
    """(kind, input dim, raw) of the named raw sweep; helpful errors."""
    if not isinstance(name, str) or name not in sweeps:
        close = difflib.get_close_matches(str(name), list(sweeps), n=2, cutoff=0.5)
        hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(f"unknown sweep {name!r} (defined: {sorted(sweeps)}){hint}")
    sw = sweeps[name]
    if not isinstance(sw, Mapping):
        raise ValueError(f"sweep {name!r} must be a mapping")
    kind = sw.get("kind")
    if kind == "corner":
        return "corner", "length", sw
    if kind == "vehicle" and sw.get("input") in VEHICLE_INPUT_DIM:
        return "vehicle", VEHICLE_INPUT_DIM[sw["input"]], sw
    raise ValueError(f"sweep {name!r} has an invalid kind/input")


def prep_constraint(raw: Any, sweeps: Mapping[str, Any]) -> Any:
    """Parse the unit-carrying fields of a raw constraint into SI (needs the sweeps)."""
    if not isinstance(raw, Mapping):
        return raw
    out = dict(raw)
    t = raw.get("type")
    name = raw.get("name", "?")

    def conv(key: str, dim: Dim, pair: bool = False) -> None:
        if out.get(key) is None:
            return
        try:
            out[key] = parse_pair(out[key], dim) if pair else parse_scalar(out[key], dim)
        except ValueError as exc:
            raise ValueError(f"constraint {name!r}, field {key!r}: {exc}") from exc

    try:
        if t in ("band", "slope", "value"):
            kind, in_dim, sw = raw_sweep(sweeps, raw.get("sweep"))
            metric = raw.get("metric")
            if not isinstance(metric, str):
                raise ValueError("metric must be a string")
            mdim = corner_metric_dim(metric) if kind == "corner" else vehicle_metric_dim(metric)
            odim = slope_dim(mdim, in_dim) if t == "slope" else mdim
            if t == "value":
                conv("at", in_dim)
            else:
                conv("where", in_dim, pair=True)
            for k in ("min", "max", "scale"):
                conv(k, odim)
            del sw
        elif t == "summary":
            metric = raw.get("metric")
            if not isinstance(metric, str):
                raise ValueError("metric must be a string")
            sdim = summary_metric_dim(metric)
            for k in ("min", "max", "scale"):
                conv(k, sdim)
        elif t == "reach":
            conv("travel", "length", pair=True)
            conv("min_toggle_margin", "angle")
            if raw.get("scale") is not None:
                raise ValueError(
                    "reach has two terms with different units: use `toggle_scale` (angle) "
                    "and/or `stroke_scale` (length) instead of `scale`"
                )
            conv("toggle_scale", "angle")
            conv("stroke_scale", "length")
        elif t in ("inside", "outside", "distance"):
            for k in ("min", "max", "scale"):
                conv(k, "length")
            if raw.get("over") is not None and raw_sweep(sweeps, raw["over"])[0] != "corner":
                raise ValueError(
                    f"`over` must name a corner sweep, {raw['over']!r} is a vehicle sweep"
                )
        elif t == "relation":
            conv("offset", "length")
            conv("scale", "length")
            rhs = raw.get("rhs")
            if isinstance(rhs, str) and COORD_RE.match(rhs) is None:
                out["rhs"] = parse_scalar(rhs, "length")
        elif t == "joint_angle":
            raw_sweep(sweeps, raw.get("sweep"))
            for k in ("min", "max", "scale"):
                conv(k, "angle")
    except MetricNameError as exc:
        raise ValueError(f"constraint {name!r}: {exc}") from exc
    except ValueError as exc:
        if f"constraint {name!r}" in str(exc):
            raise
        raise ValueError(f"constraint {name!r}: {exc}") from exc
    return out


# --------------------------------------------------------------------------------------
# objectives and optimizer
# --------------------------------------------------------------------------------------


class CurveTarget(Frozen):
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


class _Objective(Frozen):
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
            self.target.x[0] - RANGE_TOL <= self.where[0]
            and self.where[1] <= self.target.x[-1] + RANGE_TOL
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


class Budget(Frozen):
    """Run budget: number of evaluations and/or wall time in seconds (SI)."""

    evaluations: int | None = Field(default=None, gt=0)
    time: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> Budget:
        if self.evaluations is None and self.time is None:
            raise ValueError("budget needs `evaluations` and/or `time`")
        return self


class Optimizer(Frozen):
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


def prep_optimizer(raw: Any) -> Any:
    """Parse the unit-carrying ``budget.time`` of a raw optimizer block into seconds."""
    if not isinstance(raw, Mapping) or not isinstance(raw.get("budget"), Mapping):
        return raw
    budget = dict(raw["budget"])
    if budget.get("time") is not None:
        try:
            budget["time"] = parse_scalar(budget["time"], "time")
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


def prep_objective(raw: Any, sweeps: Mapping[str, Any]) -> Any:
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
            out[key] = parse_pair(out[key], dim) if pair else parse_scalar(out[key], dim)
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
            kind, in_dim, _ = raw_sweep(sweeps, raw["sweep"])
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
                "x": [parse_scalar(v, in_dim) for v in _seq_any(tgt["x"], "target x")],
                "y": [parse_scalar(v, mdim) for v in _seq_any(tgt["y"], "target y")],
            }
    except MetricNameError as exc:
        raise ValueError(f"objective {name!r}: {exc}") from exc
    except ValueError as exc:
        if f"objective {name!r}" in str(exc):
            raise
        raise ValueError(f"objective {name!r}: {exc}") from exc
    return out
