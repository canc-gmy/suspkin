"""Study constraints (schema 1)."""

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
    _COORD_RE,
    _VEHICLE_INPUT_DIM,
    HexahedronRegion,
    Pair,
    Region,
    Vec3,
    _check_point_ref,
    _Frozen,
    _pair,
    _scalar,
    parse_coord_ref,
)
from suspkin.units import Dim, slope_dim

# --------------------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------------------


class _Constraint(_Frozen):
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
        return _check_point_ref(v)

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
        return _check_point_ref(v)


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


def _raw_sweep(sweeps: Mapping[str, Any], name: Any) -> tuple[str, Dim, Mapping[str, Any]]:
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
    if kind == "vehicle" and sw.get("input") in _VEHICLE_INPUT_DIM:
        return "vehicle", _VEHICLE_INPUT_DIM[sw["input"]], sw
    raise ValueError(f"sweep {name!r} has an invalid kind/input")


def _prep_constraint(raw: Any, sweeps: Mapping[str, Any]) -> Any:
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
            out[key] = _pair(out[key], dim) if pair else _scalar(out[key], dim)
        except ValueError as exc:
            raise ValueError(f"constraint {name!r}, field {key!r}: {exc}") from exc

    try:
        if t in ("band", "slope", "value"):
            kind, in_dim, sw = _raw_sweep(sweeps, raw.get("sweep"))
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
            if raw.get("over") is not None and _raw_sweep(sweeps, raw["over"])[0] != "corner":
                raise ValueError(
                    f"`over` must name a corner sweep, {raw['over']!r} is a vehicle sweep"
                )
        elif t == "relation":
            conv("offset", "length")
            conv("scale", "length")
            rhs = raw.get("rhs")
            if isinstance(rhs, str) and _COORD_RE.match(rhs) is None:
                out["rhs"] = _scalar(rhs, "length")
        elif t == "joint_angle":
            _raw_sweep(sweeps, raw.get("sweep"))
            for k in ("min", "max", "scale"):
                conv(k, "angle")
    except MetricNameError as exc:
        raise ValueError(f"constraint {name!r}: {exc}") from exc
    except ValueError as exc:
        if f"constraint {name!r}" in str(exc):
            raise
        raise ValueError(f"constraint {name!r}: {exc}") from exc
    return out
