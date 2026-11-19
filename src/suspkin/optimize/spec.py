"""Study specification: design variables, sweeps and constraints (YAML, schema 1).

Frozen pydantic models (``extra="forbid"``). Values in the YAML may carry units
(``"30 mm"``, ``"-1.5 deg"``, ``"0.5 deg/deg"``); everything is converted to SI on load and
the unit dimension is checked against the metric registry. Regions are in the vehicle frame
on the left side (y > 0); for an asymmetric right point they are on the right side.

Hexahedron vertex order: bottom face 0-1-2-3 counter-clockwise seen from +z, then the top
face 4-5-6-7 above 0-1-2-3 (see :mod:`suspkin.optimize.regions`).
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
    model_validator,
)

from suspkin.geometry.io import VehicleConfigError, load_vehicle
from suspkin.geometry.models import HARDPOINT_NAMES, Vehicle
from suspkin.optimize.metric_registry import (
    MetricNameError,
    corner_metric_dim,
    summary_metric_dim,
    vehicle_metric_dim,
)
from suspkin.optimize.quantise import has_grid_node, node_range
from suspkin.optimize.regions import check_hexahedron, is_convex_hexahedron
from suspkin.units import Dim, normalize_dim, parse_quantity, slope_dim

Vec3 = tuple[float, float, float]
Pair = tuple[float, float]

POINT_NAMES: tuple[str, ...] = (*HARDPOINT_NAMES, "rocker_arb")
JOINT_NAMES: tuple[str, ...] = (
    "uca_outer",
    "lca_outer",
    "tie_inner",
    "tie_outer",
    "pushrod_inner",
    "pushrod_outer",
    "spring_chassis",
    "rocker_spring",
)
_VEHICLE_INPUT_DIM: dict[str, Dim] = {
    "heave": "length",
    "roll": "angle",
    "pitch": "angle",
    "rack": "length",
}
_AXES = ("x", "y", "z")
#: Default resolution of the hardpoint grid (m): designs are producible to 0.5 mm.
DEFAULT_RESOLUTION = 5e-4
_RANGE_TOL = 1e-12
_COORD_RE = re.compile(r"^(front|rear)\.([a-z_]+)\.([xyz])$")


_DEG = math.pi / 180.0
_DEFAULT_SCALE: dict[str, float] = {
    "length": 1e-3,
    "angle": 0.1 * _DEG,
    "ratio": 0.01,
    "percent": 1.0,
    "force": 10.0,
    "force/length": 1000.0,
}


def default_scale(dim: Dim) -> float | None:
    """Default normalisation scale (SI) of a dimension; ``None`` if it has none."""
    v = _DEFAULT_SCALE.get(dim)  # canonical names hit directly; aliases need normalising
    return v if v is not None else _DEFAULT_SCALE.get(normalize_dim(dim))


class StudyConfigError(ValueError):
    """Raised when a study file cannot be parsed or validated."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------------------
# point references
# --------------------------------------------------------------------------------------


def parse_point_ref(ref: str) -> tuple[Literal["front", "rear"], str]:
    """Split ``"<axle>.<hardpoint>"``; raises ``ValueError`` with a suggestion if unknown."""
    axle, sep, name = ref.partition(".")
    if sep == "" or axle not in ("front", "rear"):
        raise ValueError(f"point {ref!r} must look like '<front|rear>.<hardpoint>'")
    if name not in POINT_NAMES:
        close = difflib.get_close_matches(name, POINT_NAMES, n=3, cutoff=0.5)
        hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(f"unknown hardpoint {name!r} in {ref!r}{hint}")
    return axle, name  # type: ignore[return-value]


def parse_coord_ref(ref: str) -> tuple[Literal["front", "rear"], str, int]:
    """Split ``"<axle>.<hardpoint>.<x|y|z>"`` into axle, hardpoint, axis index."""
    m = _COORD_RE.match(ref)
    if m is None:
        raise ValueError(f"coordinate {ref!r} must look like '<front|rear>.<hardpoint>.<x|y|z>'")
    axle, name = parse_point_ref(f"{m.group(1)}.{m.group(2)}")
    return axle, name, _AXES.index(m.group(3))


def _check_point_ref(ref: str) -> str:
    parse_point_ref(ref)
    return ref


# --------------------------------------------------------------------------------------
# unit-aware raw parsing helpers
# --------------------------------------------------------------------------------------


def _scalar(v: Any, dim: Dim, factor: float = 1.0) -> float:
    """Number or unit string -> SI; ``factor`` scales bare numbers (``units: mm``)."""
    if isinstance(v, str):
        return parse_quantity(v, dim)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"expected a number or a string with unit, got {v!r}")
    return float(v) * factor


def _seq(v: Any, n: int, what: str) -> list[Any]:
    if not isinstance(v, list | tuple) or len(v) != n:
        raise ValueError(f"{what} must be a list of {n} values, got {v!r}")
    return list(v)


def _pair(v: Any, dim: Dim, factor: float = 1.0) -> list[float]:
    return [_scalar(x, dim, factor) for x in _seq(v, 2, "range")]


def _vec(v: Any, dim: Dim, factor: float = 1.0) -> list[float]:
    return [_scalar(x, dim, factor) for x in _seq(v, 3, "vector")]


def _unit_factor(units: Any) -> float:
    if units is None:
        return 1.0
    if not isinstance(units, str):
        raise ValueError(f"units must be a string like 'mm', got {units!r}")
    return parse_quantity(f"1 {units}", "length")


# --------------------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------------------


class BoxRegion(_Frozen):
    """Axis-aligned box in the vehicle frame."""

    kind: Literal["box"] = "box"
    lo: Vec3
    hi: Vec3

    @model_validator(mode="after")
    def _check(self) -> BoxRegion:
        if any(a >= b for a, b in zip(self.lo, self.hi, strict=True)):
            raise ValueError(f"box min must be below max on every axis, got {self.lo}, {self.hi}")
        return self


class HexahedronRegion(_Frozen):
    """Hexahedron by 8 vertices (order: see module docstring)."""

    kind: Literal["hexahedron"] = "hexahedron"
    vertices: Annotated[tuple[Vec3, ...], Field(min_length=8, max_length=8)]

    @model_validator(mode="after")
    def _check(self) -> HexahedronRegion:
        check_hexahedron(self.vertices)
        return self


class OffsetRegion(_Frozen):
    """Box relative to the baseline point: ``x/y/z`` -> ``(lo, hi)`` offsets (variable only)."""

    kind: Literal["offset"] = "offset"
    x: Pair | None = None
    y: Pair | None = None
    z: Pair | None = None

    @model_validator(mode="after")
    def _check(self) -> OffsetRegion:
        ranges = [r for r in (self.x, self.y, self.z) if r is not None]
        if not ranges:
            raise ValueError("offset needs at least one of x, y, z")
        if any(lo > hi for lo, hi in ranges):
            raise ValueError("offset ranges must be [lo, hi] with lo <= hi")
        if any(lo == hi for lo, hi in ranges):
            raise ValueError(
                "offset axis with min == max: omit the axis (the point stays fixed) "
                "or give a non-zero range"
            )
        return self


class SphereRegion(_Frozen):
    """Sphere (keep-in or keep-out)."""

    kind: Literal["sphere"] = "sphere"
    center: Vec3
    radius: float = Field(gt=0)


class CylinderRegion(_Frozen):
    """Finite capped cylinder between the axis endpoints ``p0`` and ``p1``."""

    kind: Literal["cylinder"] = "cylinder"
    p0: Vec3
    p1: Vec3
    radius: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> CylinderRegion:
        if self.p0 == self.p1:
            raise ValueError("cylinder p0 and p1 must differ")
        return self


def _region_raw(raw: Any, allowed: tuple[str, ...]) -> Any:
    """``{box: {...}, units: mm}`` -> ``{kind: box, lo: ..., hi: ...}`` in SI."""
    if not isinstance(raw, Mapping) or "kind" in raw:
        return raw
    keys = [k for k in raw if k != "units"]
    if len(keys) != 1 or keys[0] not in allowed:
        raise ValueError(f"region must have exactly one key out of {list(allowed)}, got {keys}")
    kind, payload = keys[0], raw[keys[0]]
    units = raw.get("units")
    if isinstance(payload, Mapping) and "units" in payload:
        units = payload["units"]
    f = _unit_factor(units)
    if kind == "hexahedron":
        verts = payload["vertices"] if isinstance(payload, Mapping) else payload
        return {
            "kind": kind,
            "vertices": [_vec(p, "length", f) for p in _seq(verts, 8, "hexahedron vertices")],
        }
    if not isinstance(payload, Mapping):
        raise ValueError(f"{kind} region must be a mapping, got {payload!r}")
    body = {k: v for k, v in payload.items() if k != "units"}
    if kind == "box":
        out: dict[str, Any] = {
            "kind": kind,
            "lo": body.pop("min", None),
            "hi": body.pop("max", None),
        }
        out["lo"], out["hi"] = _vec(out["lo"], "length", f), _vec(out["hi"], "length", f)
    elif kind == "sphere":
        out = {
            "kind": kind,
            "center": _vec(body.pop("center", None), "length", f),
            "radius": _scalar(body.pop("radius", None), "length", f),
        }
    elif kind == "cylinder":
        out = {
            "kind": kind,
            "p0": _vec(body.pop("p0", None), "length", f),
            "p1": _vec(body.pop("p1", None), "length", f),
            "radius": _scalar(body.pop("radius", None), "length", f),
        }
    else:  # offset
        out = {"kind": kind}
        for a in _AXES:
            if a in body:
                out[a] = _pair(body.pop(a), "length", f)
    if body:
        raise ValueError(f"unknown keys {sorted(body)} in {kind} region")
    return out


def _domain_before(raw: Any) -> Any:
    return _region_raw(raw, ("hexahedron", "box", "offset"))


def _region_before(raw: Any) -> Any:
    return _region_raw(raw, ("hexahedron", "box", "sphere", "cylinder"))


Domain = Annotated[
    Annotated[BoxRegion | HexahedronRegion | OffsetRegion, Field(discriminator="kind")],
    BeforeValidator(_domain_before),
]
Region = Annotated[
    Annotated[
        BoxRegion | HexahedronRegion | SphereRegion | CylinderRegion, Field(discriminator="kind")
    ],
    BeforeValidator(_region_before),
]


# --------------------------------------------------------------------------------------
# variables
# --------------------------------------------------------------------------------------


def _parse_resolution(v: Any) -> Any:
    """``0.5 mm`` / number (SI) / ``None`` -> metres or ``None``; must be positive."""
    if v is None:
        return None
    out = _scalar(v, "length")
    if not out > 0.0:
        raise ValueError(f"resolution must be a positive length, got {v!r}")
    return out


class Variable(_Frozen):
    """A hardpoint that moves inside ``domain`` (left point; the right side mirrors it)."""

    point: Annotated[
        str, BeforeValidator(lambda v: _check_point_ref(v) if isinstance(v, str) else v)
    ]
    domain: Domain
    asymmetric: bool = False
    side: Literal["left", "right"] | None = None
    resolution: float | None = None
    """Grid override (m) of this variable; an explicit ``null`` means continuous."""

    @field_validator("resolution", mode="before")
    @classmethod
    def _resolution(cls, v: Any) -> Any:
        return _parse_resolution(v)

    @property
    def effective_side(self) -> Literal["left", "right"]:
        """``side`` of an asymmetric variable (default left); ``left`` when symmetric."""
        return self.side or "left"

    @model_validator(mode="after")
    def _check(self) -> Variable:
        if self.side is not None and not self.asymmetric:
            raise ValueError("'side' is only allowed together with asymmetric: true")
        if self.asymmetric and self.side == "right":
            d = self.domain
            if isinstance(d, HexahedronRegion):
                bad = max(v[1] for v in d.vertices) >= 0.0
            elif isinstance(d, BoxRegion):
                bad = d.hi[1] >= 0.0
            else:
                bad = False
            if bad:
                raise ValueError(
                    f"{self.point}: asymmetric right-side region must lie at y < 0 (vehicle frame)"
                )
        return self


# --------------------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------------------


class CornerSweep(_Frozen):
    """Sweep of one corner over wheel travel (bump positive)."""

    kind: Literal["corner"]
    corner: Literal["FL", "FR", "RL", "RR"]
    travel: Pair
    steps: int = Field(ge=3)

    @field_validator("travel", mode="before")
    @classmethod
    def _travel(cls, v: Any) -> Any:
        return _pair(v, "length")

    @model_validator(mode="after")
    def _check(self) -> CornerSweep:
        if not self.travel[0] < self.travel[1]:
            raise ValueError("travel must be [lo, hi] with lo < hi")
        return self


class VehicleSweep(_Frozen):
    """Sweep of one body input (heave, roll, pitch or rack) with the others at ``base``."""

    kind: Literal["vehicle"]
    input: Literal["heave", "roll", "pitch", "rack"]
    range: Pair | Literal["full"]
    steps: int = Field(ge=3)
    base: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _units(cls, raw: Any) -> Any:
        if not isinstance(raw, Mapping) or raw.get("input") not in _VEHICLE_INPUT_DIM:
            return raw
        out = dict(raw)
        dim = _VEHICLE_INPUT_DIM[raw["input"]]
        if "range" in out and out["range"] != "full":
            out["range"] = _pair(out["range"], dim)
        if isinstance(out.get("base"), Mapping):
            base = {}
            for k, v in out["base"].items():
                if k not in _VEHICLE_INPUT_DIM:
                    raise ValueError(f"unknown base key {k!r}; expected {list(_VEHICLE_INPUT_DIM)}")
                base[k] = _scalar(v, _VEHICLE_INPUT_DIM[k])
            out["base"] = base
        return out

    @model_validator(mode="after")
    def _check(self) -> VehicleSweep:
        if self.range == "full" and self.input != "rack":
            raise ValueError("range: full is only allowed for input: rack")
        if self.range != "full" and not self.range[0] < self.range[1]:
            raise ValueError("range must be [lo, hi] with lo < hi")
        if self.input in self.base:
            raise ValueError(f"base must not set the swept input {self.input!r}")
        return self


Sweep = Annotated[CornerSweep | VehicleSweep, Field(discriminator="kind")]


def sweep_input_dim(sweep: CornerSweep | VehicleSweep) -> Dim:
    """Dimension of the sweep input (corner travel is a length)."""
    return "length" if isinstance(sweep, CornerSweep) else _VEHICLE_INPUT_DIM[sweep.input]


def sweep_range(sweep: CornerSweep | VehicleSweep, vehicle: Vehicle | None = None) -> Pair | None:
    """SI range of a sweep; ``None`` for ``range: full`` without a vehicle."""
    if isinstance(sweep, CornerSweep):
        return sweep.travel
    if sweep.range != "full":
        return sweep.range
    st = None if vehicle is None else vehicle.front.steering
    return None if st is None else (float(st.rack_travel[0]), float(st.rack_travel[1]))


def sweep_metric_dim(sweep: CornerSweep | VehicleSweep, metric: str) -> Dim:
    """Dimension of ``metric`` in the result table of ``sweep``."""
    return (
        corner_metric_dim(metric) if isinstance(sweep, CornerSweep) else vehicle_metric_dim(metric)
    )


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


# --------------------------------------------------------------------------------------
# surrogate
# --------------------------------------------------------------------------------------


class AxisGrid(_Frozen):
    """Uniform 1-D grid ``[min, max]`` with ``points`` nodes (SI), containing 0 (static pose)."""

    min: float
    max: float
    points: int = Field(ge=3)

    @model_validator(mode="before")
    @classmethod
    def _units(cls, raw: Any) -> Any:
        if not isinstance(raw, Mapping):
            return raw
        out = dict(raw)
        for k in ("min", "max"):
            if k in out:
                out[k] = _scalar(out[k], "length")
        return out

    @model_validator(mode="after")
    def _check(self) -> AxisGrid:
        if not self.min < self.max:
            raise ValueError("grid min must be below max")
        if not self.min <= 0.0 <= self.max:
            raise ValueError("grid must contain 0 (the static pose)")
        return self


class GridSpec(_Frozen):
    """Output grid of the surrogate: wheel travel and (front axle only) rack."""

    travel: AxisGrid
    rack: AxisGrid | None = None


class DatasetSpec(_Frozen):
    """Initial Latin-hypercube dataset: samples per axle, seed and rows per shard file."""

    n_initial: int = Field(default=20000, ge=1)
    seed: int = 1
    shard_size: int = Field(default=500, ge=1)
    lhs_optimization: Literal["auto", "random-cd", "none"] = "auto"


class TrainingSpec(_Frozen):
    """Network training settings (the ``surrogate.training`` block)."""

    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_epochs: int = Field(default=2000, ge=1)
    patience: int = Field(default=100, ge=1)
    num_threads: int | None = Field(default=None, ge=1)
    finetune_lr_factor: float = Field(default=0.1, gt=0.0)


class HpoSpec(_Frozen):
    """Architecture search budget (the ``surrogate.hpo`` block); ``timeout`` in seconds (SI).

    ``max_epochs`` caps each search trial (``None``: ``training.max_epochs``; the effective cap
    is the smaller of the two) and ``subsample`` is the number of training rows each trial
    uses (``None``: all). The final retrain of the best configuration ignores both.
    """

    n_trials: int = Field(default=100, ge=1)
    timeout: float | None = Field(default=4 * 3600.0, gt=0.0)
    parallel: Literal["auto"] | StrictBool = "auto"
    max_epochs: int | None = Field(default=300, ge=1)
    subsample: int | None = Field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _units(cls, raw: Any) -> Any:
        if isinstance(raw, Mapping) and raw.get("timeout") is not None:
            out = dict(raw)
            try:
                out["timeout"] = _scalar(out["timeout"], "time")
            except ValueError as exc:
                raise ValueError(f"hpo timeout: {exc}") from exc
            return out
        return raw


class InfillSpec(_Frozen):
    """Infill points per active-learning iteration: extra candidates and perturbations."""

    top: int = Field(default=0, ge=0)
    perturb: int = Field(default=4, ge=0)


class FinetuneSpec(_Frozen):
    """Fine-tuning of the axle networks between active-learning iterations.

    ``epochs`` caps one fine-tune, ``patience`` stops it after that many epochs without a
    new best score of the run. ``weight`` is the sample weight of the new rows, ``replay``
    the number of earlier train rows replayed (drawn once, then kept on the device).
    ``trigger`` is the residual (in tolerance units) of the new verified point above which an
    axle is fine-tuned at all. After ``skip_after`` consecutive fine-tunes that kept the
    previous weights an axle is skipped for ``skip_for`` iterations, unless its residual
    exceeds ``stall_override * trigger``. ``trigger=0`` and ``skip_for=0`` always fine-tune.
    ``on_cpu`` decides what happens when the networks run on the CPU: ``"skip"`` (default)
    never fine-tunes (it takes minutes per iteration), ``"run"`` applies the rules above.
    """

    epochs: int = Field(default=60, ge=1)
    weight: float = Field(default=5.0, gt=0.0)
    replay: int = Field(default=2048, ge=0)
    patience: int = Field(default=30, ge=1)
    trigger: float = Field(default=1.0, ge=0.0)
    skip_after: int = Field(default=2, ge=1)
    skip_for: int = Field(default=2, ge=0)
    stall_override: float = Field(default=3.0, ge=1.0)
    on_cpu: Literal["skip", "run"] = "skip"


class TrustRegionSpec(_Frozen):
    """Trust-region radius in the unit cube: initial, minimum and maximum."""

    initial: float = Field(default=0.2, gt=0.0, le=1.0)
    min: float = Field(default=0.01, gt=0.0, le=1.0)
    max: float = Field(default=0.5, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> TrustRegionSpec:
        if not self.min <= self.initial <= self.max:
            raise ValueError("trust_region needs min <= initial <= max")
        return self


class StopSpec(_Frozen):
    """Active-learning stopping: relative agreement ``eps_f`` and ``patience`` iterations."""

    eps_f: float = Field(default=0.01, gt=0.0)
    patience: int = Field(default=5, ge=1)


class ActiveLearningSpec(_Frozen):
    """Active-learning optimizer (``surrogate.active_learning``); ``time`` in seconds (SI)."""

    max_iterations: int = Field(default=50, ge=1)
    fom_evaluations: int = Field(default=2000, ge=1)
    time: float | None = Field(default=None, gt=0.0)
    surrogate_evaluations: int = Field(default=20000, ge=1)
    infill: InfillSpec = Field(default_factory=InfillSpec)
    finetune: FinetuneSpec = Field(default_factory=FinetuneSpec)
    margin: float = Field(default=0.05, ge=0.0)
    trust_region: TrustRegionSpec = Field(default_factory=TrustRegionSpec)
    stop: StopSpec = Field(default_factory=StopSpec)

    @model_validator(mode="before")
    @classmethod
    def _units(cls, raw: Any) -> Any:
        if isinstance(raw, Mapping) and raw.get("time") is not None:
            out = dict(raw)
            try:
                out["time"] = _scalar(out["time"], "time")
            except ValueError as exc:
                raise ValueError(f"active_learning time: {exc}") from exc
            return out
        return raw


class SurrogateSpec(_Frozen):
    """Surrogate block of a study: grid, dataset, training, search and active learning."""

    grid: GridSpec
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    hpo: HpoSpec = Field(default_factory=HpoSpec)
    active_learning: ActiveLearningSpec = Field(default_factory=ActiveLearningSpec)


# --------------------------------------------------------------------------------------
# study
# --------------------------------------------------------------------------------------


class Study(_Frozen):
    """Optimisation study: variables, sweeps and constraints for one vehicle file."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_version: int = Field(alias="schema")

    @field_validator("schema_version")
    @classmethod
    def _check_schema(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"unsupported study schema {v}; this suspkin supports 1")
        return v

    vehicle: str
    resolution: float | None = DEFAULT_RESOLUTION
    """Hardpoint grid (m) of the chosen designs; ``None`` (``null``) means continuous."""
    variables: tuple[Variable, ...] = ()
    sweeps: dict[str, Sweep] = Field(default_factory=dict)
    constraints: tuple[Constraint, ...] = ()
    objectives: tuple[Objective, ...] = ()
    optimizer: Optimizer | None = None
    surrogate: SurrogateSpec | None = None

    @field_validator("resolution", mode="before")
    @classmethod
    def _resolution(cls, v: Any) -> Any:
        return _parse_resolution(v)

    @model_validator(mode="before")
    @classmethod
    def _units(cls, raw: Any) -> Any:
        if not isinstance(raw, Mapping):
            return raw
        out = dict(raw)
        sweeps = raw.get("sweeps") or {}
        cons = raw.get("constraints")
        if isinstance(sweeps, Mapping) and isinstance(cons, list | tuple):
            out["constraints"] = [_prep_constraint(c, sweeps) for c in cons]
        objs = raw.get("objectives")
        if isinstance(sweeps, Mapping) and isinstance(objs, list | tuple):
            out["objectives"] = [_prep_objective(o, sweeps) for o in objs]
        if "optimizer" in raw:
            out["optimizer"] = _prep_optimizer(raw["optimizer"])
        return out

    @model_validator(mode="after")
    def _cross_check(self) -> Study:
        seen: set[str] = set()
        for c in self.constraints:
            if c.name in seen:
                raise ValueError(f"duplicate constraint name {c.name!r}")
            seen.add(c.name)
        obj_names: set[str] = set()
        for o in self.objectives:
            if o.name in obj_names:
                raise ValueError(f"duplicate objective name {o.name!r}")
            if o.name in seen:
                raise ValueError(f"objective name {o.name!r} clashes with a constraint name")
            obj_names.add(o.name)
        opt = self.optimizer
        if opt is not None:
            if opt.mode == "pareto" and len(self.objectives) < 2:
                raise ValueError("optimizer: mode pareto needs at least 2 objectives")
            if opt.mode == "weighted" and len(self.objectives) < 1:
                raise ValueError("optimizer: mode weighted needs at least 1 objective")
        pts = [v.point for v in self.variables]
        sides = [(v.point, v.effective_side if v.asymmetric else "both") for v in self.variables]
        if len(set(sides)) != len(sides) or len(set(pts)) != len(pts):
            dup = sorted({p for p in pts if pts.count(p) > 1})
            raise ValueError(f"duplicate variable point(s): {dup}")
        _check_sweep_refs(self)
        _check_ranges(self, None)
        _check_scales(self)
        _check_objective_scales(self)
        return self


def _check_objective_scales(study: Study) -> None:
    """Objectives on dimensions without a default scale need ``scale`` (or a target)."""
    for o in study.objectives:
        dim = objective_dim(study.sweeps, o)
        if objective_scale(o, dim) is None:
            raise ValueError(f"objective {o.name!r}: add `scale` (no default for dim {dim})")


def _check_scales(study: Study) -> None:
    """Constraints on dimensions without a default scale need a usable limit or a scale."""
    for c in study.constraints:
        if not isinstance(
            c, BandConstraint | SlopeConstraint | ValueConstraint | SummaryConstraint
        ):
            continue
        if c.scale is not None:
            continue
        if isinstance(c, SummaryConstraint):
            dim = summary_metric_dim(c.metric)
        else:
            sw = study.sweeps[c.sweep]
            dim = sweep_metric_dim(sw, c.metric)
            if isinstance(c, SlopeConstraint):
                dim = slope_dim(dim, sweep_input_dim(sw))
        if default_scale(dim) is not None:
            continue
        lo, hi = c.min, c.max
        if lo is not None and hi is not None and hi != lo:
            continue
        if any(v for v in (lo, hi) if v is not None):
            continue
        raise ValueError(
            f"constraint {c.name!r}: add `scale` (no default for dim {dim} with zero limit)"
        )


def _sweep(study: Study, name: str | None) -> CornerSweep | VehicleSweep | None:
    return None if name is None else study.sweeps.get(name)


def _check_sweep_refs(study: Study) -> None:
    for o in study.objectives:
        if o.sweep is not None and o.sweep not in study.sweeps:
            raise ValueError(
                f"objective {o.name!r}: unknown sweep {o.sweep!r} (defined: {sorted(study.sweeps)})"
            )
    for c in study.constraints:
        if isinstance(c, JointAngleConstraint):
            sw = study.sweeps.get(c.sweep)
            if not isinstance(sw, CornerSweep) or sw.corner != c.corner:
                raise ValueError(
                    f"constraint {c.name!r}: sweep {c.sweep!r} must be a corner sweep of {c.corner}"
                )
        if isinstance(c, InsideConstraint | OutsideConstraint | DistanceConstraint) and c.over:
            sw2 = study.sweeps.get(c.over)
            if not isinstance(sw2, CornerSweep):
                raise ValueError(f"constraint {c.name!r}: `over` must name a corner sweep")
            left = sw2.corner[1] == "L"
            sweep_axle = "front" if sw2.corner[0] == "F" else "rear"
            refs = [c.a, c.b] if isinstance(c, DistanceConstraint) else [c.point]
            for ref in refs:
                axle = parse_point_ref(ref)[0]
                if axle != sweep_axle:
                    raise ValueError(
                        f"constraint {c.name!r}: sweep {c.over!r} is on the {sweep_axle} "
                        f"axle, point {ref!r} is on the {axle} axle"
                    )
            if not isinstance(c, DistanceConstraint):
                cy = region_centroid(c.region)[1]
                if cy == 0.0 or (cy > 0.0) != left:
                    side = "left (y > 0)" if left else "right (y < 0)"
                    raise ValueError(
                        f"constraint {c.name!r}: sweep {c.over!r} is a {side} corner but the "
                        f"region centroid is at y = {cy:g}"
                    )


def variable_resolution(study: Study, var: Variable) -> float | None:
    """Grid resolution (m) of a variable: its own override, else the study's (``None``: none)."""
    return var.resolution if "resolution" in var.model_fields_set else study.resolution


def _base_point(vehicle: Vehicle, var: Variable) -> tuple[float, float, float]:
    axle, name = parse_point_ref(var.point)
    ax = getattr(vehicle, axle)
    left = getattr(ax.hardpoints, name)
    if not (var.asymmetric and var.effective_side == "right"):
        return (float(left[0]), float(left[1]), float(left[2]))
    ov = ax.right_overrides.get(name)
    p = ov if ov is not None else (left[0], -left[1], left[2])
    return (float(p[0]), float(p[1]), float(p[2]))


def check_grid(study: Study, vehicle: Vehicle) -> None:
    """Every gridded variable must have at least one grid node inside its domain."""
    for var in study.variables:
        res = variable_resolution(study, var)
        if res is None:
            continue
        d = var.domain
        err = f"variable {var.point!r}: its domain contains no node of the {res * 1e3:g} mm grid"
        if isinstance(d, HexahedronRegion):
            if not has_grid_node(d.vertices, res):
                raise ValueError(err)
            continue
        if isinstance(d, BoxRegion):
            lo, hi = list(d.lo), list(d.hi)
        else:
            base = _base_point(vehicle, var)
            lo, hi = [], []
            for a, rng in enumerate((d.x, d.y, d.z)):
                if rng is not None and rng[1] > rng[0]:
                    lo.append(base[a] + rng[0])
                    hi.append(base[a] + rng[1])
        kmin, kmax = node_range(lo, hi, res)
        if any(a > b for a, b in zip(kmin, kmax, strict=True)):
            raise ValueError(err)


def region_centroid(region: Any) -> tuple[float, float, float]:
    """Centroid of a region (vertex mean for a hexahedron)."""
    if isinstance(region, BoxRegion):
        return tuple(0.5 * (a + b) for a, b in zip(region.lo, region.hi, strict=True))  # type: ignore[return-value]
    if isinstance(region, HexahedronRegion):
        n = len(region.vertices)
        return tuple(sum(v[i] for v in region.vertices) / n for i in range(3))  # type: ignore[return-value]
    if isinstance(region, SphereRegion):
        return tuple(region.center)  # type: ignore[return-value]
    assert isinstance(region, CylinderRegion)
    return tuple(0.5 * (a + b) for a, b in zip(region.p0, region.p1, strict=True))  # type: ignore[return-value]


def _check_ranges(study: Study, vehicle: Vehicle | None) -> None:
    def inside(
        label: str, c: str, v: float, sw: CornerSweep | VehicleSweep, what: str = "constraint"
    ) -> None:
        rng = sweep_range(sw, vehicle)
        if rng is not None and not rng[0] - _RANGE_TOL <= v <= rng[1] + _RANGE_TOL:
            raise ValueError(
                f"{what} {c!r}: {label} = {v:g} lies outside the sweep range "
                f"[{rng[0]:g}, {rng[1]:g}] (SI)"
            )

    for c in study.constraints:
        if isinstance(c, BandConstraint | SlopeConstraint | ValueConstraint):
            sw = study.sweeps[c.sweep]
            pts = [("at", c.at)] if isinstance(c, ValueConstraint) else []
            if not isinstance(c, ValueConstraint) and c.where is not None:
                if c.where[0] >= c.where[1]:
                    raise ValueError(f"constraint {c.name!r}: where must be [lo, hi] with lo < hi")
                pts = [("where[0]", c.where[0]), ("where[1]", c.where[1])]
            for label, v in pts:
                inside(label, c.name, v, sw)

    for o in study.objectives:
        if o.sweep is None:
            continue
        osw = study.sweeps[o.sweep]
        opts: list[tuple[str, float]] = []
        at = getattr(o, "at", None)
        where = getattr(o, "where", None)
        if at is not None:
            opts.append(("at", at))
        if where is not None:
            opts += [("where[0]", where[0]), ("where[1]", where[1])]
        for label, v in opts:
            inside(label, o.name, v, osw, "objective")
        orng = sweep_range(osw, vehicle)
        if (
            isinstance(o, CurveObjective)
            and where is None
            and orng is not None
            and min(orng[1], o.target.x[-1]) <= max(orng[0], o.target.x[0])
        ):
            raise ValueError(
                f"objective {o.name!r}: the target x range does not overlap the sweep range"
            )


def check_against_vehicle(study: Study, vehicle: Vehicle) -> None:
    """Checks that need the vehicle: points exist, ``range: full`` ranges, rack present."""
    refs: list[str] = [v.point for v in study.variables]
    for c in study.constraints:
        if isinstance(c, InsideConstraint | OutsideConstraint):
            refs.append(c.point)
        elif isinstance(c, DistanceConstraint):
            refs += [c.a, c.b]
        elif isinstance(c, RelationConstraint):
            refs.append(".".join(c.lhs.split(".")[:2]))
            if isinstance(c.rhs, str):
                refs.append(".".join(c.rhs.split(".")[:2]))
    for ref in refs:
        axle, name = parse_point_ref(ref)
        if getattr(getattr(vehicle, axle).hardpoints, name) is None:
            raise ValueError(f"point {ref!r} does not exist in the vehicle (not defined)")
    for sname, sw in study.sweeps.items():
        if isinstance(sw, VehicleSweep) and sw.input == "rack" and vehicle.front.steering is None:
            raise ValueError(f"sweep {sname!r}: the vehicle has no steering")
    _check_ranges(study, vehicle)
    check_grid(study, vehicle)


def load_study(path: str | Path) -> tuple[Study, Vehicle]:
    """Load a study YAML and the vehicle it points to (path relative to the study file).

    Raises
    ------
    StudyConfigError
        On I/O, YAML, validation or cross-validation errors; the message includes the path.
    """
    p = Path(path)
    try:
        study = Study.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
        vehicle = load_vehicle(p.parent / study.vehicle)
        check_against_vehicle(study, vehicle)
    except (OSError, yaml.YAMLError, ValidationError, VehicleConfigError, ValueError) as exc:
        raise StudyConfigError(f"{p}: {exc}") from exc
    return study, vehicle
