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
from suspkin.optimize.regions import check_hexahedron, is_convex_hexahedron
from suspkin.units import Dim, parse_quantity, slope_dim

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
_RANGE_TOL = 1e-12
_COORD_RE = re.compile(r"^(front|rear)\.([a-z_]+)\.([xyz])$")


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


class Variable(_Frozen):
    """A hardpoint that moves inside ``domain`` (left point; the right side mirrors it)."""

    point: Annotated[
        str, BeforeValidator(lambda v: _check_point_ref(v) if isinstance(v, str) else v)
    ]
    domain: Domain
    asymmetric: bool = False
    side: Literal["left", "right"] | None = None

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
            conv("scale", "ratio")
        elif t in ("inside", "outside", "distance"):
            for k in ("min", "max", "scale"):
                conv(k, "length")
            if raw.get("over") is not None:
                _raw_sweep(sweeps, raw["over"])
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
# study
# --------------------------------------------------------------------------------------


class Study(_Frozen):
    """Optimisation study: variables, sweeps and constraints for one vehicle file."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(alias="schema")
    vehicle: str
    variables: tuple[Variable, ...] = ()
    sweeps: dict[str, Sweep] = Field(default_factory=dict)
    constraints: tuple[Constraint, ...] = ()

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
        return out

    @model_validator(mode="after")
    def _cross_check(self) -> Study:
        seen: set[str] = set()
        for c in self.constraints:
            if c.name in seen:
                raise ValueError(f"duplicate constraint name {c.name!r}")
            seen.add(c.name)
        pts = [v.point for v in self.variables]
        sides = [(v.point, v.effective_side if v.asymmetric else "both") for v in self.variables]
        if len(set(sides)) != len(sides) or len(set(pts)) != len(pts):
            dup = sorted({p for p in pts if pts.count(p) > 1})
            raise ValueError(f"duplicate variable point(s): {dup}")
        _check_sweep_refs(self)
        _check_ranges(self, None)
        return self


def _sweep(study: Study, name: str | None) -> CornerSweep | VehicleSweep | None:
    return None if name is None else study.sweeps.get(name)


def _check_sweep_refs(study: Study) -> None:
    for c in study.constraints:
        if isinstance(c, JointAngleConstraint):
            sw = study.sweeps.get(c.sweep)
            if not isinstance(sw, CornerSweep) or sw.corner != c.corner:
                raise ValueError(
                    f"constraint {c.name!r}: sweep {c.sweep!r} must be a corner sweep of {c.corner}"
                )
        if isinstance(c, InsideConstraint | OutsideConstraint | DistanceConstraint) and c.over:
            sw2 = study.sweeps.get(c.over)
            axle = parse_point_ref(c.point if not isinstance(c, DistanceConstraint) else c.a)[0]
            want = (
                "front" if isinstance(sw2, CornerSweep) and sw2.corner in ("FL", "FR") else "rear"
            )
            if isinstance(sw2, CornerSweep) and want != axle:
                raise ValueError(
                    f"constraint {c.name!r}: sweep {c.over!r} is on the {want} axle, "
                    f"the point is on the {axle} axle"
                )


def _check_ranges(study: Study, vehicle: Vehicle | None) -> None:
    def inside(label: str, c: str, v: float, sw: CornerSweep | VehicleSweep) -> None:
        rng = sweep_range(sw, vehicle)
        if rng is not None and not rng[0] - _RANGE_TOL <= v <= rng[1] + _RANGE_TOL:
            raise ValueError(
                f"constraint {c!r}: {label} = {v:g} lies outside the sweep range "
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
