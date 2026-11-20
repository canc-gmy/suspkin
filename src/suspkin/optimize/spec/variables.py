"""Study variables, regions and sweeps (schema 1)."""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from suspkin.geometry.model import HARDPOINT_NAMES, Vehicle
from suspkin.optimize.regions import check_hexahedron
from suspkin.optimize.registry import (
    corner_metric_dim,
    vehicle_metric_dim,
)
from suspkin.units import Dim, normalize_dim, parse_quantity


class Frozen(BaseModel):
    """Base of all spec models: immutable and rejecting unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def parse_scalar(v: Any, dim: Dim, factor: float = 1.0) -> float:
    """Number or unit string -> SI; ``factor`` scales bare numbers (``units: mm``).

    Args:
        v: Number or quantity string such as ``"12 mm"``.
        dim: Expected physical dimension (see :mod:`suspkin.units`).
        factor: SI multiplier for bare numbers (from a ``units:`` key).

    Returns:
        Value in SI units.

    Raises:
        ValueError: ``v`` cannot be parsed for ``dim``.
    """
    if isinstance(v, str):
        return parse_quantity(v, dim)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"expected a number or a string with unit, got {v!r}")
    return float(v) * factor


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
VEHICLE_INPUT_DIM: dict[str, Dim] = {
    "heave": "length",
    "roll": "angle",
    "pitch": "angle",
    "rack": "length",
}
_AXES = ("x", "y", "z")
#: Default resolution of the hardpoint grid (m): designs are producible to 0.5 mm.
DEFAULT_RESOLUTION = 5e-4
RANGE_TOL = 1e-12
COORD_RE = re.compile(r"^(front|rear)\.([a-z_]+)\.([xyz])$")


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
    m = COORD_RE.match(ref)
    if m is None:
        raise ValueError(f"coordinate {ref!r} must look like '<front|rear>.<hardpoint>.<x|y|z>'")
    axle, name = parse_point_ref(f"{m.group(1)}.{m.group(2)}")
    return axle, name, _AXES.index(m.group(3))


def check_point_ref(ref: str) -> str:
    """Validate a ``front.<hardpoint>`` / ``rear.<hardpoint>`` reference and return it unchanged.

    Raises:
        ValueError: If ``ref`` is not a known axle/hardpoint name.
    """
    parse_point_ref(ref)
    return ref


# --------------------------------------------------------------------------------------
# unit-aware raw parsing helpers
# --------------------------------------------------------------------------------------


def _seq(v: Any, n: int, what: str) -> list[Any]:
    if not isinstance(v, list | tuple) or len(v) != n:
        raise ValueError(f"{what} must be a list of {n} values, got {v!r}")
    return list(v)


def parse_pair(v: Any, dim: Dim, factor: float = 1.0) -> list[float]:
    """Parse a two-element ``[lo, hi]`` range to SI floats.

    Args:
        v: List or tuple of two numbers or unit strings.
        dim: Physical dimension of the entries (see :mod:`suspkin.units`).
        factor: SI multiplier applied to bare numbers (from a ``units:`` key).

    Returns:
        ``[lo, hi]`` in SI units.

    Raises:
        ValueError: If ``v`` is not a two-element list or an entry fails to parse.
    """
    return [parse_scalar(x, dim, factor) for x in _seq(v, 2, "range")]


def _vec(v: Any, dim: Dim, factor: float = 1.0) -> list[float]:
    return [parse_scalar(x, dim, factor) for x in _seq(v, 3, "vector")]


def _unit_factor(units: Any) -> float:
    if units is None:
        return 1.0
    if not isinstance(units, str):
        raise ValueError(f"units must be a string like 'mm', got {units!r}")
    return parse_quantity(f"1 {units}", "length")


# --------------------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------------------


class BoxRegion(Frozen):
    """Axis-aligned box in the vehicle frame."""

    kind: Literal["box"] = "box"
    lo: Vec3
    hi: Vec3

    @model_validator(mode="after")
    def _check(self) -> BoxRegion:
        if any(a >= b for a, b in zip(self.lo, self.hi, strict=True)):
            raise ValueError(f"box min must be below max on every axis, got {self.lo}, {self.hi}")
        return self


class HexahedronRegion(Frozen):
    """Hexahedron by 8 vertices (order: see module docstring)."""

    kind: Literal["hexahedron"] = "hexahedron"
    vertices: Annotated[tuple[Vec3, ...], Field(min_length=8, max_length=8)]

    @model_validator(mode="after")
    def _check(self) -> HexahedronRegion:
        check_hexahedron(self.vertices)
        return self


class OffsetRegion(Frozen):
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


class SphereRegion(Frozen):
    """Sphere (keep-in or keep-out)."""

    kind: Literal["sphere"] = "sphere"
    center: Vec3
    radius: float = Field(gt=0)


class CylinderRegion(Frozen):
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
            "radius": parse_scalar(body.pop("radius", None), "length", f),
        }
    elif kind == "cylinder":
        out = {
            "kind": kind,
            "p0": _vec(body.pop("p0", None), "length", f),
            "p1": _vec(body.pop("p1", None), "length", f),
            "radius": parse_scalar(body.pop("radius", None), "length", f),
        }
    else:  # offset
        out = {"kind": kind}
        for a in _AXES:
            if a in body:
                out[a] = parse_pair(body.pop(a), "length", f)
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


def parse_resolution(v: Any) -> Any:
    """``0.5 mm`` / number (SI) / ``None`` -> metres or ``None``; must be positive."""
    if v is None:
        return None
    out = parse_scalar(v, "length")
    if not out > 0.0:
        raise ValueError(f"resolution must be a positive length, got {v!r}")
    return out


class Variable(Frozen):
    """A hardpoint that moves inside ``domain`` (left point; the right side mirrors it)."""

    point: Annotated[
        str, BeforeValidator(lambda v: check_point_ref(v) if isinstance(v, str) else v)
    ]
    domain: Domain
    asymmetric: bool = False
    side: Literal["left", "right"] | None = None
    resolution: float | None = None
    """Grid override (m) of this variable; an explicit ``null`` means continuous."""

    @field_validator("resolution", mode="before")
    @classmethod
    def _resolution(cls, v: Any) -> Any:
        return parse_resolution(v)

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


class CornerSweep(Frozen):
    """Sweep of one corner over wheel travel (bump positive)."""

    kind: Literal["corner"]
    corner: Literal["FL", "FR", "RL", "RR"]
    travel: Pair
    steps: int = Field(ge=3)

    @field_validator("travel", mode="before")
    @classmethod
    def _travel(cls, v: Any) -> Any:
        return parse_pair(v, "length")

    @model_validator(mode="after")
    def _check(self) -> CornerSweep:
        if not self.travel[0] < self.travel[1]:
            raise ValueError("travel must be [lo, hi] with lo < hi")
        return self


class VehicleSweep(Frozen):
    """Sweep of one body input (heave, roll, pitch or rack) with the others at ``base``."""

    kind: Literal["vehicle"]
    input: Literal["heave", "roll", "pitch", "rack"]
    range: Pair | Literal["full"]
    steps: int = Field(ge=3)
    base: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _units(cls, raw: Any) -> Any:
        if not isinstance(raw, Mapping) or raw.get("input") not in VEHICLE_INPUT_DIM:
            return raw
        out = dict(raw)
        dim = VEHICLE_INPUT_DIM[raw["input"]]
        if "range" in out and out["range"] != "full":
            out["range"] = parse_pair(out["range"], dim)
        if isinstance(out.get("base"), Mapping):
            base = {}
            for k, v in out["base"].items():
                if k not in VEHICLE_INPUT_DIM:
                    raise ValueError(f"unknown base key {k!r}; expected {list(VEHICLE_INPUT_DIM)}")
                base[k] = parse_scalar(v, VEHICLE_INPUT_DIM[k])
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
    return "length" if isinstance(sweep, CornerSweep) else VEHICLE_INPUT_DIM[sweep.input]


def sweep_range(sweep: CornerSweep | VehicleSweep, vehicle: Vehicle | None = None) -> Pair | None:
    """SI range of a sweep; ``None`` for ``range: full`` without a vehicle.

    Args:
        sweep: Corner or vehicle sweep.
        vehicle: Needed to resolve ``range: full`` (rack travel); optional otherwise.

    Returns:
        ``(lo, hi)`` in SI units, or ``None`` when ``range: full`` cannot be resolved.
    """
    if isinstance(sweep, CornerSweep):
        return sweep.travel
    if sweep.range != "full":
        return sweep.range
    st = None if vehicle is None else vehicle.front.steering
    return None if st is None else (float(st.rack_travel[0]), float(st.rack_travel[1]))


def sweep_metric_dim(sweep: CornerSweep | VehicleSweep, metric: str) -> Dim:
    """Dimension of ``metric`` in the result table of ``sweep``.

    Args:
        sweep: Corner or vehicle sweep.
        metric: Column name of the sweep's result table.

    Returns:
        Physical dimension of that column (SI).
    """
    return (
        corner_metric_dim(metric) if isinstance(sweep, CornerSweep) else vehicle_metric_dim(metric)
    )
