"""Pydantic models of the vehicle geometry (ISO 8855, SI units, angles in rad).

Hardpoints are given for the LEFT side (y > 0); the right side is mirrored.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vec3 = tuple[float, float, float]

HARDPOINT_NAMES: tuple[str, ...] = (
    "uca_front",
    "uca_rear",
    "uca_outer",
    "lca_front",
    "lca_rear",
    "lca_outer",
    "tie_inner",
    "tie_outer",
    "pushrod_outer",
    "pushrod_inner",
    "rocker_pivot",
    "rocker_spring",
    "spring_chassis",
    "wheel_center",
)

_Y_NONNEG = ("tie_inner", "rocker_pivot", "rocker_spring", "spring_chassis", "pushrod_inner")
_MIN_LINK = 0.01
_OUTER_POINTS = ("uca_outer", "lca_outer", "tie_outer", "pushrod_outer", "wheel_center")


def _norm(v: Vec3) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _normalized(v: Vec3, what: str) -> Vec3:
    n = _norm(v)
    if n < 1e-9:
        raise ValueError(f"{what} must be non-zero")
    return (v[0] / n, v[1] / n, v[2] / n)


def _dist(a: Vec3, b: Vec3) -> float:
    return _norm((a[0] - b[0], a[1] - b[1], a[2] - b[2]))


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CornerHardpoints(_Frozen):
    """Hardpoints of one corner (left side convention)."""

    uca_front: Vec3
    uca_rear: Vec3
    uca_outer: Vec3
    lca_front: Vec3
    lca_rear: Vec3
    lca_outer: Vec3
    tie_inner: Vec3
    tie_outer: Vec3
    pushrod_outer: Vec3
    pushrod_inner: Vec3
    rocker_pivot: Vec3
    rocker_spring: Vec3
    spring_chassis: Vec3
    wheel_center: Vec3
    rocker_arb: Vec3 | None = None
    rocker_axis: Vec3

    @field_validator("rocker_axis")
    @classmethod
    def _normalize_axis(cls, v: Vec3) -> Vec3:
        return _normalized(v, "rocker_axis")

    @model_validator(mode="after")
    def _check(self) -> CornerHardpoints:
        for name in HARDPOINT_NAMES:
            y = getattr(self, name)[1]
            if name in _Y_NONNEG:
                if y < 0:
                    raise ValueError(f"{name} must have y >= 0 (left side), got {y}")
            elif y <= 0:
                raise ValueError(f"{name} must have y > 0 (left side), got {y}")
        if self.rocker_arb is not None and self.rocker_arb[1] < 0:
            raise ValueError(f"rocker_arb must have y >= 0 (left side), got {self.rocker_arb[1]}")
        links = {
            "uca_front-uca_outer": (self.uca_front, self.uca_outer),
            "uca_rear-uca_outer": (self.uca_rear, self.uca_outer),
            "lca_front-lca_outer": (self.lca_front, self.lca_outer),
            "lca_rear-lca_outer": (self.lca_rear, self.lca_outer),
            "tie rod": (self.tie_inner, self.tie_outer),
            "pushrod": (self.pushrod_inner, self.pushrod_outer),
        }
        for name, (a, b) in links.items():
            if _dist(a, b) <= _MIN_LINK:
                raise ValueError(f"link {name} shorter than {_MIN_LINK} m")
        if self.uca_outer[2] <= self.lca_outer[2]:
            raise ValueError("uca_outer z must be above lca_outer z")
        if _dist(self.uca_front, self.uca_rear) <= _MIN_LINK:
            raise ValueError(f"uca_front and uca_rear closer than {_MIN_LINK} m (degenerate axis)")
        if _dist(self.lca_front, self.lca_rear) <= _MIN_LINK:
            raise ValueError(f"lca_front and lca_rear closer than {_MIN_LINK} m (degenerate axis)")
        rocker_points: dict[str, Vec3 | None] = {
            "pushrod_inner": self.pushrod_inner,
            "rocker_spring": self.rocker_spring,
            "rocker_arb": self.rocker_arb,
        }
        for name, p in rocker_points.items():
            if p is None:
                continue
            v = (
                p[0] - self.rocker_pivot[0],
                p[1] - self.rocker_pivot[1],
                p[2] - self.rocker_pivot[2],
            )
            a = self.rocker_axis
            along = v[0] * a[0] + v[1] * a[1] + v[2] * a[2]
            perp = _norm((v[0] - along * a[0], v[1] - along * a[1], v[2] - along * a[2]))
            if perp <= _MIN_LINK:
                raise ValueError(
                    f"{name} lies within {_MIN_LINK} m of the rocker axis (degenerate rocker arm)"
                )
        return self


class Wheel(_Frozen):
    """Wheel and tyre data."""

    tyre_radius: float = Field(gt=0)
    rim_width: float = Field(gt=0)
    static_camber: float = Field(gt=-0.2, lt=0.2)
    static_toe: float = Field(gt=-0.1, lt=0.1)
    tyre_vertical_stiffness: float = Field(gt=0)


class Coilover(_Frozen):
    """Spring/damper unit."""

    spring_rate: float = Field(gt=0)
    free_length: float = Field(gt=0)
    preload: float = Field(ge=0)
    length_min: float = Field(gt=0)
    length_max: float

    @model_validator(mode="after")
    def _check(self) -> Coilover:
        if self.length_max <= self.length_min:
            raise ValueError("length_max must be greater than length_min")
        return self


class TorsionalARB(_Frozen):
    """Torsion-bar anti-roll bar."""

    type: Literal["torsional"]
    arm_end: Vec3
    bar_pivot: Vec3
    bar_axis: Vec3 = (0.0, 1.0, 0.0)
    bar_length: float = Field(gt=0)
    outer_diameter: float = Field(gt=0)
    inner_diameter: float = Field(ge=0)
    shear_modulus: float = Field(gt=0)

    @field_validator("bar_axis")
    @classmethod
    def _normalize_axis(cls, v: Vec3) -> Vec3:
        return _normalized(v, "bar_axis")

    @model_validator(mode="after")
    def _check(self) -> TorsionalARB:
        if self.inner_diameter >= self.outer_diameter:
            raise ValueError("inner_diameter must be smaller than outer_diameter")
        return self


class FlexuralARB(_Frozen):
    """Blade-type (flexural) anti-roll bar."""

    type: Literal["flexural"]
    arm_end: Vec3
    blade_root: Vec3
    blade_pivot_axis: Vec3
    blade_length: float = Field(gt=0)
    blade_width: float = Field(gt=0)
    blade_thickness: float = Field(gt=0)
    youngs_modulus: float = Field(gt=0)

    @field_validator("blade_pivot_axis")
    @classmethod
    def _normalize_axis(cls, v: Vec3) -> Vec3:
        return _normalized(v, "blade_pivot_axis")


class Steering(_Frozen):
    """Rack-and-pinion steering.

    Positive rack travel moves both ``tie_inner`` points along ``+rack_axis``
    (toward +y, left). For front-steer (tie rods ahead of the axle) that steers
    left, i.e. positive steer (plan section 2). The right corner uses the same
    global ``rack_axis`` (not mirrored), since the rack is one rigid part.
    """

    c_factor: float = Field(gt=0)
    rack_travel: tuple[float, float]
    rack_axis: Vec3 = (0.0, 1.0, 0.0)

    @field_validator("rack_axis")
    @classmethod
    def _normalize_axis(cls, v: Vec3) -> Vec3:
        return _normalized(v, "rack_axis")

    @model_validator(mode="after")
    def _check(self) -> Steering:
        if not self.rack_travel[0] < 0 < self.rack_travel[1]:
            raise ValueError("rack_travel must satisfy min < 0 < max")
        return self


class Axle(_Frozen):
    """One axle: left-side hardpoints plus components.

    ``pushrod_mount`` selects the member that carries ``pushrod_outer``: the
    ``"upright"`` (default), the lower wishbone ``"lca"`` or the upper wishbone
    ``"uca"``.
    """

    brakes: Literal["inboard", "outboard"] = "outboard"
    pushrod_mount: Literal["upright", "lca", "uca"] = "upright"
    unsprung_mass: float = Field(gt=0)
    hardpoints: CornerHardpoints
    wheel: Wheel
    coilover: Coilover
    arb: Annotated[TorsionalARB | FlexuralARB | None, Field(discriminator="type")] = None
    steering: Steering | None = None
    right_overrides: dict[str, Vec3] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Axle:
        if self.arb is not None and self.hardpoints.rocker_arb is None:
            raise ValueError("an ARB requires hardpoints.rocker_arb to be set")
        allowed = (*HARDPOINT_NAMES, "rocker_arb")
        for key, val in self.right_overrides.items():
            if key not in allowed:
                raise ValueError(f"right_overrides key {key!r} is not a hardpoint name")
            if val[1] > 0:
                raise ValueError(f"right_overrides[{key!r}] must have y <= 0 (right side)")
            if key in _OUTER_POINTS and val[1] >= 0:
                raise ValueError(f"right_overrides[{key!r}] must have y < 0 (right side)")
            if key == "rocker_arb" and self.hardpoints.rocker_arb is None:
                raise ValueError(
                    "right_overrides has rocker_arb but hardpoints.rocker_arb is unset"
                )
        h = self.hardpoints
        length = _dist(h.rocker_spring, h.spring_chassis)
        if not self.coilover.length_min <= length <= self.coilover.length_max:
            raise ValueError(
                f"static coilover length {length:.4f} m outside "
                f"[{self.coilover.length_min}, {self.coilover.length_max}]"
            )
        if isinstance(self.arb, TorsionalARB):
            span = abs(2.0 * self.arb.bar_pivot[1] - self.arb.bar_length)
            if span >= _MIN_LINK:
                raise ValueError(
                    "torsional ARB: |2 * bar_pivot.y - bar_length| must be < 0.01 m "
                    f"(bar runs between the mirrored pivots), got {span:.4f}"
                )
        return self


class VehicleParams(_Frozen):
    """Vehicle-level parameters."""

    name: str
    mass: float = Field(gt=0)
    cg_height: float = Field(gt=0)
    front_mass_ratio: float = Field(gt=0, lt=1)
    drive: Literal["rwd", "fwd", "awd"] = "rwd"


class Vehicle(_Frozen):
    """Complete vehicle description.

    Hardpoints define the static ride-height pose. ``wheel_center`` z is the
    loaded radius, which may be a little below the unloaded ``tyre_radius``
    (checked within 0.02 m).
    """

    vehicle: VehicleParams
    front: Axle
    rear: Axle

    @model_validator(mode="after")
    def _check(self) -> Vehicle:
        if self.front.steering is None:
            raise ValueError("front axle requires steering")
        if self.rear.steering is not None:
            raise ValueError("rear axle must not have steering")
        fx = self.front.hardpoints.wheel_center[0]
        rx = self.rear.hardpoints.wheel_center[0]
        if fx <= rx:
            raise ValueError("front wheel_center x must be greater than rear wheel_center x")
        if abs(fx) >= 0.005:
            raise ValueError("front wheel_center x must be 0 (origin at mid front axle)")
        for label, axle in (("front", self.front), ("rear", self.rear)):
            z = axle.hardpoints.wheel_center[2]
            if abs(z - axle.wheel.tyre_radius) >= 0.02:
                raise ValueError(
                    f"{label} wheel_center z ({z}) must match tyre_radius "
                    f"({axle.wheel.tyre_radius}) within 20 mm"
                )
        return self

    @property
    def wheelbase(self) -> float:
        """Wheelbase in metres."""
        return self.front.hardpoints.wheel_center[0] - self.rear.hardpoints.wheel_center[0]

    @property
    def front_track(self) -> float:
        """Front track width in metres."""
        return 2.0 * self.front.hardpoints.wheel_center[1]

    @property
    def rear_track(self) -> float:
        """Rear track width in metres."""
        return 2.0 * self.rear.hardpoints.wheel_center[1]
