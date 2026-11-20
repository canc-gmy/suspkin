"""The :class:`Study` model, cross-checks and the YAML loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from suspkin.geometry.io import VehicleConfigError, load_vehicle
from suspkin.geometry.model import Vehicle
from suspkin.optimize.regions import has_grid_node, node_range
from suspkin.optimize.registry import (
    summary_metric_dim,
)
from suspkin.optimize.spec.constraints import (
    BandConstraint,
    Constraint,
    DistanceConstraint,
    InsideConstraint,
    JointAngleConstraint,
    OutsideConstraint,
    RelationConstraint,
    SlopeConstraint,
    SummaryConstraint,
    ValueConstraint,
    _prep_constraint,
)
from suspkin.optimize.spec.objectives import (
    CurveObjective,
    Objective,
    Optimizer,
    _prep_objective,
    _prep_optimizer,
    objective_dim,
    objective_scale,
)
from suspkin.optimize.spec.surrogate import SurrogateSpec
from suspkin.optimize.spec.variables import (
    _RANGE_TOL,
    DEFAULT_RESOLUTION,
    BoxRegion,
    CornerSweep,
    CylinderRegion,
    HexahedronRegion,
    SphereRegion,
    StudyConfigError,
    Sweep,
    Variable,
    VehicleSweep,
    _Frozen,
    _parse_resolution,
    default_scale,
    parse_point_ref,
    sweep_input_dim,
    sweep_metric_dim,
    sweep_range,
)
from suspkin.units import slope_dim

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
