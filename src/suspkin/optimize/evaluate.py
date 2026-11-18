"""Constraint evaluation of one design: normalised ``g`` values (``g <= 0`` is feasible).

``evaluate(space, x)`` builds the vehicle, runs every sweep that an enabled constraint uses
(at most once per call), and returns an :class:`Evaluation`. Values are SI; display units are
only used by :meth:`Evaluation.display_frame`.

Normalisation: for limits ``[min, max]`` (either may be missing)
``g = max((v - max) / s, (min - v) / s)`` using only the given limits. The scale ``s`` is the
user ``scale`` if given; else ``max - min`` if both limits are given and differ; else a
per-dimension default: length 1 mm, angle 0.1 deg, ratio 0.01, percent 1, force 10 N,
stiffness 1 N/mm; any other dimension ``|limit|`` (a zero limit needs an explicit
``scale``, checked at load).

Semantics per constraint type:

* ``band``: metric over ``where`` (default whole sweep): interpolated values at the ends plus
  the interior samples; ``g`` and ``value`` are those of the worst point.
* ``slope``: least-squares slope of the metric against the sweep input over the same points.
* ``value``: linear interpolation at ``at``.
* ``reach``: the corner is solved over its own travel (13 evenly spaced steps). A solver
  failure gives ``unsolved``. ``g`` is the largest of the active terms: ``min_toggle_margin -
  min(toggle_margin)`` divided by ``toggle_scale`` (default 0.1 deg) and, with ``stroke``,
  ``-min(bump_stroke_remaining)`` and ``-min(droop_stroke_remaining)`` divided by
  ``stroke_scale`` (default 1 mm); the plain ``scale`` key is rejected for reach.
  ``value``/``dim`` belong to the term with the largest ``g`` (named in ``detail``: the
  minimum toggle margin [rad] or the minimum remaining stroke [m]); with no active
  term the constraint only requires the solve (``g = -1``, value = minimum toggle margin).
* ``summary``: dotted field of :func:`suspkin.metrics.vehicle_summary`; ``None`` is
  ``unsolved``.
* ``inside`` / ``outside``: signed distance ``sd`` of the point to the region (``sd <= 0`` is
  inside): ``g = sd / s`` (inside) or ``-sd / s`` (outside); value is the worst ``sd`` [m].
  Without ``over`` the static point is used: the left point, or the right point if the region
  lies at ``y < 0``. With ``over`` the point comes from the states of that corner sweep (as
  solved, i.e. left or right corner coordinates) and the worst state counts.
* ``distance``: ``|a - b|``; with ``over`` the worst over the sweep (a point of another axle
  stays static, left side or the sweep side).
* ``relation``: ``lhs op rhs + offset`` on static left-side coordinates; ``g = (rhs + offset
  - lhs) / s`` for ``>=`` and the negative for ``<=``; value is ``lhs``.
* ``joint_angle``: worst ``joint_angle`` over the sweep states (``axis`` override honoured).

Default scales make ``g`` a "margin in display-sized units" (mm, 0.1 deg, ...), so ``g`` is
not comparable across constraints of different dimensions. Objectives are weighted through
`scale` and `weight`; constraints stay hard constraints (``g <= 0``), so ``g`` scaling only
affects margins, the display and the SLSQP conditioning, never the cost.

Failures: a :class:`~suspkin.kinematics.KinematicsError` (or a non-finite value, a missing
column or point) marks the constraint ``unsolved`` with ``g = G_FAIL``; an invalid design
(:class:`InfeasibleDesign`) marks every enabled constraint ``infeasible`` with ``g = G_FAIL``.

Objectives (``study.objectives``) are evaluated on the same sweep cache, so a sweep still runs
once per call. Each has a raw SI value and a normalised ``f = raw / scale`` (sign-flipped for
``maximize``; the scale is the user ``scale``, else the dimension default, else ``|target|``
for a ``target`` objective). Raw values per type:

* ``curve``: RMS of ``metric - target(x)`` (linear interpolation) over the window (default:
  overlap of the target x range with the sweep); window ends are interpolated.
* ``target``: ``|value - target|`` with the value at ``at`` of a sweep, or a summary metric.
* ``minimize`` / ``maximize``: ``of`` = ``value`` (at ``at``), ``slope`` (least squares),
  ``max``, ``min``, ``max_abs`` or ``mean`` (trapezoidal) over the window; or a summary metric.
* ``linearity``: RMS residual of a least-squares line through the metric over the window.

A failure gives ``f = F_FAIL`` (status ``unsolved`` or ``infeasible``, raw ``nan``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from suspkin.geometry.corners import Corner, build_corners
from suspkin.geometry.models import Vehicle
from suspkin.kinematics import KinematicsError
from suspkin.kinematics.corner import solve_corner, sweep_corner
from suspkin.metrics import add_gradients, vehicle_summary
from suspkin.metrics.joints import body_rotations, joint_angle
from suspkin.optimize.design import DesignSpace, InfeasibleDesign
from suspkin.optimize.metric_registry import summary_metric_dim
from suspkin.optimize.regions import sd_box, sd_cylinder, sd_hexahedron, sd_sphere
from suspkin.optimize.spec import (
    BandConstraint,
    BoxRegion,
    Constraint,
    CornerSweep,
    CurveObjective,
    CylinderRegion,
    DistanceConstraint,
    ExtremumObjective,
    HexahedronRegion,
    InsideConstraint,
    JointAngleConstraint,
    LinearityObjective,
    Objective,
    OutsideConstraint,
    ReachConstraint,
    RelationConstraint,
    SlopeConstraint,
    SphereRegion,
    SummaryConstraint,
    TargetObjective,
    ValueConstraint,
    VehicleSweep,
    default_scale,
    objective_dim,
    objective_scale,
    parse_coord_ref,
    parse_point_ref,
    region_centroid,
    sweep_input_dim,
    sweep_metric_dim,
    sweep_range,
)
from suspkin.sweeps import states_to_frame, sweep_vehicle, vehicle_frame
from suspkin.units import Dim, display_factor, display_unit, slope_dim

__all__ = [
    "F_FAIL",
    "G_FAIL",
    "ConstraintResult",
    "Evaluation",
    "ObjectiveResult",
    "evaluate",
    "failed_evaluation",
]

F64 = NDArray[np.float64]
Status = Literal["pass", "fail", "unsolved", "infeasible", "disabled"]
ObjStatus = Literal["ok", "unsolved", "infeasible"]

G_FAIL = 1e3
F_FAIL = 1e3
_REACH_STEPS = 13
_DEG = math.pi / 180.0
_TOGGLE_SCALE = 0.1 * _DEG
_STROKE_SCALE = 1e-3


class _Unsolved(Exception):  # noqa: N818
    """Internal: the constraint cannot be evaluated (solver failure, NaN, missing data)."""


@dataclass(frozen=True)
class ConstraintResult:
    """Evaluation of one constraint (SI values; ``g <= 0`` is feasible, ``nan`` if disabled)."""

    name: str
    type: str
    value: float
    min: float | None
    max: float | None
    g: float
    status: Status
    dim: Dim
    detail: str = ""


@dataclass(frozen=True)
class ObjectiveResult:
    """Evaluation of one objective: ``raw`` in SI, ``f`` normalised (``F_FAIL`` on failure)."""

    name: str
    type: str
    raw: float
    f: float
    weight: float
    status: ObjStatus
    dim: Dim
    detail: str = ""


@dataclass(frozen=True)
class Evaluation:
    """All constraint and objective results of one design, in YAML order."""

    results: tuple[ConstraintResult, ...]
    objectives: tuple[ObjectiveResult, ...] = ()

    def _enabled(self) -> list[ConstraintResult]:
        return [r for r in self.results if r.status != "disabled"]

    @property
    def feasible(self) -> bool:
        """True if every enabled constraint has ``g <= 0``."""
        return all(r.g <= 0.0 for r in self._enabled())

    @property
    def max_g(self) -> float:
        """Largest ``g`` of the enabled constraints (``-inf`` if there are none)."""
        gs = [r.g for r in self._enabled()]
        return max(gs) if gs else float("-inf")

    def g_vector(self) -> F64:
        """``g`` of the enabled constraints in YAML order (disabled ones are skipped)."""
        return np.array([r.g for r in self._enabled()], dtype=np.float64)

    def f_vector(self) -> F64:
        """Normalised ``f`` of every objective in YAML order."""
        return np.array([o.f for o in self.objectives], dtype=np.float64)

    @property
    def f_total(self) -> float:
        """Weighted sum ``sum(weight * f)`` over the objectives (0 if there are none)."""
        return float(sum(o.weight * o.f for o in self.objectives))

    def objectives_frame(self) -> pd.DataFrame:
        """Tidy objective table in SI units: name, type, raw, f, weight, status, dim, detail."""
        cols = ["name", "type", "raw", "f", "weight", "status", "dim", "detail"]
        return pd.DataFrame(
            [
                {
                    "name": o.name,
                    "type": o.type,
                    "raw": o.raw,
                    "f": o.f,
                    "weight": o.weight,
                    "status": o.status,
                    "dim": o.dim,
                    "detail": o.detail,
                }
                for o in self.objectives
            ],
            columns=cols,
        )

    def objectives_display_frame(self) -> pd.DataFrame:
        """Like :meth:`objectives_frame` with ``raw`` in display units and a ``unit`` column."""
        df = self.objectives_frame()
        f = np.array([display_factor(d) for d in df["dim"]], dtype=np.float64)
        df["raw"] = df["raw"].to_numpy(dtype=np.float64) * f
        df["unit"] = [display_unit(d) for d in df["dim"]]
        return df.drop(columns=["dim"])

    def frame(self) -> pd.DataFrame:
        """Tidy table in SI units: name, type, value, min, max, g, status, dim, detail."""
        return pd.DataFrame(
            [
                {
                    "name": r.name,
                    "type": r.type,
                    "value": r.value,
                    "min": np.nan if r.min is None else r.min,
                    "max": np.nan if r.max is None else r.max,
                    "g": r.g,
                    "status": r.status,
                    "dim": r.dim,
                    "detail": r.detail,
                }
                for r in self.results
            ],
            columns=["name", "type", "value", "min", "max", "g", "status", "dim", "detail"],
        )

    def display_frame(self) -> pd.DataFrame:
        """Like :meth:`frame` with value/min/max in display units and a ``unit`` column."""
        df = self.frame()
        f = np.array([display_factor(d) for d in df["dim"]], dtype=np.float64)
        for col in ("value", "min", "max"):
            df[col] = df[col].to_numpy(dtype=np.float64) * f
        df["unit"] = [display_unit(d) for d in df["dim"]]
        return df.drop(columns=["dim"])


# --------------------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------------------


def _scale(user: float | None, dim: Dim, lo: float | None, hi: float | None) -> float:
    if user is not None:
        return float(user)
    if lo is not None and hi is not None and hi != lo:
        return float(hi - lo)
    d = default_scale(dim)
    if d is not None:
        return d
    lims = [abs(v) for v in (lo, hi) if v is not None and v != 0.0]
    return lims[0] if lims else 1.0  # unreachable via load_study (needs an explicit scale)


def _worst(
    v: ArrayLike, lo: float | None, hi: float | None, s: float, signed: float = 1.0
) -> tuple[float, float]:
    """(g, value) of the worst entry of ``v``; ``signed=-1`` flips g (outside constraints)."""
    if isinstance(v, float | int):  # scalar fast path, same arithmetic as the array path
        x = float(v)
        if not math.isfinite(x):
            raise _Unsolved("non-finite value")
        gs = -math.inf
        if hi is not None:
            gs = max(gs, (x - hi) / s)
        if lo is not None:
            gs = max(gs, (lo - x) / s)
        return signed * gs, x
    arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if not np.isfinite(arr).all():
        raise _Unsolved("non-finite value")
    if hi is not None and lo is not None:
        g = np.maximum((arr - hi) / s, (lo - arr) / s)
    elif hi is not None:
        g = (arr - hi) / s
    elif lo is not None:
        g = (lo - arr) / s
    else:
        g = np.full(arr.shape, -np.inf)
    if signed != 1.0:
        g = signed * g
    i = int(np.argmax(g))
    return float(g[i]), float(arr[i])


# --------------------------------------------------------------------------------------
# sweep cache
# --------------------------------------------------------------------------------------


@dataclass
class _CornerData:
    corner: Corner
    states: list[Any]
    frame: pd.DataFrame


@dataclass
class _Ctx:
    study_sweeps: dict[str, CornerSweep | VehicleSweep]
    vehicle: Vehicle
    store: dict[Any, tuple[Any, Exception | None]] = field(default_factory=dict)

    def memo(self, key: Any, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` at most once per key; a failure is cached and re-raised."""
        if key not in self.store:
            try:
                self.store[key] = (fn(), None)
            except (KinematicsError, ValueError, KeyError) as exc:
                self.store[key] = (None, exc)
        val, err = self.store[key]
        if err is not None:
            raise _Unsolved(str(err)) from err
        return val

    def corners(self) -> dict[str, Corner]:
        out: dict[str, Corner] = self.memo("corners", lambda: dict(build_corners(self.vehicle)))
        return out

    def corner_run(
        self, key: Any, corner_id: str, travel: tuple[float, float], steps: int
    ) -> _CornerData:
        def run() -> _CornerData:
            corner = self.corners()[corner_id]
            states = sweep_corner(corner, np.linspace(travel[0], travel[1], steps))
            frame = add_gradients(states_to_frame(states, corner, metrics=True))
            return _CornerData(corner, states, frame)

        data: _CornerData = self.memo(("corner", key), run)
        return data

    def sweep_corner_data(self, name: str) -> _CornerData:
        sw = self.study_sweeps[name]
        if not isinstance(sw, CornerSweep):
            raise _Unsolved(f"sweep {name!r} is not a corner sweep")
        return self.corner_run(name, sw.corner, sw.travel, sw.steps)

    def vehicle_data(self, name: str) -> pd.DataFrame:
        sw = self.study_sweeps[name]
        if not isinstance(sw, VehicleSweep):
            raise _Unsolved(f"sweep {name!r} is not a vehicle sweep")

        def run() -> pd.DataFrame:
            rng = sweep_range(sw, self.vehicle)
            if rng is None:
                raise ValueError(f"sweep {name!r}: range 'full' needs a steering rack")
            vals = np.linspace(rng[0], rng[1], sw.steps)
            states = sweep_vehicle(self.vehicle, **{sw.input: vals}, base=sw.base)
            df = vehicle_frame(states, self.vehicle)
            extra: dict[str, float] = {"wheelbase": float(self.vehicle.wheelbase)}
            st = self.vehicle.front.steering
            if st is not None:
                extra["steer_c_factor"] = float(st.c_factor)
            return df.assign(**extra)

        df: pd.DataFrame = self.memo(("vehicle", name), run)
        return df

    def sweep_table(self, name: str) -> tuple[pd.DataFrame, str]:
        """(frame, input column) of a named sweep."""
        sw = self.study_sweeps[name]
        if isinstance(sw, CornerSweep):
            return self.sweep_corner_data(name).frame, "travel"
        return self.vehicle_data(name), sw.input

    def summary(self) -> Any:
        return self.memo("summary", lambda: vehicle_summary(self.vehicle))

    def joint_series(self, name: str, c: JointAngleConstraint) -> F64:
        data = self.sweep_corner_data(name)

        def run() -> F64:
            static = next(
                (s for s in data.states if abs(s.travel) < 1e-12 and abs(s.rack) < 1e-12),
                None,
            )
            if static is None:
                static = solve_corner(data.corner, 0.0)
            out = []
            for s in data.states:
                rots = body_rotations(s, static, data.corner)
                out.append(joint_angle(s, static, data.corner, c.joint, c.axis, _rots=rots))
            return np.array(out, dtype=np.float64)

        arr: F64 = self.memo(("joint", name, c.joint, c.axis), run)
        return arr


# --------------------------------------------------------------------------------------
# per-constraint evaluation
# --------------------------------------------------------------------------------------


def _column(df: pd.DataFrame, metric: str) -> F64:
    if metric not in df.columns:
        raise _Unsolved(f"metric column {metric!r} missing from the sweep table")
    return df[metric].to_numpy(dtype=np.float64)


def _sorted_xy(x: F64, y: F64) -> tuple[F64, F64]:
    """``(x, y)`` ordered by ``x``; no copy when ``x`` is already strictly increasing."""
    if x.size < 2 or bool((x[1:] > x[:-1]).all()):
        return x, y
    order = np.argsort(x)
    return x[order], y[order]


def _window(x: F64, y: F64, where: tuple[float, float] | None) -> tuple[F64, F64]:
    """Samples of ``y(x)`` within ``where`` plus interpolated values at its ends."""
    x, y = _sorted_xy(x, y)
    if where is None:
        return x, y
    lo, hi = where
    inner = (x > lo) & (x < hi)
    n = int(inner.sum())
    ends = np.interp((lo, hi), x, y)
    xs = np.empty(n + 2)
    ys = np.empty(n + 2)
    xs[0], xs[-1] = lo, hi
    ys[0], ys[-1] = ends[0], ends[1]
    xs[1:-1] = x[inner]
    ys[1:-1] = y[inner]
    return xs, ys


def _interp_at(x: F64, y: F64, at: float) -> float:
    """Linear interpolation of ``y(x)`` at ``at`` (finite data required)."""
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise _Unsolved("non-finite sweep data")
    x, y = _sorted_xy(x, y)
    return float(np.interp(at, x, y))


def _summary_value(ctx: _Ctx, metric: str) -> float:
    obj = ctx.summary()
    for part in metric.split("."):
        obj = getattr(obj, part, None)
    if obj is None:
        raise _Unsolved(f"summary field {metric!r} is undefined")
    return float(obj)


def _sd(region: Any, pts: F64) -> F64:
    if isinstance(region, BoxRegion):
        return sd_box(pts, region.lo, region.hi)
    if isinstance(region, HexahedronRegion):
        return sd_hexahedron(pts, region.vertices)
    if isinstance(region, SphereRegion):
        return sd_sphere(pts, region.center, region.radius)
    assert isinstance(region, CylinderRegion)
    return sd_cylinder(pts, region.p0, region.p1, region.radius)


def _corner_id(axle: str, side: str) -> str:
    return ("F" if axle == "front" else "R") + ("L" if side == "left" else "R")


def _points(ctx: _Ctx, ref: str, over: str | None, side: str) -> F64:
    """``(n, 3)`` positions of a hardpoint: static (n = 1) or over a corner sweep."""
    axle, name = parse_point_ref(ref)
    if over is not None:
        sw = ctx.study_sweeps[over]
        if isinstance(sw, CornerSweep) and (axle == "front") == (sw.corner[0] == "F"):
            data = ctx.sweep_corner_data(over)
            try:
                return np.array([s.points[name] for s in data.states], dtype=np.float64).reshape(
                    -1, 3
                )
            except KeyError as exc:
                raise _Unsolved(f"point {ref!r} is missing from the solved states") from exc
        if isinstance(sw, CornerSweep):
            side = "left" if sw.corner[1] == "L" else "right"
    cid = _corner_id(axle, side)
    corner = ctx.corners()[cid]
    if name not in corner.points:
        raise _Unsolved(f"point {ref!r} is not defined")
    return np.asarray(corner.points[name], dtype=np.float64).reshape(1, 3)


def _eval_one(c: Constraint, ctx: _Ctx) -> ConstraintResult:
    """Evaluate an enabled constraint; raises ``_Unsolved``."""
    lo: float | None = getattr(c, "min", None)
    hi: float | None = getattr(c, "max", None)
    user = c.scale

    def res(
        g: float, v: float, dim: Dim, vmin: float | None, vmax: float | None, detail: str = ""
    ) -> ConstraintResult:
        return ConstraintResult(
            c.name, c.type, v, vmin, vmax, g, "pass" if g <= 0.0 else "fail", dim, detail
        )

    if isinstance(c, BandConstraint | SlopeConstraint | ValueConstraint):
        sw = ctx.study_sweeps[c.sweep]
        df, xcol = ctx.sweep_table(c.sweep)
        x, y = _column(df, xcol), _column(df, c.metric)
        mdim = sweep_metric_dim(sw, c.metric)
        if isinstance(c, ValueConstraint):
            v = _interp_at(x, y, c.at)
            g, val = _worst(v, lo, hi, _scale(user, mdim, lo, hi))
            return res(g, val, mdim, lo, hi)
        xs, ys = _window(x, y, c.where)
        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            raise _Unsolved("non-finite sweep data")
        if isinstance(c, BandConstraint):
            g, val = _worst(ys, lo, hi, _scale(user, mdim, lo, hi))
            return res(g, val, mdim, lo, hi)
        sdim = slope_dim(mdim, sweep_input_dim(sw))
        slope = float(_fit_line(xs, ys)[0])
        g, val = _worst(slope, lo, hi, _scale(user, sdim, lo, hi))
        return res(g, val, sdim, lo, hi)

    if isinstance(c, ReachConstraint):
        key = ("reach", c.corner, c.travel)
        data = ctx.corner_run(key, c.corner, c.travel, _REACH_STEPS)
        terms: list[tuple[float, float, Dim, str]] = []  # (g, value, dim, label)
        tm = float(np.min(_column(data.frame, "toggle_margin")))
        if c.min_toggle_margin is not None:
            s = _scale(
                c.toggle_scale if c.toggle_scale is not None else _TOGGLE_SCALE, "angle", None, None
            )
            terms.append(((c.min_toggle_margin - tm) / s, tm, "angle", "toggle margin"))
        if c.stroke:
            s = _scale(
                c.stroke_scale if c.stroke_scale is not None else _STROKE_SCALE,
                "length",
                None,
                None,
            )
            for col, label in (
                ("bump_stroke_remaining", "bump stroke remaining"),
                ("droop_stroke_remaining", "droop stroke remaining"),
            ):
                rem = float(np.min(_column(data.frame, col)))
                terms.append((-rem / s, rem, "length", label))
        if not np.all(np.isfinite([t[0] for t in terms] + [tm])):
            raise _Unsolved("non-finite reach data")
        if not terms:
            return res(-1.0, tm, "angle", c.min_toggle_margin, None, "solve only")
        g, val, dim, label = max(terms, key=lambda t: t[0])
        return res(g, val, dim, c.min_toggle_margin if dim == "angle" else 0.0, None, label)

    if isinstance(c, SummaryConstraint):
        sdim = summary_metric_dim(c.metric)
        g, val = _worst(_summary_value(ctx, c.metric), lo, hi, _scale(user, sdim, lo, hi))
        return res(g, val, sdim, lo, hi)

    if isinstance(c, InsideConstraint | OutsideConstraint):
        side = "right" if region_centroid(c.region)[1] < 0.0 else "left"
        sd = _sd(c.region, _points(ctx, c.point, c.over, side))
        s = _scale(user, "length", None, None)
        if isinstance(c, InsideConstraint):
            g, val = _worst(sd, None, 0.0, s)
            return res(g, val, "length", None, 0.0)
        g, val = _worst(sd, None, 0.0, s, signed=-1.0)
        # outside: the worst state is the smallest sd (g = -sd / s is largest there)
        return res(g, val, "length", 0.0, None)

    if isinstance(c, DistanceConstraint):
        d = np.linalg.norm(
            _points(ctx, c.a, c.over, "left") - _points(ctx, c.b, c.over, "left"), axis=1
        )
        g, val = _worst(d, lo, hi, _scale(user, "length", lo, hi))
        return res(g, val, "length", lo, hi)

    if isinstance(c, RelationConstraint):
        lhs = _coord(ctx, c.lhs)
        rhs = (_coord(ctx, c.rhs) if isinstance(c.rhs, str) else float(c.rhs)) + c.offset
        s = _scale(user, "length", None, None)
        if c.op == ">=":
            g, val = _worst(lhs, rhs, None, s)
            return res(g, val, "length", rhs, None)
        g, val = _worst(lhs, None, rhs, s)
        return res(g, val, "length", None, rhs)

    assert isinstance(c, JointAngleConstraint)
    ang = ctx.joint_series(c.sweep, c)
    g, val = _worst(ang, lo, hi, _scale(user, "angle", lo, hi))
    return res(g, val, "angle", lo, hi)


def _coord(ctx: _Ctx, ref: str) -> float:
    axle, name, ax = parse_coord_ref(ref)
    return float(_points(ctx, f"{axle}.{name}", None, "left")[0, ax])


def _constraint_dim(c: Constraint, ctx: _Ctx) -> Dim:
    """Best-effort dimension for failed results."""
    try:
        if isinstance(c, BandConstraint | ValueConstraint):
            return sweep_metric_dim(ctx.study_sweeps[c.sweep], c.metric)
        if isinstance(c, SlopeConstraint):
            sw = ctx.study_sweeps[c.sweep]
            return slope_dim(sweep_metric_dim(sw, c.metric), sweep_input_dim(sw))
        if isinstance(c, SummaryConstraint):
            return summary_metric_dim(c.metric)
        if isinstance(c, JointAngleConstraint):
            return "angle"
        if isinstance(c, ReachConstraint):
            return "angle" if c.min_toggle_margin is not None else "length"
    except (KeyError, ValueError):
        pass
    return "length"


def _failed(c: Constraint, ctx: _Ctx, status: Status, detail: str = "") -> ConstraintResult:
    return ConstraintResult(
        c.name,
        c.type,
        float("nan"),
        getattr(c, "min", None),
        getattr(c, "max", None),
        G_FAIL,
        status,
        _constraint_dim(c, ctx),
        detail,
    )


def _fit_line(xs: F64, ys: F64) -> F64:
    """Least-squares line ``[slope, intercept]``; needs at least 2 distinct x values."""
    if xs.size < 2 or not np.min(xs) < np.max(xs):
        raise _Unsolved("window has fewer than 2 distinct x values (slope/linearity undefined)")
    xm = float(xs.mean())
    dx = xs - xm
    sxx = float(dx @ dx)
    if sxx <= 0.0:
        raise _Unsolved("window has fewer than 2 distinct x values (slope/linearity undefined)")
    ym = float(ys.mean())
    slope = float(dx @ (ys - ym)) / sxx
    return np.array([slope, ym - slope * xm])


def _trapz_mean(xs: F64, ys: F64) -> float:
    span = float(xs[-1] - xs[0])
    if span <= 0.0:
        return float(np.mean(ys))
    return float(np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs)) / span)


def _eval_objective(o: Objective, ctx: _Ctx) -> ObjectiveResult:
    """Evaluate an objective; raises ``_Unsolved``."""
    dim = objective_dim(ctx.study_sweeps, o)
    scale = objective_scale(o, dim) or 1.0
    detail = ""
    if o.sweep is None:
        assert isinstance(o, TargetObjective | ExtremumObjective)
        v = _summary_value(ctx, o.metric)
        raw = abs(v - o.target) if isinstance(o, TargetObjective) else v
    else:
        df, xcol = ctx.sweep_table(o.sweep)
        x, y = _column(df, xcol), _column(df, o.metric)
        if isinstance(o, TargetObjective):
            assert o.at is not None
            raw = abs(_interp_at(x, y, o.at) - o.target)
        elif isinstance(o, ExtremumObjective) and o.of == "value":
            assert o.at is not None
            raw = _interp_at(x, y, o.at)
            detail = "value"
        else:
            where = o.where
            if isinstance(o, CurveObjective) and where is None:
                lo, hi = max(float(np.min(x)), o.target.x[0]), min(float(np.max(x)), o.target.x[-1])
                if lo >= hi:
                    raise _Unsolved("target x range does not overlap the sweep")
                where = (lo, hi)
            xs, ys = _window(x, y, where)
            if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
                raise _Unsolved("non-finite sweep data")
            if isinstance(o, CurveObjective):
                tgt = np.interp(xs, np.array(o.target.x), np.array(o.target.y))
                raw = float(np.sqrt(np.mean((ys - tgt) ** 2)))
            elif isinstance(o, LinearityObjective):
                slope, icpt = _fit_line(xs, ys)
                resid = ys - (slope * xs + icpt)
                raw = float(np.sqrt(np.mean(resid**2)))
            else:
                assert isinstance(o, ExtremumObjective)
                detail = o.of
                stat = {
                    "slope": lambda: float(_fit_line(xs, ys)[0]),
                    "max": lambda: float(np.max(ys)),
                    "min": lambda: float(np.min(ys)),
                    "max_abs": lambda: float(np.max(np.abs(ys))),
                    "mean": lambda: _trapz_mean(xs, ys),
                }
                raw = stat[o.of]()
    if not math.isfinite(raw):
        raise _Unsolved("non-finite objective value")
    f = (-raw if o.type == "maximize" else raw) / scale
    return ObjectiveResult(o.name, o.type, raw, f, o.weight, "ok", dim, detail)


def _failed_objective(
    o: Objective, ctx: _Ctx, status: ObjStatus, detail: str = ""
) -> ObjectiveResult:
    try:
        dim = objective_dim(ctx.study_sweeps, o)
    except (KeyError, ValueError):
        dim = "length"
    return ObjectiveResult(o.name, o.type, float("nan"), F_FAIL, o.weight, status, dim, detail)


def evaluate(
    space: DesignSpace,
    x: ArrayLike | None = None,
    *,
    vehicle: Vehicle | None = None,
    ctx: _Ctx | None = None,
) -> Evaluation:
    """Evaluate all enabled constraints of ``space.study`` for a design.

    Give either ``x`` (normalised design vector) or ``vehicle``; with neither, the baseline
    vehicle is used. An ``x`` that gives an invalid vehicle makes every enabled constraint
    ``infeasible`` (``g = G_FAIL``).

    ``ctx`` (internal, default ``None``) replaces the FOM sweep context; the surrogate
    evaluator passes a context whose sweeps come from the networks.
    """
    if x is not None and vehicle is not None:
        raise ValueError("give either x or vehicle, not both")
    infeasible = False
    veh = space.baseline
    if vehicle is not None:
        veh = vehicle
    elif x is not None:
        try:
            veh = space.to_vehicle(x)
        except InfeasibleDesign:
            infeasible = True
    study = space.study
    if ctx is None:
        ctx = _Ctx(dict(study.sweeps), veh)
    results: list[ConstraintResult] = []
    for c in study.constraints:
        if not c.enabled:
            results.append(
                ConstraintResult(
                    c.name,
                    c.type,
                    float("nan"),
                    getattr(c, "min", None),
                    getattr(c, "max", None),
                    float("nan"),
                    "disabled",
                    _constraint_dim(c, ctx),
                )
            )
        elif infeasible:
            results.append(_failed(c, ctx, "infeasible"))
        else:
            try:
                results.append(_eval_one(c, ctx))
            except _Unsolved as exc:
                results.append(_failed(c, ctx, "unsolved", str(exc)))
    objectives: list[ObjectiveResult] = []
    for o in study.objectives:
        if infeasible:
            objectives.append(_failed_objective(o, ctx, "infeasible"))
            continue
        try:
            objectives.append(_eval_objective(o, ctx))
        except _Unsolved as exc:
            objectives.append(_failed_objective(o, ctx, "unsolved", str(exc)))
    return Evaluation(tuple(results), tuple(objectives))


def failed_evaluation(space: DesignSpace, detail: str) -> Evaluation:
    """All-failed evaluation (``unsolved``, ``G_FAIL`` / ``F_FAIL``) carrying ``detail``."""
    study = space.study
    ctx = _Ctx(dict(study.sweeps), space.baseline)
    results = tuple(
        _failed(c, ctx, "unsolved", detail)
        if c.enabled
        else ConstraintResult(
            c.name,
            c.type,
            float("nan"),
            getattr(c, "min", None),
            getattr(c, "max", None),
            float("nan"),
            "disabled",
            _constraint_dim(c, ctx),
        )
        for c in study.constraints
    )
    objs = tuple(_failed_objective(o, ctx, "unsolved", detail) for o in study.objectives)
    return Evaluation(results, objs)
