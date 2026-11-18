"""Surrogate evaluator: network curves -> exact corner states -> the v0.8 constraint evaluator.

Experimental (see ``docs/guides/stability.md``).

``SurrogateEvaluator.evaluate(X)`` takes a batch of design vectors and returns the same
:class:`~suspkin.optimize.evaluate.Evaluation` objects as the FOM evaluator:

1. each design is mapped to a vehicle (``DesignSpace.to_vehicle``) and the per-axle design
   vectors go through the axle predictors in one batch (curves ``Y`` and validity ``p``);
2. **corner level:** ``theta_uca``, ``theta_lca``, ``psi`` and the rocker angle are
   interpolated (cubic Hermite, rack 0) at the travel values of every corner sweep and the
   hardpoint positions are rebuilt with the exact forward kinematics of
   :mod:`suspkin.kinematics.corner` (vectorised over designs and sweep states; the
   rocker uses its closed form with the predicted angle as branch guess). Every corner
   column (metrics, gradients, joint angles, point coordinates) then follows from the
   rebuilt points with the formulas of :mod:`suspkin.metrics.batch`;
3. **vehicle level:** sweeps and ``summary`` come from :mod:`suspkin.surrogate.vehicle_layer`;
4. the unchanged evaluator runs per design on a context that serves these tables.

A node with predicted validity below :data:`P_VALID_MIN`, or any non-finite value a sweep
needs, makes the dependent constraints and objectives ``unsolved`` (FOM semantics). A
vehicle-sweep column the layer does not provide makes its constraint ``unsolved`` too.

Private helpers of ``optimize.evaluate``, ``geometry.corners`` and ``metrics`` are imported
on purpose: this module is the only other implementer of the sweep context.
"""

# ruff: noqa: N803  (X follows the dataset contract)
from __future__ import annotations

import gc
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from suspkin.geom_math.vectors import wheel_spin_axis
from suspkin.geometry.corners import Corner, _build
from suspkin.geometry.models import Axle, Vehicle
from suspkin.metrics.batch import _ic_arrays, _swing, roll_line_arrays
from suspkin.metrics.joints import _BODY_POINTS, _LINKS, _housing
from suspkin.metrics.table import GRADIENTS
from suspkin.optimize.design import DesignSpace, InfeasibleDesign
from suspkin.optimize.evaluate import (
    Evaluation,
    _CornerData,
    _Ctx,
    _Unsolved,
    evaluate,
    failed_evaluation,
)
from suspkin.optimize.spec import (
    CornerSweep,
    JointAngleConstraint,
    ReachConstraint,
    StudyConfigError,
    SummaryConstraint,
    VehicleSweep,
    parse_point_ref,
    sweep_range,
)
from suspkin.surrogate.channels import channels_for
from suspkin.surrogate.extract import extract_corner
from suspkin.surrogate.grid import AxleGrid
from suspkin.surrogate.sampling import AxleSpace
from suspkin.surrogate.vehicle_layer import (
    AxleBatch,
    LayerResult,
    VehicleStatics,
    vehicle_layer,
    vehicle_statics,
)

if TYPE_CHECKING:
    from suspkin.surrogate.predict import SurrogatePredictor

Arr = NDArray[np.float64]
P_VALID_MIN = 0.5
_NAN = float("nan")
_TOL = 1e-9
_STATE_CHANNELS = ("theta_uca", "theta_lca", "psi", "rocker_angle")
_POINT_RE = re.compile(r"^(.+)_([xyz])$")
_AXLES: tuple[Literal["front", "rear"], ...] = ("front", "rear")
#: Hardpoints that enter ``vehicle_statics`` (the others never change its output).
_STATICS_POINTS = frozenset({"wheel_center", "rocker_pivot", "rocker_arb", "rocker_axis"})
_T = TypeVar("_T")

__all__ = [
    "P_VALID_MIN",
    "AxlePredictor",
    "BaselineAxle",
    "NetworkPredictor",
    "OraclePredictor",
    "SurrogateEvaluator",
]


# --------------------------------------------------------------------------------------
# predictors
# --------------------------------------------------------------------------------------


class AxlePredictor(Protocol):
    """Anything that maps axle design vectors to curves: ``predict(X) -> (Y, p_valid)``.

    ``Y`` is ``(B, C, R, T)`` in the physical units of ``channels``; ``p_valid`` is
    ``(B, R, T)`` (a node is valid when it is at least :data:`P_VALID_MIN`).
    """

    channels: tuple[str, ...]
    travel: Arr
    rack: Arr

    def predict(self, X: Arr) -> tuple[Arr, Arr]:
        """Curves and validity probabilities of the rows of ``X`` (unit cube, axle slots)."""
        ...


class NetworkPredictor:
    """Adapter giving a trained :class:`SurrogatePredictor` the :class:`AxlePredictor` shape."""

    def __init__(self, inner: SurrogatePredictor) -> None:
        self.inner = inner
        self.channels: tuple[str, ...] = tuple(inner.model.channels)
        self.travel: Arr = np.asarray(inner.model.travel, dtype=np.float64)
        self.rack: Arr = np.asarray(inner.model.rack, dtype=np.float64)

    def predict(self, X: Arr) -> tuple[Arr, Arr]:
        """Forward pass of the network."""
        return self.inner.predict(X)


class OraclePredictor:
    """Test double: the FOM channels of :func:`extract_corner` as the "prediction".

    ``p_valid`` is the kinematic mask (0 / 1). The other axle stays at the baseline.
    """

    def __init__(self, space: DesignSpace, axle: Literal["front", "rear"], grid: AxleGrid) -> None:
        self.space = space
        self.axle = axle
        self.grid = grid
        self._asp = AxleSpace(space, axle)
        self.channels = tuple(c.name for c in channels_for(axle))
        self.travel = grid.travel
        self.rack = grid.rack

    def predict(self, X: Arr) -> tuple[Arr, Arr]:
        """Run the FOM per row; rows that cannot be built are all-invalid."""
        r, t = self.grid.shape
        ys: list[Arr] = []
        ms: list[Arr] = []
        for x in np.atleast_2d(X):
            try:
                veh = self.space.to_vehicle(self._asp.to_full(x))
            except InfeasibleDesign:
                ys.append(np.full((len(self.channels), r, t), _NAN))
                ms.append(np.zeros((r, t)))
                continue
            s = extract_corner(veh, self.axle, self.grid)
            ys.append(s.Y)
            ms.append(s.mask.astype(np.float64))
        if not ys:
            return np.zeros((0, len(self.channels), r, t)), np.zeros((0, r, t))
        return np.stack(ys), np.stack(ms)


class BaselineAxle:
    """An axle without design variables: its curves are the baseline's, computed once."""

    def __init__(self, space: DesignSpace, axle: Literal["front", "rear"], grid: AxleGrid) -> None:
        self.channels = tuple(c.name for c in channels_for(axle))
        self.travel = grid.travel
        self.rack = grid.rack
        s = extract_corner(space.baseline, axle, grid)
        self._y = s.Y
        self._p = s.mask.astype(np.float64)

    def predict(self, X: Arr) -> tuple[Arr, Arr]:
        """The baseline curves, repeated for every row."""
        b = np.atleast_2d(X).shape[0]
        return np.repeat(self._y[None], b, axis=0), np.repeat(self._p[None], b, axis=0)


# --------------------------------------------------------------------------------------
# small vector helpers (last axis = xyz)
# --------------------------------------------------------------------------------------


def _cross(a: Arr, b: Arr) -> Arr:
    out: Arr = np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )
    return out


def _dot(a: Arr, b: Arr) -> Arr:
    out: Arr = np.sum(a * b, axis=-1, keepdims=True)
    return out


def _unit(a: Arr) -> Arr:
    out: Arr = a / np.linalg.norm(a, axis=-1, keepdims=True)
    return out


def _rod(k: Arr, cs: Arr, sn: Arr, v: Arr) -> Arr:
    """Rotate ``v`` about the unit axis ``k`` (``cs`` / ``sn`` = cos / sin of the angle)."""
    out: Arr = v * cs + _cross(k, v) * sn + k * _dot(k, v) * (1.0 - cs)
    return out


def _wrap(a: Arr) -> Arr:
    out: Arr = (a + np.pi) % (2.0 * np.pi) - np.pi
    return out


def _norm(a: Arr) -> Arr:
    out: Arr = np.sqrt(np.einsum("ij,ij->i", a, a))
    return out


def _grad(y: Arr, x: Arr) -> Arr:
    """``np.gradient`` along axis 1 (typed as an array)."""
    out: Arr = np.gradient(y, x, axis=1)
    return out


def _rowdot(a: Arr, b: Arr) -> Arr:
    out: Arr = np.einsum("ij,ij->i", a, b)
    return out


def _tile(obj: _T, n: int) -> _T:
    """``obj`` (a statics dataclass with ``B = 1``) repeated to ``n`` designs."""
    kw: dict[str, Any] = {}
    for f in fields(obj):  # type: ignore[arg-type]
        v = getattr(obj, f.name)
        if isinstance(v, np.ndarray):
            kw[f.name] = np.repeat(v, n, axis=0)
        elif is_dataclass(v):
            kw[f.name] = _tile(v, n)
    return replace(obj, **kw)  # type: ignore[type-var]


# --------------------------------------------------------------------------------------
# curve interpolation
# --------------------------------------------------------------------------------------


def hermite(y: Arr, x: Arr, q: Arr) -> Arr:
    """Local cubic Hermite interpolation of ``y (..., L)`` along the last axis.

    Slopes by ``np.gradient`` (as the vehicle layer and ``add_gradients``). The result is
    ``(..., len(q))``; NaN outside ``[x[0], x[-1]]`` (``1e-9`` tolerance) and wherever a node
    that the interval or its slopes touch is NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    ok = (q >= x[0] - _TOL) & (q <= x[-1] + _TOL)
    if x.size == 1:
        out1: Arr = np.where(np.abs(q - x[0]) <= _TOL, y[..., :1], _NAN)
        return out1
    m = np.gradient(y, x, axis=-1)
    idx = np.clip(np.searchsorted(x, q, side="right") - 1, 0, x.size - 2)
    x0 = x[idx]
    h = x[idx + 1] - x0
    t = (q - x0) / h
    t2 = t * t
    t3 = t2 * t
    out: Arr = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y[..., idx]
        + (t3 - 2.0 * t2 + t) * h * m[..., idx]
        + (3.0 * t2 - 2.0 * t3) * y[..., idx + 1]
        + (t3 - t2) * h * m[..., idx + 1]
    )
    return np.where(ok, out, _NAN)


def _rack0_curves(batch: AxleBatch, names: Sequence[str]) -> Arr:
    """Channels ``names`` versus travel at rack 0, ``(B, len(names), T)``; NaN at masked nodes."""
    ix = [batch.channels.index(n) for n in names]
    y: Arr = np.where(batch.mask[:, None], batch.Y[:, ix], _NAN)  # (B, K, R, T)
    nodes = batch.grid.rack
    j = int(np.argmin(np.abs(nodes)))
    if abs(nodes[j]) <= _TOL:
        out: Arr = y[:, :, j, :]
        return out
    if nodes.size == 1:
        return np.full(y[:, :, 0, :].shape, _NAN)
    red: Arr = hermite(y.transpose(0, 1, 3, 2), nodes, np.zeros(1))[..., 0]
    return red


# --------------------------------------------------------------------------------------
# exact forward kinematics, vectorised over designs and states
# --------------------------------------------------------------------------------------


@dataclass
class _CStatic:
    """Static corner data of ``B`` designs (side-resolved hardpoints, ``(B, 3)``)."""

    cid: str
    side: str
    axle: str
    sign: float
    p: dict[str, Arr]
    axis: Arr
    cfg: Axle
    spin0: Arr
    scp: Arr
    len0: Arr
    corner0: Corner


def _make_static(corners: Sequence[Corner]) -> _CStatic:
    c0 = corners[0]
    p = {
        n: np.array([c.points[n] for c in corners], dtype=np.float64).reshape(-1, 3)
        for n in c0.points
    }
    cfg = c0.axle_cfg
    spin0 = wheel_spin_axis(c0.side, cfg.wheel.static_camber, cfg.wheel.static_toe)
    fz = spin0[2]
    up = _unit(np.array([-fz * spin0[0], -fz * spin0[1], 1.0 - fz * spin0[2]]))
    scp = p["wheel_center"] - cfg.wheel.tyre_radius * up
    len0 = np.linalg.norm(p["rocker_spring"] - p["spring_chassis"], axis=-1)
    return _CStatic(
        c0.id,
        c0.side,
        c0.axle,
        1.0 if c0.side == "left" else -1.0,
        p,
        np.array([c.rocker_axis for c in corners], dtype=np.float64).reshape(-1, 3),
        cfg,
        np.asarray(spin0, dtype=np.float64),
        scp,
        len0,
        c0,
    )


@dataclass
class _Rebuilt:
    """Rebuilt corner states: ``(B, K, ...)`` arrays of the moved points and solver outputs."""

    st: _CStatic
    moved: dict[str, Arr]
    spin: Arr
    coil: Arr
    toggle: Arr
    rocker: Arr
    theta_uca: Arr
    theta_lca: Arr
    psi: Arr

    def pt(self, name: str) -> Arr:
        """Point ``name`` as ``(B, K, 3)`` (static points are broadcast)."""
        if name in self.moved:
            return self.moved[name]
        if name not in self.st.p:
            raise KeyError(name)
        a = self.st.p[name]
        return np.broadcast_to(a[:, None, :], (a.shape[0], self.rocker.shape[1], 3))


def _rebuild(st: _CStatic, th_u: Arr, th_l: Arr, psi: Arr, guess: Arr) -> _Rebuilt:
    """Vectorised ``corner._assemble`` at rack 0 (``th_*``, ``psi``, ``guess``: ``(B, K)``).

    Invalid states (flipped upright, pushrod out of reach, non-finite angles) give NaN.
    """

    def b3(a: Arr) -> Arr:
        return a[:, None, :]

    s = st.sign
    p = st.p
    with np.errstate(all="ignore"):
        uf, lf = b3(p["uca_front"]), b3(p["lca_front"])
        uca_hat = _unit(b3(p["uca_rear"] - p["uca_front"]))
        lca_hat = _unit(b3(p["lca_rear"] - p["lca_front"]))
        cu, su = np.cos(s * th_u)[..., None], np.sin(s * th_u)[..., None]
        cl, sl = np.cos(s * th_l)[..., None], np.sin(s * th_l)[..., None]
        ubj = uf + _rod(uca_hat, cu, su, b3(p["uca_outer"] - p["uca_front"]))
        lbj = lf + _rod(lca_hat, cl, sl, b3(p["lca_outer"] - p["lca_front"]))
        d = ubj - lbj
        n = _unit(d)
        u0 = _unit(b3(p["uca_outer"] - p["lca_outer"]))
        cc = _dot(u0, n)
        ik = np.where(cc < -1.0 + 1e-12, _NAN, 1.0 / (1.0 + cc))
        w = _cross(u0, n)
        cph, sph = np.cos(s * psi)[..., None], np.sin(s * psi)[..., None]

        def rot(v: Arr) -> Arr:
            wr = _cross(w, v)
            return _rod(n, cph, sph, v + wr + _cross(w, wr) * ik)

        tie_outer = lbj + rot(b3(p["tie_outer"] - p["lca_outer"]))
        wc = lbj + rot(b3(p["wheel_center"] - p["lca_outer"]))
        mount = st.cfg.pushrod_mount
        if mount == "upright":
            po = lbj + rot(b3(p["pushrod_outer"] - p["lca_outer"]))
        elif mount == "lca":
            po = lf + _rod(lca_hat, cl, sl, b3(p["pushrod_outer"] - p["lca_front"]))
        else:
            po = uf + _rod(uca_hat, cu, su, b3(p["pushrod_outer"] - p["uca_front"]))

        # rocker: closed form (RockerCtx.solve), branch nearest to the predicted angle
        pivot = b3(p["rocker_pivot"])
        axis = b3(st.axis)
        k = _unit(axis)
        r_in = b3(p["pushrod_inner"]) - pivot
        w_r = r_in - _dot(r_in, axis) * axis
        ww = _dot(w_r, w_r)
        axw = _cross(axis, w_r)
        length2 = np.sum((p["pushrod_inner"] - p["pushrod_outer"]) ** 2, axis=-1)[:, None, None]
        r = (po - pivot) - _dot(r_in, axis) * axis
        ca, cb = _dot(r, w_r)[..., 0], _dot(r, axw)[..., 0]
        c2 = 0.5 * (_dot(r, r) + ww - length2)[..., 0]
        amp = np.hypot(ca, cb)
        bad = (amp < 1e-12) | (np.abs(c2) > amp * (1.0 + 1e-12))
        half = np.arccos(np.clip(c2 / amp, -1.0, 1.0))
        base = np.arctan2(cb, ca)
        g = s * guess
        c_a, c_b = base + half, base - half
        best = np.where(np.abs(_wrap(c_a - g)) <= np.abs(_wrap(c_b - g)), c_a, c_b)
        phys = np.where(bad, _NAN, g + _wrap(best - g))
        cph_r, sph_r = np.cos(phys)[..., None], np.sin(phys)[..., None]

        def rocker_rot(v: Arr) -> Arr:
            return pivot + _rod(k, cph_r, sph_r, v)

        p_in = rocker_rot(r_in)
        p_spring = rocker_rot(b3(p["rocker_spring"]) - pivot)
        lever = p_in - pivot
        lever = lever - _dot(lever, axis) * axis
        push = po - p_in
        push = push - _dot(push, axis) * axis
        nl, npu = np.linalg.norm(lever, axis=-1), np.linalg.norm(push, axis=-1)
        cosang = np.where(nl * npu > 1e-12, _dot(lever, push)[..., 0] / (nl * npu), 1.0)
        a = np.arccos(np.clip(cosang, -1.0, 1.0))
        toggle = np.minimum(a, np.pi - a)

        spin = _unit(rot(st.spin0[None, None, :]))
        fz = spin[..., 2:3]
        up = _unit(
            np.concatenate(
                [-fz * spin[..., 0:1], -fz * spin[..., 1:2], 1.0 - fz * spin[..., 2:3]], -1
            )
        )
        moved: dict[str, Arr] = {
            "uca_outer": ubj,
            "lca_outer": lbj,
            "tie_outer": tie_outer,
            "wheel_center": wc,
            "pushrod_outer": po,
            "pushrod_inner": p_in,
            "rocker_spring": p_spring,
            "contact_patch": wc - st.cfg.wheel.tyre_radius * up,
        }
        if "rocker_arb" in p:
            moved["rocker_arb"] = rocker_rot(b3(p["rocker_arb"]) - pivot)
        coil = np.linalg.norm(p_spring - b3(p["spring_chassis"]), axis=-1)
    return _Rebuilt(st, moved, spin, coil, toggle, s * phys, th_u, th_l, psi)


# --------------------------------------------------------------------------------------
# corner columns (vectorised ``metrics.batch.corner_arrays`` with per-design statics)
# --------------------------------------------------------------------------------------


def _corner_columns(rb: _Rebuilt, travel: Arr) -> dict[str, Arr]:
    """All ``states_to_frame(metrics=True)`` + ``add_gradients`` columns, each ``(B, K)``."""
    st = rb.st
    b, k = rb.rocker.shape
    m = b * k

    def flat(a: Arr) -> Arr:
        return a.reshape(m, *a.shape[2:])

    def rep(a: Arr) -> Arr:
        return np.repeat(a, k, axis=0)

    cache: dict[str, Arr] = {}

    def pts(name: str) -> Arr:
        if name not in cache:
            cache[name] = flat(rb.pt(name))
        return cache[name]

    o = st.sign
    cfg = st.cfg
    coil = cfg.coilover
    spin = flat(rb.spin)
    length, rocker = flat(rb.coil), flat(rb.rocker)
    scp, swc = rep(st.scp), rep(st.p["wheel_center"])
    len0 = np.repeat(st.len0, k)
    out: dict[str, Arr] = {}
    with np.errstate(all="ignore"):
        cp, wc, lbj = pts("contact_patch"), pts("wheel_center"), pts("lca_outer")
        u = pts("uca_outer") - lbj

        def ground(z: Arr) -> Arr:
            t = (z - lbj[:, 2]) / u[:, 2]
            gr: Arr = lbj + t[:, None] * u
            return gr

        g_cp, g_wc = ground(cp[:, 2]), ground(wc[:, 2])
        un = u / _norm(u)[:, None]
        r = wc - lbj
        perp = r - _rowdot(r, un)[:, None] * un
        dx = cp[:, 0] - scp[:, 0]
        ht = np.abs(cp[:, 1])
        out["camber"] = -np.arcsin(np.clip(spin[:, 2], -1.0, 1.0))
        out["toe"] = np.arctan2(spin[:, 0], o * spin[:, 1])
        out["steer_angle"] = np.arctan2(-o * spin[:, 0], o * spin[:, 1])
        out["caster"] = np.arctan2(-u[:, 0], u[:, 2])
        out["kpi"] = np.arctan2(-o * u[:, 1], u[:, 2])
        out["scrub_radius"] = o * (cp[:, 1] - g_cp[:, 1])
        out["mechanical_trail"] = g_cp[:, 0] - cp[:, 0]
        out["kingpin_offset_wc"] = o * (wc[:, 1] - g_wc[:, 1])
        out["spindle_length"] = _norm(perp)
        out["half_track"] = ht
        out["half_track_change"] = ht - np.abs(scp[:, 1])
        out["contact_patch_dx"] = dx
        out["contact_patch_dy"] = o * (cp[:, 1] - scp[:, 1])
        out["wheel_center_recession"] = wc[:, 0] - swc[:, 0]
        out["wheelbase_change_contribution"] = dx if st.axle == "front" else -dx

        fv = _ic_arrays(pts, 0, cp[:, 0])
        sv = _ic_arrays(pts, 1, cp[:, 1])
        inboard = -1.0 if st.side == "left" else 1.0
        anti = -1.0 if st.axle == "front" else 1.0
        out["fvic_y"] = fv.h
        out["fvic_z"] = fv.z
        out["fvsa"] = np.where(fv.finite, np.abs(fv.h - cp[:, 1]), np.inf)
        out["fv_swing_angle"] = _swing(fv, cp[:, 1], cp[:, 2], inboard)
        out["svic_x"] = sv.h
        out["svic_z"] = sv.z
        out["svsa"] = np.where(sv.finite, np.abs(sv.h - cp[:, 0]), np.inf)
        out["sv_swing_angle"] = _swing(sv, cp[:, 0], cp[:, 2], anti)
        out["sv_wc_angle"] = _swing(sv, wc[:, 0], wc[:, 2], anti)
        rd0, rd1 = roll_line_arrays(cp[:, 1], cp[:, 2], fv)
        out["rc_height"] = np.where(
            np.abs(rd0) < 1e-12, _NAN, cp[:, 2] + rd1 / rd0 * (0.0 - cp[:, 1])
        )
        rod = pts("pushrod_inner") - pts("pushrod_outer")
        out["pushrod_angle"] = np.arctan2(rod[:, 2], np.hypot(rod[:, 0], rod[:, 1]))
        out["pushrod_angle_fv"] = np.arctan2(rod[:, 2], np.abs(rod[:, 1]))
        out["rocker_rotation"] = rocker
        out["coilover_compression"] = len0 - length
        out["bump_stroke_remaining"] = length - coil.length_min
        out["droop_stroke_remaining"] = coil.length_max - length
        out["spring_force"] = coil.preload + coil.spring_rate * (coil.free_length - length)
    out["spring_rate"] = np.full(m, float(coil.spring_rate))
    out["tyre_stiffness"] = np.full(m, float(cfg.wheel.tyre_vertical_stiffness))
    cols = {name: a.reshape(b, k) for name, a in out.items()}

    cols["travel"] = np.broadcast_to(travel, (b, k))
    cols["rack"] = np.zeros((b, k))
    cols["theta_uca"], cols["theta_lca"], cols["psi"] = rb.theta_uca, rb.theta_lca, rb.psi
    cols["rocker_angle"] = rb.rocker
    cols["coilover_length"] = rb.coil
    cols["toggle_margin"] = rb.toggle
    cols["max_residual"] = np.zeros((b, k))
    for i, axname in enumerate("xyz"):
        cols[f"spin_{axname}"] = rb.spin[..., i]
    for metric, name in GRADIENTS["travel"].items():
        cols[name] = _grad(cols[metric], travel)
    mr = _grad(cols["coilover_compression"], travel)
    prog = _grad(mr, travel)
    force, kk, kt = cols["spring_force"], cols["spring_rate"], cols["tyre_stiffness"]
    wr = kk * mr**2 + force * prog
    cols["mr_spring"] = mr
    cols["mr_progression"] = prog
    cols["wheel_load_spring"] = force * mr
    cols["wheel_rate_linear"] = kk * mr**2
    cols["wheel_rate"] = wr
    cols["ride_rate"] = wr * kt / (wr + kt)
    return cols


# --------------------------------------------------------------------------------------
# joint angles (vectorised ``metrics.batch.joint_angle_arrays`` with per-design statics)
# --------------------------------------------------------------------------------------


def _angle(u: Arr, v: Arr) -> Arr:
    out: Arr = np.arctan2(np.linalg.norm(_cross(u, v), axis=-1), np.sum(u * v, axis=-1))
    return out


def _frames(p: Arr) -> Arr:
    """Orthonormal frames ``(..., 3, 3)`` (columns) of the point triples ``p (..., 3, 3)``.

    NaN for coincident or collinear points.
    """
    e1 = _unit(p[..., 1, :] - p[..., 0, :])
    v = p[..., 2, :] - p[..., 0, :]
    w = v - _dot(v, e1) * e1
    nw = np.linalg.norm(w, axis=-1, keepdims=True)
    e2 = np.where(nw > 1e-9 * np.linalg.norm(v, axis=-1, keepdims=True), w / nw, _NAN)
    out: Arr = np.stack([e1, e2, _cross(e1, e2)], axis=-1)
    return out


def _body_rotations(p0: Arr, p1: Arr) -> Arr:
    """Rotations ``(B, K, 3, 3)`` of a rigid body from its static points ``p0 (B, k, 3)`` to
    ``p1 (B, K, k, 3)``.

    The rebuilt bodies are exactly rigid, so the frame of the first three points gives the
    same rotation as a Kabsch fit without the SVD.
    """
    with np.errstate(all="ignore"):
        f0 = _frames(p0[:, :3])[:, None]
        f1 = _frames(p1[:, :, :3])
        rot: Arr = f1 @ f0.swapaxes(-1, -2)
    return rot


def _joint_angles(rb: _Rebuilt, joint: str, axis: ArrayLike | None) -> Arr:
    """Joint misalignment angle ``(B, K)`` of ``joint`` relative to the static pose."""
    st = rb.st
    names = dict(_BODY_POINTS)
    if "rocker_arb" in st.p:
        names["rocker"] = (*names["rocker"], "rocker_arb")
    rots: dict[str, Arr] = {}
    for body, pn in names.items():
        p0 = np.stack([st.p[n] for n in pn], axis=1)
        p1 = np.stack([rb.pt(n) for n in pn], axis=2)
        rots[body] = _body_rotations(p0, p1)
    b, k = rb.rocker.shape
    rots["chassis"] = np.broadcast_to(np.eye(3), (b, k, 3, 3))
    rh = rots[_housing(st.corner0, joint)]
    p = st.p
    with np.errstate(all="ignore"):
        if joint in ("uca_outer", "lca_outer"):
            a0 = (
                _unit(p["uca_outer"] - p["lca_outer"])
                if axis is None
                else np.broadcast_to(_unit(np.asarray(axis, dtype=np.float64)), (b, 3))
            )
            return _angle(
                np.einsum("bkij,bj->bki", rots["upright"], a0), np.einsum("bkij,bj->bki", rh, a0)
            )
        a_name, b_name = _LINKS[joint]
        d0 = _unit(p[b_name] - p[a_name])
        d1 = _unit(rb.pt(b_name) - rb.pt(a_name))
        if axis is None:
            return _angle(d1, np.einsum("bkij,bj->bki", rh, d0))
        a0 = np.broadcast_to(_unit(np.asarray(axis, dtype=np.float64)), (b, 3))
        d0k = d0[:, None, :]
        wv = _cross(d0k, d1)
        carried = (
            a0[:, None, :]
            + _cross(wv, a0[:, None, :])
            + _cross(wv, _cross(wv, a0[:, None, :])) / (1.0 + _dot(d0k, d1))
        )
        return _angle(carried, np.einsum("bkij,bj->bki", rh, a0))


# --------------------------------------------------------------------------------------
# table views served to the evaluator
# --------------------------------------------------------------------------------------


class _Col:
    """Minimal ``pandas.Series`` stand-in: ``to_numpy(dtype)``."""

    __slots__ = ("a",)

    def __init__(self, a: Arr) -> None:
        self.a = a

    def to_numpy(self, dtype: Any = None) -> Arr:
        return np.asarray(self.a, dtype=dtype or np.float64)


class _Cols:
    """Container for ``name in df.columns``."""

    def __init__(self, has: Any, strict_detail: str | None = None) -> None:
        self._has = has
        self._detail = strict_detail

    def __contains__(self, name: object) -> bool:
        ok = bool(self._has(name))
        if not ok and self._detail is not None:
            raise _Unsolved(f"column {name!r} {self._detail}")
        return ok


class _Row:
    """One design's table: column name -> 1-D array (a stand-in for the sweep DataFrame)."""

    def __init__(self, get: Any, has: Any, detail: str | None = None) -> None:
        self._get = get
        self.columns = _Cols(has, detail)

    def __getitem__(self, name: str) -> _Col:
        return _Col(self._get(name))


@dataclass
class _Run:
    """One corner run (corner id, travel grid) for ``B`` designs."""

    rb: _Rebuilt
    travel: Arr
    cols: dict[str, Arr]
    valid: NDArray[np.bool_]
    joints: dict[tuple[str, Any], Arr] = field(default_factory=dict)

    def column(self, name: str) -> Arr | None:
        """``(B, K)`` array of a column, ``None`` if the name is unknown."""
        if name in self.cols:
            return self.cols[name]
        m = _POINT_RE.match(name)
        if m is not None:
            try:
                return self.rb.pt(m.group(1))[..., "xyz".index(m.group(2))]
            except KeyError:
                return None
        return None

    def row(self, i: int) -> _Row:
        def get(name: str) -> Arr:
            a = self.column(name)
            if a is None:
                raise KeyError(name)
            out: Arr = a[i]
            return out

        return _Row(get, lambda n: isinstance(n, str) and self.column(n) is not None)


class _LazyPoints(Mapping[str, Arr]):
    def __init__(self, rb: _Rebuilt, i: int, k: int) -> None:
        self._rb, self._i, self._k = rb, i, k

    def __getitem__(self, name: str) -> Arr:
        out: Arr = self._rb.pt(name)[self._i, self._k]
        return out

    def __iter__(self) -> Iterator[str]:
        yield from (*self._rb.st.p, *self._rb.moved)

    def __len__(self) -> int:
        return len(set(self._rb.st.p) | set(self._rb.moved))


@dataclass
class _StateView:
    travel: float
    points: Mapping[str, Arr]


class _States(Sequence[_StateView]):
    """Lazy states of one design: only ``travel`` and ``points`` are available."""

    def __init__(self, rb: _Rebuilt, i: int, travel: Arr) -> None:
        self._rb, self._i, self._travel = rb, i, travel

    def __len__(self) -> int:
        return int(self._travel.size)

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, slice):
            return [self[j] for j in range(*k.indices(len(self)))]
        return _StateView(float(self._travel[k]), _LazyPoints(self._rb, self._i, int(k)))


class _SummaryView:
    """``VehicleSummary``-like attribute access over the flat layer summary of one design."""

    def __init__(self, flat: Mapping[str, Arr], i: int, prefix: str = "") -> None:
        self._flat, self._i, self._prefix = flat, i, prefix

    def __getattr__(self, name: str) -> Any:
        key = self._prefix + name
        if key in self._flat:
            return float(self._flat[key][self._i])
        if not self._prefix and name in ("front", "rear"):
            return _SummaryView(self._flat, self._i, name + ".")
        return None


# --------------------------------------------------------------------------------------
# batch state and per-design context
# --------------------------------------------------------------------------------------


class _LazyCorners(Mapping[str, Corner]):
    def __init__(self, batch: _Batch, i: int) -> None:
        self._b, self._i = batch, i

    def __getitem__(self, cid: str) -> Corner:
        if cid not in ("FL", "FR", "RL", "RR"):
            raise KeyError(cid)
        return self._b.corner(self._i, cid)

    def __iter__(self) -> Iterator[str]:
        yield from ("FL", "FR", "RL", "RR")

    def __len__(self) -> int:
        return 4


class _Batch:
    """Per-batch shared data: predicted curves, layer output, lazily built corner runs."""

    def __init__(
        self,
        vehicles: list[Vehicle],
        axle_batches: dict[str, AxleBatch],
        layer: LayerResult | None,
        layer_error: str | None,
        vehicle_tables: dict[str, dict[str, Arr]],
        vehicle_errors: dict[str, str],
        timers: dict[str, float],
    ) -> None:
        self.vehicles = vehicles
        self.axle_batches = axle_batches
        self.layer = layer
        self.layer_error = layer_error
        self.vehicle_tables = vehicle_tables
        self.vehicle_errors = vehicle_errors
        self.timers = timers
        self._corners: dict[tuple[int, str], Corner] = {}
        self._runs: dict[tuple[str, tuple[float, float], int], _Run] = {}
        self._curves: dict[str, Arr] = {}

    def corner(self, i: int, cid: str) -> Corner:
        key = (i, cid)
        if key not in self._corners:
            axle = "front" if cid[0] == "F" else "rear"
            v = self.vehicles[i]
            self._corners[key] = _build(cid, axle, getattr(v, axle))  # type: ignore[arg-type]
        return self._corners[key]

    def run(self, cid: str, travel: tuple[float, float], steps: int) -> _Run:
        key = (cid, travel, steps)
        if key in self._runs:
            return self._runs[key]
        t0 = time.perf_counter()
        axle = "front" if cid[0] == "F" else "rear"
        ab = self.axle_batches[axle]
        if axle not in self._curves:
            self._curves[axle] = _rack0_curves(ab, _STATE_CHANNELS)
        q = np.linspace(travel[0], travel[1], steps)
        v = hermite(self._curves[axle], ab.grid.travel, q)  # (B, 4, K)
        st = _make_static([self.corner(i, cid) for i in range(len(self.vehicles))])
        rb = _rebuild(st, v[:, 0], v[:, 1], v[:, 2], v[:, 3])
        cols = _corner_columns(rb, q)
        keys = (rb.rocker, rb.coil, rb.moved["wheel_center"], rb.moved["contact_patch"])
        valid = np.ones(len(self.vehicles), dtype=bool)
        for a in keys:
            valid &= np.isfinite(a).reshape(a.shape[0], -1).all(axis=1)
        run = _Run(rb, q, cols, valid)
        self._runs[key] = run
        self.timers["rebuild"] = self.timers.get("rebuild", 0.0) + time.perf_counter() - t0
        return run

    def joint(
        self, cid: str, travel: tuple[float, float], steps: int, joint: str, axis: Any
    ) -> Arr:
        run = self.run(cid, travel, steps)
        key = (joint, None if axis is None else tuple(axis))
        if key not in run.joints:
            run.joints[key] = _joint_angles(run.rb, joint, axis)
        return run.joints[key]


class _SurrogateCtx(_Ctx):
    """Sweep context of one design that serves the batch tables instead of solving."""

    def __init__(
        self, study_sweeps: dict[str, Any], vehicle: Vehicle, batch: _Batch, i: int
    ) -> None:
        super().__init__(study_sweeps, vehicle)
        self.batch = batch
        self.i = i

    def corners(self) -> dict[str, Corner]:
        return cast(dict[str, Corner], _LazyCorners(self.batch, self.i))

    def corner_run(
        self, key: Any, corner_id: str, travel: tuple[float, float], steps: int
    ) -> _CornerData:
        def run() -> _CornerData:
            r = self.batch.run(corner_id, (float(travel[0]), float(travel[1])), steps)
            if not r.valid[self.i]:
                raise ValueError(
                    f"corner {corner_id}: invalid surrogate nodes in the sweep "
                    f"(masked, outside the grid or not assemblable)"
                )
            return _CornerData(
                self.batch.corner(self.i, corner_id),
                cast(list[Any], _States(r.rb, self.i, r.travel)),
                cast(pd.DataFrame, r.row(self.i)),
            )

        data: _CornerData = self.memo(("corner", key), run)
        return data

    def vehicle_data(self, name: str) -> pd.DataFrame:
        sw = self.study_sweeps[name]
        if not isinstance(sw, VehicleSweep):
            raise _Unsolved(f"sweep {name!r} is not a vehicle sweep")

        def run() -> pd.DataFrame:
            b = self.batch
            if name in b.vehicle_errors:
                raise ValueError(b.vehicle_errors[name])
            if b.layer_error is not None:
                raise ValueError(b.layer_error)
            tab = b.vehicle_tables[name]
            n = next(iter(tab.values())).shape[1]
            extra = {
                "wheelbase": np.full(n, float(self.vehicle.wheelbase)),
            }
            if self.vehicle.front.steering is not None:
                extra["steer_c_factor"] = np.full(n, float(self.vehicle.front.steering.c_factor))

            def has(name_: object) -> bool:
                return name_ in tab or name_ in extra

            def get(name_: str) -> Arr:
                return tab[name_][self.i] if name_ in tab else extra[name_]

            detail = f"is not provided by the surrogate vehicle layer ({sw.input} sweep)"
            return cast(pd.DataFrame, _Row(get, has, detail))

        df: pd.DataFrame = self.memo(("vehicle", name), run)
        return df

    def summary(self) -> Any:
        def run() -> Any:
            b = self.batch
            if b.layer_error is not None:
                raise ValueError(b.layer_error)
            if b.layer is None:
                raise ValueError("the vehicle layer was not run for this study")
            return _SummaryView(b.layer.summary, self.i)

        return self.memo("summary", run)

    def joint_series(self, name: str, c: JointAngleConstraint) -> Arr:
        sw = self.study_sweeps[name]
        assert isinstance(sw, CornerSweep)
        self.sweep_corner_data(name)  # raises _Unsolved for an invalid run

        def run() -> Arr:
            ang = self.batch.joint(
                sw.corner, (float(sw.travel[0]), float(sw.travel[1])), sw.steps, c.joint, c.axis
            )
            return np.asarray(ang[self.i], dtype=np.float64)

        out: Arr = self.memo(("joint", name, c.joint, c.axis), run)
        return out


# --------------------------------------------------------------------------------------
# the evaluator
# --------------------------------------------------------------------------------------


def _used_sweeps(space: DesignSpace) -> tuple[set[str], bool]:
    """Names of the sweeps that enabled constraints / objectives use; summary needed?"""
    names: set[str] = set()
    summary = False
    for c in space.study.constraints:
        if not c.enabled:
            continue
        summary |= isinstance(c, SummaryConstraint)
        for attr in ("sweep", "over"):
            v = getattr(c, attr, None)
            if isinstance(v, str):
                names.add(v)
    for o in space.study.objectives:
        if o.sweep is None:
            summary = True
        else:
            names.add(o.sweep)
    return names, summary


class SurrogateEvaluator:
    """Evaluate batches of designs on the axle predictors (see the module docstring).

    Parameters
    ----------
    space
        Design space of the study (symmetric variables only).
    predictors
        ``{"front": ..., "rear": ...}`` axle predictors with the same grids as the study.
    chunk
        Maximum designs per internal batch (memory bound).

    Attributes
    ----------
    timers
        Cumulative seconds per stage: ``vehicles``, ``predict``, ``layer``, ``rebuild``
        (inside ``evaluator``), ``evaluator``.
    """

    def __init__(
        self, space: DesignSpace, predictors: Mapping[str, AxlePredictor], *, chunk: int = 2048
    ) -> None:
        self.space = space
        self.predictors = dict(predictors)
        self.chunk = chunk
        self.timers: dict[str, float] = {}
        self.asp = {a: AxleSpace(space, a) for a in _AXLES}
        self.grids: dict[str, AxleGrid] = {}
        for a in _AXLES:
            if a not in self.predictors:
                raise ValueError(f"missing predictor for the {a} axle")
            p = self.predictors[a]
            need = set(_STATE_CHANNELS)
            if not need <= set(p.channels):
                raise ValueError(
                    f"{a} model lacks the channels {sorted(need - set(p.channels))} "
                    f"(train on a dataset sampled with the current channel registry)"
                )
            self.grids[a] = AxleGrid(a, np.asarray(p.travel), np.asarray(p.rack))
        self._check_study_grids()
        used, self.needs_summary = _used_sweeps(space)
        self.vehicle_sweeps: dict[str, tuple[str, VehicleSweep]] = {}
        self.vehicle_errors: dict[str, str] = {}
        base = space.baseline
        for name in sorted(used):
            sw = space.study.sweeps.get(name)
            if isinstance(sw, VehicleSweep):
                rng = sweep_range(sw, base)
                if rng is None:
                    self.vehicle_errors[name] = (
                        f"sweep {name!r}: range 'full' needs a steering rack"
                    )
                elif sw.base:
                    self.vehicle_errors[name] = (
                        f"sweep {name!r}: a non-empty 'base' is not supported by the layer"
                    )
                else:
                    self.vehicle_sweeps[name] = (sw.input, sw)
        self._groups: list[dict[str, str]] = []  # layer calls: kind -> sweep name
        for name, (kind, _) in self.vehicle_sweeps.items():
            for g in self._groups:
                if kind not in g:
                    g[kind] = name
                    break
            else:
                self._groups.append({kind: name})
        self._check_ranges(used)
        # Statics depend on the design only through a few hardpoints; if no variable moves
        # one of them the baseline statics are repeated (saves a per-design Python loop).
        moved = {parse_point_ref(v.point)[1] for v in space.study.variables}
        self._const_statics: VehicleStatics | None = (
            None if moved & _STATICS_POINTS else vehicle_statics([space.baseline])
        )

    # ---- setup checks ---------------------------------------------------------------

    def _check_study_grids(self) -> None:
        spec = self.space.study.surrogate
        if spec is None:
            return
        from suspkin.surrogate.grid import axle_grids

        for a, g in axle_grids(spec.grid).items():
            mine = self.grids[a]
            if mine.shape != g.shape or not (
                np.allclose(mine.travel, g.travel) and np.allclose(mine.rack, g.rack)
            ):
                raise StudyConfigError(f"the {a} model grid differs from the study surrogate.grid")

    def _check_ranges(self, used: set[str]) -> None:
        """Corner sweeps and reach travels must lie inside the model's travel grid."""
        ranges: list[tuple[str, str, tuple[float, float]]] = []
        for name in used:
            sw = self.space.study.sweeps.get(name)
            if isinstance(sw, CornerSweep):
                ranges.append((f"sweep {name!r}", sw.corner, sw.travel))
        for c in self.space.study.constraints:
            if c.enabled and isinstance(c, ReachConstraint):
                ranges.append((f"constraint {c.name!r}", c.corner, c.travel))
        for what, cid, (lo, hi) in ranges:
            g = self.grids["front" if cid[0] == "F" else "rear"]
            if lo < g.travel[0] - _TOL or hi > g.travel[-1] + _TOL:
                raise StudyConfigError(
                    f"{what}: travel [{lo}, {hi}] m exceeds the surrogate grid "
                    f"[{g.travel[0]}, {g.travel[-1]}] m"
                )

    # ---- evaluation -----------------------------------------------------------------

    def _run_layer(
        self, vehicles: list[Vehicle], batches: dict[str, AxleBatch]
    ) -> tuple[LayerResult, dict[str, dict[str, Arr]]]:
        """Layer output for the batch. Sweep ranges come from each design's own vehicle
        (``range: full`` follows its rack travel); designs with different ranges run in
        separate layer calls and are scattered back."""
        names = list(self.vehicle_sweeps)
        keys = [tuple(sweep_range(self.vehicle_sweeps[n][1], v) for n in names) for v in vehicles]
        by_key: dict[Any, list[int]] = {}
        for i, rk in enumerate(keys):
            by_key.setdefault(rk, []).append(i)
        b = len(vehicles)
        summary: dict[str, Arr] = {}
        tables: dict[str, dict[str, Arr]] = {}
        for key, idx in by_key.items():
            ii = np.asarray(idx)
            sub = {
                a: AxleBatch(x.Y[ii], x.mask[ii], x.channels, x.grid) for a, x in batches.items()
            }
            statics = (
                vehicle_statics([vehicles[i] for i in idx])
                if self._const_statics is None
                else _tile(self._const_statics, len(idx))
            )
            values = {
                n: np.linspace(r[0], r[1], self.vehicle_sweeps[n][1].steps)
                for n, r in zip(names, key, strict=True)
            }
            for gi, group in enumerate(self._groups or [{}]):
                res = vehicle_layer(sub, statics, sweeps={k: values[n] for k, n in group.items()})
                if gi == 0:
                    for sk, sv in res.summary.items():
                        summary.setdefault(sk, np.full(b, _NAN))[ii] = sv
                for kind, name in group.items():
                    for c, arr in res.sweeps[kind].items():
                        tab = tables.setdefault(name, {})
                        tab.setdefault(c, np.full((b, arr.shape[1]), _NAN))[ii] = arr
        return LayerResult(summary, {}), tables

    def _tick(self, key: str, t0: float) -> float:
        t1 = time.perf_counter()
        self.timers[key] = self.timers.get(key, 0.0) + t1 - t0
        return t1

    def evaluate(self, X: ArrayLike) -> list[Evaluation]:
        """Evaluations of the rows of ``X`` (never raises for a single bad design)."""
        xm = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out: list[Evaluation] = []
        # The cyclic collector costs ~10 % here (many small short-lived objects); the
        # reference cycles of a batch are collected when it is re-enabled.
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for lo in range(0, len(xm), self.chunk):
                out.extend(self._evaluate_chunk(xm[lo : lo + self.chunk]))
        finally:
            if was_enabled:
                gc.enable()
        return out

    def _evaluate_chunk(self, xm: Arr) -> list[Evaluation]:
        space = self.space
        t0 = time.perf_counter()
        results: list[Evaluation | None] = [None] * len(xm)
        vehicles: list[Vehicle] = []
        rows: list[int] = []
        for i, x in enumerate(xm):
            try:
                vehicles.append(space.to_vehicle(x))
                rows.append(i)
            except InfeasibleDesign:
                results[i] = evaluate(space, x)
            except Exception as exc:  # a crashing design must not abort the run
                results[i] = failed_evaluation(space, f"{type(exc).__name__}: {exc}")
        t0 = self._tick("vehicles", t0)
        if not rows:
            return [r for r in results if r is not None]

        xf = np.clip(xm[rows], 0.0, 1.0)
        batches: dict[str, AxleBatch] = {}
        for a in _AXLES:
            y, p = self.predictors[a].predict(np.ascontiguousarray(xf[:, self.asp[a].idx]))
            batches[a] = AxleBatch(
                np.asarray(y, dtype=np.float64),
                np.asarray(p) >= P_VALID_MIN,
                tuple(self.predictors[a].channels),
                self.grids[a],
            )
        t0 = self._tick("predict", t0)

        layer: LayerResult | None = None
        layer_error: str | None = None
        tables: dict[str, dict[str, Arr]] = {}
        if self.needs_summary or self._groups:
            try:
                layer, tables = self._run_layer(vehicles, batches)
            except Exception as exc:  # layer failure -> every dependent constraint unsolved
                layer_error = f"vehicle layer failed: {type(exc).__name__}: {exc}"
        t0 = self._tick("layer", t0)

        timers: dict[str, float] = {}
        batch = _Batch(
            vehicles, batches, layer, layer_error, tables, dict(self.vehicle_errors), timers
        )
        sweeps_map = dict(space.study.sweeps)
        for j, (i, veh) in enumerate(zip(rows, vehicles, strict=True)):
            try:
                ctx = _SurrogateCtx(sweeps_map, veh, batch, j)
                results[i] = evaluate(space, vehicle=veh, ctx=ctx)
            except Exception as exc:
                results[i] = failed_evaluation(space, f"{type(exc).__name__}: {exc}")
        t1 = time.perf_counter()
        self.timers["rebuild"] = self.timers.get("rebuild", 0.0) + timers.get("rebuild", 0.0)
        self.timers["evaluator"] = self.timers.get("evaluator", 0.0) + (t1 - t0)
        return cast(list[Evaluation], results)
