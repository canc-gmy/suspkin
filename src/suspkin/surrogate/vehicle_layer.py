"""Analytic vehicle layer: per-axle corner curves + per-design statics -> vehicle results.

The surrogate predicts only left-corner curves on a fixed grid (see
:mod:`suspkin.surrogate.channels`). This module turns those curves into everything
:func:`suspkin.metrics.vehicle.vehicle_summary` and (a subset of)
:func:`suspkin.sweeps.vehicle_frame` return, vectorised over a batch of
designs. SI units and the sign conventions of ``vehicle_summary`` throughout.

Method (DESIGN_PLAN section 13.5):

* The right corner is the mirrored left corner (the right wheel at rack ``r`` is the
  mirrored left wheel at ``-r``). Overrides of right-hand hardpoints are not representable.
* Curves are interpolated with a local cubic Hermite spline (C1; slopes from
  ``np.gradient`` like ``add_gradients``). Cubic rather than linear because the wheel and
  ARB rates and the roll-centre curvature need smooth first/second differences; local
  (4 nodes) so a masked node only spoils its neighbourhood. A query outside the grid or
  touching a masked node gives NaN for the affected outputs, never an extrapolation.
* Body pose -> wheel travel: for each wheel the travel ``s`` solves "tyre contact height
  = static contact height" on the tilted ground, exactly like ``solve_vehicle``, with the
  wheel centre and spin axis rebuilt from the channels (camber, toe, contact-patch and
  wheel-centre offsets) and the static data.
* Swing-angle channels hold the line angle wrapped to (-pi/2, pi/2] (smooth in the design);
  every use below needs only the line (roll centre and pitch centre are line intersections,
  anti-dive/squat/lift use tan, which has period pi), never the direction.
* Roll centre: the two front-view lines through the (tilted-ground) contact patches with
  the front-view swing-angle line; a line whose swing angle was taken at the
  vertical contact patch differs slightly under roll because the instant centre itself is
  not a channel (error ~ patch shift / FVSA).
* Anti-features, pitch centre: static swing angles with the ``metrics.vehicle`` formulas.
* Wheel rate: ``d(coilover compression)/d(travel)`` and its derivative by ``np.gradient``
  on the grid, spring force from the static data. ARB rate: drop-link point rebuilt by
  rigid rocker rotation, ARB solved in closed form as in ``kinematics.arb``.
* Steering: travel-0 row of the front grid versus rack (no ground re-solve).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from suspkin.geometry.model import Axle, BeamARB, Vehicle
from suspkin.geometry.vectors import rodrigues_rows, wheel_spin_axis, wrap_angle
from suspkin.kinematics.arb import arb_stiffness
from suspkin.kinematics.vehicle import cg_pivot
from suspkin.metrics import vehicle as vm
from suspkin.metrics.vehicle import AxleSummary, VehicleSummary, vehicle_summary
from suspkin.surrogate.channels import CornerSample, extract_axles
from suspkin.surrogate.data import AxleGrid

Arr = NDArray[np.float64]
_NAN = float("nan")
_TOL = 1e-9
_SIDE = np.array([1.0, -1.0]).reshape(1, 2, 1)  # left, right
_FV_CORNERS = {"front": ("FL", "FR"), "rear": ("RL", "RR")}
_ARB_DELTA = 2e-4  # m, travel step of the ARB motion ratio (central difference)
_N_RACK = 41  # rack samples of the steering sweep (as vehicle_summary)
_ACKERMANN_AT = (5.0, 10.0, 15.0)
_POSE_ITERS = 8  # secant iterations of the pose solve
_POSE_TOL = 1e-9  # m, accepted contact-height residual of the pose solve


class UnsupportedColumnError(ValueError):
    """A ``vehicle_frame`` column the surrogate layer does not provide."""


# ----------------------------------------------------------------------------- batches
@dataclass(frozen=True)
class AxleBatch:
    """Curve tensors of one axle for ``B`` designs.

    ``Y`` is ``(B, C, R, T)`` (channel order ``channels``), ``mask`` is ``(B, R, T)``
    (valid points); invalid points are treated as NaN whatever ``Y`` holds there.
    """

    Y: Arr
    mask: NDArray[np.bool_]
    channels: tuple[str, ...]
    grid: AxleGrid

    def __post_init__(self) -> None:
        r, t = self.grid.shape
        if self.Y.ndim != 4 or self.Y.shape[1:] != (len(self.channels), r, t):
            raise ValueError(
                f"Y shape {self.Y.shape} does not match channels/grid {(len(self.channels), r, t)}"
            )
        if self.mask.shape != (self.Y.shape[0], r, t):
            raise ValueError(f"mask shape {self.mask.shape} must be (B, R, T)")

    @property
    def n(self) -> int:
        """Number of designs ``B``."""
        return int(self.Y.shape[0])


def stack_samples(samples: Sequence[CornerSample], grid: AxleGrid) -> AxleBatch:
    """Stack the :class:`CornerSample` of ``B`` designs (one axle) into an :class:`AxleBatch`."""
    if not samples:
        raise ValueError("samples must not be empty")
    chans = samples[0].channels
    if any(s.channels != chans for s in samples):
        raise ValueError("all samples must share the same channels")
    return AxleBatch(
        np.stack([s.Y for s in samples]).astype(np.float64),
        np.stack([s.mask for s in samples]).astype(bool),
        chans,
        grid,
    )


# ----------------------------------------------------------------------------- statics
@dataclass(frozen=True)
class AxleStatics:
    """Static per-design data of one axle (left corner, YAML hardpoints = static pose).

    Arrays are ``(B,)`` or ``(B, 3)``. ARB arrays are zero when ``arb_type`` is None;
    for a beam bar ``arb_bar_pivot`` / ``arb_bar_axis`` are unused.
    """

    scp: Arr
    swc: Arr
    tyre_radius: Arr
    tyre_stiffness: Arr
    spring_rate: Arr
    free_length: Arr
    preload: Arr
    unsprung_mass: Arr
    unsprung_cg_height: Arr
    brakes_outboard: NDArray[np.bool_]
    arb_type: Literal["torsional", "blade", "beam"] | None
    arb_k: Arr
    arb_arm_end: Arr
    arb_bar_pivot: Arr
    arb_bar_axis: Arr
    arb_drop_length: Arr
    rocker_pivot: Arr
    rocker_axis: Arr
    rocker_arb: Arr

    @property
    def track(self) -> Arr:
        """Track ``2 y`` of the static wheel centre."""
        out: Arr = 2.0 * self.swc[:, 1]
        return out


@dataclass(frozen=True)
class VehicleStatics:
    """Static per-design data of the vehicle (see :func:`vehicle_statics`)."""

    front: AxleStatics
    rear: AxleStatics
    wheelbase: Arr
    mass: Arr
    cg_height: Arr
    front_mass_ratio: Arr
    brake_bias_front: Arr
    drive_split_front: Arr
    pivot: Arr
    rack_travel: Arr
    c_factor: Arr

    @property
    def n(self) -> int:
        """Number of designs ``B``."""
        return int(self.wheelbase.shape[0])


def _static_contact(axle: Axle) -> Arr:
    """Static contact patch of the left wheel (as ``kinematics.vehicle._static_contact``)."""
    s = wheel_spin_axis("left", axle.wheel.static_camber, axle.wheel.static_toe)
    up = np.array([0.0, 0.0, 1.0]) - s[2] * s
    up = up / np.linalg.norm(up)
    out: Arr = np.asarray(axle.hardpoints.wheel_center) - axle.wheel.tyre_radius * up
    return out


def _axle_statics(vehicles: Sequence[Vehicle], name: Literal["front", "rear"]) -> AxleStatics:
    axles = [getattr(v, name) for v in vehicles]
    kinds = {None if a.arb is None else a.arb.type for a in axles}
    if len(kinds) != 1:
        raise ValueError(f"{name}: all designs must have the same ARB type, got {kinds}")
    kind = kinds.pop()
    n = len(axles)

    def col(f: Callable[[Axle], float]) -> Arr:
        return np.array([f(a) for a in axles], dtype=np.float64)

    def vec(f: Callable[[Axle], ArrayLike]) -> Arr:
        return np.array([f(a) for a in axles], dtype=np.float64).reshape(n, 3)

    arb_k = np.zeros(n)
    arm = np.zeros((n, 3))
    piv = np.zeros((n, 3))
    axis = np.zeros((n, 3))
    drop = np.zeros(n)
    r_arb = np.zeros((n, 3))
    if kind is not None:
        for i, a in enumerate(axles):
            arb = a.arb
            assert arb is not None and a.hardpoints.rocker_arb is not None
            arb_k[i] = arb_stiffness(a)
            arm[i] = arb.arm_end
            if not isinstance(arb, BeamARB):
                piv[i] = arb.bar_pivot
                axis[i] = arb.bar_axis
            r_arb[i] = a.hardpoints.rocker_arb
            drop[i] = np.linalg.norm(r_arb[i] - arm[i])
    hu = [v.vehicle.unsprung_cg_height or getattr(v, name).wheel.tyre_radius for v in vehicles]
    return AxleStatics(
        scp=vec(_static_contact),
        swc=vec(lambda a: a.hardpoints.wheel_center),
        tyre_radius=col(lambda a: a.wheel.tyre_radius),
        tyre_stiffness=col(lambda a: a.wheel.tyre_vertical_stiffness),
        spring_rate=col(lambda a: a.coilover.spring_rate),
        free_length=col(lambda a: a.coilover.free_length),
        preload=col(lambda a: a.coilover.preload),
        unsprung_mass=col(lambda a: a.unsprung_mass),
        unsprung_cg_height=np.array(hu, dtype=np.float64),
        brakes_outboard=np.array([a.brakes == "outboard" for a in axles], dtype=bool),
        arb_type=kind,
        arb_k=arb_k,
        arb_arm_end=arm,
        arb_bar_pivot=piv,
        arb_bar_axis=axis,
        arb_drop_length=drop,
        rocker_pivot=vec(lambda a: a.hardpoints.rocker_pivot),
        rocker_axis=vec(lambda a: a.hardpoints.rocker_axis),
        rocker_arb=r_arb,
    )


def vehicle_statics(vehicles: Sequence[Vehicle]) -> VehicleStatics:
    """Build :class:`VehicleStatics` for ``vehicles`` (loops over designs in Python).

    All designs must have the same ARB type per axle (arrays are rectangular).
    """
    if not vehicles:
        raise ValueError("vehicles must not be empty")

    def col(f: Callable[[Vehicle], float]) -> Arr:
        return np.array([f(v) for v in vehicles], dtype=np.float64)

    return VehicleStatics(
        front=_axle_statics(vehicles, "front"),
        rear=_axle_statics(vehicles, "rear"),
        wheelbase=col(lambda v: v.wheelbase),
        mass=col(lambda v: v.vehicle.mass),
        cg_height=col(lambda v: v.vehicle.cg_height),
        front_mass_ratio=col(lambda v: v.vehicle.front_mass_ratio),
        brake_bias_front=col(lambda v: v.vehicle.brake_bias_front),
        drive_split_front=col(lambda v: v.vehicle.effective_drive_split_front),
        pivot=np.array([cg_pivot(v) for v in vehicles], dtype=np.float64).reshape(-1, 3),
        rack_travel=np.array(
            [v.front.steering.rack_travel for v in vehicles],  # type: ignore[union-attr]
            dtype=np.float64,
        ).reshape(-1, 2),
        c_factor=col(lambda v: v.front.steering.c_factor),  # type: ignore[union-attr]
    )


# ----------------------------------------------------------------------------- interpolation
class Hermite:
    """Local cubic Hermite interpolation of ``y (B, K, L)`` along the last axis.

    Slopes by ``np.gradient``. Queries ``q`` are ``(N,)`` (shared) or ``(B, N)``; the result
    is ``(B, K, N)``, NaN outside ``[x[0], x[-1]]`` (a ``1e-9`` tolerance at the ends).
    """

    def __init__(self, y: Arr, x: Arr, m: Arr | None = None) -> None:
        self.y, self.x = y, x
        self.m = m if m is not None else (np.gradient(y, x, axis=-1) if x.size > 1 else 0.0 * y)

    def take(self, ix: Sequence[int]) -> Hermite:
        """Same interpolant restricted to the channels ``ix``."""
        sel = list(ix)
        return Hermite(self.y[:, sel], self.x, self.m[:, sel])

    def __call__(self, q: Arr) -> Arr:
        y, x, m = self.y, self.x, self.m
        q2 = np.broadcast_to(q, (y.shape[0], q.shape[-1])) if q.ndim == 1 else q
        if x.size == 1:
            ok1 = np.abs(q2 - x[0]) <= _TOL
            out1: Arr = np.where(ok1[:, None, :], y[..., :1], np.nan)
            return out1
        idx = np.clip(np.searchsorted(x, q2, side="right") - 1, 0, x.size - 2)
        x0 = x[idx]
        h = x[idx + 1] - x0
        t = ((q2 - x0) / h)[:, None, :]
        ok = ((q2 >= x[0] - _TOL) & (q2 <= x[-1] + _TOL))[:, None, :]
        ib = idx[:, None, :]
        y0 = np.take_along_axis(y, ib, axis=-1)
        y1 = np.take_along_axis(y, ib + 1, axis=-1)
        m0 = np.take_along_axis(m, ib, axis=-1)
        m1 = np.take_along_axis(m, ib + 1, axis=-1)
        t2 = t * t
        t3 = t2 * t
        hh = h[:, None, :]
        out: Arr = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * y0
            + (t3 - 2.0 * t2 + t) * hh * m0
            + (3.0 * t2 - 2.0 * t3) * y1
            + (t3 - t2) * hh * m1
        )
        return np.where(ok, out, np.nan)


def _masked(batch: AxleBatch, names: Sequence[str]) -> Arr:
    ix = [batch.channels.index(n) for n in names]
    y: Arr = np.where(batch.mask[:, None], batch.Y[:, ix], np.nan)
    return y


def travel_curves(batch: AxleBatch, names: Sequence[str], rack: float = 0.0) -> Hermite:
    """Curves versus travel at a fixed rack (interpolated along the rack axis if needed)."""
    y = _masked(batch, names)  # (B, K, R, T)
    nodes = batch.grid.rack
    j = int(np.argmin(np.abs(nodes - rack)))
    b, k, r, t = y.shape
    if abs(nodes[j] - rack) <= _TOL:
        red = y[:, :, j, :]
    elif r == 1:
        red = np.full((b, k, t), np.nan)
    else:
        flat = y.transpose(0, 1, 3, 2).reshape(b, k * t, r)
        red = Hermite(flat, nodes)(np.array([rack])).reshape(b, k, t)
    return Hermite(red, batch.grid.travel)


def _rack_curves(batch: AxleBatch, name: str, travel: float = 0.0) -> Hermite:
    """Curve of one channel versus rack at a fixed travel."""
    y = _masked(batch, [name])  # (B, 1, R, T)
    b, _, r, t = y.shape
    at = Hermite(y.reshape(b, r, t), batch.grid.travel)(np.array([travel]))  # (B, R, 1)
    return Hermite(at.reshape(b, 1, r), batch.grid.rack)


def _zero_node(x: Arr, what: str) -> int:
    """Index of the grid node at 0; the layer requires travel 0 and rack 0 to be nodes."""
    j = int(np.argmin(np.abs(x)))
    if abs(x[j]) > _TOL:
        raise ValueError(
            f"{what} grid must contain 0 as a node (static pose); nearest node is {x[j]:.6g}"
        )
    return j


def _at_zero(y: Arr, x: Arr) -> Arr:
    """Value of ``y (B, T)`` at the grid node ``x = 0`` (see :func:`_zero_node`)."""
    out: Arr = y[:, _zero_node(x, "travel")]
    return out


# ----------------------------------------------------------------------------- pose solve
_POSE_CORE = ("camber", "toe", "contact_patch_dy", "wheel_center_recession")
_POSE_FULL = (*_POSE_CORE, "contact_patch_dx", "fv_swing_angle")


@dataclass
class _Pose:
    """Both wheels of one axle for ``N`` body poses; side axis 0 = left, 1 = right."""

    travel: Arr  # (B, 2, N)
    camber: Arr
    toe: Arr
    camber_ground: Arr
    steer_ground: Arr
    cp: Arr  # (B, 2, 3, N) tilted-ground contact point, body frame
    fv_angle: Arr  # (B, 2, N) front-view swing angle of the wheel at its travel


def _normal(roll: Arr, pitch: Arr) -> tuple[Arr, Arr, Arr]:
    """Ground normal in the body frame, ``R^T e_z`` for ``R = R_y(pitch) R_x(roll)``."""
    return -np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)


def _wheel(
    v: Arr, ix: Mapping[str, int], st: AxleStatics, s: Arr
) -> tuple[tuple[Arr, Arr, Arr], tuple[Arr, Arr, Arr]]:
    """Wheel centre and spin axis (body frame, per side) from channel values ``v``."""

    def ch(name: str) -> Arr:
        return v[:, ix[name]]

    cam, toe = ch("camber"), ch("toe")
    c = np.cos(cam)
    sx, sy, sz = c * np.sin(toe), c * np.cos(toe), -np.sin(cam)
    nrm = np.sqrt(sz * sz * (sx * sx + sy * sy) + (1.0 - sz * sz) ** 2)
    scp_y = st.scp[:, 1][:, None, None]
    r = st.tyre_radius[:, None, None]
    wcy = scp_y + ch("contact_patch_dy") - r * sz * sy / nrm
    wcx = st.swc[:, 0][:, None, None] + ch("wheel_center_recession")
    wcz = st.swc[:, 2][:, None, None] + s
    return (wcx, _SIDE * wcy, wcz), (sx, _SIDE * sy, sz)


def _contact(
    wc: tuple[Arr, Arr, Arr],
    sp: tuple[Arr, Arr, Arr],
    n: tuple[Arr, Arr, Arr],
    radius: Arr,
) -> tuple[Arr, tuple[Arr, Arr, Arr], Arr]:
    """Tyre point lowest along ``-n``: ``(n . spin, contact point, |n_perp|)``."""
    ns = n[0] * sp[0] + n[1] * sp[1] + n[2] * sp[2]
    nu = np.sqrt(np.maximum(1.0 - ns * ns, 1e-300))
    cp = tuple(wc[i] - radius * (n[i] - ns * sp[i]) / nu for i in range(3))
    return ns, (cp[0], cp[1], cp[2]), nu


def _solve_pose(
    batch: AxleBatch,
    st: AxleStatics,
    pivot: Arr,
    heave: Arr,
    roll: Arr,
    pitch: Arr,
) -> _Pose:
    """Wheel travels for prescribed body poses (rack 0) and the resulting wheel geometry."""
    full = travel_curves(batch, _POSE_FULL)
    core = full.take(range(len(_POSE_CORE)))
    ix_core = {n: i for i, n in enumerate(_POSE_CORE)}
    ix_full = {n: i for i, n in enumerate(_POSE_FULL)}
    b, n_pose = batch.n, roll.size
    n0, n1, n2 = (a.reshape(1, 1, -1) for a in _normal(roll, pitch))
    nv = (n0, n1, n2)
    pv = pivot[:, :, None, None]
    off = (
        pv[:, 2]
        - (nv[0] * pv[:, 0] + nv[1] * pv[:, 1] + nv[2] * pv[:, 2])
        + heave.reshape(1, 1, -1)
    )  # (B, 1, N)
    radius = st.tyre_radius[:, None, None]
    z_static = st.scp[:, 2][:, None, None]

    def height(s: Arr) -> Arr:
        v = core(s.reshape(b, -1)).reshape(b, len(_POSE_CORE), 2, n_pose)
        wc, sp = _wheel(v, ix_core, st, s)
        _, _, nu = _contact(wc, sp, nv, radius)
        out: Arr = nv[0] * wc[0] + nv[1] * wc[1] + nv[2] * wc[2] - radius * nu + off - z_static
        return out

    s_prev = np.zeros((b, 2, n_pose))
    g_prev = height(s_prev)
    s = s_prev - g_prev / nv[2]
    for _ in range(_POSE_ITERS):
        g = height(s)
        if not np.any(np.abs(g) > 1e-13):
            break
        ds, dg = s - s_prev, g - g_prev
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where((np.abs(ds) > 1e-15) & (np.abs(dg) > 1e-18), dg / ds, nv[2])
        s_prev, g_prev = s, g
        s = s - g / slope
    # no silent last iterate: a pose that did not reach the ground (or is NaN) has no travel
    with np.errstate(invalid="ignore"):
        s = np.where(np.abs(height(s)) <= _POSE_TOL, s, np.nan)

    v = full(s.reshape(b, -1)).reshape(b, len(_POSE_FULL), 2, n_pose)
    wc, sp = _wheel(v, ix_full, st, s)
    ns, cp, _ = _contact(wc, sp, nv, radius)
    sr, cr = np.sin(roll).reshape(1, 1, -1), np.cos(roll).reshape(1, 1, -1)
    sp_, cp_ = np.sin(pitch).reshape(1, 1, -1), np.cos(pitch).reshape(1, 1, -1)
    sw_x = cp_ * sp[0] + sp_ * sr * sp[1] + sp_ * cr * sp[2]
    sw_y = cr * sp[1] - sr * sp[2]
    ang = v[:, ix_full["fv_swing_angle"]]
    return _Pose(
        travel=s,
        camber=v[:, ix_full["camber"]],
        toe=v[:, ix_full["toe"]],
        camber_ground=-np.arcsin(np.clip(ns, -1.0, 1.0)),
        steer_ground=np.arctan2(-_SIDE * sw_x, _SIDE * sw_y),
        cp=np.stack(cp, axis=2),
        fv_angle=ang,
    )


def _roll_centre(pose: _Pose, pivot: Arr, heave: Arr, roll: Arr, pitch: Arr) -> tuple[Arr, Arr]:
    """Roll centre ``(y_body, z_ground)`` of one axle, ``(B, N)`` (NaN if undefined)."""
    cp, al = pose.cp, pose.fv_angle
    py, pz = cp[:, :, 1], cp[:, :, 2]  # (B, 2, N)
    dy = np.cos(al) * (-_SIDE)  # inboard direction: left -y, right +y
    dz = np.sin(al)
    a, bb = -dz, dy  # unit normals (|d| = 1)
    c = a * py + bb * pz
    a1, a2, b1, b2, c1, c2 = a[:, 0], a[:, 1], bb[:, 0], bb[:, 1], c[:, 0], c[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        det = a1 * b2 - b1 * a2
        valid = np.all(np.isfinite(dy) & np.isfinite(dz), axis=1)
        parallel = np.abs(det) < _TOL
        hit = valid & ~parallel
        y = np.where(hit, (c1 * b2 - b1 * c2) / det, _NAN)
        z = np.where(hit, (a1 * c2 - a2 * c1) / det, _NAN)
        p1y, p1z, d1y, d1z = py[:, 0], pz[:, 0], dy[:, 0], dz[:, 0]
        coincident = (
            valid & parallel & ~((np.abs(a2 * p1y + b2 * p1z - c2) > _TOL) | (np.abs(d1y) < 1e-12))
        )
        y = np.where(coincident, 0.0, y)
        z = np.where(coincident, p1z + d1z / d1y * (0.0 - p1y), z)
        ok = np.isfinite(y) & np.isfinite(z)
        n = tuple(a_.reshape(1, -1) for a_ in _normal(roll, pitch))
        pv = [pivot[:, i][:, None] for i in range(3)]
        x_mid = 0.5 * (cp[:, 0, 0] + cp[:, 1, 0])
        w_z = (
            n[0] * (x_mid - pv[0])
            + n[1] * (np.where(ok, y, 0.0) - pv[1])
            + n[2] * (np.where(ok, z, 0.0) - pv[2])
            + pv[2]
            + heave.reshape(1, -1)
        )
    return np.where(ok, y, _NAN), np.where(ok, w_z, _NAN)


# ----------------------------------------------------------------------------- ARB
def _arb_wheel_rate_roll(st: AxleStatics, rocker_plus: Arr, rocker_minus: Arr) -> Arr:
    """Roll-mode ARB wheel rate ``k MR^2 / 2`` at travel 0 (zero without ARB), ``(B,)``."""
    if st.arb_type is None:
        return np.zeros(st.arb_k.shape)
    ends = [
        st.rocker_pivot
        + rodrigues_rows(
            st.rocker_axis,
            np.cos(ang)[:, None],
            np.sin(ang)[:, None],
            st.rocker_arb - st.rocker_pivot,
        )
        for ang in (rocker_plus, rocker_minus)
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        if st.arb_type == "beam":
            vals = []
            for tgt in ends:
                d = tgt - st.arb_arm_end
                q = st.arb_drop_length**2 - d[:, 0] ** 2 - d[:, 1] ** 2
                root = np.sqrt(np.where(q < 0.0, _NAN, q))
                lo, hi = d[:, 2] - root, d[:, 2] + root
                vals.append(np.where(np.abs(lo) <= np.abs(hi), lo, hi))
            mr = (vals[0] - vals[1]) / (2.0 * _ARB_DELTA)
        else:
            betas = []
            for tgt in ends:
                v0 = st.arb_arm_end - st.arb_bar_pivot
                v_par = np.sum(v0 * st.arb_bar_axis, axis=-1, keepdims=True) * st.arb_bar_axis
                w = v0 - v_par
                r = (tgt - st.arb_bar_pivot) - v_par
                ca = np.sum(r * w, axis=-1)
                cb = np.sum(r * np.cross(st.arb_bar_axis, w), axis=-1)
                cc = 0.5 * (np.sum(r * r, axis=-1) + np.sum(w * w, axis=-1) - st.arb_drop_length**2)
                amp = np.hypot(ca, cb)
                bad = (amp < 1e-12) | (np.abs(cc) > amp * (1.0 + 1e-12))
                half = np.arccos(np.clip(cc / amp, -1.0, 1.0))
                base = np.arctan2(cb, ca)
                a1, a2 = wrap_angle(base + half), wrap_angle(base - half)
                betas.append(np.where(bad, _NAN, np.where(np.abs(a1) <= np.abs(a2), a1, a2)))
            mr = (betas[0] - betas[1]) / _ARB_DELTA
    out: Arr = 0.5 * st.arb_k * mr**2
    return out


# ----------------------------------------------------------------------------- steering
def _steering(front: AxleBatch, st: VehicleStatics) -> dict[str, Arr]:
    """Steering results from the travel-0 row of the front grid versus rack.

    NaN policy (conservative): if any of the 41 rack samples of a design (or the +-1 mm
    samples of the steering ratio) is NaN, i.e. outside the rack grid or touching a masked
    node, every steering output of that design is NaN. The lock side and the Ackermann
    interpolation need the whole sweep, so a partial sweep is never used silently.
    """
    b = front.n
    names = (
        "max_steer_inner",
        "max_steer_outer",
        "ackermann_5deg",
        "ackermann_10deg",
        "ackermann_15deg",
        "steering_ratio",
        "turning_radius",
    )
    nan = np.full(b, _NAN)
    if "steer_angle" not in front.channels:
        return {k: nan.copy() for k in names}
    curve = _rack_curves(front, "steer_angle")
    lo, hi = st.rack_travel[:, 0], st.rack_travel[:, 1]
    r = np.linspace(lo, hi, _N_RACK, axis=-1)  # (B, K)
    h = 1e-3
    q = np.concatenate([r, -r, np.full((b, 1), h), np.full((b, 1), -h), np.zeros((b, 1))], axis=1)
    v = curve(q)[:, 0]
    k = _N_RACK
    s_p, s_m, s_hp, s_hm, s0 = v[:, :k], v[:, k : 2 * k], v[:, -3], v[:, -2], v[:, -1]
    dl = s_p - s0[:, None]
    dr = -(s_m - s0[:, None])
    mean = 0.5 * (dl + dr)
    good = np.all(np.isfinite(mean), axis=1) & np.isfinite(s_hp) & np.isfinite(s_hm)
    mean = np.where(good[:, None], mean, 0.0)
    dl = np.where(good[:, None], dl, 0.0)
    dr = np.where(good[:, None], dr, 0.0)
    rows = np.arange(b)
    i_lock = np.argmax(np.abs(mean), axis=1)
    sgn = np.where(mean[rows, i_lock] >= 0.0, 1.0, -1.0)
    left_inner = (dl + dr) >= 0.0
    inner = sgn[:, None] * np.where(left_inner, dl, dr)
    outer = sgn[:, None] * np.where(left_inner, dr, dl)
    sel = (sgn[:, None] * mean > 0.0) & (outer > 1e-4)
    wb, track = st.wheelbase[:, None], st.front.track[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ideal = np.arctan2(1.0, 1.0 / np.tan(outer) - track / wb)
        ack = np.where(sel, (inner - outer) / (ideal - outer) * 100.0, _NAN)
        o_sorted_key = np.where(sel, outer, np.inf)
        order = np.argsort(o_sorted_key, axis=1)
        o_s = np.take_along_axis(o_sorted_key, order, axis=1)
        a_s = np.take_along_axis(ack, order, axis=1)
        count = sel.sum(axis=1)
        out: dict[str, Arr] = {}
        lock_in, lock_out = inner[rows, i_lock], outer[rows, i_lock]
        out["max_steer_inner"] = lock_in
        out["max_steer_outer"] = lock_out
        t = np.tan(0.5 * (lock_in + lock_out))
        out["turning_radius"] = np.where(t == 0.0, np.inf, st.wheelbase / t)
        for deg in _ACKERMANN_AT:
            tgt = math.radians(deg)
            j = np.clip((o_s <= tgt).sum(axis=1) - 1, 0, np.maximum(count - 2, 0))
            x0, x1 = o_s[rows, j], o_s[rows, np.minimum(j + 1, k - 1)]
            y0, y1 = a_s[rows, j], a_s[rows, np.minimum(j + 1, k - 1)]
            val = y0 + (tgt - x0) / (x1 - x0) * (y1 - y0)
            in_range = (
                (count >= 2) & (o_s[:, 0] <= tgt) & (tgt <= o_s[rows, np.maximum(count - 1, 0)])
            )
            out[f"ackermann_{int(deg)}deg"] = np.where(in_range, val, _NAN)
        m = 0.5 * (s_hp - s_hm)  # mean road-wheel angle per 1 mm rack (s0 and mirror cancel)
        out["steering_ratio"] = np.where(
            np.abs(m) < 1e-9, _NAN, (h / st.c_factor * 2.0 * math.pi) / m
        )
    return {kk: np.where(good, vv, _NAN) for kk, vv in out.items()}


# ----------------------------------------------------------------------------- summary
def _poly_weights(x: Arr) -> tuple[Arr, Arr]:
    """Weights ``(w1, w2)`` so that ``y @ w1`` / ``y @ w2`` are the slope / second derivative at
    ``x = 0`` of the cubic least-squares fit (as ``np.polyfit(x, y, 3)``)."""
    xm = float(np.max(np.abs(x)))
    xs = x / xm
    vander = np.stack([xs**3, xs**2, xs, np.ones_like(xs)], axis=1)
    w = np.linalg.pinv(vander)
    return w[2] / xm, 2.0 * w[1] / xm**2


@dataclass
class _AxleStatic:
    wheel_rate: Arr
    arb_wheel_rate_roll: Arr
    sv_cp: Arr  # static side-view swing angle at the contact patch
    sv_wc: Arr  # static side-view swing angle at the wheel centre


def _axle_static(batch: AxleBatch, st: AxleStatics) -> _AxleStatic:
    names = (
        "coilover_length",
        "rocker_angle",
        "sv_swing_angle",
        "sv_wc_angle",
    )
    curves = travel_curves(batch, names)
    length = curves.y[:, 0]  # (B, T)
    x = batch.grid.travel
    mr = -np.gradient(length, x, axis=-1)
    prog = np.gradient(mr, x, axis=-1)
    force = st.preload + st.spring_rate * (st.free_length - _at_zero(length, x))
    wheel_rate = st.spring_rate * _at_zero(mr, x) ** 2 + force * _at_zero(prog, x)
    v = curves(np.array([_ARB_DELTA, -_ARB_DELTA, 0.0]))  # (B, K, 3)
    arb = _arb_wheel_rate_roll(st, v[:, 1, 0], v[:, 1, 1])
    sv_cp = v[:, 2, 2]
    sv_wc = v[:, 3, 2]
    return _AxleStatic(wheel_rate, arb, sv_cp, sv_wc)


def _flat_names() -> list[str]:
    out: list[str] = []
    for f in fields(VehicleSummary):
        if f.name in ("front", "rear"):
            out += [f"{f.name}.{g.name}" for g in fields(AxleSummary)]
        else:
            out.append(f.name)
    return out


def summary_fields() -> tuple[str, ...]:
    """Flattened ``VehicleSummary`` field names (``front.rc_height`` ...) the layer returns."""
    return tuple(_flat_names())


def _tan_anti(angle: Arr, share: Arr, st: VehicleStatics) -> Arr:
    out: Arr = np.tan(angle) * share * st.wheelbase / st.cg_height * 100.0
    return out


def _pitch_centre(st: VehicleStatics, af: Arr, ar: Arr) -> tuple[Arr, Arr]:
    """Intersection of the static side-view lines of the front and rear contact patches."""
    p1 = np.stack([st.front.scp[:, 0], st.front.scp[:, 2]], axis=1)
    p2 = np.stack([st.rear.scp[:, 0], st.rear.scp[:, 2]], axis=1)
    d1 = np.stack([-np.cos(af), np.sin(af)], axis=1)
    d2 = np.stack([np.cos(ar), np.sin(ar)], axis=1)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    scale = np.maximum(np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1), 1e-300)
    bad = np.abs(cross) < 1e-12 * scale
    w = p2 - p1
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (w[:, 0] * d2[:, 1] - w[:, 1] * d2[:, 0]) / cross
        pt = p1 + s[:, None] * d1
    return np.where(bad, _NAN, pt[:, 0]), np.where(bad, _NAN, pt[:, 1])


def _summary(
    batches: Mapping[str, AxleBatch],
    st: VehicleStatics,
    roll_range: float,
    n: int,
    ay_ref: float,
) -> dict[str, Arr]:
    b = st.n
    rolls = np.linspace(-roll_range, roll_range, n | 1)
    i0 = int(np.argmin(np.abs(rolls)))
    w1, w2 = _poly_weights(rolls)
    zeros = np.zeros_like(rolls)
    out: dict[str, Arr] = {}
    ax: dict[str, dict[str, Arr]] = {}
    stat: dict[str, _AxleStatic] = {}
    for name, ast in (("front", st.front), ("rear", st.rear)):
        batch = batches[name]
        stat[name] = _axle_static(batch, ast)
        pose = _solve_pose(batch, ast, st.pivot, zeros, rolls, zeros)
        rc_y, rc_z = _roll_centre(pose, st.pivot, zeros, rolls, zeros)
        track = ast.track
        k_s = stat[name].wheel_rate * track**2 / 2.0
        k_a = track**2 / 2.0 * stat[name].arb_wheel_rate_roll
        k_t = ast.tyre_stiffness * track**2 / 2.0
        k_susp = k_s + k_a
        k_roll = k_susp * k_t / (k_susp + k_t)
        mean_steer = 0.5 * (pose.steer_ground[:, 0] + pose.steer_ground[:, 1])
        ax[name] = {
            "track": track,
            "wheel_rate": stat[name].wheel_rate,
            "arb_wheel_rate_roll": stat[name].arb_wheel_rate_roll,
            "roll_stiffness_springs": k_s,
            "roll_stiffness_arb": k_a,
            "roll_stiffness_tyres": k_t,
            "roll_stiffness": k_roll,
            "rc_height": rc_z[:, i0],
            "roll_camber": pose.camber_ground[:, 1] @ w1,
            "roll_steer": mean_steer @ w1,
            "rc_migration": rc_y @ w1,
            "rc_height_curvature": rc_z @ w2,
        }
        for key, val in ax[name].items():
            out[f"{name}.{key}"] = val
    fa, ra = ax["front"], ax["rear"]
    wb = st.wheelbase
    m_u_f = 2.0 * st.front.unsprung_mass
    m_u_r = 2.0 * st.rear.unsprung_mass
    m_s = st.mass - m_u_f - m_u_r
    hu_f, hu_r = st.front.unsprung_cg_height, st.rear.unsprung_cg_height
    a_total = (1.0 - st.front_mass_ratio) * wb
    h_s = (st.mass * st.cg_height - m_u_f * hu_f - m_u_r * hu_r) / m_s
    x_s = (st.mass * (-a_total) - m_u_f * st.front.swc[:, 0] - m_u_r * st.rear.swc[:, 0]) / m_s
    a = -x_s
    t_ra = x_s / (-wb)
    h_ra = fa["rc_height"] + t_ra * (ra["rc_height"] - fa["rc_height"])
    k_f, k_r = fa["roll_stiffness"], ra["roll_stiffness"]
    moment = m_s * vm.G * (h_s - h_ra)
    denom = k_f + k_r - moment
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(denom > 0.0, denom, _NAN)
        gradient = moment / denom
        phi = m_s * ay_ref * (h_s - h_ra) / denom
        msf = m_s * (wb - a) / wb
        msr = m_s * a / wb
        tf, tr = fa["track"], ra["track"]
        lt_f = k_f * phi / tf + msf * ay_ref * fa["rc_height"] / tf + m_u_f * ay_ref * hu_f / tf
        lt_r = k_r * phi / tr + msr * ay_ref * ra["rc_height"] / tr + m_u_r * ay_ref * hu_r / tr
        total = lt_f + lt_r
        tlltd = np.where(total != 0.0, lt_f / total, _NAN)

    sv_f, sv_r = stat["front"], stat["rear"]
    ang_f_brake = np.where(st.front.brakes_outboard, sv_f.sv_cp, sv_f.sv_wc)
    ang_r_brake = np.where(st.rear.brakes_outboard, sv_r.sv_cp, sv_r.sv_wc)
    split = st.drive_split_front
    pc_x, pc_z = _pitch_centre(st, sv_f.sv_cp, sv_r.sv_cp)
    steer = _steering(batches["front"], st)
    out.update(
        {
            "wheelbase": wb,
            "sprung_mass": m_s,
            "sprung_cg_height": h_s,
            "roll_axis_height_at_cg": h_ra,
            "roll_stiffness_total": k_f + k_r,
            "roll_gradient": gradient,
            "ay_ref": np.full(b, float(ay_ref)),
            "load_transfer_front": lt_f,
            "load_transfer_rear": lt_r,
            "tlltd_front": tlltd,
            "anti_dive_front": _tan_anti(ang_f_brake, st.brake_bias_front, st),
            "anti_lift_rear": _tan_anti(ang_r_brake, 1.0 - st.brake_bias_front, st),
            "anti_squat_rear": _tan_anti(sv_r.sv_wc, 1.0 - split, st),
            "anti_lift_front_accel": np.where(split > 0.0, _tan_anti(sv_f.sv_wc, split, st), _NAN),
            "pitch_centre_x": pc_x,
            "pitch_centre_z": pc_z,
            **steer,
        }
    )
    return {k: out[k] for k in _flat_names()}


# ----------------------------------------------------------------------------- sweeps
_POSE_KINDS = ("heave", "roll", "pitch")
_CORNER_COLS = ("travel", "camber", "toe", "steer_angle", "camber_ground", "steer_ground")
_RACK_COLS = ("camber", "toe", "steer_angle", "camber_ground", "steer_ground")


def supported_sweep_columns(kind: str) -> tuple[str, ...]:
    """``vehicle_frame`` columns the layer provides for a sweep of ``kind``.

    ``heave`` / ``roll`` / ``pitch``: the four input columns, per corner ``travel``,
    ``camber``, ``toe``, ``steer_angle``, ``camber_ground``, ``steer_ground``,
    ``cp_x/_y/_z`` and ``rc_front_y/_z``, ``rc_rear_y/_z``. ``rack``: the input columns and,
    for FL and FR, ``camber``, ``toe``, ``steer_angle``, ``camber_ground``,
    ``steer_ground`` (travel-0 row, no ground re-solve). All other columns (scrub radius,
    caster, instant-centre coordinates, ARB elastic, ...) are not supported.
    """
    base = ("heave", "roll", "pitch", "rack")
    if kind == "rack":
        return (*base, *(f"{c}_{m}" for c in ("FL", "FR") for m in _RACK_COLS))
    if kind not in _POSE_KINDS:
        raise ValueError(f"sweep kind must be one of heave, roll, pitch, rack; got {kind!r}")
    cols = list(base)
    for cid in ("FL", "FR", "RL", "RR"):
        cols += [f"{cid}_{m}" for m in _CORNER_COLS]
        cols += [f"{cid}_cp_{a}" for a in "xyz"]
    cols += ["rc_front_y", "rc_front_z", "rc_rear_y", "rc_rear_z"]
    return tuple(cols)


def _pose_sweep(
    batches: Mapping[str, AxleBatch], st: VehicleStatics, kind: str, values: Arr
) -> dict[str, Arr]:
    b = st.n
    zeros = np.zeros_like(values)
    heave, roll, pitch = (values if k == kind else zeros for k in _POSE_KINDS)
    cols: dict[str, Arr] = {}
    for k, arr in (("heave", heave), ("roll", roll), ("pitch", pitch), ("rack", zeros)):
        cols[k] = np.tile(arr, (b, 1))
    for name, ast in (("front", st.front), ("rear", st.rear)):
        pose = _solve_pose(batches[name], ast, st.pivot, heave, roll, pitch)
        ids = _FV_CORNERS[name]
        for i, cid in enumerate(ids):
            cols[f"{cid}_travel"] = pose.travel[:, i]
            cols[f"{cid}_camber"] = pose.camber[:, i]
            cols[f"{cid}_toe"] = pose.toe[:, i]
            cols[f"{cid}_steer_angle"] = -pose.toe[:, i] if i == 0 else pose.toe[:, i]
            cols[f"{cid}_camber_ground"] = pose.camber_ground[:, i]
            cols[f"{cid}_steer_ground"] = pose.steer_ground[:, i]
            for j, a in enumerate("xyz"):
                cols[f"{cid}_cp_{a}"] = pose.cp[:, i, j]
        y, z = _roll_centre(pose, st.pivot, heave, roll, pitch)
        cols[f"rc_{name}_y"], cols[f"rc_{name}_z"] = y, z
    return cols


def _rack_sweep(batches: Mapping[str, AxleBatch], st: VehicleStatics, rack: Arr) -> dict[str, Arr]:
    b = st.n
    front = batches["front"]
    names = ("camber", "toe")
    ix = {n: i for i, n in enumerate(names)}
    y = _masked(front, names)
    at = Hermite(y.reshape(b, 2 * front.grid.rack.size, -1), front.grid.travel)(np.array([0.0]))
    curves = Hermite(at.reshape(b, 2, front.grid.rack.size), front.grid.rack)
    zeros = np.zeros((b, rack.size))
    cols: dict[str, Arr] = {
        "heave": zeros.copy(),
        "roll": zeros.copy(),
        "pitch": zeros.copy(),
        "rack": np.tile(rack, (b, 1)),
    }
    for cid, o, q in (("FL", 1.0, rack), ("FR", -1.0, -rack)):
        v = curves(q)  # right wheel at rack r = mirrored left wheel at -r
        cam, toe = v[:, ix["camber"]], v[:, ix["toe"]]
        steer = -toe if o > 0 else toe
        cols[f"{cid}_camber"] = cam
        cols[f"{cid}_toe"] = toe
        cols[f"{cid}_steer_angle"] = steer
        cols[f"{cid}_camber_ground"] = cam
        cols[f"{cid}_steer_ground"] = steer
    return cols


# ----------------------------------------------------------------------------- public layer
@dataclass(frozen=True)
class LayerResult:
    """Output of :func:`vehicle_layer`.

    ``summary``: flattened ``VehicleSummary`` fields (``front.rc_height``, ``tlltd_front`` ...),
    each ``(B,)`` in SI units. ``sweeps``: per requested sweep kind, ``vehicle_frame`` column
    name -> ``(B, N)`` array.
    """

    summary: dict[str, Arr]
    sweeps: dict[str, dict[str, Arr]]


def vehicle_layer(
    samples: Mapping[str, AxleBatch],
    statics: VehicleStatics,
    *,
    roll_range: float = 0.035,
    n: int = 15,
    ay_ref: float = 9.81,
    sweeps: Mapping[str, ArrayLike] | None = None,
    columns: Mapping[str, Sequence[str]] | None = None,
) -> LayerResult:
    """Vehicle summary (and optional sweeps) from the corner curves of ``B`` designs.

    Parameters
    ----------
    samples : mapping
        ``"front"`` and ``"rear"`` :class:`AxleBatch` with the same ``B`` as ``statics``.
    statics : VehicleStatics
        Output of :func:`vehicle_statics`.
    roll_range, n, ay_ref
        As :func:`suspkin.metrics.vehicle.vehicle_summary`.
    sweeps : mapping, optional
        Sweep kind (``heave``, ``roll``, ``pitch``, ``rack``) -> input values (SI).
    columns : mapping, optional
        Sweep kind -> ``vehicle_frame`` columns to return (default: all supported).

    Raises
    ------
    ValueError
        If the travel or rack grid of an axle has no node at 0 (the static pose must be a
        grid node; wheel/ARB rates and statics are read there).
    UnsupportedColumnError
        If a requested column is not in :func:`supported_sweep_columns`.
    """
    for axle in ("front", "rear"):
        if axle not in samples:
            raise ValueError(f"samples is missing the {axle!r} axle")
        if samples[axle].n != statics.n:
            raise ValueError(f"{axle}: batch size {samples[axle].n} != statics {statics.n}")
        if samples[axle].grid.travel.size < 3:
            raise ValueError("the travel grid needs at least 3 points")
        _zero_node(samples[axle].grid.travel, f"{axle} travel")
        _zero_node(samples[axle].grid.rack, f"{axle} rack")
    summary = _summary(samples, statics, roll_range, n, ay_ref)
    out: dict[str, dict[str, Arr]] = {}
    for kind, values in (sweeps or {}).items():
        supported = supported_sweep_columns(kind)
        wanted = tuple((columns or {}).get(kind, supported))
        bad = [c for c in wanted if c not in supported]
        if bad:
            raise UnsupportedColumnError(
                f"vehicle_frame column(s) {bad} not supported for a {kind!r} sweep by the "
                f"surrogate layer; supported: {list(supported)}"
            )
        vals = np.atleast_1d(np.asarray(values, dtype=np.float64))
        cols = (
            _rack_sweep(samples, statics, vals)
            if kind == "rack"
            else _pose_sweep(samples, statics, kind, vals)
        )
        out[kind] = {c: cols[c] for c in wanted}
    return LayerResult(summary, out)


# ----------------------------------------------------------------------------- layer error
_ERROR_SWEEPS: dict[str, Arr] = {
    "heave": np.linspace(-0.02, 0.02, 5),
    "roll": np.linspace(-0.035, 0.035, 15),
    "pitch": np.linspace(-0.02, 0.02, 5),
    "rack": np.linspace(-0.03, 0.03, 9),
}


def _flatten(d: Mapping[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            out.update({f"{k}.{kk}": vv for kk, vv in _flatten(v).items()})
        else:
            out[k] = _NAN if v is None else float(v)  # type: ignore[arg-type]
    return out


def _abs_err(a: Arr, b: Arr) -> float:
    """Max ``|a - b|``; both NaN counts as agreement, one NaN as NaN."""
    both = np.isnan(a) & np.isnan(b)
    d = np.where(both, 0.0, np.abs(a - b))
    return float(np.max(d)) if d.size else 0.0


def layer_error(vehicles: Sequence[Vehicle], grids: Mapping[str, AxleGrid]) -> pd.DataFrame:
    """Absolute error of the layer fed with FOM curves, one row per vehicle (SI units).

    Columns: every summary field (versus ``vehicle_summary`` with its defaults) and, for
    each sweep kind and supported column, ``sweep.<kind>.<column>`` = max over the sweep of
    the error versus ``vehicle_frame`` (heave +-20 mm, roll +-0.035 rad, pitch +-0.02 rad,
    rack +-30 mm). A field that is NaN on only one side gives NaN.
    """
    from suspkin.sweeps import sweep_vehicle, vehicle_frame

    per = [extract_axles(v, dict(grids)) for v in vehicles]
    batches = {a: stack_samples([p[a] for p in per], grids[a]) for a in ("front", "rear")}
    res = vehicle_layer(batches, vehicle_statics(vehicles), sweeps=_ERROR_SWEEPS)
    rows: list[dict[str, float]] = []
    for i, veh in enumerate(vehicles):
        fom = _flatten(vehicle_summary(veh).to_dict())
        row = {k: _abs_err(res.summary[k][i : i + 1], np.array([fom[k]])) for k in fom}
        for kind, vals in _ERROR_SWEEPS.items():
            df = vehicle_frame(sweep_vehicle(veh, **{kind: vals}), veh)  # type: ignore[arg-type]
            for col, arr in res.sweeps[kind].items():
                row[f"sweep.{kind}.{col}"] = _abs_err(arr[i], df[col].to_numpy(dtype=np.float64))
        rows.append(row)
    return pd.DataFrame(rows)
