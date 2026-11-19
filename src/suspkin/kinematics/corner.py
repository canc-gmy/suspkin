"""Single-corner kinematic solver (rigid bodies, ideal joints).

Body frame = ISO 8855 vehicle frame; the ground moves relative to the body.
YAML hardpoints are the static pose (travel 0, rack 0).

Formulation (3 unknowns): the UCA rotates by ``theta_uca`` and the LCA by
``theta_lca`` about their chassis axes, which fixes both ball joints. The upright
pose is the minimal rotation aligning the static ``UBJ - LBJ`` with the new one,
followed by a spin ``psi`` about the new ``LBJ -> UBJ`` axis, anchored at the LBJ.
Solved by a damped Newton with an analytic Jacobian (scipy hybr as fallback).
Closing equations: ball-joint distance, tie rod length, prescribed wheel-centre
height. Rocker, spring and toggle margin follow in closed form.

Sign convention: positive ``theta_uca``/``theta_lca``/``psi``/``rocker_angle`` are
right-hand rotations about the left corner's axes. Right corners are reflections
of left corners, so every angle is reported with a per-side sign (left/right equal
for mirrored motion); the physical rotation is ``side_sign * angle``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import cos, sin, sqrt
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root

from suspkin.geom_math.vectors import wheel_spin_axis
from suspkin.geometry.corners import Corner, CornerId
from suspkin.kinematics import _fast
from suspkin.kinematics.rocker import KinematicsError, RockerCtx, RockerTuple, side_sign

_RESIDUAL_TOL = 1e-9
_MAX_HALVINGS = 6
_COLD_LIMIT = 0.005
_STAGE_STEP = 0.005
_MAX_ANGLE_STEP = 0.05  # rad; larger jump from the guess = branch jump, step is halved
_NEWTON_MAX_ITER = 30
_NEWTON_F_TOL = 1e-12
_NEWTON_X_TOL = 1e-14
_NEWTON_STALL_F_TOL = 1e-10
_MAX_BACKTRACKS = 10

Vec = NDArray[np.float64]
_V3 = tuple[float, float, float]


@dataclass(frozen=True)
class CornerState:
    """Solved corner pose. Lengths in m, angles in rad.

    ``coilover_in_range`` flags whether ``coilover_length`` lies within the
    coilover's ``[length_min, length_max]``.
    """

    corner_id: CornerId
    travel: float
    rack: float
    theta_uca: float
    theta_lca: float
    psi: float
    rocker_angle: float
    points: Mapping[str, Vec]
    spin_axis: Vec
    coilover_length: float
    toggle_margin: float
    max_residual: float
    coilover_in_range: bool = True


def _tup(v: ArrayLike) -> _V3:
    a = np.asarray(v, dtype=np.float64)
    return (float(a[0]), float(a[1]), float(a[2]))


def _sub(a: _V3, b: _V3) -> _V3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _unit(a: _V3) -> _V3:
    n = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    return (a[0] / n, a[1] / n, a[2] / n)


def _cross(a: _V3, b: _V3) -> _V3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: _V3, b: _V3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _rodrigues(k: _V3, cs: float, sn: float, v: _V3) -> _V3:
    """Rotate ``v`` about the unit axis ``k`` (``cs``/``sn`` = cos/sin of the angle)."""
    k0, k1, k2 = k
    v0, v1, v2 = v
    f = (k0 * v0 + k1 * v1 + k2 * v2) * (1.0 - cs)
    return (
        v0 * cs + (k1 * v2 - k2 * v1) * sn + k0 * f,
        v1 * cs + (k2 * v0 - k0 * v2) * sn + k1 * f,
        v2 * cs + (k0 * v1 - k1 * v0) * sn + k2 * f,
    )


def _min_rot(w: _V3, ik: float, r: _V3) -> _V3:
    """Apply ``I + K + K^2 / (1 + c)`` (minimal rotation, ``w = u x v``) to ``r``."""
    w0, w1, w2 = w
    r0, r1, r2 = r
    a0 = w1 * r2 - w2 * r1
    a1 = w2 * r0 - w0 * r2
    a2 = w0 * r1 - w1 * r0
    return (
        r0 + a0 + (w1 * a2 - w2 * a1) * ik,
        r1 + a1 + (w2 * a0 - w0 * a2) * ik,
        r2 + a2 + (w0 * a1 - w1 * a0) * ik,
    )


class _Ctx:
    """Precomputed static data of a corner (plain floats/tuples for the scalar fast path)."""

    def __init__(self, corner: Corner) -> None:
        p = corner.points
        self.corner = corner
        self.sign = side_sign(corner)
        self.p = p
        self.uca_front = _tup(p["uca_front"])
        self.lca_front = _tup(p["lca_front"])
        self.uca_hat = _unit(_tup(p["uca_rear"] - p["uca_front"]))
        self.lca_hat = _unit(_tup(p["lca_rear"] - p["lca_front"]))
        self.ubj0 = np.asarray(p["uca_outer"])
        self.lbj0 = np.asarray(p["lca_outer"])
        d0 = self.ubj0 - self.lbj0
        self.len_ubj = float(np.linalg.norm(d0))
        self.len_tie = float(np.linalg.norm(p["tie_outer"] - p["tie_inner"]))
        self.wc0 = np.asarray(p["wheel_center"])
        self.r_ubj = _sub(_tup(self.ubj0), self.uca_front)
        self.r_lbj = _sub(_tup(self.lbj0), self.lca_front)
        self.d0_hat = _unit(_tup(d0))
        self.r_tie = _tup(p["tie_outer"] - self.lbj0)
        self.r_wc = _tup(self.wc0 - self.lbj0)
        self.wc0_z = float(self.wc0[2])
        cfg = corner.axle_cfg
        self.spin0 = wheel_spin_axis(corner.side, cfg.wheel.static_camber, cfg.wheel.static_toe)
        self.radius = cfg.wheel.tyre_radius
        self.mount = cfg.pushrod_mount
        self.rack_axis = None if cfg.steering is None else np.asarray(cfg.steering.rack_axis)
        self.rocker = RockerCtx(corner)
        self.po0 = _tup(p["pushrod_outer"])
        self.r_po = _sub(self.po0, _tup(self.lbj0))
        self.spin0_t = _tup(self.spin0)
        self.static_push = float(np.linalg.norm(p["pushrod_outer"] - p["pushrod_inner"]))
        self.spring_chassis = _tup(p["spring_chassis"])
        self.fast_arrays: tuple[Vec, Vec] | None = None  # packed lazily by ``_fast.arrays``


def _eval(
    c: _Ctx, a: float, b: float, psi: float, travel: float, ti: _V3, jac: bool
) -> tuple[_V3, list[list[float]] | None]:
    """Residuals F and (optionally) the analytic Jacobian dF/d(a, b, psi).

    ``a``/``b``/``psi`` are ``theta_uca``/``theta_lca``/``psi``. The upright rotation is
    ``R = Q M`` with ``M`` the minimal rotation ``d0 -> d`` and ``Q`` the spin about
    ``n = d / |d|``. For a change ``dd`` of ``d`` the spatial angular velocity of the
    upright is ``sin(phi) n' + (1 - cos(phi)) n x n' + Q w_M`` with
    ``n' = (dd - n (n.dd)) / |d|`` and
    ``w_M = u x n' - w (u.n') / (1 + c) + w x (u x n') / (1 + c)``
    (``u = d0_hat``, ``w = u x n``, ``c = u.n``); the spin adds ``s n``.
    """
    s = c.sign
    ru = _rodrigues(c.uca_hat, cos(s * a), sin(s * a), c.r_ubj)
    lr = _rodrigues(c.lca_hat, cos(s * b), sin(s * b), c.r_lbj)
    uf, lf = c.uca_front, c.lca_front
    ubj = (uf[0] + ru[0], uf[1] + ru[1], uf[2] + ru[2])
    lbj = (lf[0] + lr[0], lf[1] + lr[1], lf[2] + lr[2])
    d = _sub(ubj, lbj)
    dn = sqrt(_dot(d, d))
    n = (d[0] / dn, d[1] / dn, d[2] / dn)
    u0 = c.d0_hat
    cc = _dot(u0, n)
    if cc < -1.0 + 1e-12:
        raise ValueError("upright axis flipped")
    ik = 1.0 / (1.0 + cc)
    w = _cross(u0, n)
    phi = s * psi
    cph, sph = cos(phi), sin(phi)
    pt = _rodrigues(n, cph, sph, _min_rot(w, ik, c.r_tie))
    pw = _rodrigues(n, cph, sph, _min_rot(w, ik, c.r_wc))
    e = (lbj[0] + pt[0] - ti[0], lbj[1] + pt[1] - ti[1], lbj[2] + pt[2] - ti[2])
    en = sqrt(_dot(e, e))
    f = (dn - c.len_ubj, en - c.len_tie, lbj[2] + pw[2] - c.wc0_z - travel)
    if not jac:
        return f, None

    n0, n1, n2 = n
    u00, u01, u02 = u0
    w0, w1, w2 = w
    pt0, pt1, pt2 = pt
    pw0, pw1 = pw[0], pw[1]
    e0, e1, e2 = e
    vc = 1.0 - cph

    def column(dd: _V3, dl: _V3) -> _V3:
        # Vector helpers inlined (hot path); same operation order as _cross/_dot.
        k = n0 * dd[0] + n1 * dd[1] + n2 * dd[2]
        p0 = (dd[0] - n0 * k) / dn
        p1 = (dd[1] - n1 * k) / dn
        p2 = (dd[2] - n2 * k) / dn
        t0 = u01 * p2 - u02 * p1
        t1 = u02 * p0 - u00 * p2
        t2 = u00 * p1 - u01 * p0
        kk = (u00 * p0 + u01 * p1 + u02 * p2) * ik
        m0 = t0 - w0 * kk + (w1 * t2 - w2 * t1) * ik
        m1 = t1 - w1 * kk + (w2 * t0 - w0 * t2) * ik
        m2 = t2 - w2 * kk + (w0 * t1 - w1 * t0) * ik
        q0, q1, q2 = _rodrigues(n, cph, sph, (m0, m1, m2))
        o0 = sph * p0 + vc * (n1 * p2 - n2 * p1) + q0
        o1 = sph * p1 + vc * (n2 * p0 - n0 * p2) + q1
        o2 = sph * p2 + vc * (n0 * p1 - n1 * p0) + q2
        return (
            k,
            (
                e0 * (dl[0] + (o1 * pt2 - o2 * pt1))
                + e1 * (dl[1] + (o2 * pt0 - o0 * pt2))
                + e2 * (dl[2] + (o0 * pt1 - o1 * pt0))
            )
            / en,
            dl[2] + (o0 * pw1 - o1 * pw0),
        )

    da = _cross(c.uca_hat, ru)
    db = _cross(c.lca_hat, lr)
    dl_b = (s * db[0], s * db[1], s * db[2])
    col_a = column((s * da[0], s * da[1], s * da[2]), (0.0, 0.0, 0.0))
    col_b = column((-dl_b[0], -dl_b[1], -dl_b[2]), dl_b)
    om = (s * n[0], s * n[1], s * n[2])
    vt = _cross(om, pt)
    vw = _cross(om, pw)
    col_p = (0.0, (e[0] * vt[0] + e[1] * vt[1] + e[2] * vt[2]) / en, vw[2])
    return f, [
        [col_a[0], col_b[0], col_p[0]],
        [col_a[1], col_b[1], col_p[1]],
        [col_a[2], col_b[2], col_p[2]],
    ]


def _upright_map(
    c: _Ctx, th_u: float, th_l: float, psi: float
) -> tuple[_V3, _V3, Callable[[_V3], _V3]]:
    """Return ``(ubj, lbj, rot)``; ``rot(v)`` applies the upright rotation to a vector."""
    s = c.sign
    ru = _rodrigues(c.uca_hat, cos(s * th_u), sin(s * th_u), c.r_ubj)
    lr = _rodrigues(c.lca_hat, cos(s * th_l), sin(s * th_l), c.r_lbj)
    uf, lf = c.uca_front, c.lca_front
    ubj = (uf[0] + ru[0], uf[1] + ru[1], uf[2] + ru[2])
    lbj = (lf[0] + lr[0], lf[1] + lr[1], lf[2] + lr[2])
    n = _unit(_sub(ubj, lbj))
    cc = _dot(c.d0_hat, n)
    if cc < -1.0 + 1e-12:
        raise ValueError("upright axis flipped")
    ik = 1.0 / (1.0 + cc)
    w = _cross(c.d0_hat, n)
    cph, sph = cos(s * psi), sin(s * psi)

    def rot(v: _V3) -> _V3:
        return _rodrigues(n, cph, sph, _min_rot(w, ik, v))

    return ubj, lbj, rot


def _residuals(c: _Ctx, x: Vec, travel: float, tie_inner: Vec) -> Vec:
    f, _ = _eval(c, float(x[0]), float(x[1]), float(x[2]), travel, _tup(tie_inner), False)
    return np.array(f)


def _jacobian(c: _Ctx, x: Vec, travel: float, tie_inner: Vec) -> Vec:
    _, jm = _eval(c, float(x[0]), float(x[1]), float(x[2]), travel, _tup(tie_inner), True)
    assert jm is not None
    return np.array(jm)


def _solve3(j: list[list[float]], f: _V3) -> _V3 | None:
    """Solve ``J dx = -F`` for a 3x3 system by Cramer's rule (None if singular)."""
    (a, b, c), (d, e, g), (h, i, k) = j
    c0 = e * k - g * i
    c1 = g * h - d * k
    c2 = d * i - e * h
    det = a * c0 + b * c1 + c * c2
    if abs(det) < 1e-30:
        return None
    r0, r1, r2 = -f[0], -f[1], -f[2]
    x0 = (r0 * c0 + b * (g * r2 - r1 * k) + c * (r1 * i - e * r2)) / det
    x1 = (a * (r1 * k - g * r2) + r0 * c1 + c * (d * r2 - r1 * h)) / det
    x2 = (a * (e * r2 - r1 * i) + b * (r1 * h - d * r2) + r0 * c2) / det
    return (x0, x1, x2)


def _newton(c: _Ctx, x0: _V3, travel: float, ti: _V3) -> tuple[_V3, _V3] | None:
    """Damped Newton on the 3x3 closing system; None if it does not converge."""
    if _fast.available():
        handled, res = _fast.newton(c, x0, travel, ti)
        if handled:
            return res
    x = x0
    try:
        f, jm = _eval(c, x[0], x[1], x[2], travel, ti, True)
        f2 = f[0] * f[0] + f[1] * f[1] + f[2] * f[2]
        for _ in range(_NEWTON_MAX_ITER):
            if max(abs(f[0]), abs(f[1]), abs(f[2])) < _NEWTON_F_TOL:
                return x, f
            assert jm is not None
            dx = _solve3(jm, f)
            if dx is None:
                return None
            lam = 1.0
            for _ in range(_MAX_BACKTRACKS):
                xn = (x[0] + lam * dx[0], x[1] + lam * dx[1], x[2] + lam * dx[2])
                try:
                    fn, jn = _eval(c, xn[0], xn[1], xn[2], travel, ti, True)
                except ValueError:
                    lam *= 0.5
                    continue
                f2n = fn[0] * fn[0] + fn[1] * fn[1] + fn[2] * fn[2]
                if f2n < f2:
                    break
                lam *= 0.5
            else:
                return None
            step = lam * max(abs(dx[0]), abs(dx[1]), abs(dx[2]))
            x, f, jm, f2 = xn, fn, jn, f2n
            if step < _NEWTON_X_TOL:
                stalled_ok = max(abs(f[0]), abs(f[1]), abs(f[2])) < _NEWTON_STALL_F_TOL
                return (x, f) if stalled_ok else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return None


def _hybr(c: _Ctx, x0: _V3, travel: float, tie_inner: Vec) -> Vec:
    """Fallback: scipy MINPACK hybrid solve with the analytic Jacobian."""
    start = np.array(x0)
    if float(np.linalg.norm(start)) < 1e-4:
        # hybr bounds its first step by |x0|; a tiny non-zero x0 would stall it.
        start = np.zeros(3)
    sol = root(
        lambda x: _residuals(c, x, travel, tie_inner),
        start,
        jac=lambda x: _jacobian(c, x, travel, tie_inner),
        method="hybr",
        tol=1e-14,
    )
    return np.asarray(sol.x, dtype=np.float64)


def solve_corner(
    corner: Corner,
    travel: float,
    rack: float = 0.0,
    *,
    guess: CornerState | None = None,
    _ctx: _Ctx | None = None,
) -> CornerState:
    """Solve one corner for a wheel travel [m] (bump positive) and rack offset [m].

    Parameters
    ----------
    corner : Corner
        Side-resolved corner from ``build_corners``.
    travel, rack : float
        Wheel-centre vertical travel and rack displacement along ``rack_axis``.
    guess : CornerState, optional
        Neighbouring solution used as the starting point (continuation). Without a
        guess, large ``travel``/``rack`` (> 5 mm) are reached by internal continuation
        from the static pose, so a cold solve lands on the same branch as a sweep. A far
        guess (direct solve fails or jumps branch) is reached by staged continuation
        from the guess.

    Raises
    ------
    ValueError
        If ``rack != 0`` on an axle without steering.
    KinematicsError
        If the mechanism cannot be assembled or the solver does not converge.
    """
    c = _Ctx(corner) if _ctx is None else _ctx
    return _state(c, _solve_light(c, travel, rack, guess))


def _state(c: _Ctx, s: CornerState | _Pose) -> CornerState:
    return s if isinstance(s, CornerState) else _finish(c, s)


def _solve_light(
    c: _Ctx, travel: float, rack: float, guess: CornerState | _Pose | None
) -> CornerState | _Pose:
    """``solve_corner`` without building the point table on the direct path.

    Returns a ``_Pose`` when the solve is direct and a ``CornerState`` when it went
    through continuation; ``_state`` turns either into the ``solve_corner`` result.
    """
    if guess is None:
        if abs(travel) > _COLD_LIMIT or abs(rack) > _COLD_LIMIT:
            return _staged(c, travel, rack)
        return _solve_pose(c, travel, rack, None)
    try:
        return _solve_pose(c, travel, rack, guess)
    except KinematicsError:
        # Far guess: walk from the guess to the target in guarded steps.
        start = _state(c, guess)
        state = _walk(_Advance(c, start.rack, 0.0, "travel"), start, travel)
        return _walk(_Advance(c, 0.0, travel, "rack"), state, rack)


class _Guess(Protocol):
    """Fields of a neighbouring solution used to seed a solve (``CornerState`` or ``_Pose``)."""

    @property
    def travel(self) -> float: ...
    @property
    def rack(self) -> float: ...
    @property
    def theta_uca(self) -> float: ...
    @property
    def theta_lca(self) -> float: ...
    @property
    def psi(self) -> float: ...
    @property
    def rocker_angle(self) -> float: ...


@dataclass(frozen=True)
class _Pose:
    """Solved corner pose before the point table is built (see ``_pose`` / ``_finish``).

    Enough to seed the next solve and to locate the wheel (``wheel_center``,
    ``spin_axis``); ``_finish`` turns it into the ``CornerState`` of the same solution.
    """

    travel: float
    rack: float
    theta_uca: float
    theta_lca: float
    psi: float
    rocker_angle: float
    tie_inner: Vec
    residual: _V3
    ubj: _V3
    lbj: _V3
    tie_outer: _V3
    wheel_center: _V3
    pushrod_outer: _V3
    rocker: RockerTuple
    spin_axis: _V3


def _tie_inner(c: _Ctx, rack: float) -> Vec:
    if c.rack_axis is None:
        if rack != 0.0:
            raise ValueError("rack must be 0 on an axle without steering")
        return np.asarray(c.p["tie_inner"])
    out: Vec = c.p["tie_inner"] + rack * c.rack_axis
    return out


def _solve_raw(
    c: _Ctx,
    travel: float,
    rack: float,
    guess: _Guess | None,
    x_pred: _V3 | None = None,
) -> CornerState:
    return _finish(c, _solve_pose(c, travel, rack, guess, x_pred))


def _solve_pose(
    c: _Ctx,
    travel: float,
    rack: float,
    guess: _Guess | None,
    x_pred: _V3 | None = None,
) -> _Pose:
    """Solve the closing angles and the rocker (no point table, see ``_finish``)."""
    tie_inner = _tie_inner(c, rack)
    ti = _tup(tie_inner)

    if guess is None:
        starts: list[_V3] = [(0.0, 0.0, 0.0)]
        phi0 = 0.0
    else:
        xg = (guess.theta_uca, guess.theta_lca, guess.psi)
        starts = [xg] if x_pred is None else [x_pred, xg]
        phi0 = guess.rocker_angle

    def check(x: _V3, worst: float) -> None:
        if not (worst <= _RESIDUAL_TOL):
            raise KinematicsError(
                f"no convergence at travel {travel:.6f} m, rack {rack:.6f} m "
                f"(residual {worst:.2e} m)"
            )
        if guess is not None:
            jump = max(
                abs(x[0] - guess.theta_uca),
                abs(x[1] - guess.theta_lca),
                abs(x[2] - guess.psi),
            )
            if jump > _MAX_ANGLE_STEP:
                raise KinematicsError(
                    f"branch jump at travel {travel:.6f} m, rack {rack:.6f} m "
                    f"(angle change {jump:.3f} rad from the guess)"
                )

    converged = False
    err: KinematicsError | None = None
    for x0 in starts:  # predictor first, then the plain guess
        result = _newton(c, x0, travel, ti)
        if result is None:
            continue
        converged = True
        xs, f = result
        try:
            check(xs, max(abs(f[0]), abs(f[1]), abs(f[2])))
        except KinematicsError as exc:
            err = exc
            continue
        return _pose(c, xs, f, travel, rack, tie_inner, phi0)
    if converged and err is not None:
        raise err
    try:
        x = _hybr(c, starts[-1], travel, tie_inner)
        if not np.all(np.isfinite(x)):
            raise KinematicsError(f"solver diverged at travel {travel:.6f} m")
        xs = (float(x[0]), float(x[1]), float(x[2]))
        f, _ = _eval(c, xs[0], xs[1], xs[2], travel, ti, False)
        worst = max(abs(f[0]), abs(f[1]), abs(f[2]))
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise KinematicsError(f"solver failed at travel {travel:.6f} m: {exc}") from exc
    check(xs, worst)
    return _pose(c, xs, f, travel, rack, tie_inner, phi0)


def _pose(
    c: _Ctx, x: _V3, f: _V3, travel: float, rack: float, tie_inner: Vec, phi0: float
) -> _Pose:
    """Upright points, rocker and spin axis of the solved angles ``x`` (residuals ``f``)."""
    th_u, th_l, psi = x
    s = c.sign
    ubj, lbj, rot = _upright_map(c, th_u, th_l, psi)

    def on_upright(r: _V3) -> _V3:
        v = rot(r)
        return (lbj[0] + v[0], lbj[1] + v[1], lbj[2] + v[2])

    if c.mount == "upright":
        po = on_upright(c.r_po)
    elif c.mount == "lca":
        rr = _rodrigues(c.lca_hat, cos(s * th_l), sin(s * th_l), _sub(c.po0, c.lca_front))
        po = (rr[0] + c.lca_front[0], rr[1] + c.lca_front[1], rr[2] + c.lca_front[2])
    else:
        rr = _rodrigues(c.uca_hat, cos(s * th_u), sin(s * th_u), _sub(c.po0, c.uca_front))
        po = (rr[0] + c.uca_front[0], rr[1] + c.uca_front[1], rr[2] + c.uca_front[2])
    try:
        rk = c.rocker.solve_tuple(po, phi0)
    except KinematicsError as exc:
        raise KinematicsError(f"{exc} at travel {travel:.6f} m") from exc
    return _Pose(
        travel=float(travel),
        rack=float(rack),
        theta_uca=th_u,
        theta_lca=th_l,
        psi=psi,
        rocker_angle=rk[0],
        tie_inner=tie_inner,
        residual=f,
        ubj=ubj,
        lbj=lbj,
        tie_outer=on_upright(c.r_tie),
        wheel_center=on_upright(c.r_wc),
        pushrod_outer=po,
        rocker=rk,
        spin_axis=_unit(rot(c.spin0_t)),
    )


def _finish(c: _Ctx, q: _Pose) -> CornerState:
    """Build the ``CornerState`` (point table, coilover, residual) of a solved pose."""
    angle, p_in, p_spring, p_arb, toggle = q.rocker
    wc, spin = q.wheel_center, q.spin_axis
    fz = spin[2]
    up = _unit((-fz * spin[0], -fz * spin[1], 1.0 - fz * spin[2]))
    # Static points are shared with the corner (read-only); moved points are new arrays.
    pts: dict[str, Vec] = dict(c.p)
    moved: dict[str, Vec] = {
        "uca_outer": np.array(q.ubj),
        "lca_outer": np.array(q.lbj),
        "tie_outer": np.array(q.tie_outer),
        "tie_inner": np.array(q.tie_inner),
        "wheel_center": np.array(wc),
        "pushrod_outer": np.array(q.pushrod_outer),
        "pushrod_inner": np.array(p_in),
        "rocker_spring": np.array(p_spring),
        "contact_patch": np.array(
            (wc[0] - c.radius * up[0], wc[1] - c.radius * up[1], wc[2] - c.radius * up[2])
        ),
    }
    if p_arb is not None:
        moved["rocker_arb"] = np.array(p_arb)
    for v in moved.values():
        v.setflags(write=False)
    pts.update(moved)

    d_coil = _sub(p_spring, c.spring_chassis)
    cl = sqrt(_dot(d_coil, d_coil))
    coil = c.corner.axle_cfg.coilover
    d_push = _sub(q.pushrod_outer, p_in)
    push_len = sqrt(_dot(d_push, d_push))
    f = q.residual
    max_res = max(max(abs(f[0]), abs(f[1]), abs(f[2])), abs(push_len - c.static_push))
    return CornerState(
        corner_id=c.corner.id,
        travel=q.travel,
        rack=q.rack,
        theta_uca=q.theta_uca,
        theta_lca=q.theta_lca,
        psi=q.psi,
        rocker_angle=angle,
        points=pts,
        spin_axis=np.array(spin),
        coilover_length=cl,
        toggle_margin=toggle,
        max_residual=max_res,
        coilover_in_range=bool(coil.length_min <= cl <= coil.length_max),
    )


class _Advance:
    """Continuation helper solving over either travel or rack (one shared ``_Ctx``)."""

    def __init__(
        self, ctx: _Ctx, rack: float, travel: float, over: Literal["travel", "rack"]
    ) -> None:
        self.ctx, self.rack, self.travel, self.over = ctx, rack, travel, over

    def solve(
        self, value: float, guess: CornerState | None, prior: CornerState | None = None
    ) -> CornerState:
        pred = None if guess is None or prior is None else self._predict(value, guess, prior)
        if self.over == "travel":
            return _solve_raw(self.ctx, value, self.rack, guess, pred)
        return _solve_raw(self.ctx, self.travel, value, guess, pred)

    def _predict(self, value: float, state: CornerState, prior: CornerState) -> _V3 | None:
        """Linear extrapolation of the angles from the two previous states."""
        dv = self.value(state) - self.value(prior)
        if abs(dv) < 1e-12:
            return None
        r = (value - self.value(state)) / dv
        if abs(r) > 4.0:
            return None
        return (
            state.theta_uca + r * (state.theta_uca - prior.theta_uca),
            state.theta_lca + r * (state.theta_lca - prior.theta_lca),
            state.psi + r * (state.psi - prior.psi),
        )

    def value(self, state: CornerState) -> float:
        return state.travel if self.over == "travel" else state.rack


def _reach(
    adv: _Advance,
    state: CornerState,
    target: float,
    prior: CornerState | None = None,
    depth: int = 0,
) -> CornerState:
    """Solve at ``target`` from ``state``; halve the step on failure (max 6 halvings)."""
    try:
        return adv.solve(target, state, prior)
    except KinematicsError:
        if depth >= _MAX_HALVINGS:
            raise
    mid = 0.5 * (adv.value(state) + target)
    half = _reach(adv, state, mid, prior, depth + 1)
    return _reach(adv, half, target, state, depth + 1)


def _walk(adv: _Advance, state: CornerState, target: float) -> CornerState:
    """Step from ``state`` to ``target`` in increments of at most ``_STAGE_STEP``."""
    start = adv.value(state)
    n = max(1, int(np.ceil(abs(target - start) / _STAGE_STEP)))
    prior: CornerState | None = None
    for k in range(1, n + 1):
        new = _reach(adv, state, start + (target - start) * k / n, prior)
        prior, state = state, new
    return state


def _staged(c: _Ctx, travel: float, rack: float) -> CornerState:
    """Reach (travel, rack) from the static pose: travel first at rack 0, then rack."""
    state = _solve_raw(c, 0.0, 0.0, None)
    state = _walk(_Advance(c, 0.0, 0.0, "travel"), state, travel)
    return _walk(_Advance(c, 0.0, travel, "rack"), state, rack)


def _cold(adv: _Advance, value: float) -> CornerState:
    if adv.over == "travel":
        return solve_corner(adv.ctx.corner, value, adv.rack)
    return solve_corner(adv.ctx.corner, adv.travel, value)


def _continuation(
    adv: _Advance, values: ArrayLike, first_guess: CornerState | None = None
) -> list[CornerState]:
    vals = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if vals.size == 0:
        return []
    start = int(np.argmin(np.abs(vals)))
    out: dict[int, CornerState] = {}
    try:
        out[start] = (
            _reach(adv, first_guess, float(vals[start]))
            if first_guess is not None
            else _cold(adv, float(vals[start]))
        )
    except KinematicsError:
        if abs(vals[start]) < 1e-12:
            raise
        base = first_guess if first_guess is not None else _solve_raw(adv.ctx, 0.0, 0.0, None)
        out[start] = _reach(adv, base, float(vals[start]))
    for direction in (1, -1):
        prev = out[start]
        prior: CornerState | None = None
        i = start + direction
        while 0 <= i < vals.size:
            new = _reach(adv, prev, float(vals[i]), prior)
            out[i] = new
            prior, prev = prev, new
            i += direction
    return [out[i] for i in range(vals.size)]


def sweep_corner(corner: Corner, travels: ArrayLike, rack: float = 0.0) -> list[CornerState]:
    """Solve a travel sweep by continuation outward from the point nearest to 0.

    Returns one state per requested travel, in input order. On non-convergence the
    step is halved (up to 6 times) before ``KinematicsError`` is raised.
    """
    return _continuation(_Advance(_Ctx(corner), rack, 0.0, "travel"), travels)


def sweep_rack(corner: Corner, racks: ArrayLike, travel: float = 0.0) -> list[CornerState]:
    """Solve a rack sweep at fixed travel by continuation (see ``sweep_corner``).

    A non-zero ``travel`` is first reached at rack 0 from the static pose.
    """
    c = _Ctx(corner)
    adv = _Advance(c, 0.0, travel, "rack")
    first = None
    if abs(travel) > 0.0:
        first = _walk(_Advance(c, 0.0, 0.0, "travel"), _solve_raw(c, 0.0, 0.0, None), travel)
    return _continuation(adv, racks, first)
