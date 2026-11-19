"""Optional numba fast path of the corner solver (pure-Python fallback when unavailable).

numba is imported lazily on first use (never at module import). ``SUSPKIN_NO_NUMBA`` (any
value except empty or ``0``) disables the fast path. This module holds both the numba
kernels (scalar ports of ``corner._eval`` / ``_solve3`` / ``_newton`` and of the secant
iteration of ``vehicle._solve_corner_ground``) and the wrappers that pack the static geometry
into flat arrays and call them. Kernels are plain Python functions until :func:`kernels`
compiles them with ``njit``; they never raise: every failure is mapped to a status code and
the caller falls back to the pure-Python path.

Status of ``ground_kernel``: 0 ok, 1 travel out of range, 2 Newton failed, 3 residual too
large, 4 branch jump, 5 flipped upright, 6 rocker unreachable, 7 non-finite, 8 no convergence.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable
from math import acos, atan2, cos, hypot, pi, sin, sqrt
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.vectors import _V3

if TYPE_CHECKING:
    from suspkin.kinematics.corner import _Ctx

Vec = NDArray[np.float64]

_mod: ModuleType | None = None
_tried = False
_enabled: bool | None = None  # cached decision of ``available()``
_MOUNT = {"upright": 0.0, "lca": 1.0, "uca": 2.0}
_RAW: dict[str, Callable[..., Any]] = {}  # kernel name -> plain-Python function
_jitted = False


def _kernel(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register ``fn`` as a kernel; ``_compile`` swaps it for its numba dispatcher."""
    _RAW[fn.__name__] = fn
    return fn


def _compile() -> None:
    """Import numba (``ImportError`` if missing) and replace every kernel by its dispatcher.

    The module globals are rebound before any call, so kernels that call each other resolve
    to dispatchers at compile time.
    """
    global _jitted
    from numba import njit

    if _jitted:
        return
    jit = njit(cache=True, error_model="numpy")
    g = globals()
    for name, fn in _RAW.items():
        g[name] = jit(fn)
    _jitted = True


# --------------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------------

# Keep in sync with corner.py / vehicle.py (a test asserts equality).
RESIDUAL_TOL = 1e-9
MAX_ANGLE_STEP = 0.05
NEWTON_MAX_ITER = 30
NEWTON_F_TOL = 1e-12
NEWTON_X_TOL = 1e-14
NEWTON_STALL_F_TOL = 1e-10
MAX_BACKTRACKS = 10
G_TOL = 1e-11
MAX_ITER = 30
MAX_TRAVEL = 0.25


@_kernel
def _rod(k0, k1, k2, cs, sn, v0, v1, v2):  # type: ignore[no-untyped-def]
    f = (k0 * v0 + k1 * v1 + k2 * v2) * (1.0 - cs)
    return (
        v0 * cs + (k1 * v2 - k2 * v1) * sn + k0 * f,
        v1 * cs + (k2 * v0 - k0 * v2) * sn + k1 * f,
        v2 * cs + (k0 * v1 - k1 * v0) * sn + k2 * f,
    )


@_kernel
def _minrot(w0, w1, w2, ik, r0, r1, r2):  # type: ignore[no-untyped-def]
    a0 = w1 * r2 - w2 * r1
    a1 = w2 * r0 - w0 * r2
    a2 = w0 * r1 - w1 * r0
    return (
        r0 + a0 + (w1 * a2 - w2 * a1) * ik,
        r1 + a1 + (w2 * a0 - w0 * a2) * ik,
        r2 + a2 + (w0 * a1 - w1 * a0) * ik,
    )


@_kernel
def _column(  # type: ignore[no-untyped-def]
    dd0, dd1, dd2, dl0, dl1, dl2,
    n0, n1, n2, dn, u00, u01, u02, ik, w0, w1, w2,
    cph, sph, vc, pt0, pt1, pt2, pw0, pw1, e0, e1, e2, en,
):  # fmt: skip
    k = n0 * dd0 + n1 * dd1 + n2 * dd2
    p0 = (dd0 - n0 * k) / dn
    p1 = (dd1 - n1 * k) / dn
    p2 = (dd2 - n2 * k) / dn
    t0 = u01 * p2 - u02 * p1
    t1 = u02 * p0 - u00 * p2
    t2 = u00 * p1 - u01 * p0
    kk = (u00 * p0 + u01 * p1 + u02 * p2) * ik
    m0 = t0 - w0 * kk + (w1 * t2 - w2 * t1) * ik
    m1 = t1 - w1 * kk + (w2 * t0 - w0 * t2) * ik
    m2 = t2 - w2 * kk + (w0 * t1 - w1 * t0) * ik
    q0, q1, q2 = _rod(n0, n1, n2, cph, sph, m0, m1, m2)
    o0 = sph * p0 + vc * (n1 * p2 - n2 * p1) + q0
    o1 = sph * p1 + vc * (n2 * p0 - n0 * p2) + q1
    o2 = sph * p2 + vc * (n0 * p1 - n1 * p0) + q2
    return (
        k,
        (
            e0 * (dl0 + (o1 * pt2 - o2 * pt1))
            + e1 * (dl1 + (o2 * pt0 - o0 * pt2))
            + e2 * (dl2 + (o0 * pt1 - o1 * pt0))
        )
        / en,
        dl2 + (o0 * pw1 - o1 * pw0),
    )


@_kernel
def _eval(g, a, b, psi, travel, ti0, ti1, ti2):  # type: ignore[no-untyped-def]
    """Residuals and Jacobian; returns ``(status, f0..f2, j00, j01, ..., j22)``.

    Status 0 ok, 1 flipped upright (``ValueError`` in the Python path), 2 division by zero.
    """
    z = 0.0
    bad1 = (1, z, z, z, z, z, z, z, z, z, z, z, z)
    bad2 = (2, z, z, z, z, z, z, z, z, z, z, z, z)
    s = g[0]
    ru0, ru1, ru2 = _rod(g[7], g[8], g[9], cos(s * a), sin(s * a), g[13], g[14], g[15])
    lr0, lr1, lr2 = _rod(g[10], g[11], g[12], cos(s * b), sin(s * b), g[16], g[17], g[18])
    ubj0, ubj1, ubj2 = g[1] + ru0, g[2] + ru1, g[3] + ru2
    lbj0, lbj1, lbj2 = g[4] + lr0, g[5] + lr1, g[6] + lr2
    d0, d1, d2 = ubj0 - lbj0, ubj1 - lbj1, ubj2 - lbj2
    dn = sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    if dn == 0.0:
        return bad2
    n0, n1, n2 = d0 / dn, d1 / dn, d2 / dn
    u00, u01, u02 = g[19], g[20], g[21]
    cc = u00 * n0 + u01 * n1 + u02 * n2
    if cc < -1.0 + 1e-12:
        return bad1
    ik = 1.0 / (1.0 + cc)
    w0 = u01 * n2 - u02 * n1
    w1 = u02 * n0 - u00 * n2
    w2 = u00 * n1 - u01 * n0
    phi = s * psi
    cph, sph = cos(phi), sin(phi)
    m0, m1, m2 = _minrot(w0, w1, w2, ik, g[22], g[23], g[24])
    pt0, pt1, pt2 = _rod(n0, n1, n2, cph, sph, m0, m1, m2)
    m0, m1, m2 = _minrot(w0, w1, w2, ik, g[25], g[26], g[27])
    pw0, pw1, pw2 = _rod(n0, n1, n2, cph, sph, m0, m1, m2)
    e0 = lbj0 + pt0 - ti0
    e1 = lbj1 + pt1 - ti1
    e2 = lbj2 + pt2 - ti2
    en = sqrt(e0 * e0 + e1 * e1 + e2 * e2)
    if en == 0.0:
        return bad2
    f0 = dn - g[28]
    f1 = en - g[29]
    f2 = lbj2 + pw2 - g[30] - travel
    vc = 1.0 - cph

    da0 = g[8] * ru2 - g[9] * ru1
    da1 = g[9] * ru0 - g[7] * ru2
    da2 = g[7] * ru1 - g[8] * ru0
    db0 = g[11] * lr2 - g[12] * lr1
    db1 = g[12] * lr0 - g[10] * lr2
    db2 = g[10] * lr1 - g[11] * lr0
    dl0, dl1, dl2 = s * db0, s * db1, s * db2
    ca0, ca1, ca2 = _column(
        s * da0, s * da1, s * da2, 0.0, 0.0, 0.0,
        n0, n1, n2, dn, u00, u01, u02, ik, w0, w1, w2,
        cph, sph, vc, pt0, pt1, pt2, pw0, pw1, e0, e1, e2, en,
    )  # fmt: skip
    cb0, cb1, cb2 = _column(
        -dl0, -dl1, -dl2, dl0, dl1, dl2,
        n0, n1, n2, dn, u00, u01, u02, ik, w0, w1, w2,
        cph, sph, vc, pt0, pt1, pt2, pw0, pw1, e0, e1, e2, en,
    )  # fmt: skip
    om0, om1, om2 = s * n0, s * n1, s * n2
    vt0 = om1 * pt2 - om2 * pt1
    vt1 = om2 * pt0 - om0 * pt2
    vt2 = om0 * pt1 - om1 * pt0
    vw2 = om0 * pw1 - om1 * pw0
    cp1 = (e0 * vt0 + e1 * vt1 + e2 * vt2) / en
    return (0, f0, f1, f2, ca0, cb0, 0.0, ca1, cb1, cp1, ca2, cb2, vw2)


@_kernel
def _solve3(a, b, c, d, e, g, h, i, k, f0, f1, f2):  # type: ignore[no-untyped-def]
    c0 = e * k - g * i
    c1 = g * h - d * k
    c2 = d * i - e * h
    det = a * c0 + b * c1 + c * c2
    if abs(det) < 1e-30:
        return (False, 0.0, 0.0, 0.0)
    r0, r1, r2 = -f0, -f1, -f2
    x0 = (r0 * c0 + b * (g * r2 - r1 * k) + c * (r1 * i - e * r2)) / det
    x1 = (a * (r1 * k - g * r2) + r0 * c1 + c * (d * r2 - r1 * h)) / det
    x2 = (a * (e * r2 - r1 * i) + b * (r1 * h - d * r2) + r0 * c2) / det
    return (True, x0, x1, x2)


@_kernel
def newton_kernel(g, a, b, psi, travel, ti0, ti1, ti2):  # type: ignore[no-untyped-def]
    """Damped Newton, same algorithm as ``corner._newton``.

    Returns ``(ok, x0, x1, x2, f0, f1, f2)`` with ``ok`` 1 on success and 0 on failure.
    """
    z = 0.0
    r = _eval(g, a, b, psi, travel, ti0, ti1, ti2)
    if r[0] != 0:
        return (0, z, z, z, z, z, z)
    xa, xb, xp = a, b, psi
    f0, f1, f2 = r[1], r[2], r[3]
    j0, j1, j2, j3, j4, j5, j6, j7, j8 = r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12]
    ff = f0 * f0 + f1 * f1 + f2 * f2
    for _ in range(NEWTON_MAX_ITER):
        if max(abs(f0), abs(f1), abs(f2)) < NEWTON_F_TOL:
            return (1, xa, xb, xp, f0, f1, f2)
        sol = _solve3(j0, j1, j2, j3, j4, j5, j6, j7, j8, f0, f1, f2)
        if not sol[0]:
            return (0, z, z, z, z, z, z)
        lam = 1.0
        found = False
        ffn = ff
        rn = r
        na, nb, npsi = xa, xb, xp
        for _ in range(MAX_BACKTRACKS):
            na = xa + lam * sol[1]
            nb = xb + lam * sol[2]
            npsi = xp + lam * sol[3]
            rn = _eval(g, na, nb, npsi, travel, ti0, ti1, ti2)
            if rn[0] == 1:
                lam *= 0.5
                continue
            if rn[0] == 2:
                return (0, z, z, z, z, z, z)
            ffn = rn[1] * rn[1] + rn[2] * rn[2] + rn[3] * rn[3]
            if ffn < ff:
                found = True
                break
            lam *= 0.5
        if not found:
            return (0, z, z, z, z, z, z)
        step = lam * max(abs(sol[1]), abs(sol[2]), abs(sol[3]))
        xa, xb, xp = na, nb, npsi
        f0, f1, f2 = rn[1], rn[2], rn[3]
        j0, j1, j2, j3, j4, j5 = rn[4], rn[5], rn[6], rn[7], rn[8], rn[9]
        j6, j7, j8 = rn[10], rn[11], rn[12]
        ff = ffn
        if step < NEWTON_X_TOL:
            if max(abs(f0), abs(f1), abs(f2)) < NEWTON_STALL_F_TOL:
                return (1, xa, xb, xp, f0, f1, f2)
            return (0, z, z, z, z, z, z)
    return (0, z, z, z, z, z, z)


@_kernel
def _rocker(rk, po0, po1, po2, guess):  # type: ignore[no-untyped-def]
    """Rocker angle nearest to ``guess`` (``RockerCtx.solve_tuple``); ``(ok, angle)``."""
    sign = rk[0]
    r0 = po0 - rk[1] - rk[4]
    r1 = po1 - rk[2] - rk[5]
    r2 = po2 - rk[3] - rk[6]
    ca = r0 * rk[7] + r1 * rk[8] + r2 * rk[9]
    cb = r0 * rk[10] + r1 * rk[11] + r2 * rk[12]
    cc = 0.5 * (r0 * r0 + r1 * r1 + r2 * r2 + rk[13] - rk[14] * rk[14])
    amp = hypot(ca, cb)
    if amp < 1e-12 or abs(cc) > amp * (1.0 + 1e-12):
        return (False, 0.0)
    half = acos(max(-1.0, min(1.0, cc / amp)))
    base = atan2(cb, ca)
    gg = sign * guess
    a1 = base + half
    a2 = base - half
    d1 = abs((a1 - gg + pi) % (2.0 * pi) - pi)
    d2 = abs((a2 - gg + pi) % (2.0 * pi) - pi)
    best = a2 if d2 < d1 else a1
    phys = gg + ((best - gg + pi) % (2.0 * pi) - pi)
    return (True, sign * phys)


@_kernel
def _iterate(  # type: ignore[no-untyped-def]
    g, rk, travel, ti0, ti1, ti2, xa, xb, xp, phi_in, n0, n1, n2, offset, z_static,
):  # fmt: skip
    """One ``_solve_pose`` plus the contact height; ``(status, a, b, psi, f0..f2, phi, gval)``."""
    z = 0.0
    r = newton_kernel(g, xa, xb, xp, travel, ti0, ti1, ti2)
    if r[0] == 0:
        return (2, z, z, z, z, z, z, z, z)
    a, b, psi = r[1], r[2], r[3]
    f0, f1, f2 = r[4], r[5], r[6]
    worst = max(abs(f0), abs(f1), abs(f2))
    if not (worst <= RESIDUAL_TOL):
        return (3, z, z, z, z, z, z, z, z)
    if max(abs(a - xa), abs(b - xb), abs(psi - xp)) > MAX_ANGLE_STEP:
        return (4, z, z, z, z, z, z, z, z)

    s = g[0]
    ru0, ru1, ru2 = _rod(g[7], g[8], g[9], cos(s * a), sin(s * a), g[13], g[14], g[15])
    lr0, lr1, lr2 = _rod(g[10], g[11], g[12], cos(s * b), sin(s * b), g[16], g[17], g[18])
    lbj0, lbj1, lbj2 = g[4] + lr0, g[5] + lr1, g[6] + lr2
    d0 = g[1] + ru0 - lbj0
    d1 = g[2] + ru1 - lbj1
    d2 = g[3] + ru2 - lbj2
    dn = sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    if dn == 0.0:
        return (7, z, z, z, z, z, z, z, z)
    q0, q1, q2 = d0 / dn, d1 / dn, d2 / dn
    cc = g[19] * q0 + g[20] * q1 + g[21] * q2
    if cc < -1.0 + 1e-12:
        return (5, z, z, z, z, z, z, z, z)
    ik = 1.0 / (1.0 + cc)
    w0 = g[20] * q2 - g[21] * q1
    w1 = g[21] * q0 - g[19] * q2
    w2 = g[19] * q1 - g[20] * q0
    cph, sph = cos(s * psi), sin(s * psi)

    mount = g[43]
    if mount == 0.0:
        m0, m1, m2 = _minrot(w0, w1, w2, ik, g[34], g[35], g[36])
        v0, v1, v2 = _rod(q0, q1, q2, cph, sph, m0, m1, m2)
        po0, po1, po2 = lbj0 + v0, lbj1 + v1, lbj2 + v2
    elif mount == 1.0:
        v0, v1, v2 = _rod(g[10], g[11], g[12], cos(s * b), sin(s * b), g[37], g[38], g[39])
        po0, po1, po2 = v0 + g[4], v1 + g[5], v2 + g[6]
    else:
        v0, v1, v2 = _rod(g[7], g[8], g[9], cos(s * a), sin(s * a), g[40], g[41], g[42])
        po0, po1, po2 = v0 + g[1], v1 + g[2], v2 + g[3]
    rr = _rocker(rk, po0, po1, po2, phi_in)
    if not rr[0]:
        return (6, z, z, z, z, z, z, z, z)

    m0, m1, m2 = _minrot(w0, w1, w2, ik, g[25], g[26], g[27])
    v0, v1, v2 = _rod(q0, q1, q2, cph, sph, m0, m1, m2)
    wc0, wc1, wc2 = lbj0 + v0, lbj1 + v1, lbj2 + v2
    m0, m1, m2 = _minrot(w0, w1, w2, ik, g[31], g[32], g[33])
    v0, v1, v2 = _rod(q0, q1, q2, cph, sph, m0, m1, m2)
    sn = sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    if sn == 0.0:
        return (7, z, z, z, z, z, z, z, z)
    s0, s1, s2 = v0 / sn, v1 / sn, v2 / sn

    k = n0 * s0 + n1 * s1 + n2 * s2
    u0, u1, u2 = n0 - k * s0, n1 - k * s1, n2 - k * s2
    nu = sqrt(u0 * u0 + u1 * u1 + u2 * u2)
    if nu == 0.0:
        return (7, z, z, z, z, z, z, z, z)
    radius = g[44]
    c0 = wc0 - radius * (u0 / nu)
    c1 = wc1 - radius * (u1 / nu)
    c2 = wc2 - radius * (u2 / nu)
    gval = n0 * c0 + n1 * c1 + n2 * c2 + offset - z_static
    if not (abs(gval) < 1e300):
        return (7, z, z, z, z, z, z, z, z)
    return (0, a, b, psi, f0, f1, f2, rr[1], gval)


@_kernel
def ground_kernel(  # type: ignore[no-untyped-def]
    g, rk, n0, n1, n2, offset, z_static, ti0, ti1, ti2, t_guess, xa, xb, xp, phi0, slope0,
):  # fmt: skip
    """Secant iteration on travel (``vehicle._solve_corner_ground`` with a guess).

    ``slope0`` is NaN when there is no warm-start slope. Returns
    ``(status, travel, a, b, psi, f0, f1, f2, phi0_final, g, slope)``.
    """
    z = 0.0
    nz = n2
    if nz == 0.0:
        return (7, z, z, z, z, z, z, z, z, z, z)
    t = t_guess
    if not (abs(t) <= MAX_TRAVEL):
        return (1, z, z, z, z, z, z, z, z, z, z)
    it = _iterate(g, rk, t, ti0, ti1, ti2, xa, xb, xp, phi0, n0, n1, n2, offset, z_static)
    if it[0] != 0:
        return (it[0], z, z, z, z, z, z, z, z, z, z)
    phi_used = phi0
    t_st = t
    gv = it[8]
    slope = nz
    if 0.5 < slope0 / nz < 2.0:
        slope = slope0
    if slope == 0.0:
        return (7, z, z, z, z, z, z, z, z, z, z)
    t_prev, g_prev = t, gv
    t = t - gv / slope
    for _ in range(MAX_ITER):
        if abs(gv) < G_TOL:
            return (0, t_st, it[1], it[2], it[3], it[4], it[5], it[6], phi_used, gv, slope)
        if not (abs(t) <= MAX_TRAVEL):
            return (1, z, z, z, z, z, z, z, z, z, z)
        nxt = _iterate(
            g, rk, t, ti0, ti1, ti2, it[1], it[2], it[3], it[7], n0, n1, n2, offset, z_static
        )
        if nxt[0] != 0:
            return (nxt[0], z, z, z, z, z, z, z, z, z, z)
        phi_used = it[7]
        it = nxt
        t_st = t
        gv = it[8]
        if abs(gv) < G_TOL:
            return (0, t_st, it[1], it[2], it[3], it[4], it[5], it[6], phi_used, gv, slope)
        dg = gv - g_prev
        slope = dg / (t - t_prev) if abs(t - t_prev) > 1e-15 and abs(dg) > 1e-18 else nz
        if slope == 0.0:
            return (7, z, z, z, z, z, z, z, z, z, z)
        t_prev, g_prev = t, gv
        t = t - gv / slope
    return (8, z, z, z, z, z, z, z, z, z, z)


# --------------------------------------------------------------------------------------
# wrappers
# --------------------------------------------------------------------------------------


def _warmup(k: ModuleType) -> None:
    """Compile and call both kernels on dummy inputs (they return a failure status)."""
    geom, rk = np.zeros(45), np.zeros(15)
    k.newton_kernel(geom, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    k.ground_kernel(geom, rk, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _disable(reason: str) -> None:
    """Switch the fast path off for this process and say why (once)."""
    global _mod, _enabled
    if _mod is not None or _enabled is not False:
        warnings.warn(
            f"suspkin: numba fast path disabled, using pure Python ({reason})",
            RuntimeWarning,
            stacklevel=3,
        )
    _mod = None
    _enabled = False


def kernels() -> ModuleType | None:
    """Return the kernel module (this module), compiling the kernels on first call.

    numba is imported on first call; ``None`` if it is unavailable.

    A missing numba is silent; any other failure (import, compile, warm-up) disables the
    fast path for the process with a ``RuntimeWarning``.
    """
    global _mod, _tried
    if not _tried:
        _tried = True
        try:
            _compile()
            k = sys.modules[__name__]
        except ImportError:
            return None
        except Exception as exc:
            _disable(f"import failed: {exc!r}")
            return None
        try:
            _warmup(k)
        except Exception as exc:
            _disable(f"compilation failed: {exc!r}")
            return None
        _mod = k
    return _mod


def reset() -> None:
    """Forget the cached decision so ``SUSPKIN_NO_NUMBA`` is read again (for tests)."""
    global _enabled
    _enabled = None


def available() -> bool:
    """True if numba works and ``SUSPKIN_NO_NUMBA`` does not disable it (cached)."""
    global _enabled
    if _enabled is None:
        off = os.environ.get("SUSPKIN_NO_NUMBA", "") not in ("", "0")
        _enabled = False if off else kernels() is not None
    return _enabled


def pack_geom(c: _Ctx) -> Vec:
    """Flat corner geometry (45 floats): see the index use in the kernels."""
    po0 = np.asarray(c.po0)
    out = np.concatenate(
        (
            [c.sign],
            c.uca_front,
            c.lca_front,
            c.uca_hat,
            c.lca_hat,
            c.r_ubj,
            c.r_lbj,
            c.d0_hat,
            c.r_tie,
            c.r_wc,
            [c.len_ubj, c.len_tie, c.wc0_z],
            c.spin0_t,
            c.r_po,
            po0 - np.asarray(c.lca_front),
            po0 - np.asarray(c.uca_front),
            [_MOUNT[c.mount], c.radius],
        )
    ).astype(np.float64)
    assert out.shape == (45,)
    return out


def pack_rocker(c: _Ctx) -> Vec:
    """Flat rocker data (15 floats): sign, pivot, v_par, w, axis x w, |w|^2, pushrod length."""
    r = c.rocker
    return np.concatenate(([r.sign], r.pivot, r.v_par, r.w, r.axw, [r.ww, r.length])).astype(
        np.float64
    )


def arrays(c: _Ctx) -> tuple[Vec, Vec]:
    """Packed ``(geom, rocker)`` arrays of a corner context (cached on the context)."""
    if c.fast_arrays is None:
        c.fast_arrays = (pack_geom(c), pack_rocker(c))
    return c.fast_arrays


def newton(c: _Ctx, x0: _V3, travel: float, ti: _V3) -> tuple[bool, tuple[_V3, _V3] | None]:
    """Kernel version of ``corner._newton``: ``(handled, result)``.

    ``handled`` is False if the kernel raised (fast path now disabled): run the Python body.
    ``result`` is None if Newton does not converge.
    """
    k = kernels()
    assert k is not None
    try:
        r = k.newton_kernel(arrays(c)[0], x0[0], x0[1], x0[2], travel, ti[0], ti[1], ti[2])
    except Exception as exc:
        _disable(f"kernel call failed: {exc!r}")
        return False, None
    if r[0] == 0:
        return True, None
    return True, ((r[1], r[2], r[3]), (r[4], r[5], r[6]))


def ground(
    c: _Ctx,
    n: Vec,
    offset: float,
    z_static: float,
    ti: _V3,
    guess: object,
    slope0: float | None,
) -> tuple[float, _V3, _V3, float, float] | None:
    """Kernel secant of ``vehicle._solve_corner_ground`` (warm-started by ``guess``).

    Returns ``(travel, angles, residuals, phi0, slope)`` or None on any failure status or
    exception, in which case the caller runs the pure-Python path.
    """
    k = kernels()
    assert k is not None
    try:
        geom, rk = arrays(c)
        r = k.ground_kernel(
            geom,
            rk,
            float(n[0]),
            float(n[1]),
            float(n[2]),
            offset,
            z_static,
            ti[0],
            ti[1],
            ti[2],
            guess.travel,  # type: ignore[attr-defined]
            guess.theta_uca,  # type: ignore[attr-defined]
            guess.theta_lca,  # type: ignore[attr-defined]
            guess.psi,  # type: ignore[attr-defined]
            guess.rocker_angle,  # type: ignore[attr-defined]
            float("nan") if slope0 is None else slope0,
        )
    except Exception as exc:
        _disable(f"kernel call failed: {exc!r}")
        return None
    if r[0] != 0:
        return None
    return r[1], (r[2], r[3], r[4]), (r[5], r[6], r[7]), r[8], r[10]
