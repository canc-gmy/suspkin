"""Rocker kinematics: closed-form pushrod-length constraint on a rotating rocker."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin.geometry.corners import Corner
from suspkin.geometry.vectors import _V3, _as3, _cross3, _dot3, _sub3


class KinematicsError(RuntimeError):
    """Raised when the mechanism cannot be assembled or a solve does not converge."""


@dataclass(frozen=True)
class RockerResult:
    """Rocker pose for a given pushrod outer point.

    ``angle`` is the reported (side-signed) rocker angle [rad]; the physical
    rotation about the stored ``corner.rocker_axis`` is ``side_sign * angle``.
    ``toggle_margin`` is min(a, pi - a) [rad] with ``a`` the angle between the
    pushrod and the rocker lever in the plane normal to the rocker axis.
    """

    angle: float
    pushrod_inner: NDArray[np.float64]
    rocker_spring: NDArray[np.float64]
    rocker_arb: NDArray[np.float64] | None
    toggle_margin: float


def side_sign(corner: Corner) -> float:
    """Return +1 for left corners and -1 for right corners (reflection sense)."""
    return 1.0 if corner.side == "left" else -1.0


RockerTuple = tuple[float, _V3, _V3, _V3 | None, float]


class RockerCtx:
    """Static rocker data of a corner as plain floats (scalar fast path of the solver)."""

    def __init__(self, corner: Corner) -> None:
        pts = corner.points
        self.sign = side_sign(corner)
        self.pivot = _as3(pts["rocker_pivot"])
        self.axis = _as3(corner.rocker_axis)
        n = math.sqrt(_dot3(self.axis, self.axis))
        if n < 1e-12:
            raise ValueError("rotation axis must be non-zero")
        self.k: _V3 = (self.axis[0] / n, self.axis[1] / n, self.axis[2] / n)
        pin = _as3(pts["pushrod_inner"])
        d = _sub3(pin, _as3(pts["pushrod_outer"]))
        self.length = math.sqrt(_dot3(d, d))
        self.r_in = _sub3(pin, self.pivot)
        self.r_spring = _sub3(_as3(pts["rocker_spring"]), self.pivot)
        arb = pts.get("rocker_arb")
        self.r_arb = None if arb is None else _sub3(_as3(arb), self.pivot)
        ax = self.axis
        f = _dot3(self.r_in, ax)
        self.v_par: _V3 = (f * ax[0], f * ax[1], f * ax[2])
        self.w = _sub3(self.r_in, self.v_par)
        self.ww = _dot3(self.w, self.w)
        self.axw = _cross3(ax, self.w)

    def _rotate(self, angle: float, v: _V3) -> _V3:
        """``pivot + R(sign * angle) v`` for a vector ``v`` relative to the pivot."""
        a = self.sign * angle
        cs, sn = math.cos(a), math.sin(a)
        k = self.k
        kx = _cross3(k, v)
        f = _dot3(k, v) * (1.0 - cs)
        pv = self.pivot
        return (
            pv[0] + v[0] * cs + kx[0] * sn + k[0] * f,
            pv[1] + v[1] * cs + kx[1] * sn + k[1] * f,
            pv[2] + v[2] * cs + kx[2] * sn + k[2] * f,
        )

    def solve(self, po: _V3, guess: float = 0.0) -> RockerResult:
        """See :func:`solve_rocker`; ``po`` is the pushrod outer point."""
        angle, p_in, p_spring, p_arb, toggle = self.solve_tuple(po, guess)
        return RockerResult(
            angle,
            np.array(p_in),
            np.array(p_spring),
            None if p_arb is None else np.array(p_arb),
            toggle,
        )

    def solve_tuple(self, po: _V3, guess: float = 0.0) -> RockerTuple:
        """``solve`` returning plain tuples (angle, inner, spring, arb, toggle margin)."""
        sign = self.sign
        pivot, axis, v_par = self.pivot, self.axis, self.v_par
        r = _sub3(_sub3(po, pivot), v_par)
        ca = _dot3(r, self.w)
        cb = _dot3(r, self.axw)
        cc = 0.5 * (_dot3(r, r) + self.ww - self.length**2)
        amp = math.hypot(ca, cb)
        if amp < 1e-12 or abs(cc) > amp * (1.0 + 1e-12):
            raise KinematicsError("pushrod cannot reach the rocker circle (no rocker solution)")
        half = math.acos(max(-1.0, min(1.0, cc / amp)))
        base = math.atan2(cb, ca)
        g = sign * guess  # physical rotation about the stored axis
        best = min(
            (base + half, base - half),
            key=lambda a: abs((a - g + math.pi) % (2.0 * math.pi) - math.pi),
        )
        phys = g + ((best - g + math.pi) % (2.0 * math.pi) - math.pi)
        angle = sign * phys

        p_in = self._rotate(angle, self.r_in)
        p_spring = self._rotate(angle, self.r_spring)
        p_arb = None if self.r_arb is None else self._rotate(angle, self.r_arb)

        lever = _sub3(p_in, pivot)
        f = _dot3(lever, axis)
        lever = (lever[0] - f * axis[0], lever[1] - f * axis[1], lever[2] - f * axis[2])
        push = _sub3(po, p_in)
        f = _dot3(push, axis)
        push = (push[0] - f * axis[0], push[1] - f * axis[1], push[2] - f * axis[2])
        nl, npu = math.sqrt(_dot3(lever, lever)), math.sqrt(_dot3(push, push))
        cosang = _dot3(lever, push) / (nl * npu) if nl * npu > 1e-12 else 1.0
        a = math.acos(max(-1.0, min(1.0, cosang)))
        return angle, p_in, p_spring, p_arb, min(a, math.pi - a)


def solve_rocker(corner: Corner, pushrod_outer: ArrayLike, guess: float = 0.0) -> RockerResult:
    """Solve the rocker angle so the pushrod keeps its static length.

    Closed form: the rotated ``pushrod_inner`` lies on a circle; the distance
    constraint reduces to ``A cos(a) + B sin(a) = C``. Of the two roots the one
    closest to ``guess`` is used.

    Raises
    ------
    KinematicsError
        If the pushrod cannot reach the rocker circle (no real solution).
    """
    return RockerCtx(corner).solve(_as3(pushrod_outer), guess)
