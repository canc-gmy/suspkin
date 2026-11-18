"""Rocker kinematics: closed-form pushrod-length constraint on a rotating rocker."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin.geom_math.rotations import axis_angle_matrix
from suspkin.geometry.corners import Corner

_V3 = tuple[float, float, float]


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


def rotate_rocker_points(
    corner: Corner, angle: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Rotate ``pushrod_inner``, ``rocker_spring`` and ``rocker_arb`` by the reported ``angle``."""
    pivot = np.asarray(corner.points["rocker_pivot"])
    rot = axis_angle_matrix(corner.rocker_axis, side_sign(corner) * angle)
    arb = corner.points.get("rocker_arb")
    return (
        pivot + rot @ (corner.points["pushrod_inner"] - pivot),
        pivot + rot @ (corner.points["rocker_spring"] - pivot),
        None if arb is None else pivot + rot @ (arb - pivot),
    )


def _t(v: ArrayLike) -> _V3:
    a = np.asarray(v, dtype=np.float64)
    return (float(a[0]), float(a[1]), float(a[2]))


def _sub(a: _V3, b: _V3) -> _V3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: _V3, b: _V3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: _V3, b: _V3) -> _V3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


class RockerCtx:
    """Static rocker data of a corner as plain floats (scalar fast path of the solver)."""

    def __init__(self, corner: Corner) -> None:
        pts = corner.points
        self.sign = side_sign(corner)
        self.pivot = _t(pts["rocker_pivot"])
        self.axis = _t(corner.rocker_axis)
        n = math.sqrt(_dot(self.axis, self.axis))
        if n < 1e-12:
            raise ValueError("rotation axis must be non-zero")
        self.k: _V3 = (self.axis[0] / n, self.axis[1] / n, self.axis[2] / n)
        pin = _t(pts["pushrod_inner"])
        d = _sub(pin, _t(pts["pushrod_outer"]))
        self.length = math.sqrt(_dot(d, d))
        self.r_in = _sub(pin, self.pivot)
        self.r_spring = _sub(_t(pts["rocker_spring"]), self.pivot)
        arb = pts.get("rocker_arb")
        self.r_arb = None if arb is None else _sub(_t(arb), self.pivot)
        ax = self.axis
        f = _dot(self.r_in, ax)
        self.v_par: _V3 = (f * ax[0], f * ax[1], f * ax[2])
        self.w = _sub(self.r_in, self.v_par)
        self.ww = _dot(self.w, self.w)
        self.axw = _cross(ax, self.w)

    def _rotate(self, angle: float, v: _V3) -> _V3:
        """``pivot + R(sign * angle) v`` for a vector ``v`` relative to the pivot."""
        a = self.sign * angle
        cs, sn = math.cos(a), math.sin(a)
        k = self.k
        kx = _cross(k, v)
        f = _dot(k, v) * (1.0 - cs)
        pv = self.pivot
        return (
            pv[0] + v[0] * cs + kx[0] * sn + k[0] * f,
            pv[1] + v[1] * cs + kx[1] * sn + k[1] * f,
            pv[2] + v[2] * cs + kx[2] * sn + k[2] * f,
        )

    def solve(self, po: _V3, guess: float = 0.0) -> RockerResult:
        """See :func:`solve_rocker`; ``po`` is the pushrod outer point."""
        sign = self.sign
        pivot, axis, v_par = self.pivot, self.axis, self.v_par
        r = _sub(_sub(po, pivot), v_par)
        ca = _dot(r, self.w)
        cb = _dot(r, self.axw)
        cc = 0.5 * (_dot(r, r) + self.ww - self.length**2)
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

        lever = _sub(p_in, pivot)
        f = _dot(lever, axis)
        lever = (lever[0] - f * axis[0], lever[1] - f * axis[1], lever[2] - f * axis[2])
        push = _sub(po, p_in)
        f = _dot(push, axis)
        push = (push[0] - f * axis[0], push[1] - f * axis[1], push[2] - f * axis[2])
        nl, npu = math.sqrt(_dot(lever, lever)), math.sqrt(_dot(push, push))
        cosang = _dot(lever, push) / (nl * npu) if nl * npu > 1e-12 else 1.0
        a = math.acos(max(-1.0, min(1.0, cosang)))
        return RockerResult(
            angle,
            np.array(p_in),
            np.array(p_spring),
            None if p_arb is None else np.array(p_arb),
            min(a, math.pi - a),
        )


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
    return RockerCtx(corner).solve(_t(pushrod_outer), guess)
