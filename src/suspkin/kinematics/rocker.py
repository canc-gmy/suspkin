"""Rocker kinematics: closed-form pushrod-length constraint on a rotating rocker."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin.geom_math.rotations import axis_angle_matrix
from suspkin.geometry.corners import Corner


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
    pts = corner.points
    pivot = np.asarray(pts["rocker_pivot"])
    axis = np.asarray(corner.rocker_axis)
    po = np.asarray(pushrod_outer, dtype=np.float64)
    length = float(np.linalg.norm(pts["pushrod_inner"] - pts["pushrod_outer"]))
    sign = side_sign(corner)

    v0 = pts["pushrod_inner"] - pivot
    v_par = (v0 @ axis) * axis
    w = v0 - v_par
    r = (po - pivot) - v_par
    ca = float(r @ w)
    cb = float(r @ np.cross(axis, w))
    cc = 0.5 * (float(r @ r) + float(w @ w) - length**2)
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

    p_in, p_spring, p_arb = rotate_rocker_points(corner, angle)

    lever = p_in - pivot
    lever = lever - (lever @ axis) * axis
    push = po - p_in
    push = push - (push @ axis) * axis
    nl, npu = float(np.linalg.norm(lever)), float(np.linalg.norm(push))
    cosang = float(lever @ push) / (nl * npu) if nl * npu > 1e-12 else 1.0
    a = math.acos(max(-1.0, min(1.0, cosang)))
    return RockerResult(angle, p_in, p_spring, p_arb, min(a, math.pi - a))
