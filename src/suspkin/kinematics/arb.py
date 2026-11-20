"""Anti-roll bar kinematics, stiffness and energy (pure functions).

Three bar types (``docs/concepts/corner-kinematics.md``; signs in
``docs/concepts/conventions.md`` section 5.10):

* ``torsional`` / ``blade``: each side's arm rotates rigidly about the bar axis
  line through that side's ``bar_pivot`` by an angle ``beta`` such that the drop
  link keeps its static length. Reported angles follow the rocker convention
  (physical rotation = ``side_sign * beta`` about the stored, side-mirrored axis),
  so symmetric heave gives ``beta_left == beta_right``. Elastic coordinate:
  ``twist = beta_left - beta_right`` [rad] (zero in pure heave).
* ``beam``: the beam ends move vertically (small deflection): left end
  ``arm_end + (c + delta) z``, right end ``mirror(arm_end) + (c - delta) z``.
  ``c`` is the heave of the central slider, elastic coordinate ``delta`` [m].

The static pose (travel 0) is the stress-free state of the bar.

Stiffness ``k`` of the elastic coordinate ``q`` (``E = 1/2 k q^2``):

* torsional: ``1/k = L/(G J) + 2 r / (3 E I)`` with ``J = pi/32 (D^4 - d^4)`` and,
  for round solid arms, ``I = pi d_a^4 / 64``; ``r`` is the arm length (arm tip to
  bar axis). The second term is two cantilever arms: tip stiffness ``3EI/r^3``
  acting with lever ``r`` gives a twist compliance ``1/(k_tip r^2)`` each. Rigid
  arms (``arm_diameter`` None) omit it. Unit: N m/rad.
* blade: same, with the effective inertia in compliance (harmonic) form
  ``I_eff = 1 / (cos^2 a / I_flat + sin^2 a / I_edge)`` with ``I_flat = w t^3/12`` and
  ``I_edge = t w^3/12``: tip force along the load direction, lateral deflection free.
  Exact at a = 0 (flat, soft) and a = pi/2 (edge-on, stiff).
* beam: each half is a cantilever of length ``a`` with tip stiffness
  ``k_end = 3 E I / a^3``; the energy is ``2 * 1/2 k_end delta^2``, hence
  ``k = 2 k_end = 6 E I / a^3`` for ``E = 1/2 k delta^2``. Unit: N/m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.model import ARB, Axle, BeamARB, TorsionalARB
from suspkin.geometry.vectors import _as3, mirror_y
from suspkin.kinematics.corner import CornerState
from suspkin.kinematics.rocker import KinematicsError

Vec = NDArray[np.float64]
_TOL = 1e-9


@dataclass(frozen=True)
class ArbState:
    """Solved ARB pose.

    ``beta_left/right`` [rad] are used by torsional and blade bars, ``heave`` [m]
    by beams (unused fields are 0). ``elastic`` is ``twist`` [rad] or ``deflection``
    [m]. ``arm_end_left/right`` are the bar-side drop-link points.

    Attributes:
        type: ``"torsional"``, ``"blade"`` or ``"beam"``.
        beta_left: Left arm angle in rad (per-side signed; 0 for beams).
        beta_right: Right arm angle in rad (per-side signed; 0 for beams).
        heave: Beam central-slider heave ``c`` in m (0 for tube bars).
        elastic: Elastic coordinate: twist ``beta_left - beta_right`` [rad] for tube bars,
            deflection ``delta`` [m] for beams. Zero in pure heave.
        arm_end_left: Moved bar-side drop-link point, shape ``(3,)``, m (read-only).
        arm_end_right: Same for the right side.
        max_residual: Largest drop-link length error in m.
    """

    type: Literal["torsional", "blade", "beam"]
    beta_left: float
    beta_right: float
    heave: float
    elastic: float
    arm_end_left: Vec
    arm_end_right: Vec
    max_residual: float


def _require_arb(axle_cfg: Axle) -> ARB:
    """The axle's ARB; ``ValueError`` if the axle has none."""
    if axle_cfg.arb is None:
        raise ValueError("axle has no ARB")
    return axle_cfg.arb


def _drop_length(axle_cfg: Axle, side_left: bool) -> float:
    """Static drop-link length in m (``rocker_arb`` to ``arm_end``, right override honoured)."""
    arb = _require_arb(axle_cfg)
    rocker = axle_cfg.hardpoints.rocker_arb
    assert rocker is not None
    arm = np.array(arb.arm_end, dtype=np.float64)
    r = np.array(rocker, dtype=np.float64)
    if not side_left:
        arm = mirror_y(arm)
        r = np.array(axle_cfg.right_overrides.get("rocker_arb", mirror_y(r)), dtype=np.float64)
    return float(np.linalg.norm(r - arm))


def _circle_angle(
    pivot: Vec, axis: Vec, p0: Vec, target: Vec, length: float, guess: float
) -> float:
    """Rotation angle of ``p0`` about the axis so that ``|target - p| = length``.

    Closed form ``A cos + B sin = C``; of the two roots the nearest to ``guess``.
    Scalar arithmetic (3-vectors are too small for numpy to pay off).
    """
    px, py, pz = _as3(pivot)
    a0, a1, a2 = _as3(axis)
    q0, q1, q2 = _as3(p0)
    t0, t1, t2 = _as3(target)
    v0, v1, v2 = q0 - px, q1 - py, q2 - pz
    f = v0 * a0 + v1 * a1 + v2 * a2
    p0_, p1_, p2_ = f * a0, f * a1, f * a2
    w0, w1, w2 = v0 - p0_, v1 - p1_, v2 - p2_
    r0, r1, r2 = (t0 - px) - p0_, (t1 - py) - p1_, (t2 - pz) - p2_
    ca = r0 * w0 + r1 * w1 + r2 * w2
    cb = r0 * (a1 * w2 - a2 * w1) + r1 * (a2 * w0 - a0 * w2) + r2 * (a0 * w1 - a1 * w0)
    cc = 0.5 * ((r0 * r0 + r1 * r1 + r2 * r2) + (w0 * w0 + w1 * w1 + w2 * w2) - length**2)
    amp = math.hypot(ca, cb)
    if amp < 1e-12 or abs(cc) > amp * (1.0 + 1e-12):
        raise KinematicsError("ARB drop link cannot reach the arm circle (no ARB solution)")
    half = math.acos(max(-1.0, min(1.0, cc / amp)))
    base = math.atan2(cb, ca)
    best = min(
        (base + half, base - half),
        key=lambda a: abs((a - guess + math.pi) % (2.0 * math.pi) - math.pi),
    )
    return guess + ((best - guess + math.pi) % (2.0 * math.pi) - math.pi)


def _rotate_about(pivot: Vec, axis: Vec, angle: float, p: Vec) -> Vec:
    """``pivot + R(axis, angle) (p - pivot)`` (Rodrigues, axis normalized)."""
    a0, a1, a2 = _as3(axis)
    na = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    if na < 1e-12:
        raise ValueError("rotation axis must be non-zero")
    k0, k1, k2 = a0 / na, a1 / na, a2 / na
    c0, c1, c2 = _as3(pivot)
    q0, q1, q2 = _as3(p)
    v0, v1, v2 = q0 - c0, q1 - c1, q2 - c2
    cs, sn = math.cos(angle), math.sin(angle)
    f = (k0 * v0 + k1 * v1 + k2 * v2) * (1.0 - cs)
    return np.array(
        (
            c0 + v0 * cs + (k1 * v2 - k2 * v1) * sn + k0 * f,
            c1 + v1 * cs + (k2 * v0 - k0 * v2) * sn + k1 * f,
            c2 + v2 * cs + (k0 * v1 - k1 * v0) * sn + k2 * f,
        )
    )


def _frozen(v: Vec) -> Vec:
    """Read-only float64 copy of ``v``."""
    out = np.array(v, dtype=np.float64)
    out.setflags(write=False)
    return out


def _residual(
    left: CornerState, right: CornerState, ends: list[Vec], lengths: list[float]
) -> float:
    """Largest absolute drop-link length error in m over both sides."""
    worst = 0.0
    for i, state in enumerate((left, right)):
        r = state.points["rocker_arb"]
        e = ends[i]
        link = math.sqrt(
            (float(r[0]) - float(e[0])) ** 2
            + (float(r[1]) - float(e[1])) ** 2
            + (float(r[2]) - float(e[2])) ** 2
        )
        worst = max(worst, abs(link - lengths[i]))
    return worst


def _solve_tube(
    arb: ARB,
    axle_cfg: Axle,
    left: CornerState,
    right: CornerState,
    guess: ArbState | None,
) -> ArbState:
    """Torsional/blade bar: per-side arm angle so the drop link keeps its static length."""
    assert not isinstance(arb, BeamARB)
    betas: list[float] = []
    ends: list[Vec] = []
    lengths: list[float] = []
    for state, sign, g in (
        (left, 1.0, 0.0 if guess is None else guess.beta_left),
        (right, -1.0, 0.0 if guess is None else guess.beta_right),
    ):
        arm = np.array(arb.arm_end, dtype=np.float64)
        pivot = np.array(arb.bar_pivot, dtype=np.float64)
        axis = np.array(arb.bar_axis, dtype=np.float64)
        if sign < 0:
            arm, pivot, axis = mirror_y(arm), mirror_y(pivot), mirror_y(axis)
        target = np.asarray(state.points["rocker_arb"], dtype=np.float64)
        length = _drop_length(axle_cfg, sign > 0)
        lengths.append(length)
        phys = _circle_angle(pivot, axis, arm, target, length, sign * g)
        betas.append(sign * phys)
        ends.append(_rotate_about(pivot, axis, phys, arm))
    return ArbState(
        arb.type,
        betas[0],
        betas[1],
        0.0,
        betas[0] - betas[1],
        _frozen(ends[0]),
        _frozen(ends[1]),
        _residual(left, right, ends, lengths),
    )


def _solve_beam(
    arb: BeamARB,
    axle_cfg: Axle,
    left: CornerState,
    right: CornerState,
    guess: ArbState | None,
) -> ArbState:
    """Beam bar: vertical end offsets so each drop link keeps its static length."""
    offsets: list[float] = []
    ends: list[Vec] = []
    lengths: list[float] = []
    g_l = 0.0 if guess is None else guess.heave + guess.elastic
    g_r = 0.0 if guess is None else guess.heave - guess.elastic
    for state, sign, g in ((left, 1.0, g_l), (right, -1.0, g_r)):
        arm = np.array(arb.arm_end, dtype=np.float64)
        if sign < 0:
            arm = mirror_y(arm)
        d = np.asarray(state.points["rocker_arb"], dtype=np.float64) - arm
        length = _drop_length(axle_cfg, sign > 0)
        lengths.append(length)
        q = length**2 - d[0] ** 2 - d[1] ** 2
        if q < 0.0:
            raise KinematicsError("ARB drop link cannot reach the beam end (no ARB solution)")
        s = math.sqrt(q)
        e = min((d[2] - s, d[2] + s), key=lambda x: abs(x - g))
        offsets.append(e)
        ends.append(arm + np.array([0.0, 0.0, e]))
    return ArbState(
        "beam",
        0.0,
        0.0,
        0.5 * (offsets[0] + offsets[1]),
        0.5 * (offsets[0] - offsets[1]),
        _frozen(ends[0]),
        _frozen(ends[1]),
        _residual(left, right, ends, lengths),
    )


def solve_arb(
    axle_cfg: Axle,
    left: CornerState,
    right: CornerState,
    *,
    guess: ArbState | None = None,
) -> ArbState:
    """Solve the ARB pose from the solved left/right corner states.

    Args:
        axle_cfg: Axle with an ARB.
        left: Solved left corner; ``points["rocker_arb"]`` drives the left drop link.
        right: Solved right corner; ``points["rocker_arb"]`` drives the right drop link.
        guess: Neighbouring solution used to pick the root branch.

    Returns:
        :class:`ArbState` (arm angles in rad per-side signed, or beam offsets in m).

    Raises:
        ValueError: If the axle has no ARB.
        KinematicsError: If a corner carries no ``rocker_arb`` point, a drop link cannot be
            closed, or the residual exceeds 1e-9 m.
    """
    arb = _require_arb(axle_cfg)
    if "rocker_arb" not in left.points or "rocker_arb" not in right.points:
        raise KinematicsError("corner states carry no rocker_arb point")
    if isinstance(arb, BeamARB):
        state = _solve_beam(arb, axle_cfg, left, right, guess)
    else:
        state = _solve_tube(arb, axle_cfg, left, right, guess)
    if state.max_residual > _TOL:
        raise KinematicsError(f"ARB drop-link residual {state.max_residual:.2e} m")
    return state


def _rect_inertia(width: float, thickness: float, angle: float) -> float:
    """Effective second moment of area in m^4 of a rotated rectangle (harmonic blend)."""
    i_flat = width * thickness**3 / 12.0
    i_edge = thickness * width**3 / 12.0
    c, sn = math.cos(angle), math.sin(angle)
    return 1.0 / (c * c / i_flat + sn * sn / i_edge)


def arm_length(arb: ARB) -> float:
    """Arm length [m] of a torsional or blade bar (arm tip to the bar axis).

    Args:
        arb: A ``TorsionalARB`` or ``BladeARB`` (a ``BeamARB`` is not supported).

    Returns:
        Perpendicular distance in m of ``arm_end`` from the bar axis line through
        ``bar_pivot``.
    """
    assert not isinstance(arb, BeamARB)
    d = np.array(arb.arm_end, dtype=np.float64) - np.array(arb.bar_pivot, dtype=np.float64)
    axis = np.array(arb.bar_axis, dtype=np.float64)
    return float(np.linalg.norm(d - (d @ axis) * axis))


def arb_stiffness(axle_cfg: Axle) -> float:
    """Stiffness of the elastic coordinate: N m/rad (twist) or N/m (beam deflection).

    Args:
        axle_cfg: Axle with an ARB (SI material and section data).

    Returns:
        ``k`` with bar energy ``E = 1/2 k q^2``: N m/rad for torsional/blade bars,
        N/m for beams.

    Raises:
        ValueError: If the axle has no ARB.
    """
    arb = _require_arb(axle_cfg)
    if isinstance(arb, BeamARB):
        a = float((np.array(arb.arm_end) - np.array(arb.beam_center)) @ np.array(arb.beam_axis))
        inertia = _rect_inertia(arb.beam_width, arb.beam_thickness, arb.beam_angle)
        return 6.0 * arb.beam_youngs_modulus * inertia / a**3
    j = math.pi / 32.0 * (arb.outer_diameter**4 - arb.inner_diameter**4)
    compliance = arb.bar_length / (arb.shear_modulus * j)
    r = arm_length(arb)
    if isinstance(arb, TorsionalARB):
        if arb.arm_diameter is not None:
            inertia = math.pi * arb.arm_diameter**4 / 64.0
            compliance += 2.0 * r / (3.0 * arb.arm_youngs_modulus * inertia)
    else:
        inertia = _rect_inertia(arb.blade_width, arb.blade_thickness, arb.blade_angle)
        compliance += 2.0 * r / (3.0 * arb.blade_youngs_modulus * inertia)
    return 1.0 / compliance


def arb_energy(axle_cfg: Axle, arb_state: ArbState) -> float:
    """Elastic energy [J] stored in the bar, ``1/2 k q^2``.

    Args:
        axle_cfg: Axle with an ARB.
        arb_state: Solved ARB pose; its ``elastic`` coordinate is ``q``.

    Returns:
        Energy in J (zero for the static pose and for pure heave).

    Raises:
        ValueError: If the axle has no ARB.
    """
    return 0.5 * arb_stiffness(axle_cfg) * arb_state.elastic**2
