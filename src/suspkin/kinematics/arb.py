"""Anti-roll bar kinematics, stiffness and energy (pure functions).

Three bar types (plan decision O3):

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
* blade: same, with ``I = (w t^3 cos^2 a + t w^3 sin^2 a) / 12``, the standard
  compliance (harmonic) form ``I_eff = 1 / (cos^2 a / I_flat + sin^2 a / I_edge)``
  with ``I_flat = w t^3/12`` and ``I_edge = t w^3/12``: tip force along the load
  direction, lateral deflection free. Exact at a = 0 (flat, soft) and a = pi/2
  (edge-on, stiff).
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

from suspkin.geom_math.rotations import axis_angle_matrix
from suspkin.geom_math.vectors import mirror_y
from suspkin.geometry.models import ARB, Axle, BeamARB, TorsionalARB
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
    if axle_cfg.arb is None:
        raise ValueError("axle has no ARB")
    return axle_cfg.arb


def _drop_length(axle_cfg: Axle, side_left: bool) -> float:
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
    """
    v0 = p0 - pivot
    v_par = (v0 @ axis) * axis
    w = v0 - v_par
    r = (target - pivot) - v_par
    ca = float(r @ w)
    cb = float(r @ np.cross(axis, w))
    cc = 0.5 * (float(r @ r) + float(w @ w) - length**2)
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


def _frozen(v: Vec) -> Vec:
    out = np.array(v, dtype=np.float64)
    out.setflags(write=False)
    return out


def _residual(axle_cfg: Axle, left: CornerState, right: CornerState, ends: list[Vec]) -> float:
    worst = 0.0
    for i, state in enumerate((left, right)):
        link = float(np.linalg.norm(np.asarray(state.points["rocker_arb"]) - ends[i]))
        worst = max(worst, abs(link - _drop_length(axle_cfg, i == 0)))
    return worst


def _solve_tube(
    arb: ARB,
    axle_cfg: Axle,
    left: CornerState,
    right: CornerState,
    guess: ArbState | None,
) -> ArbState:
    assert not isinstance(arb, BeamARB)
    betas: list[float] = []
    ends: list[Vec] = []
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
        phys = _circle_angle(pivot, axis, arm, target, length, sign * g)
        betas.append(sign * phys)
        ends.append(pivot + axis_angle_matrix(axis, phys) @ (arm - pivot))
    return ArbState(
        arb.type,
        betas[0],
        betas[1],
        0.0,
        betas[0] - betas[1],
        _frozen(ends[0]),
        _frozen(ends[1]),
        _residual(axle_cfg, left, right, ends),
    )


def _solve_beam(
    arb: BeamARB,
    axle_cfg: Axle,
    left: CornerState,
    right: CornerState,
    guess: ArbState | None,
) -> ArbState:
    offsets: list[float] = []
    ends: list[Vec] = []
    g_l = 0.0 if guess is None else guess.heave + guess.elastic
    g_r = 0.0 if guess is None else guess.heave - guess.elastic
    for state, sign, g in ((left, 1.0, g_l), (right, -1.0, g_r)):
        arm = np.array(arb.arm_end, dtype=np.float64)
        if sign < 0:
            arm = mirror_y(arm)
        d = np.asarray(state.points["rocker_arb"], dtype=np.float64) - arm
        length = _drop_length(axle_cfg, sign > 0)
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
        _residual(axle_cfg, left, right, ends),
    )


def solve_arb(
    axle_cfg: Axle,
    left: CornerState,
    right: CornerState,
    *,
    guess: ArbState | None = None,
) -> ArbState:
    """Solve the ARB pose from the solved left/right corner states.

    Parameters
    ----------
    axle_cfg : Axle
        Axle with an ARB.
    left, right : CornerState
        Solved corners; ``points["rocker_arb"]`` drives the drop links.
    guess : ArbState, optional
        Neighbouring solution used to pick the root branch.

    Raises
    ------
    KinematicsError
        If a drop link cannot be closed or the residual exceeds 1e-9 m.
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
    i_flat = width * thickness**3 / 12.0
    i_edge = thickness * width**3 / 12.0
    c, sn = math.cos(angle), math.sin(angle)
    return 1.0 / (c * c / i_flat + sn * sn / i_edge)


def arm_length(arb: ARB) -> float:
    """Arm length [m] of a torsional or blade bar (arm tip to the bar axis)."""
    assert not isinstance(arb, BeamARB)
    d = np.array(arb.arm_end, dtype=np.float64) - np.array(arb.bar_pivot, dtype=np.float64)
    axis = np.array(arb.bar_axis, dtype=np.float64)
    return float(np.linalg.norm(d - (d @ axis) * axis))


def arb_stiffness(axle_cfg: Axle) -> float:
    """Stiffness of the elastic coordinate: N m/rad (twist) or N/m (beam deflection)."""
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
    """Elastic energy [J] stored in the bar, ``1/2 k q^2``."""
    return 0.5 * arb_stiffness(axle_cfg) * arb_state.elastic**2
