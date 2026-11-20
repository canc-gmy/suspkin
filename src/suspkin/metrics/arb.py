"""ARB metrics: elastic coordinate, drop-link angles, motion ratios and wheel rates.

Virtual work: the bar energy is ``E = 1/2 k q^2`` with elastic coordinate ``q``.

* Roll mode (left ``+z``, right ``-z``): ``q = MR_roll z``; the two wheels carry
  equal and opposite forces ``F`` with ``2 F dz = dE``, so the per-wheel force is
  ``F = 1/2 dE/dz`` and the per-wheel rate ``k_w = 1/2 d2E/dz2 = k MR_roll^2 / 2``.
* Single-wheel mode (left ``+z``, right fixed): ``F = dE/dz``, rate
  ``k_w = k MR_single^2``.

The ``q * d2q/dz2`` term (motion-ratio progression) is neglected; it vanishes where
the elastic coordinate ``q = 0`` (static and symmetric-heave poses, which is where
``arb_rates`` is evaluated).

Units: rad, m, N, N/m, J. Sign of the twist coordinate: ``docs/concepts/conventions.md``
section 5.10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.geometry.model import Axle
from suspkin.kinematics.arb import ArbState, arb_energy, arb_stiffness, solve_arb
from suspkin.kinematics.corner import CornerState, solve_corner


@dataclass(frozen=True)
class ArbRates:
    """ARB motion ratios [rad/m or m/m] and per-wheel rates [N/m] at one travel.

    Attributes:
        stiffness: Bar stiffness ``k`` of the elastic coordinate: N m/rad (twist) or N/m
            (beam).
        motion_ratio_roll: ``dq/dz`` for left ``+z`` and right ``-z`` (rad/m, or m/m for beams).
        motion_ratio_single: ``dq/dz`` for left ``+z`` with the right wheel fixed.
        wheel_rate_roll: Per-wheel roll-mode rate ``k MR_roll^2 / 2`` in N/m.
        wheel_rate_single: Single-wheel rate ``k MR_single^2`` in N/m.
        heave_coupling: ``dq/dz`` for both wheels moving together (zero when the geometry
            is symmetric), same unit as the motion ratios.
    """

    stiffness: float
    motion_ratio_roll: float
    motion_ratio_single: float
    wheel_rate_roll: float
    wheel_rate_single: float
    heave_coupling: float


def _angle_from_vertical(link: NDArray[np.float64]) -> float:
    """Unsigned angle in rad (0 to pi/2) between a link vector and the vertical."""
    n = float(np.linalg.norm(link))
    return math.acos(max(-1.0, min(1.0, abs(float(link[2])) / n)))


def arb_metrics(
    axle_cfg: Axle, left: CornerState, right: CornerState, static_arb: ArbState
) -> dict[str, float]:
    """ARB quantities of one pose.

    Re-solves the bar for the given corner states and returns its elastic coordinate, arm
    angles / heave, stored energy, residual and the absolute drop-link angles from vertical.

    Args:
        axle_cfg: Axle with an ARB.
        left: Solved left corner.
        right: Solved right corner.
        static_arb: Static-pose ARB state, used only as the starting guess (root branch).

    Returns:
        Dict with ``arb_elastic`` (twist in rad, or beam deflection in m), ``arb_beta_left``
        and ``arb_beta_right`` (rad, per-side signed; 0 for beams), ``arb_heave`` (m; 0 for
        tube bars), ``arb_energy`` (J), ``arb_residual`` (m) and ``drop_link_angle_left`` /
        ``drop_link_angle_right`` (rad, absolute angle of the drop link from vertical in
        ``[0, pi/2]``, not the change from the static pose).

    Raises:
        KinematicsError: If the bar cannot be solved (see
            :func:`suspkin.kinematics.arb.solve_arb`).
        ValueError: If the axle has no ARB.
    """
    st = solve_arb(axle_cfg, left, right, guess=static_arb)
    out: dict[str, float] = {
        "arb_elastic": st.elastic,
        "arb_beta_left": st.beta_left,
        "arb_beta_right": st.beta_right,
        "arb_heave": st.heave,
        "arb_energy": arb_energy(axle_cfg, st),
        "arb_residual": st.max_residual,
    }
    for name, s, e in (
        ("left", left, st.arm_end_left),
        ("right", right, st.arm_end_right),
    ):
        out[f"drop_link_angle_{name}"] = _angle_from_vertical(
            np.asarray(s.points["rocker_arb"]) - e
        )
    return out


def arb_rates(
    corner_left: Corner,
    corner_right: Corner,
    axle_cfg: Axle,
    travel: float = 0.0,
    h: float = 1e-5,
) -> ArbRates:
    """ARB motion ratios and wheel rates by central differences around ``travel``.

    Args:
        corner_left: Side-resolved left corner.
        corner_right: Side-resolved right corner of the same axle.
        axle_cfg: Axle with an ARB.
        travel: Symmetric wheel travel in m around which to differentiate (default static).
        h: Central-difference step in m.

    Returns:
        :class:`ArbRates` (motion ratios in rad/m or m/m, wheel rates in N/m).

    Raises:
        KinematicsError: If a corner or the bar cannot be solved.
        ValueError: If the axle has no ARB.
    """
    k = arb_stiffness(axle_cfg)
    base_l = solve_corner(corner_left, travel)
    base_r = solve_corner(corner_right, travel)
    guess = solve_arb(axle_cfg, base_l, base_r)

    def q(zl: float, zr: float) -> float:
        sl = solve_corner(corner_left, zl, guess=base_l)
        sr = solve_corner(corner_right, zr, guess=base_r)
        return solve_arb(axle_cfg, sl, sr, guess=guess).elastic

    t = travel
    mr_roll = (q(t + h, t - h) - q(t - h, t + h)) / (2.0 * h)
    mr_single = (q(t + h, t) - q(t - h, t)) / (2.0 * h)
    coupling = (q(t + h, t + h) - q(t - h, t - h)) / (2.0 * h)
    return ArbRates(
        stiffness=k,
        motion_ratio_roll=mr_roll,
        motion_ratio_single=mr_single,
        wheel_rate_roll=0.5 * k * mr_roll**2,
        wheel_rate_single=k * mr_single**2,
        heave_coupling=coupling,
    )
