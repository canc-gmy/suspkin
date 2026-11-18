"""Channel registry: the quantities the surrogate predicts per grid point.

Each channel names a column of ``metrics_frame`` / ``joint_angles_frame`` (or of the
solver state, see :mod:`suspkin.surrogate.extract`), an encoding and a tolerance in SI
(m, rad; swing angles are encoded as cos/sin with a dimensionless tolerance).

Deliberately absent:

- ARB drop-link displacement: the drop-link point follows exactly from the rocker angle
  (rigid rotation about the fixed rocker axis), so the vehicle layer rebuilds it.
- Wheel-centre z: it equals the static z plus the travel by definition.

``theta_uca``, ``theta_lca`` and ``psi`` are the solver coordinates of the corner state
(side-signed like ``CornerState``): the surrogate evaluator rebuilds the hardpoint
positions, and from them every other corner column, with the exact forward kinematics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from suspkin.metrics.joints import JOINTS

_DEG = math.pi / 180.0
_MM = 1e-3
_BOTH = frozenset({"front", "rear"})
_FRONT = frozenset({"front"})


@dataclass(frozen=True)
class Channel:
    """One network output channel.

    ``source`` is the table column; ``encoding`` maps it to the channel value (``value``
    as is, ``cos`` / ``sin`` of an angle); ``tol`` is the engineering tolerance (SI).
    """

    name: str
    source: str
    encoding: Literal["value", "cos", "sin"]
    tol: float
    axles: frozenset[str] = _BOTH


def _build() -> tuple[Channel, ...]:
    ch: list[Channel] = [
        Channel("camber", "camber", "value", 0.02 * _DEG),
        Channel("toe", "toe", "value", 0.02 * _DEG),
        Channel("steer_angle", "steer_angle", "value", 0.02 * _DEG, _FRONT),
        Channel("caster", "caster", "value", 0.05 * _DEG),
        Channel("kpi", "kpi", "value", 0.05 * _DEG),
        Channel("scrub_radius", "scrub_radius", "value", 0.1 * _MM),
        Channel("mechanical_trail", "mechanical_trail", "value", 0.1 * _MM),
        Channel("half_track_change", "half_track_change", "value", 0.1 * _MM),
        Channel("contact_patch_dx", "contact_patch_dx", "value", 0.1 * _MM),
        Channel("contact_patch_dy", "contact_patch_dy", "value", 0.1 * _MM),
        Channel("wheel_center_recession", "wheel_center_recession", "value", 0.1 * _MM),
    ]
    for name in ("fv_swing_angle", "sv_swing_angle", "sv_wc_angle"):
        ch.append(Channel(f"{name}_cos", name, "cos", 1e-3))
        ch.append(Channel(f"{name}_sin", name, "sin", 1e-3))
    ch += [
        Channel("coilover_length", "coilover_length", "value", 0.05 * _MM),
        Channel("rocker_angle", "rocker_rotation", "value", 0.05 * _DEG),
        Channel("pushrod_angle", "pushrod_angle", "value", 0.05 * _DEG),
        Channel("toggle_margin", "toggle_margin", "value", 0.05 * _DEG),
    ]
    ch += [Channel(f"joint_{j}", f"joint_{j}", "value", 0.1 * _DEG) for j in JOINTS]
    # Solver coordinates: the evaluator rebuilds exact corner states from them.
    ch += [Channel(n, n, "value", 0.01 * _DEG) for n in ("theta_uca", "theta_lca", "psi")]
    return tuple(ch)


CHANNELS: tuple[Channel, ...] = _build()


def channels_for(axle: str) -> tuple[Channel, ...]:
    """Channels of an axle (``"front"`` or ``"rear"``), in registry order."""
    if axle not in ("front", "rear"):
        raise ValueError(f"axle must be 'front' or 'rear', got {axle!r}")
    return tuple(c for c in CHANNELS if axle in c.axles)
