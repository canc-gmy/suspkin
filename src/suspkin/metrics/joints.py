"""Joint misalignment angles over a corner sweep (neutral = static pose).

Body rotations relative to the static pose come from a Kabsch fit (SVD with a
determinant fix) of each body's hardpoint set; the chassis is the identity.

Definitions (angle = ``atan2(|a x b|, a . b)``):

* Rod ends (two-force links ``tie``, ``pushrod``, ``coilover``): the link direction
  ``d = unit(b - a)`` is compared with the static direction carried by the housing
  body, ``angle(d_now, R_housing @ d_static)``.
* Ball joints ``uca_outer`` / ``lca_outer``: the stud axis ``a0`` (static kingpin
  ``lca_outer -> uca_outer``) is carried by the upright; the housing is the wishbone.
  Angle = ``angle(R_upright @ a0, R_wishbone @ a0)``.
* ``axis`` override: a static vector replacing ``a0``. For ball joints the formula
  is unchanged. For rod ends the carrier rotation is the minimal rotation taking
  ``d_static`` onto ``d_now`` and the angle is ``angle(R_carrier @ a0, R_housing @ a0)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from suspkin.geometry.corners import Corner
from suspkin.geometry.vectors import rotation_between, unit
from suspkin.kinematics.corner import CornerState

Vec = NDArray[np.float64]

JOINTS: tuple[str, ...] = (
    "tie_inner",
    "tie_outer",
    "pushrod_outer",
    "pushrod_inner",
    "spring_chassis",
    "rocker_spring",
    "uca_outer",
    "lca_outer",
)

# joint -> (link end a, link end b) for rod ends; direction a -> b
_LINKS: dict[str, tuple[str, str]] = {
    "tie_inner": ("tie_inner", "tie_outer"),
    "tie_outer": ("tie_inner", "tie_outer"),
    "pushrod_outer": ("pushrod_outer", "pushrod_inner"),
    "pushrod_inner": ("pushrod_outer", "pushrod_inner"),
    "spring_chassis": ("spring_chassis", "rocker_spring"),
    "rocker_spring": ("spring_chassis", "rocker_spring"),
}
_BODY_POINTS: dict[str, tuple[str, ...]] = {
    "upright": ("uca_outer", "lca_outer", "tie_outer", "wheel_center"),
    "uca": ("uca_front", "uca_rear", "uca_outer"),
    "lca": ("lca_front", "lca_rear", "lca_outer"),
    "rocker": ("rocker_pivot", "rocker_spring", "pushrod_inner"),
}


def _kabsch(p0: Vec, p1: Vec) -> Vec:
    """Proper rotation R minimising ``|R (p0 - c0) - (p1 - c1)|`` for (n, 3) point sets."""
    a = p0 - p0.mean(axis=0)
    b = p1 - p1.mean(axis=0)
    u, sv, vt = np.linalg.svd(a.T @ b)
    if sv[1] < 1e-9 * sv[0] or sv[0] <= 0.0:
        raise ValueError("degenerate point set for rotation fit (collinear or coincident points)")
    d = np.sign(np.linalg.det(vt.T @ u.T))
    fix = np.diag([1.0, 1.0, d if d != 0.0 else 1.0])
    out: Vec = vt.T @ fix @ u.T
    return out


def _rot(static: CornerState, state: CornerState, names: Sequence[str]) -> Vec:
    p0 = np.array([static.points[n] for n in names], dtype=np.float64)
    p1 = np.array([state.points[n] for n in names], dtype=np.float64)
    return _kabsch(p0, p1)


def body_rotations(state: CornerState, static: CornerState, corner: Corner) -> dict[str, Vec]:
    """Rotation of each body from the static pose to ``state``.

    Keys: ``upright``, ``uca``, ``lca``, ``rocker``, ``chassis`` (identity). Each is
    a Kabsch fit of the body's hardpoints (``rocker_arb`` joins the rocker set if
    present). The pushrod mount body needs no entry of its own: ``pushrod_outer``
    moves with the body named by ``corner.axle_cfg.pushrod_mount``.
    """
    names = dict(_BODY_POINTS)
    if "rocker_arb" in state.points:
        names["rocker"] = (*names["rocker"], "rocker_arb")
    out = {k: _rot(static, state, v) for k, v in names.items()}
    out["chassis"] = np.eye(3)
    return out


def _housing(corner: Corner, joint: str) -> str:
    if joint in ("tie_inner", "spring_chassis"):
        return "chassis"
    if joint == "tie_outer":
        return "upright"
    if joint == "pushrod_outer":
        return str(corner.axle_cfg.pushrod_mount)
    if joint in ("pushrod_inner", "rocker_spring"):
        return "rocker"
    return "uca" if joint == "uca_outer" else "lca"


def _angle(a: ArrayLike, b: ArrayLike) -> float:
    u = np.asarray(a, dtype=np.float64)
    v = np.asarray(b, dtype=np.float64)
    return float(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v))


def joint_angle(
    state: CornerState,
    static: CornerState,
    corner: Corner,
    joint: str,
    axis: ArrayLike | None = None,
    _rots: dict[str, Vec] | None = None,
) -> float:
    """Misalignment angle [rad] of ``joint`` at ``state`` relative to ``static``.

    See the module docstring for the definitions. ``axis`` is an optional static
    vector (normalised internally) replacing the default stud/bore axis. For rod
    ends the override uses a minimal-rotation carrier convention (the carrier turns
    ``d_static`` onto ``d_now`` with no spin about the link), so it is approximate.
    ``_rots`` is an internal cache of ``body_rotations`` for this state.
    """
    if joint not in JOINTS:
        raise ValueError(f"unknown joint {joint!r}; expected one of {JOINTS}")
    rots = body_rotations(state, static, corner) if _rots is None else _rots
    r_h = rots[_housing(corner, joint)]
    if joint in ("uca_outer", "lca_outer"):
        a0 = (
            unit(np.asarray(static.points["uca_outer"]) - np.asarray(static.points["lca_outer"]))
            if axis is None
            else unit(axis)
        )
        return _angle(rots["upright"] @ a0, r_h @ a0)
    a_name, b_name = _LINKS[joint]
    d0 = unit(np.asarray(static.points[b_name]) - np.asarray(static.points[a_name]))
    d1 = unit(np.asarray(state.points[b_name]) - np.asarray(state.points[a_name]))
    if axis is None:
        return _angle(d1, r_h @ d0)
    a0 = unit(axis)
    return _angle(rotation_between(d0, d1) @ a0, r_h @ a0)


def joint_angles_frame(states: Sequence[CornerState], corner: Corner) -> pd.DataFrame:
    """Joint angles [rad] per state, columns ``joint_<name>`` for every joint in ``JOINTS``.

    The static pose is the state with ``travel ~ 0`` and ``rack ~ 0`` if present,
    else it is solved with ``solve_corner``.
    """
    from suspkin.metrics.batch import joint_angle_arrays

    cols = [f"joint_{j}" for j in JOINTS]
    if len(states) == 0:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(joint_angle_arrays(states, corner))
