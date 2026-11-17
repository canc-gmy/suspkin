"""Single-corner kinematic solver (rigid bodies, ideal joints).

Body frame = ISO 8855 vehicle frame; the ground moves relative to the body.
YAML hardpoints are the static pose (travel 0, rack 0).

Formulation (3 unknowns): the UCA rotates by ``theta_uca`` and the LCA by
``theta_lca`` about their chassis axes, which fixes both ball joints. The upright
pose is the minimal rotation aligning the static ``UBJ - LBJ`` with the new one,
followed by a spin ``psi`` about the new ``LBJ -> UBJ`` axis, anchored at the LBJ.
Closing equations: ball-joint distance, tie rod length, prescribed wheel-centre
height. Rocker, spring and toggle margin follow in closed form.

Sign convention: positive ``theta_uca``/``theta_lca``/``psi``/``rocker_angle`` are
right-hand rotations about the left corner's axes. Right corners are reflections
of left corners, so every angle is reported with a per-side sign (left/right equal
for mirrored motion); the physical rotation is ``side_sign * angle``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root

from suspkin.geom_math.rotations import axis_angle_matrix, rotation_between
from suspkin.geom_math.vectors import wheel_spin_axis
from suspkin.geometry.corners import Corner, CornerId
from suspkin.kinematics.rocker import KinematicsError, side_sign, solve_rocker

_RESIDUAL_TOL = 1e-9
_MAX_HALVINGS = 6

Vec = NDArray[np.float64]


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


class _Ctx:
    """Precomputed static data of a corner."""

    def __init__(self, corner: Corner) -> None:
        p = corner.points
        self.corner = corner
        self.sign = side_sign(corner)
        self.p = p
        self.uca_axis = p["uca_rear"] - p["uca_front"]
        self.lca_axis = p["lca_rear"] - p["lca_front"]
        self.ubj0 = np.asarray(p["uca_outer"])
        self.lbj0 = np.asarray(p["lca_outer"])
        self.d0 = self.ubj0 - self.lbj0
        self.len_ubj = float(np.linalg.norm(self.d0))
        self.len_tie = float(np.linalg.norm(p["tie_outer"] - p["tie_inner"]))
        self.wc0 = np.asarray(p["wheel_center"])
        cfg = corner.axle_cfg
        self.spin0 = wheel_spin_axis(corner.side, cfg.wheel.static_camber, cfg.wheel.static_toe)
        self.radius = cfg.wheel.tyre_radius
        self.mount = cfg.pushrod_mount
        self.rack_axis = None if cfg.steering is None else np.asarray(cfg.steering.rack_axis)


def _pose(c: _Ctx, th_u: float, th_l: float, psi: float) -> tuple[Vec, Vec, Vec]:
    """Return (ubj, lbj, R) of the upright for the given angles."""
    s = c.sign
    p = c.p
    ubj = p["uca_front"] + axis_angle_matrix(c.uca_axis, s * th_u) @ (c.ubj0 - p["uca_front"])
    lbj = p["lca_front"] + axis_angle_matrix(c.lca_axis, s * th_l) @ (c.lbj0 - p["lca_front"])
    d = ubj - lbj
    rot = axis_angle_matrix(d, s * psi) @ rotation_between(c.d0, d)
    return ubj, lbj, rot


def _on_upright(c: _Ctx, lbj: Vec, rot: Vec, point: Vec) -> Vec:
    out: Vec = lbj + rot @ (point - c.lbj0)
    return out


def _residuals(c: _Ctx, x: Vec, travel: float, tie_inner: Vec) -> Vec:
    ubj, lbj, rot = _pose(c, float(x[0]), float(x[1]), float(x[2]))
    tie_outer = _on_upright(c, lbj, rot, c.p["tie_outer"])
    wc = _on_upright(c, lbj, rot, c.wc0)
    return np.array(
        [
            float(np.linalg.norm(ubj - lbj)) - c.len_ubj,
            float(np.linalg.norm(tie_outer - tie_inner)) - c.len_tie,
            wc[2] - c.wc0[2] - travel,
        ]
    )


def solve_corner(
    corner: Corner,
    travel: float,
    rack: float = 0.0,
    *,
    guess: CornerState | None = None,
) -> CornerState:
    """Solve one corner for a wheel travel [m] (bump positive) and rack offset [m].

    Parameters
    ----------
    corner : Corner
        Side-resolved corner from ``build_corners``.
    travel, rack : float
        Wheel-centre vertical travel and rack displacement along ``rack_axis``.
    guess : CornerState, optional
        Neighbouring solution used as the starting point (continuation).

    Raises
    ------
    ValueError
        If ``rack != 0`` on an axle without steering.
    KinematicsError
        If the mechanism cannot be assembled or the solver does not converge.
    """
    c = _Ctx(corner)
    if c.rack_axis is None:
        if rack != 0.0:
            raise ValueError("rack must be 0 on an axle without steering")
        tie_inner = np.asarray(c.p["tie_inner"])
    else:
        tie_inner = c.p["tie_inner"] + rack * c.rack_axis

    if guess is None:
        x0 = np.zeros(3)
        phi0 = 0.0
    else:
        x0 = np.array([guess.theta_uca, guess.theta_lca, guess.psi])
        phi0 = guess.rocker_angle
    try:
        sol = root(
            lambda x: _residuals(c, x, travel, tie_inner),
            x0,
            method="hybr",
            tol=1e-14,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise KinematicsError(f"solver failed at travel {travel:.6f} m: {exc}") from exc
    x = np.asarray(sol.x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise KinematicsError(f"solver diverged at travel {travel:.6f} m")
    worst = float(np.max(np.abs(_residuals(c, x, travel, tie_inner))))
    if worst > _RESIDUAL_TOL:
        raise KinematicsError(
            f"no convergence at travel {travel:.6f} m, rack {rack:.6f} m (residual {worst:.2e} m)"
        )
    return _assemble(c, x, travel, rack, tie_inner, phi0)


def _assemble(
    c: _Ctx, x: Vec, travel: float, rack: float, tie_inner: Vec, phi0: float
) -> CornerState:
    th_u, th_l, psi = float(x[0]), float(x[1]), float(x[2])
    p = c.p
    s = c.sign
    ubj, lbj, rot = _pose(c, th_u, th_l, psi)
    pts: dict[str, Vec] = {k: np.array(v) for k, v in p.items()}
    pts["uca_outer"] = ubj
    pts["lca_outer"] = lbj
    pts["tie_outer"] = _on_upright(c, lbj, rot, p["tie_outer"])
    pts["tie_inner"] = np.array(tie_inner)
    pts["wheel_center"] = _on_upright(c, lbj, rot, c.wc0)
    if c.mount == "upright":
        pts["pushrod_outer"] = _on_upright(c, lbj, rot, p["pushrod_outer"])
    elif c.mount == "lca":
        pts["pushrod_outer"] = p["lca_front"] + axis_angle_matrix(c.lca_axis, s * th_l) @ (
            p["pushrod_outer"] - p["lca_front"]
        )
    else:
        pts["pushrod_outer"] = p["uca_front"] + axis_angle_matrix(c.uca_axis, s * th_u) @ (
            p["pushrod_outer"] - p["uca_front"]
        )
    try:
        rk = solve_rocker(c.corner, pts["pushrod_outer"], phi0)
    except KinematicsError as exc:
        raise KinematicsError(f"{exc} at travel {travel:.6f} m") from exc
    pts["pushrod_inner"] = rk.pushrod_inner
    pts["rocker_spring"] = rk.rocker_spring
    if rk.rocker_arb is not None:
        pts["rocker_arb"] = rk.rocker_arb

    spin = rot @ c.spin0
    spin = spin / np.linalg.norm(spin)
    up = np.array([0.0, 0.0, 1.0]) - spin[2] * spin
    up = up / np.linalg.norm(up)
    pts["contact_patch"] = pts["wheel_center"] - c.radius * up

    cl = float(np.linalg.norm(pts["rocker_spring"] - pts["spring_chassis"]))
    coil = c.corner.axle_cfg.coilover
    push_len = float(np.linalg.norm(pts["pushrod_outer"] - pts["pushrod_inner"]))
    static_push = float(np.linalg.norm(p["pushrod_outer"] - p["pushrod_inner"]))
    res = _residuals(c, x, travel, tie_inner)
    max_res = max(float(np.max(np.abs(res))), abs(push_len - static_push))
    for v in pts.values():
        v.setflags(write=False)
    return CornerState(
        corner_id=c.corner.id,
        travel=float(travel),
        rack=float(rack),
        theta_uca=th_u,
        theta_lca=th_l,
        psi=psi,
        rocker_angle=rk.angle,
        points=pts,
        spin_axis=spin,
        coilover_length=cl,
        toggle_margin=rk.toggle_margin,
        max_residual=max_res,
        coilover_in_range=bool(coil.length_min <= cl <= coil.length_max),
    )


class _Advance:
    """Continuation helper solving over either travel or rack."""

    def __init__(
        self, corner: Corner, rack: float, travel: float, over: Literal["travel", "rack"]
    ) -> None:
        self.corner, self.rack, self.travel, self.over = corner, rack, travel, over

    def solve(self, value: float, guess: CornerState | None) -> CornerState:
        if self.over == "travel":
            return solve_corner(self.corner, value, self.rack, guess=guess)
        return solve_corner(self.corner, self.travel, value, guess=guess)

    def value(self, state: CornerState) -> float:
        return state.travel if self.over == "travel" else state.rack


def _reach(adv: _Advance, state: CornerState, target: float, depth: int = 0) -> CornerState:
    """Solve at ``target`` from ``state``; halve the step on failure (max 6 halvings)."""
    try:
        return adv.solve(target, state)
    except KinematicsError:
        if depth >= _MAX_HALVINGS:
            raise
    mid = 0.5 * (adv.value(state) + target)
    return _reach(adv, _reach(adv, state, mid, depth + 1), target, depth + 1)


def _continuation(adv: _Advance, values: ArrayLike) -> list[CornerState]:
    vals = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if vals.size == 0:
        return []
    start = int(np.argmin(np.abs(vals)))
    out: dict[int, CornerState] = {}
    try:
        out[start] = adv.solve(float(vals[start]), None)
    except KinematicsError:
        if abs(vals[start]) < 1e-12:
            raise
        out[start] = _reach(adv, adv.solve(0.0, None), float(vals[start]))
    for direction in (1, -1):
        prev = out[start]
        i = start + direction
        while 0 <= i < vals.size:
            prev = _reach(adv, prev, float(vals[i]))
            out[i] = prev
            i += direction
    return [out[i] for i in range(vals.size)]


def sweep_corner(corner: Corner, travels: ArrayLike, rack: float = 0.0) -> list[CornerState]:
    """Solve a travel sweep by continuation outward from the point nearest to 0.

    Returns one state per requested travel, in input order. On non-convergence the
    step is halved (up to 6 times) before ``KinematicsError`` is raised.
    """
    return _continuation(_Advance(corner, rack, 0.0, "travel"), travels)


def sweep_rack(corner: Corner, racks: ArrayLike, travel: float = 0.0) -> list[CornerState]:
    """Solve a rack sweep at fixed travel by continuation (see ``sweep_corner``)."""
    return _continuation(_Advance(corner, 0.0, travel, "rack"), racks)
