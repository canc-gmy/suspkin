"""Shared test helpers: vehicle fixtures with a well-conditioned rocker."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from suspkin.geometry import Vehicle, build_corners, load_vehicle
from suspkin.geometry.corners import Corner, CornerId

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"
SPRING_SIGN = -1.0


def _fix_rocker(axle: object, spring_sign: float) -> object:
    """Re-place the rocker points so the pushrod is ~perpendicular to the rocker lever.

    The stock minimal fixture sits a few mm from a toggle point; this variant
    keeps the pushrod, wishbones and upright untouched.
    """
    hp = axle.hardpoints  # type: ignore[attr-defined]
    inner = np.array(hp.pushrod_inner)
    outer = np.array(hp.pushrod_outer)
    u = (outer - inner)[1:]
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n = -n
    r = 0.05
    pivot = inner.copy()
    pivot[1:] -= r * n
    n2 = spring_sign * np.array([-n[1], n[0]])
    spring = pivot + np.array([0.05, 0.05 * n2[0], 0.05 * n2[1]])
    arb = pivot + np.array([-0.04, 0.04 * n2[0], 0.04 * n2[1]])
    chassis = spring + 0.2 * np.array([-0.3, 0.0, 1.0]) / np.hypot(0.3, 1.0)
    new_hp = hp.model_copy(
        update={
            "spring_chassis": tuple(chassis),
            "rocker_pivot": tuple(pivot),
            "rocker_spring": tuple(spring),
            "rocker_arb": tuple(arb),
        }
    )
    return axle.model_copy(update={"hardpoints": new_hp})  # type: ignore[attr-defined]


def good_vehicle(spring_sign: float = SPRING_SIGN) -> Vehicle:
    """Minimal vehicle with the rocker re-placed away from toggle."""
    v = load_vehicle(FIXTURE)
    new = v.model_copy(
        update={
            "front": _fix_rocker(v.front, spring_sign),
            "rear": _fix_rocker(v.rear, spring_sign),
        }
    )
    return Vehicle.model_validate(new.model_dump())


def good_corners(spring_sign: float = SPRING_SIGN) -> dict[CornerId, Corner]:
    return build_corners(good_vehicle(spring_sign))


def max_invariant_error(corner: Corner, state: object) -> float:
    """Largest deviation [m] of any rigid-body distance from its static value."""
    sp = corner.points
    pts = state.points  # type: ignore[attr-defined]

    def d(p: dict[str, object] | object, a: str, b: str) -> float:
        return float(np.linalg.norm(p[a] - p[b]))  # type: ignore[index]

    pairs = [
        ("uca_front", "uca_outer"),
        ("uca_rear", "uca_outer"),
        ("lca_front", "lca_outer"),
        ("lca_rear", "lca_outer"),
        ("pushrod_inner", "pushrod_outer"),
        ("rocker_pivot", "pushrod_inner"),
        ("rocker_pivot", "rocker_spring"),
        ("uca_outer", "lca_outer"),
        ("uca_outer", "wheel_center"),
        ("lca_outer", "wheel_center"),
        ("uca_outer", "tie_outer"),
        ("lca_outer", "tie_outer"),
        ("tie_outer", "wheel_center"),
    ]
    if "rocker_arb" in sp:
        pairs.append(("rocker_pivot", "rocker_arb"))
    if corner.axle_cfg.pushrod_mount == "upright":
        pairs += [("pushrod_outer", "uca_outer"), ("pushrod_outer", "lca_outer")]
    err = max(abs(d(pts, a, b) - d(sp, a, b)) for a, b in pairs)
    err = max(err, abs(d(pts, "tie_inner", "tie_outer") - d(sp, "tie_inner", "tie_outer")))
    ax = corner.rocker_axis
    for n in ("pushrod_inner", "rocker_spring"):
        along = float((pts[n] - pts["rocker_pivot"]) @ ax)
        err = max(err, abs(along - float((sp[n] - sp["rocker_pivot"]) @ ax)))
    return err
