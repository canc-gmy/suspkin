"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from suspkin.geometry import Vehicle, build_corners, load_vehicle
from suspkin.geometry.corners import Corner, CornerId

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"

_NEAR_TOGGLE = {
    "rocker_pivot": (0.0, 0.18, 0.45),
    "rocker_spring": (0.05, 0.18, 0.52),
    "rocker_arb": (-0.04, 0.17, 0.48),
    "spring_chassis": (0.10, 0.12, 0.30),
}


def fixture_vehicle() -> Vehicle:
    """The minimal fixture vehicle."""
    return load_vehicle(FIXTURE)


def fixture_corners() -> dict[CornerId, Corner]:
    """The four corners of the minimal fixture."""
    return build_corners(load_vehicle(FIXTURE))


def near_toggle_corners() -> dict[CornerId, Corner]:
    """Front axle with the pushrod nearly in line with the rocker lever (in-memory)."""
    v = load_vehicle(FIXTURE)
    hp = v.front.hardpoints.model_copy(update=_NEAR_TOGGLE)
    v2 = v.model_copy(update={"front": v.front.model_copy(update={"hardpoints": hp})})
    return build_corners(v2)


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
