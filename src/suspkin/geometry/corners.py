"""Per-corner geometry with left/right mirroring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.model import HARDPOINT_NAMES, Axle, Vehicle
from suspkin.geometry.vectors import mirror_y

CornerId = Literal["FL", "FR", "RL", "RR"]


@dataclass(frozen=True)
class Corner:
    """One corner with absolute (side-resolved) hardpoints."""

    id: CornerId
    axle: Literal["front", "rear"]
    side: Literal["left", "right"]
    points: Mapping[str, NDArray[np.float64]]
    rocker_axis: NDArray[np.float64]
    axle_cfg: Axle


def _ro(v: object) -> NDArray[np.float64]:
    a = np.array(v, dtype=np.float64, copy=True)
    a.setflags(write=False)
    return a


def _build(cid: CornerId, name: Literal["front", "rear"], axle: Axle) -> Corner:
    side: Literal["left", "right"] = "left" if cid.endswith("L") else "right"
    hp = axle.hardpoints
    names = list(HARDPOINT_NAMES)
    if hp.rocker_arb is not None:
        names.append("rocker_arb")
    pts: dict[str, NDArray[np.float64]] = {}
    for n in names:
        p = np.array(getattr(hp, n), dtype=np.float64)
        pts[n] = p if side == "left" else mirror_y(p)
    axis = np.array(hp.rocker_axis, dtype=np.float64)
    if side == "right":
        axis = mirror_y(axis)
        for k, v in axle.right_overrides.items():
            pts[k] = np.array(v, dtype=np.float64)
    return Corner(
        id=cid,
        axle=name,
        side=side,
        points={k: _ro(v) for k, v in pts.items()},
        rocker_axis=_ro(axis),
        axle_cfg=axle,
    )


def build_corners(vehicle: Vehicle) -> dict[CornerId, Corner]:
    """Build the four corners from a vehicle.

    Left corners use the hardpoints as given. Right corners mirror y -> -y and then
    apply ``right_overrides`` (already in right-side coordinates). The right
    ``rocker_axis`` is (x, -y, z); since mirroring is a reflection, the rotation
    sense about the mirrored axis flips relative to the left side.
    """
    return {
        "FL": _build("FL", "front", vehicle.front),
        "FR": _build("FR", "front", vehicle.front),
        "RL": _build("RL", "rear", vehicle.rear),
        "RR": _build("RR", "rear", vehicle.rear),
    }


def arb_points(corner: Corner) -> dict[str, NDArray[np.float64]]:
    """ARB points of a corner, mirrored for right-side corners.

    Returns ``arm_end`` plus ``bar_pivot`` (torsional, blade) or ``beam_center``
    (beam; on the centreline, so identical on both sides); an empty dict if the
    axle has no ARB.
    """
    arb = corner.axle_cfg.arb
    if arb is None:
        return {}
    raw: dict[str, tuple[float, float, float]] = {"arm_end": arb.arm_end}
    if arb.type == "beam":
        raw["beam_center"] = arb.beam_center
    else:
        raw["bar_pivot"] = arb.bar_pivot
    out: dict[str, NDArray[np.float64]] = {}
    for k, v in raw.items():
        a = np.array(v, dtype=np.float64)
        out[k] = _ro(a if corner.side == "left" else mirror_y(a))
    return out
