"""Rotation centres of a solved pose for the 3D view (SI, in the scene's plotted frame).

The 3D scene draws the body frame (the body is fixed, the ground tilts), so every point here is
in that frame: the contact patches equal the scene's contact markers. The math is reused from
``suspkin.metrics`` (plane-section instant centres, roll centre pair, pitch centre) with the
state's ground contact point as the contact patch, as ``vehicle_frame`` does.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from suspkin.app.models import (
    AxleCentres,
    AxleHeights,
    CornerCentres,
    CornerLengths,
    SceneCentres,
)
from suspkin.geometry import CornerId, Vehicle, build_corners
from suspkin.geometry.vectors import body_rotation
from suspkin.kinematics import VehicleState, solve_vehicle
from suspkin.metrics import pitch_centre
from suspkin.metrics.corner import (
    front_view_ic,
    roll_centre_pair,
    side_view_ic,
)
from suspkin.metrics.vehicle import _sv_line

__all__ = ["scene_centres"]

_V3 = tuple[float, float, float]
_CORNERS: tuple[CornerId, ...] = ("FL", "FR", "RL", "RR")


def _fin(*v: float) -> bool:
    return all(math.isfinite(x) for x in v)


def scene_centres(vehicle: Vehicle, state: VehicleState | None = None) -> SceneCentres:
    """Instant centres, roll centres and pitch centre at ``state`` (static pose if None).

    Undefined centres (at infinity or degenerate) are ``None``. ``roll_height`` is the ground
    height of the roll centre (the ``rc_*_z`` column of ``vehicle_frame``); ``roll`` is its
    body-frame position, x being the mean contact-patch x of the axle.
    """
    corners = build_corners(vehicle)
    st = solve_vehicle(vehicle, corners) if state is None else state
    cs = {}
    contact: dict[str, _V3] = {}
    fv_ic: dict[str, _V3 | None] = {}
    sv_ic: dict[str, _V3 | None] = {}
    fvsa: dict[str, float | None] = {}
    svsa: dict[str, float | None] = {}
    for cid in _CORNERS:
        cp = np.asarray(st.contact_points[cid], dtype=np.float64)
        s = st.corners[cid]
        # the metrics cut and measure at ``points["contact_patch"]``: use the ground contact point
        cs[cid] = dataclasses.replace(s, points={**s.points, "contact_patch": cp})
        contact[cid] = (float(cp[0]), float(cp[1]), float(cp[2]))
        fv = front_view_ic(cs[cid], corners[cid])
        sv = side_view_ic(cs[cid], corners[cid])
        fv_ok = fv.is_finite and _fin(fv.h, fv.z)
        sv_ok = sv.is_finite and _fin(sv.h, sv.z)
        fv_ic[cid] = (float(cp[0]), fv.h, fv.z) if fv_ok else None
        sv_ic[cid] = (sv.h, float(cp[1]), sv.z) if sv_ok else None
        fvsa[cid] = abs(fv.h - float(cp[1])) if fv_ok else None
        svsa[cid] = abs(sv.h - float(cp[0])) if sv_ok else None

    rot = body_rotation(st.roll, st.pitch)
    roll_pt: dict[str, _V3 | None] = {}
    roll_h: dict[str, float | None] = {}
    pairs: tuple[tuple[str, CornerId, CornerId], ...] = (
        ("front", "FL", "FR"),
        ("rear", "RL", "RR"),
    )
    for axle, left, right in pairs:
        y, z = roll_centre_pair(cs[left], cs[right])
        if not _fin(y, z):
            roll_pt[axle], roll_h[axle] = None, None
            continue
        x = 0.5 * (contact[left][0] + contact[right][0])
        v = np.array([x, y, z])
        roll_pt[axle] = (x, float(y), float(z))
        roll_h[axle] = float(rot[2, :] @ (v - st.pivot) + st.pivot[2] + st.heave)

    px, pz = pitch_centre(_sv_line(cs["FL"], corners["FL"]), _sv_line(cs["RL"], corners["RL"]))
    pitch: _V3 | None = (float(px), 0.0, float(pz)) if _fin(px, pz) else None
    return SceneCentres(
        roll=AxleCentres(front=roll_pt["front"], rear=roll_pt["rear"]),
        roll_height=AxleHeights(front=roll_h["front"], rear=roll_h["rear"]),
        fv_ic=CornerCentres(**fv_ic),
        sv_ic=CornerCentres(**sv_ic),
        contact=CornerCentres(**contact),
        fvsa=CornerLengths(**fvsa),
        svsa=CornerLengths(**svsa),
        pitch=pitch,
    )
