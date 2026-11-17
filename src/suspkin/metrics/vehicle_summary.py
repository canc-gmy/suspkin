"""Vehicle-level summary: static rates, roll, anti-features, roll/steer coefficients.

All values SI (m, rad, N, N/m, N*m/rad, kg). Conventions:

* Roll sweep: body roll phi is right-hand about +x (positive = left side up, which is
  the roll of a LEFT turn). The outer wheel for positive roll is therefore the right
  wheel (FR / RR); ``roll_camber_*`` is d(camber_ground of that wheel)/d(roll), a
  negative value meaning the outer wheel gains negative camber relative to the ground.
* ``roll_steer_*`` = d(mean of left and right ``steer_ground``)/d(roll) [rad/rad].
* Roll centres are in the ground frame (height above ground); ``rc_migration_*`` =
  d(rc_y)/d(roll) [m/rad] and ``rc_height_gradient_*`` = d(rc_z)/d(roll) [m/rad].
* Mass: ``vehicle.mass`` is the total mass, ``cg_height`` the total CG height; the
  sprung CG height follows from removing the unsprung masses at
  ``unsprung_cg_height`` (default: the tyre radius).
* Steering (rack sweep over the full rack travel): angles are relative to the straight
  ahead pose; the lock side is the rack end with the larger mean steer. Ackermann [%] at
  outer-wheel steer of 5/10/15 deg is interpolated on that side (nan if not reached).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import build_corners
from suspkin.geometry.models import Vehicle
from suspkin.kinematics.corner import CornerState, sweep_corner
from suspkin.kinematics.vehicle import solve_vehicle
from suspkin.metrics import vehicle as vm
from suspkin.metrics.alignment import steer_angle
from suspkin.metrics.arb import arb_rates
from suspkin.metrics.centres import (
    side_view_ic,
    sv_swing_angle,
    sv_wc_angle,
)
from suspkin.metrics.table import add_gradients, metrics_frame, static_summary

_NAN = float("nan")
_ACKERMANN_AT = (5.0, 10.0, 15.0)  # deg of outer-wheel steer


@dataclass(frozen=True)
class AxleSummary:
    """Static per-axle values."""

    track: float
    wheel_rate: float
    arb_wheel_rate_roll: float
    roll_stiffness_springs: float
    roll_stiffness_arb: float
    roll_stiffness_tyres: float
    roll_stiffness: float
    rc_height: float
    roll_camber: float
    roll_steer: float
    rc_migration: float
    rc_height_gradient: float


@dataclass(frozen=True)
class VehicleSummary:
    """Vehicle summary (SI). See the module docstring for conventions."""

    front: AxleSummary
    rear: AxleSummary
    wheelbase: float
    sprung_mass: float
    sprung_cg_height: float
    roll_axis_height_at_cg: float
    roll_stiffness_total: float
    roll_gradient: float
    ay_ref: float
    load_transfer_front: float
    load_transfer_rear: float
    tlltd_front: float
    anti_dive_front: float
    anti_lift_rear: float
    anti_squat_rear: float
    anti_lift_front_accel: float | None
    pitch_centre_x: float
    pitch_centre_z: float
    max_steer_inner: float
    max_steer_outer: float
    ackermann_5deg: float
    ackermann_10deg: float
    ackermann_15deg: float
    steering_ratio: float
    turning_radius: float

    def to_dict(self) -> dict[str, object]:
        """Nested dict of SI floats (JSON-friendly; nan stays nan)."""
        return asdict(self)


def _slope0(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Derivative at x = 0 of a cubic least-squares fit."""
    if not np.all(np.isfinite(y)):
        return _NAN
    return float(np.polyfit(x, y, 3)[-2])


def _sv_line(st: CornerState, corner: object) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cp = st.points["contact_patch"]
    ic = side_view_ic(st, corner)  # type: ignore[arg-type]
    p = np.array([cp[0], cp[2]], dtype=np.float64)
    if ic.is_finite:
        return p, np.array([ic.h - p[0], ic.z - p[1]])
    return p, np.asarray(ic.direction, dtype=np.float64)


def _axle_static(vehicle: Vehicle, cid_l: str, cid_r: str, axle_name: str) -> dict[str, float]:
    corners = build_corners(vehicle)
    cl, cr = corners[cid_l], corners[cid_r]  # type: ignore[index]
    cfg = cl.axle_cfg
    states = sweep_corner(cl, np.linspace(-0.01, 0.01, 21))
    ss = static_summary(add_gradients(metrics_frame(states, cl)))
    track = vehicle.front_track if axle_name == "front" else vehicle.rear_track
    arb_w = arb_rates(cl, cr, cfg).wheel_rate_roll if cfg.arb is not None else 0.0
    ks = vm.roll_stiffness_springs(ss["wheel_rate"], track)
    ka = vm.roll_stiffness_arb(arb_w, track)
    kt = vm.roll_stiffness_tyres(cfg.wheel.tyre_vertical_stiffness, track)
    return {
        "track": track,
        "wheel_rate": ss["wheel_rate"],
        "arb_wheel_rate_roll": arb_w,
        "roll_stiffness_springs": ks,
        "roll_stiffness_arb": ka,
        "roll_stiffness_tyres": kt,
        "roll_stiffness": vm.axle_roll_stiffness(ks, ka, kt),
    }


def _steering(vehicle: Vehicle, n_rack: int = 41) -> dict[str, float]:
    from suspkin.sweeps.vehicle_sweeps import sweep_vehicle

    corners = build_corners(vehicle)
    steer = vehicle.front.steering
    assert steer is not None
    fl, fr = corners["FL"], corners["FR"]
    static = solve_vehicle(vehicle, corners)
    s0 = (steer_angle(static.corners["FL"], fl), steer_angle(static.corners["FR"], fr))
    racks = np.linspace(steer.rack_travel[0], steer.rack_travel[1], n_rack)
    states = sweep_vehicle(vehicle, rack=racks)
    dl = np.array([steer_angle(s.corners["FL"], fl) - s0[0] for s in states])
    dr = np.array([steer_angle(s.corners["FR"], fr) - s0[1] for s in states])
    mean = 0.5 * (dl + dr)
    i_lock = int(np.argmax(np.abs(mean)))
    sgn = math.copysign(1.0, mean[i_lock])
    inner = np.empty_like(dl)
    outer = np.empty_like(dl)
    for i in range(len(dl)):
        inner[i], outer[i] = vm.inner_outer(float(dl[i]), float(dr[i]))
    inner, outer = sgn * inner, sgn * outer
    sel = (sgn * mean > 0.0) & (outer > 1e-4)
    ack = np.array(
        [
            vm.ackermann_percent(float(i), float(o), vehicle.wheelbase, vehicle.front_track)
            for i, o in zip(inner[sel], outer[sel], strict=True)
        ]
    )
    o_sel = outer[sel]
    order = np.argsort(o_sel)
    out: dict[str, float] = {
        "max_steer_inner": float(inner[i_lock]),
        "max_steer_outer": float(outer[i_lock]),
        "turning_radius": vm.turning_radius(
            vehicle.wheelbase, float(inner[i_lock]), float(outer[i_lock])
        ),
    }
    for deg in _ACKERMANN_AT:
        tgt = math.radians(deg)
        key = f"ackermann_{int(deg)}deg"
        if o_sel.size < 2 or not (o_sel.min() <= tgt <= o_sel.max()):
            out[key] = _NAN
        else:
            out[key] = float(np.interp(tgt, o_sel[order], ack[order]))
    r_small = [-1e-3, 0.0, 1e-3]
    sm = sweep_vehicle(vehicle, rack=r_small)
    dm = [
        0.5 * (steer_angle(s.corners["FL"], fl) - s0[0] + steer_angle(s.corners["FR"], fr) - s0[1])
        for s in sm
    ]
    out["steering_ratio"] = vm.steering_ratio(1e-3, steer.c_factor, 0.5 * (dm[2] - dm[0]))
    return out


def vehicle_summary(
    vehicle: Vehicle, *, roll_range: float = 0.035, n: int = 15, ay_ref: float = 9.81
) -> VehicleSummary:
    """Compute the vehicle summary (see the module docstring)."""
    from suspkin.sweeps.vehicle_sweeps import sweep_vehicle, vehicle_frame

    corners = build_corners(vehicle)
    rolls = np.linspace(-roll_range, roll_range, n | 1)
    states = sweep_vehicle(vehicle, roll=rolls)
    df = vehicle_frame(states, vehicle)
    i0 = int(np.argmin(np.abs(rolls)))
    roll = df["roll"].to_numpy(dtype=float)

    ax = {
        "front": _axle_static(vehicle, "FL", "FR", "front"),
        "rear": _axle_static(vehicle, "RL", "RR", "rear"),
    }
    summaries: dict[str, AxleSummary] = {}
    for name, left, right in (("front", "FL", "FR"), ("rear", "RL", "RR")):
        rc_y = df[f"rc_{name}_y"].to_numpy(dtype=float)
        rc_z = df[f"rc_{name}_z"].to_numpy(dtype=float)
        mean_steer = 0.5 * (df[f"{left}_steer_ground"] + df[f"{right}_steer_ground"])
        summaries[name] = AxleSummary(
            **ax[name],
            rc_height=float(rc_z[i0]),
            roll_camber=_slope0(roll, df[f"{right}_camber_ground"].to_numpy(dtype=float)),
            roll_steer=_slope0(roll, mean_steer.to_numpy(dtype=float)),
            rc_migration=_slope0(roll, rc_y),
            rc_height_gradient=_slope0(roll, rc_z),
        )
    fa, ra = summaries["front"], summaries["rear"]

    vp = vehicle.vehicle
    wb = vehicle.wheelbase
    m_s = vm.sprung_mass(vehicle)
    m_u = vp.mass - m_s
    hu_f = vp.unsprung_cg_height or vehicle.front.wheel.tyre_radius
    hu_r = vp.unsprung_cg_height or vehicle.rear.wheel.tyre_radius
    h_u = 0.5 * (hu_f + hu_r)
    h_s = (vp.mass * vp.cg_height - m_u * h_u) / m_s
    a = (1.0 - vp.front_mass_ratio) * wb  # CG to front axle
    h_ra = vm.roll_axis_height_at(-a, 0.0, fa.rc_height, -wb, ra.rc_height)
    gradient = vm.roll_gradient(m_s, h_s, h_ra, fa.roll_stiffness, ra.roll_stiffness)
    lt = vm.lateral_load_transfer(
        ay_ref,
        m_s,
        2.0 * vehicle.front.unsprung_mass,
        2.0 * vehicle.rear.unsprung_mass,
        h_s,
        fa.rc_height,
        ra.rc_height,
        hu_f,
        hu_r,
        fa.roll_stiffness,
        ra.roll_stiffness,
        fa.track,
        ra.track,
        wb,
        a,
    )

    static = solve_vehicle(vehicle, corners)
    fl, rl = static.corners["FL"], static.corners["RL"]
    cf, cr = corners["FL"], corners["RL"]
    split = vp.effective_drive_split_front
    h_cg = vp.cg_height
    ang_f_brake = vm.anti_angle_for(
        vehicle.front, "brake", sv_swing_angle(fl, cf), sv_wc_angle(fl, cf)
    )
    ang_r_brake = vm.anti_angle_for(
        vehicle.rear, "brake", sv_swing_angle(rl, cr), sv_wc_angle(rl, cr)
    )
    ang_r_drive = vm.anti_angle_for(
        vehicle.rear, "drive", sv_swing_angle(rl, cr), sv_wc_angle(rl, cr)
    )
    ang_f_drive = vm.anti_angle_for(
        vehicle.front, "drive", sv_swing_angle(fl, cf), sv_wc_angle(fl, cf)
    )
    pc = vm.pitch_centre(_sv_line(fl, cf), _sv_line(rl, cr))
    steer = _steering(vehicle)
    return VehicleSummary(
        front=fa,
        rear=ra,
        wheelbase=wb,
        sprung_mass=m_s,
        sprung_cg_height=h_s,
        roll_axis_height_at_cg=h_ra,
        roll_stiffness_total=fa.roll_stiffness + ra.roll_stiffness,
        roll_gradient=gradient,
        ay_ref=ay_ref,
        load_transfer_front=lt.front,
        load_transfer_rear=lt.rear,
        tlltd_front=lt.tlltd_front,
        anti_dive_front=vm.anti_dive_front(ang_f_brake, h_cg, wb, vp.brake_bias_front),
        anti_lift_rear=vm.anti_lift_rear(ang_r_brake, h_cg, wb, vp.brake_bias_front),
        anti_squat_rear=vm.anti_squat_rear(ang_r_drive, h_cg, wb, split),
        anti_lift_front_accel=(
            vm.anti_lift_front_accel(ang_f_drive, h_cg, wb, split) if split > 0.0 else None
        ),
        pitch_centre_x=pc[0],
        pitch_centre_z=pc[1],
        max_steer_inner=steer["max_steer_inner"],
        max_steer_outer=steer["max_steer_outer"],
        ackermann_5deg=steer["ackermann_5deg"],
        ackermann_10deg=steer["ackermann_10deg"],
        ackermann_15deg=steer["ackermann_15deg"],
        steering_ratio=steer["steering_ratio"],
        turning_radius=steer["turning_radius"],
    )
