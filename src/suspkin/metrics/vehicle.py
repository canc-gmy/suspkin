"""Vehicle-level metrics: formulas (anti-features, roll, steering, pitch centre) and the summary.

Formulas are pure functions on numbers (SI, rad). Conventions follow Milliken, *Race Car
Vehicle Dynamics* (RCVD): ISO 8855 frame (x forward, z up), origin on ground at the mid
front axle, so the rear axle is at x = -wheelbase.

:func:`vehicle_summary` (end of the module) assembles the static vehicle summary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin.geometry.corners import build_corners
from suspkin.geometry.model import Axle, Vehicle
from suspkin.kinematics.corner import CornerState, sweep_corner
from suspkin.kinematics.vehicle import solve_vehicle
from suspkin.metrics.arb import arb_rates
from suspkin.metrics.corner import side_view_ic, steer_angle, sv_swing_angle, sv_wc_angle
from suspkin.metrics.table import add_gradients, metrics_frame, static_summary

G = 9.80665  # m/s^2

_NAN = float("nan")
_ACKERMANN_AT = (5.0, 10.0, 15.0)  # deg of outer-wheel steer


# --------------------------------------------------------------------------- anti-features
def anti_dive_front(
    sv_angle: float, h_cg: float, wheelbase: float, brake_bias_front: float
) -> float:
    """Front anti-dive [%] = tan(angle) * bias_f * L / h_cg * 100 (RCVD).

    ``sv_angle`` is the signed side-view swing angle (positive = anti-geometry, see
    ``corner.sv_swing_angle``). Outboard brakes: pass the contact-patch angle;
    inboard brakes: the wheel-centre angle (see `anti_angle_for`).
    """
    return math.tan(sv_angle) * brake_bias_front * wheelbase / h_cg * 100.0


def anti_lift_rear(
    sv_angle: float, h_cg: float, wheelbase: float, brake_bias_front: float
) -> float:
    """Rear anti-lift [%] = tan(angle) * (1 - bias_f) * L / h_cg * 100 (RCVD)."""
    return math.tan(sv_angle) * (1.0 - brake_bias_front) * wheelbase / h_cg * 100.0


def anti_squat_rear(
    sv_wc_angle: float, h_cg: float, wheelbase: float, drive_split_front: float
) -> float:
    """Rear anti-squat [%] = tan(angle_wc) * (1 - split_f) * L / h_cg * 100.

    Drive torque is reacted through the chassis, so the wheel-centre swing angle is
    used (positive = anti-geometry).
    """
    return math.tan(sv_wc_angle) * (1.0 - drive_split_front) * wheelbase / h_cg * 100.0


def anti_lift_front_accel(
    sv_wc_angle: float, h_cg: float, wheelbase: float, drive_split_front: float
) -> float:
    """Front anti-lift under acceleration [%] = tan(angle_wc) * split_f * L / h_cg * 100."""
    return math.tan(sv_wc_angle) * drive_split_front * wheelbase / h_cg * 100.0


def anti_angle_for(
    axle_cfg: Axle,
    phase: Literal["brake", "drive"],
    sv_swing_angle: float,
    sv_wc_angle: float,
) -> float:
    """Pick the side-view angle that governs the anti-feature.

    Braking: outboard brakes react through the contact patch (``sv_swing_angle``),
    inboard brakes through the chassis (``sv_wc_angle``). Drive uses the wheel-centre
    angle (inboard drive with halfshafts, the usual layout); hub/outboard drive (drive
    torque reacted by the upright) would use the contact-patch angle instead.
    """
    if phase == "drive":
        return sv_wc_angle
    return sv_swing_angle if axle_cfg.brakes == "outboard" else sv_wc_angle


# --------------------------------------------------------------------------- roll
def roll_stiffness_springs(wheel_rate: float, track: float) -> float:
    """Axle roll stiffness from springs [N*m/rad] = k_w * t^2 / 2 (k_w per wheel, N/m)."""
    return wheel_rate * track**2 / 2.0


def roll_stiffness_arb(arb_wheel_rate_roll: float, track: float) -> float:
    """Axle roll stiffness from the ARB [N*m/rad] = k_w,roll * t^2 / 2.

    ``arb_wheel_rate_roll`` is the roll-mode ARB wheel rate (1/2 k MR^2, N/m).
    """
    return arb_wheel_rate_roll * track**2 / 2.0


def roll_stiffness_tyres(k_t: float, track: float) -> float:
    """Axle roll stiffness of the tyres [N*m/rad] = k_t * t^2 / 2 (k_t per tyre)."""
    return k_t * track**2 / 2.0


def axle_roll_stiffness(k_spring_roll: float, k_arb_roll: float, k_tyre_roll: float) -> float:
    """Axle roll stiffness [N*m/rad]: (springs + ARB) in series with the tyres."""
    k_susp = k_spring_roll + k_arb_roll
    return k_susp * k_tyre_roll / (k_susp + k_tyre_roll)


def roll_gradient(
    m_sprung: float, h_cg: float, h_roll_axis_at_cg: float, k_front: float, k_rear: float
) -> float:
    """Roll gradient [rad per g of lateral acceleration].

    phi/(a_y/g) = m_s g d / (K_f + K_r - m_s g d), d = h_cg - h_ra (RCVD). The
    ``- m_s g d`` term is the destabilising weight moment of the sprung mass. Multiply
    by 180/pi for deg/g. Raises ValueError if the denominator is not positive.
    """
    moment = m_sprung * G * (h_cg - h_roll_axis_at_cg)
    denom = k_front + k_rear - moment
    if denom <= 0.0:
        raise ValueError("roll stiffness does not exceed the destabilising moment m_s*g*d")
    return moment / denom


def roll_axis_height_at(
    x: float, x_front: float, rc_front: float, x_rear: float, rc_rear: float
) -> float:
    """Roll-axis height at longitudinal position x: linear between the two roll centres."""
    t = (x - x_front) / (x_rear - x_front)
    return rc_front + t * (rc_rear - rc_front)


@dataclass(frozen=True)
class LoadTransfer:
    """Lateral load transfer per axle [N].

    Each value is the load shift ``dW`` per wheel: the outside wheel gains ``dW`` and the
    inside wheel loses ``dW``, so the outside-minus-inside load difference of the axle
    is ``2 dW``.
    """

    front_elastic: float
    front_geometric: float
    front_unsprung: float
    rear_elastic: float
    rear_geometric: float
    rear_unsprung: float

    @property
    def front(self) -> float:
        """Total front-axle transfer."""
        return self.front_elastic + self.front_geometric + self.front_unsprung

    @property
    def rear(self) -> float:
        """Total rear-axle transfer."""
        return self.rear_elastic + self.rear_geometric + self.rear_unsprung

    @property
    def tlltd_front(self) -> float:
        """Total lateral load transfer distribution = dW_f / (dW_f + dW_r)."""
        return tlltd_front(self.front, self.rear)


def tlltd_front(dw_front: float, dw_rear: float) -> float:
    """Front TLLTD = dW_f / (dW_f + dW_r); nan if the total is zero."""
    total = dw_front + dw_rear
    return dw_front / total if total != 0.0 else _NAN


def lateral_load_transfer(
    ay: float,
    m_sprung: float,
    m_unsprung_front: float,
    m_unsprung_rear: float,
    h_cg: float,
    rc_front: float,
    rc_rear: float,
    hu_front: float,
    hu_rear: float,
    k_front: float,
    k_rear: float,
    track_front: float,
    track_rear: float,
    wheelbase: float,
    a: float,
) -> LoadTransfer:
    """Lateral load transfer per axle for lateral acceleration ``ay`` [m/s^2].

    Standard decomposition (RCVD). With CG at x = -a (a = CG to front axle),
    h_ra the roll-axis height at the CG, d = h_cg - h_ra, K = K_f + K_r - m_s g d,
    phi = m_s ay d / K (roll angle) and sprung mass shares m_sf = m_s (L-a)/L,
    m_sr = m_s a/L, per axle (i = f, r):

        elastic   = K_i phi / t_i
        geometric = m_si ay rc_i / t_i
        unsprung  = m_ui ay hu_i / t_i       (m_ui = total unsprung mass of the axle)

    ``m_unsprung_*`` are axle totals (both corners). ``h_cg`` is the sprung-mass CG
    height. K_f and K_r are the axle roll stiffnesses [N*m/rad].
    """
    h_ra = roll_axis_height_at(-a, 0.0, rc_front, -wheelbase, rc_rear)
    d = h_cg - h_ra
    denom = k_front + k_rear - m_sprung * G * d
    if denom <= 0.0:
        raise ValueError("roll stiffness does not exceed the destabilising moment m_s*g*d")
    phi = m_sprung * ay * d / denom
    msf = m_sprung * (wheelbase - a) / wheelbase
    msr = m_sprung * a / wheelbase
    return LoadTransfer(
        front_elastic=k_front * phi / track_front,
        front_geometric=msf * ay * rc_front / track_front,
        front_unsprung=m_unsprung_front * ay * hu_front / track_front,
        rear_elastic=k_rear * phi / track_rear,
        rear_geometric=msr * ay * rc_rear / track_rear,
        rear_unsprung=m_unsprung_rear * ay * hu_rear / track_rear,
    )


def sprung_mass(vehicle: Vehicle) -> float:
    """Sprung mass [kg] = total mass - 2 * (front + rear unsprung mass per corner)."""
    return vehicle.vehicle.mass - 2.0 * (vehicle.front.unsprung_mass + vehicle.rear.unsprung_mass)


# --------------------------------------------------------------------------- steering
def inner_outer(delta_left: float, delta_right: float) -> tuple[float, float]:
    """Return (delta_inner, delta_outer) with the turn direction from the mean angle.

    Left turn (mean > 0, +delta = left): left wheel is inner. Right turn: right wheel.
    Signs of the inputs are kept.
    """
    if delta_left + delta_right >= 0.0:
        return delta_left, delta_right
    return delta_right, delta_left


def ackermann_percent(
    delta_inner: float, delta_outer: float, wheelbase: float, track: float
) -> float:
    """Ackermann [%] = (d_i - d_o) / (d_i,ideal - d_o) * 100.

    Ideal inner angle from cot d_o - cot d_i = t / L (steering-axis spacing taken as
    the track). 100 = true Ackermann, 0 = parallel steer, negative = anti-Ackermann.
    Angles are mirrored so the turn is positive. nan if |d_o| < 1e-4 rad.
    """
    if abs(delta_outer) < 1e-4:
        return _NAN
    s = math.copysign(1.0, delta_outer)
    di, do = s * delta_inner, s * delta_outer
    ideal = math.atan2(1.0, 1.0 / math.tan(do) - track / wheelbase)
    return (di - do) / (ideal - do) * 100.0


def steering_ratio(rack: float, c_factor: float, mean_road_wheel: float) -> float:
    """Overall ratio = handwheel angle / mean road-wheel angle.

    Handwheel angle = rack / c_factor * 2 pi (c_factor = rack travel per pinion
    revolution, m/rev). nan if the mean road-wheel angle is below 1e-9 rad.
    """
    if abs(mean_road_wheel) < 1e-9:
        return _NAN
    return (rack / c_factor * 2.0 * math.pi) / mean_road_wheel


def turning_radius(wheelbase: float, delta_inner: float, delta_outer: float) -> float:
    """Low-speed turning radius [m] = L / tan(mean steer angle) (bicycle model).

    Measured to the rear-axle centre; inf for zero steer.
    """
    t = math.tan(0.5 * (delta_inner + delta_outer))
    return math.inf if t == 0.0 else wheelbase / t


# --------------------------------------------------------------------------- pitch centre
def pitch_centre(
    front_line: tuple[NDArray[np.float64], NDArray[np.float64]],
    rear_line: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> tuple[float, float]:
    """Intersection (x, z) of two side-view lines, each (point, direction).

    Returns (nan, nan) if the lines are parallel.
    """
    p1 = np.asarray(front_line[0], dtype=np.float64)
    d1 = np.asarray(front_line[1], dtype=np.float64)
    p2 = np.asarray(rear_line[0], dtype=np.float64)
    d2 = np.asarray(rear_line[1], dtype=np.float64)
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12 * max(float(np.linalg.norm(d1) * np.linalg.norm(d2)), 1e-300):
        return _NAN, _NAN
    w = p2 - p1
    s = (w[0] * d2[1] - w[1] * d2[0]) / cross
    pt = p1 + s * d1
    return float(pt[0]), float(pt[1])


# --------------------------------------------------------------------------- vehicle summary
# Vehicle-level summary: static rates, roll, anti-features, roll/steer coefficients.
#
# All values SI (m, rad, N, N/m, N*m/rad, kg). Conventions:
#
# * Roll sweep: body roll phi is right-hand about +x (positive = left side up, which is
#   the roll of a LEFT turn). The outer wheel for positive roll is therefore the right
#   wheel (FR / RR); ``roll_camber_*`` is d(camber_ground of that wheel)/d(roll), a
#   negative value meaning the outer wheel gains negative camber relative to the ground.
# * ``roll_steer_*`` = d(mean of left and right ``steer_ground``)/d(roll) [rad/rad].
# * Roll centres: height above ground (ground frame); lateral position y relative to the
#   car centreline (body frame, independent of the pivot); ``rc_migration_*`` =
#   d(rc_y)/d(roll) [m/rad] and ``rc_height_curvature`` = d2(rc_z)/d(roll)2 [m/rad^2]
#   (the first derivative is zero by symmetry).
# * Mass: ``vehicle.mass`` is the total mass, ``cg_height`` the total CG height and the
#   total CG is at ``x = -a``, ``a = wheelbase (1 - front_mass_ratio)``. The sprung CG
#   follows from removing the unsprung masses (per axle: 2 * ``unsprung_mass`` at the axle
#   wheel-centre x and at ``unsprung_cg_height``, default the tyre radius): moments
#   give ``h_s = (M h - sum m_u h_u) / m_s`` and ``x_s = (M x - sum m_u x_u) / m_s``.
#   The sprung CG x/height feed the roll axis height and the load transfer.
# * Body rotations use ``pivot`` (default: the total CG, see ``cg_pivot``).
# * Steering (rack sweep over the full rack travel): angles are relative to the straight
#   ahead pose; the lock side is the rack end with the larger mean steer. Ackermann [%] at
#   outer-wheel steer of 5/10/15 deg is interpolated on that side (nan if not reached).


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
    rc_height_curvature: float


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


def _curv0(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Second derivative at x = 0 of a cubic least-squares fit."""
    if not np.all(np.isfinite(y)):
        return _NAN
    return float(2.0 * np.polyfit(x, y, 3)[-3])


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
    ks = roll_stiffness_springs(ss["wheel_rate"], track)
    ka = roll_stiffness_arb(arb_w, track)
    kt = roll_stiffness_tyres(cfg.wheel.tyre_vertical_stiffness, track)
    return {
        "track": track,
        "wheel_rate": ss["wheel_rate"],
        "arb_wheel_rate_roll": arb_w,
        "roll_stiffness_springs": ks,
        "roll_stiffness_arb": ka,
        "roll_stiffness_tyres": kt,
        "roll_stiffness": axle_roll_stiffness(ks, ka, kt),
    }


def _steering(
    vehicle: Vehicle, pivot: ArrayLike | None = None, n_rack: int = 41
) -> dict[str, float]:
    # lazy: suspkin.sweeps imports suspkin.metrics at module level (import cycle otherwise)
    from suspkin.sweeps import sweep_vehicle

    corners = build_corners(vehicle)
    steer = vehicle.front.steering
    assert steer is not None
    fl, fr = corners["FL"], corners["FR"]
    static = solve_vehicle(vehicle, corners, pivot=pivot)
    s0 = (steer_angle(static.corners["FL"], fl), steer_angle(static.corners["FR"], fr))
    racks = np.linspace(steer.rack_travel[0], steer.rack_travel[1], n_rack)
    states = sweep_vehicle(vehicle, rack=racks, pivot=pivot)
    dl = np.array([steer_angle(s.corners["FL"], fl) - s0[0] for s in states])
    dr = np.array([steer_angle(s.corners["FR"], fr) - s0[1] for s in states])
    mean = 0.5 * (dl + dr)
    i_lock = int(np.argmax(np.abs(mean)))
    sgn = math.copysign(1.0, mean[i_lock])
    inner = np.empty_like(dl)
    outer = np.empty_like(dl)
    for i in range(len(dl)):
        inner[i], outer[i] = inner_outer(float(dl[i]), float(dr[i]))
    inner, outer = sgn * inner, sgn * outer
    sel = (sgn * mean > 0.0) & (outer > 1e-4)
    ack = np.array(
        [
            ackermann_percent(float(i), float(o), vehicle.wheelbase, vehicle.front_track)
            for i, o in zip(inner[sel], outer[sel], strict=True)
        ]
    )
    o_sel = outer[sel]
    order = np.argsort(o_sel)
    out: dict[str, float] = {
        "max_steer_inner": float(inner[i_lock]),
        "max_steer_outer": float(outer[i_lock]),
        "turning_radius": turning_radius(
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
    sm = sweep_vehicle(vehicle, rack=r_small, pivot=pivot)
    dm = [
        0.5 * (steer_angle(s.corners["FL"], fl) - s0[0] + steer_angle(s.corners["FR"], fr) - s0[1])
        for s in sm
    ]
    out["steering_ratio"] = steering_ratio(1e-3, steer.c_factor, 0.5 * (dm[2] - dm[0]))
    return out


def vehicle_summary(
    vehicle: Vehicle,
    *,
    roll_range: float = 0.035,
    n: int = 15,
    ay_ref: float = 9.81,
    pivot: ArrayLike | None = None,
) -> VehicleSummary:
    """Compute the vehicle summary (see the module docstring)."""
    # lazy: suspkin.sweeps imports suspkin.metrics at module level (import cycle otherwise)
    from suspkin.sweeps import sweep_vehicle, vehicle_frame

    corners = build_corners(vehicle)
    rolls = np.linspace(-roll_range, roll_range, n | 1)
    states = sweep_vehicle(vehicle, roll=rolls, pivot=pivot)
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
            rc_height_curvature=_curv0(roll, rc_z),
        )
    fa, ra = summaries["front"], summaries["rear"]

    vp = vehicle.vehicle
    wb = vehicle.wheelbase
    m_s = sprung_mass(vehicle)
    m_uf = 2.0 * vehicle.front.unsprung_mass
    m_ur = 2.0 * vehicle.rear.unsprung_mass
    hu_f = vp.unsprung_cg_height or vehicle.front.wheel.tyre_radius
    hu_r = vp.unsprung_cg_height or vehicle.rear.wheel.tyre_radius
    x_uf = vehicle.front.hardpoints.wheel_center[0]
    x_ur = vehicle.rear.hardpoints.wheel_center[0]
    a_total = (1.0 - vp.front_mass_ratio) * wb  # total CG to front axle
    h_s = (vp.mass * vp.cg_height - m_uf * hu_f - m_ur * hu_r) / m_s
    x_s = (vp.mass * (-a_total) - m_uf * x_uf - m_ur * x_ur) / m_s
    a = -x_s  # sprung CG to front axle
    h_ra = roll_axis_height_at(x_s, 0.0, fa.rc_height, -wb, ra.rc_height)
    gradient = roll_gradient(m_s, h_s, h_ra, fa.roll_stiffness, ra.roll_stiffness)
    lt = lateral_load_transfer(
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

    static = solve_vehicle(vehicle, corners, pivot=pivot)
    fl, rl = static.corners["FL"], static.corners["RL"]
    cf, cr = corners["FL"], corners["RL"]
    split = vp.effective_drive_split_front
    h_cg = vp.cg_height
    ang_f_brake = anti_angle_for(
        vehicle.front, "brake", sv_swing_angle(fl, cf), sv_wc_angle(fl, cf)
    )
    ang_r_brake = anti_angle_for(vehicle.rear, "brake", sv_swing_angle(rl, cr), sv_wc_angle(rl, cr))
    ang_r_drive = anti_angle_for(vehicle.rear, "drive", sv_swing_angle(rl, cr), sv_wc_angle(rl, cr))
    ang_f_drive = anti_angle_for(
        vehicle.front, "drive", sv_swing_angle(fl, cf), sv_wc_angle(fl, cf)
    )
    pc = pitch_centre(_sv_line(fl, cf), _sv_line(rl, cr))
    steer = _steering(vehicle, pivot)
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
        anti_dive_front=anti_dive_front(ang_f_brake, h_cg, wb, vp.brake_bias_front),
        anti_lift_rear=anti_lift_rear(ang_r_brake, h_cg, wb, vp.brake_bias_front),
        anti_squat_rear=anti_squat_rear(ang_r_drive, h_cg, wb, split),
        anti_lift_front_accel=(
            anti_lift_front_accel(ang_f_drive, h_cg, wb, split) if split > 0.0 else None
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
