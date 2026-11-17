"""Vehicle-level metric formulas (v0.5): anti-features, roll, steering, pitch centre.

Pure functions on numbers (SI, rad). Conventions follow Milliken, *Race Car Vehicle
Dynamics* (RCVD): ISO 8855 frame (x forward, z up), origin on ground at the mid
front axle, so the rear axle is at x = -wheelbase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.models import Axle, Vehicle

G = 9.80665  # m/s^2

_NAN = float("nan")


# --------------------------------------------------------------------------- anti-features
def anti_dive_front(
    sv_angle: float, h_cg: float, wheelbase: float, brake_bias_front: float
) -> float:
    """Front anti-dive [%] = tan(angle) * bias_f * L / h_cg * 100 (RCVD).

    ``sv_angle`` is the signed side-view swing angle (positive = anti-geometry, see
    ``centres.sv_swing_angle``). Outboard brakes: pass the contact-patch angle;
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
