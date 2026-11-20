"""Vehicle-level metrics: formulas (anti-features, roll, steering, pitch centre) and the summary.

Formulas are pure functions on numbers (SI, rad). Conventions follow Milliken, *Race Car
Vehicle Dynamics* (RCVD): ISO 8855 frame (x forward, z up), origin on ground at the mid
front axle, so the rear axle is at x = -wheelbase.

:func:`vehicle_summary` (end of the module) assembles the static vehicle summary.
Signs and formulas: ``docs/concepts/conventions.md`` sections 5.5, 5.6 and 5.9; discussion in
``docs/concepts/metrics.md``.
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

    Args:
        sv_angle: Side-view swing angle in rad (positive = anti-geometry).
        h_cg: Total CG height in m.
        wheelbase: Wheelbase ``L`` in m.
        brake_bias_front: Front share of the brake force, in [0, 1].

    Returns:
        Anti-dive in percent (the percent number itself; 100 = full cancellation of dive).
    """
    return math.tan(sv_angle) * brake_bias_front * wheelbase / h_cg * 100.0


def anti_lift_rear(
    sv_angle: float, h_cg: float, wheelbase: float, brake_bias_front: float
) -> float:
    """Rear anti-lift [%] = tan(angle) * (1 - bias_f) * L / h_cg * 100 (RCVD).

    Args:
        sv_angle: Rear side-view swing angle in rad (positive = anti-geometry).
        h_cg: Total CG height in m.
        wheelbase: Wheelbase ``L`` in m.
        brake_bias_front: Front share of the brake force, in [0, 1].

    Returns:
        Anti-lift in percent.
    """
    return math.tan(sv_angle) * (1.0 - brake_bias_front) * wheelbase / h_cg * 100.0


def anti_squat_rear(
    sv_wc_angle: float, h_cg: float, wheelbase: float, drive_split_front: float
) -> float:
    """Rear anti-squat [%] = tan(angle_wc) * (1 - split_f) * L / h_cg * 100.

    Drive torque is reacted through the chassis, so the wheel-centre swing angle is
    used (positive = anti-geometry).

    Args:
        sv_wc_angle: Rear wheel-centre side-view angle in rad (positive = anti-geometry).
        h_cg: Total CG height in m.
        wheelbase: Wheelbase ``L`` in m.
        drive_split_front: Front share of the drive torque, in [0, 1].

    Returns:
        Anti-squat in percent.
    """
    return math.tan(sv_wc_angle) * (1.0 - drive_split_front) * wheelbase / h_cg * 100.0


def anti_lift_front_accel(
    sv_wc_angle: float, h_cg: float, wheelbase: float, drive_split_front: float
) -> float:
    """Front anti-lift under acceleration [%] = tan(angle_wc) * split_f * L / h_cg * 100.

    Args:
        sv_wc_angle: Front wheel-centre side-view angle in rad (positive = anti-geometry).
        h_cg: Total CG height in m.
        wheelbase: Wheelbase ``L`` in m.
        drive_split_front: Front share of the drive torque, in [0, 1].

    Returns:
        Front anti-lift under drive in percent.
    """
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

    Args:
        axle_cfg: Axle (``brakes`` is ``"outboard"`` or ``"inboard"``).
        phase: ``"brake"`` or ``"drive"``.
        sv_swing_angle: Side-view angle of the contact patch in rad.
        sv_wc_angle: Side-view angle of the wheel centre in rad.

    Returns:
        The governing angle in rad.
    """
    if phase == "drive":
        return sv_wc_angle
    return sv_swing_angle if axle_cfg.brakes == "outboard" else sv_wc_angle


# --------------------------------------------------------------------------- roll
def roll_stiffness_springs(wheel_rate: float, track: float) -> float:
    """Axle roll stiffness from springs [N*m/rad] = k_w * t^2 / 2 (k_w per wheel, N/m).

    Args:
        wheel_rate: Wheel rate per wheel in N/m.
        track: Axle track in m.

    Returns:
        Roll stiffness in N m/rad.
    """
    return wheel_rate * track**2 / 2.0


def roll_stiffness_arb(arb_wheel_rate_roll: float, track: float) -> float:
    """Axle roll stiffness from the ARB [N*m/rad] = k_w,roll * t^2 / 2.

    ``arb_wheel_rate_roll`` is the roll-mode ARB wheel rate (1/2 k MR^2, N/m).

    Args:
        arb_wheel_rate_roll: Roll-mode ARB wheel rate in N/m.
        track: Axle track in m.

    Returns:
        Roll stiffness in N m/rad.
    """
    return arb_wheel_rate_roll * track**2 / 2.0


def roll_stiffness_tyres(k_t: float, track: float) -> float:
    """Axle roll stiffness of the tyres [N*m/rad] = k_t * t^2 / 2 (k_t per tyre).

    Args:
        k_t: Tyre vertical stiffness per tyre in N/m.
        track: Axle track in m.

    Returns:
        Roll stiffness in N m/rad.
    """
    return k_t * track**2 / 2.0


def axle_roll_stiffness(k_spring_roll: float, k_arb_roll: float, k_tyre_roll: float) -> float:
    """Axle roll stiffness [N*m/rad]: (springs + ARB) in series with the tyres.

    Args:
        k_spring_roll: Spring roll stiffness in N m/rad.
        k_arb_roll: ARB roll stiffness in N m/rad.
        k_tyre_roll: Tyre roll stiffness in N m/rad.

    Returns:
        Combined axle roll stiffness in N m/rad.
    """
    k_susp = k_spring_roll + k_arb_roll
    return k_susp * k_tyre_roll / (k_susp + k_tyre_roll)


def roll_gradient(
    m_sprung: float, h_cg: float, h_roll_axis_at_cg: float, k_front: float, k_rear: float
) -> float:
    """Roll gradient [rad per g of lateral acceleration].

    phi/(a_y/g) = m_s g d / (K_f + K_r - m_s g d), d = h_cg - h_ra (RCVD). The
    ``- m_s g d`` term is the destabilising weight moment of the sprung mass. Multiply
    by 180/pi for deg/g. Positive = the body rolls outward in a turn.

    Args:
        m_sprung: Sprung mass in kg.
        h_cg: Sprung-mass CG height in m.
        h_roll_axis_at_cg: Roll-axis height at the CG position in m.
        k_front: Front axle roll stiffness in N m/rad.
        k_rear: Rear axle roll stiffness in N m/rad.

    Returns:
        Roll gradient in rad per g of lateral acceleration.

    Raises:
        ValueError: If ``k_front + k_rear`` does not exceed ``m_s g d``.
    """
    moment = m_sprung * G * (h_cg - h_roll_axis_at_cg)
    denom = k_front + k_rear - moment
    if denom <= 0.0:
        raise ValueError("roll stiffness does not exceed the destabilising moment m_s*g*d")
    return moment / denom


def roll_axis_height_at(
    x: float, x_front: float, rc_front: float, x_rear: float, rc_rear: float
) -> float:
    """Roll-axis height at longitudinal position x: linear between the two roll centres.

    Args:
        x: Longitudinal position in m (vehicle frame).
        x_front: Front roll-centre x in m.
        rc_front: Front roll-centre height in m.
        x_rear: Rear roll-centre x in m.
        rc_rear: Rear roll-centre height in m.

    Returns:
        Roll-axis height at ``x`` in m (linear extrapolation outside the axles).
    """
    t = (x - x_front) / (x_rear - x_front)
    return rc_front + t * (rc_rear - rc_front)


@dataclass(frozen=True)
class LoadTransfer:
    """Lateral load transfer per axle [N].

    Each value is the load shift ``dW`` per wheel: the outside wheel gains ``dW`` and the
    inside wheel loses ``dW``, so the outside-minus-inside load difference of the axle
    is ``2 dW``.

    Attributes:
        front_elastic: Elastic (roll-stiffness) transfer of the front axle in N.
        front_geometric: Geometric (roll-centre) transfer of the front axle in N.
        front_unsprung: Unsprung-mass transfer of the front axle in N.
        rear_elastic: Elastic transfer of the rear axle in N.
        rear_geometric: Geometric transfer of the rear axle in N.
        rear_unsprung: Unsprung-mass transfer of the rear axle in N.
    """

    front_elastic: float
    front_geometric: float
    front_unsprung: float
    rear_elastic: float
    rear_geometric: float
    rear_unsprung: float

    @property
    def front(self) -> float:
        """Total front-axle transfer ``dW`` per wheel in N."""
        return self.front_elastic + self.front_geometric + self.front_unsprung

    @property
    def rear(self) -> float:
        """Total rear-axle transfer ``dW`` per wheel in N."""
        return self.rear_elastic + self.rear_geometric + self.rear_unsprung

    @property
    def tlltd_front(self) -> float:
        """Total lateral load transfer distribution = dW_f / (dW_f + dW_r), a fraction in [0, 1]."""
        return tlltd_front(self.front, self.rear)


def tlltd_front(dw_front: float, dw_rear: float) -> float:
    """Front TLLTD = dW_f / (dW_f + dW_r); nan if the total is zero.

    Args:
        dw_front: Front axle load transfer per wheel in N.
        dw_rear: Rear axle load transfer per wheel in N.

    Returns:
        Fraction of the total transfer carried by the front axle (not a percent).
    """
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

    Args:
        ay: Lateral acceleration in m/s^2.
        m_sprung: Sprung mass in kg.
        m_unsprung_front: Front unsprung mass (both corners) in kg.
        m_unsprung_rear: Rear unsprung mass (both corners) in kg.
        h_cg: Sprung-mass CG height in m.
        rc_front: Front roll-centre height in m.
        rc_rear: Rear roll-centre height in m.
        hu_front: Front unsprung CG height in m.
        hu_rear: Rear unsprung CG height in m.
        k_front: Front axle roll stiffness in N m/rad.
        k_rear: Rear axle roll stiffness in N m/rad.
        track_front: Front track in m.
        track_rear: Rear track in m.
        wheelbase: Wheelbase in m.
        a: Sprung CG distance behind the front axle in m.

    Returns:
        :class:`LoadTransfer` with per-wheel shifts in N (outside wheel gains).

    Raises:
        ValueError: If the roll stiffness does not exceed ``m_s g d``.
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
    """Sprung mass [kg] = total mass - 2 * (front + rear unsprung mass per corner).

    Args:
        vehicle: Vehicle description.

    Returns:
        Sprung mass in kg.
    """
    return vehicle.vehicle.mass - 2.0 * (vehicle.front.unsprung_mass + vehicle.rear.unsprung_mass)


# --------------------------------------------------------------------------- steering
def inner_outer(delta_left: float, delta_right: float) -> tuple[float, float]:
    """Return (delta_inner, delta_outer) with the turn direction from the mean angle.

    Left turn (mean > 0, +delta = left): left wheel is inner. Right turn: right wheel.
    Signs of the inputs are kept.

    Args:
        delta_left: Left road-wheel steer angle in rad (positive = wheel points left).
        delta_right: Right road-wheel steer angle in rad.

    Returns:
        ``(delta_inner, delta_outer)`` in rad. A zero mean steer counts as a left turn.
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

    Args:
        delta_inner: Inner road-wheel steer angle in rad.
        delta_outer: Outer road-wheel steer angle in rad.
        wheelbase: Wheelbase ``L`` in m.
        track: Front track ``t`` in m.

    Returns:
        Ackermann in percent (the percent number itself).
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

    Args:
        rack: Rack displacement in m.
        c_factor: Rack travel per pinion revolution in m/rev.
        mean_road_wheel: Mean road-wheel steer angle in rad.

    Returns:
        Ratio handwheel angle / road-wheel angle (dimensionless, deg/deg).
    """
    if abs(mean_road_wheel) < 1e-9:
        return _NAN
    return (rack / c_factor * 2.0 * math.pi) / mean_road_wheel


def turning_radius(wheelbase: float, delta_inner: float, delta_outer: float) -> float:
    """Low-speed turning radius [m] = L / tan(mean steer angle) (bicycle model).

    Measured to the rear-axle centre; inf for zero steer.

    Args:
        wheelbase: Wheelbase in m.
        delta_inner: Inner road-wheel steer angle in rad.
        delta_outer: Outer road-wheel steer angle in rad.

    Returns:
        Turning radius in m (``inf`` for zero mean steer).
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

    Args:
        front_line: ``(point, direction)`` in the ``(x, z)`` plane, m, each shape ``(2,)``.
        rear_line: Same for the rear axle.

    Returns:
        ``(x, z)`` of the pitch centre in m, or ``(nan, nan)`` for parallel lines.
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
#   wheel (FR / RR); ``roll_camber`` is d(camber_ground of that wheel)/d(roll) [rad/rad].
#   It includes the body roll itself, so it is positive on a normal car (reference car:
#   +0.81 front, +0.76 rear): the outer wheel loses negative camber relative to the ground
#   as the body rolls, because the suspension camber gain only partly cancels the roll.
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
    """Static per-axle values (SI).

    Attributes:
        track: Axle track in m.
        wheel_rate: Spring wheel rate at static in N/m (per wheel).
        arb_wheel_rate_roll: Roll-mode ARB wheel rate in N/m (0 without ARB).
        roll_stiffness_springs: Roll stiffness from springs in N m/rad.
        roll_stiffness_arb: Roll stiffness from the ARB in N m/rad.
        roll_stiffness_tyres: Roll stiffness of the tyres in N m/rad.
        roll_stiffness: Axle roll stiffness, (springs + ARB) in series with tyres, N m/rad.
        rc_height: Static roll-centre height above the ground in m.
        roll_camber: ``d(camber_ground)/d(roll)`` of the outer (right) wheel in rad/rad.
        roll_steer: ``d(mean ground steer)/d(roll)`` in rad/rad.
        rc_migration: ``d(rc_y)/d(roll)`` in m/rad (positive = toward +y, left).
        rc_height_curvature: ``d2(rc_z)/d(roll)2`` in m/rad^2.
    """

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
    """Vehicle summary (SI). See the module docstring for conventions.

    Attributes:
        front: Front :class:`AxleSummary`.
        rear: Rear :class:`AxleSummary`.
        wheelbase: Wheelbase in m.
        sprung_mass: Sprung mass in kg.
        sprung_cg_height: Sprung-mass CG height in m.
        roll_axis_height_at_cg: Roll-axis height at the sprung CG x in m.
        roll_stiffness_total: Sum of the axle roll stiffnesses in N m/rad.
        roll_gradient: Roll gradient in rad per g of lateral acceleration.
        ay_ref: Reference lateral acceleration used for the load transfer, m/s^2.
        load_transfer_front: Front-axle lateral load transfer per wheel at ``ay_ref``, N.
        load_transfer_rear: Rear-axle lateral load transfer per wheel at ``ay_ref``, N.
        tlltd_front: Front total lateral load transfer distribution, a fraction in [0, 1].
        anti_dive_front: Front anti-dive in percent.
        anti_lift_rear: Rear anti-lift in percent.
        anti_squat_rear: Rear anti-squat in percent.
        anti_lift_front_accel: Front anti-lift under drive in percent; ``None`` when no
            drive torque goes to the front.
        pitch_centre_x: Side-view pitch-centre x in m (NaN if the lines are parallel).
        pitch_centre_z: Side-view pitch-centre height in m.
        max_steer_inner: Inner-wheel steer at full lock in rad, relative to straight ahead.
        max_steer_outer: Outer-wheel steer at full lock in rad.
        ackermann_5deg: Ackermann in percent at 5 deg outer-wheel steer (NaN if not reached).
        ackermann_10deg: Ackermann in percent at 10 deg outer-wheel steer.
        ackermann_15deg: Ackermann in percent at 15 deg outer-wheel steer.
        steering_ratio: Handwheel angle / mean road-wheel angle (dimensionless).
        turning_radius: Low-speed turning radius at full lock in m.
    """

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
        """Nested dict of SI floats (JSON-friendly; nan stays nan).

        Returns:
            Field name -> value, with the two axle summaries as nested dicts.
        """
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
    """Side-view line (point, direction) from the contact patch through the SVIC, in ``(x, z)``."""
    cp = st.points["contact_patch"]
    ic = side_view_ic(st, corner)  # type: ignore[arg-type]
    p = np.array([cp[0], cp[2]], dtype=np.float64)
    if ic.is_finite:
        return p, np.array([ic.h - p[0], ic.z - p[1]])
    return p, np.asarray(ic.direction, dtype=np.float64)


def _axle_static(vehicle: Vehicle, cid_l: str, cid_r: str, axle_name: str) -> dict[str, float]:
    """Static wheel rate and roll stiffnesses of one axle (travel sweep of +-10 mm)."""
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
    """Steering metrics from a full-rack sweep: lock angles, Ackermann, ratio, turning radius."""
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
    """Compute the vehicle summary (see the module docstring).

    Runs a roll sweep for the roll-centre, roll-camber and roll-steer coefficients, a +-10 mm
    travel sweep per axle for the wheel rates, a full rack sweep for the steering metrics and
    the static pose for the anti-features and pitch centre.

    Args:
        vehicle: Vehicle description.
        roll_range: Half range of the roll sweep in rad (default 0.035 = 2 deg).
        n: Number of roll samples (forced odd so that roll 0 is included).
        ay_ref: Reference lateral acceleration for the load transfer in m/s^2.
        pivot: Centre of the body rotations, shape ``(3,)``, m (default: the vehicle CG).

    Returns:
        :class:`VehicleSummary` in SI units (see its attributes).

    Raises:
        KinematicsError: If a sweep pose cannot be solved.
        ValueError: If the roll stiffness does not exceed the destabilising moment.
    """
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
