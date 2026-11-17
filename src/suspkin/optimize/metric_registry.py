"""Metric name -> dimension registry for study constraints (validation and display units).

Three namespaces:

* corner metrics: columns of ``states_to_frame(metrics=True)`` + ``add_gradients``;
* vehicle metrics: columns of ``vehicle_frame`` (``FL_``... prefixed corner metrics, ground
  frame columns, roll centres, ARB);
* summary metrics: dotted ``VehicleSummary`` fields (``front.rc_height``, ``roll_gradient``).

Dimensions are canonical names from :mod:`suspkin.units`. Where a column mixes units
(``arb_*_elastic`` is a twist [rad] or a deflection [m]) it is ``"ratio"`` (plain SI).
"""

from __future__ import annotations

import difflib
import re

from suspkin.geometry.models import HARDPOINT_NAMES
from suspkin.units import Dim, slope_dim

__all__ = [
    "MetricNameError",
    "corner_metric_dim",
    "corner_metric_names",
    "slope_dim",
    "summary_metric_dim",
    "summary_metric_names",
    "vehicle_metric_dim",
    "vehicle_metric_names",
]

CORNER_IDS: tuple[str, ...] = ("FL", "FR", "RL", "RR")


class MetricNameError(KeyError):
    """Unknown metric name; the message lists close matches."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return str(self.args[0])


_CORNER: dict[str, Dim] = {
    # state columns (states_to_frame)
    "travel": "length",
    "rack": "length",
    "theta_uca": "angle",
    "theta_lca": "angle",
    "psi": "angle",
    "rocker_angle": "angle",
    "coilover_length": "length",
    "toggle_margin": "angle",
    "max_residual": "length",
    "spin_x": "ratio",
    "spin_y": "ratio",
    "spin_z": "ratio",
    # alignment
    "camber": "angle",
    "toe": "angle",
    "steer_angle": "angle",
    "caster": "angle",
    "kpi": "angle",
    "scrub_radius": "length",
    "mechanical_trail": "length",
    "kingpin_offset_wc": "length",
    "spindle_length": "length",
    "half_track": "length",
    "half_track_change": "length",
    "contact_patch_dx": "length",
    "contact_patch_dy": "length",
    "wheel_center_recession": "length",
    "wheelbase_change_contribution": "length",
    # centres
    "fvic_y": "length",
    "fvic_z": "length",
    "fvsa": "length",
    "fv_swing_angle": "angle",
    "svic_x": "length",
    "svic_z": "length",
    "svsa": "length",
    "sv_swing_angle": "angle",
    "sv_wc_angle": "angle",
    "rc_height": "length",
    # actuation
    "pushrod_angle": "angle",
    "pushrod_angle_fv": "angle",
    "rocker_rotation": "angle",
    "coilover_compression": "length",
    "bump_stroke_remaining": "length",
    "droop_stroke_remaining": "length",
    "spring_force": "force",
    "spring_rate": "stiffness",
    "tyre_stiffness": "stiffness",
    # gradients vs travel / rack
    "camber_gain": "angle/length",
    "bump_steer": "angle/length",
    "caster_change": "angle/length",
    "track_change_rate": "ratio",
    "rc_migration_rate": "ratio",
    "steer_gain": "angle/length",
    "mr_spring": "ratio",
    "mr_progression": "1/length",
    "wheel_load_spring": "force",
    "wheel_rate_linear": "stiffness",
    "wheel_rate": "stiffness",
    "ride_rate": "stiffness",
}

_POINT_RE = re.compile(
    r"^(" + "|".join((*HARDPOINT_NAMES, "rocker_arb", "contact_patch")) + r")_([xyz])$"
)

_VEHICLE_OWN: dict[str, Dim] = {
    "heave": "length",
    "roll": "angle",
    "pitch": "angle",
    "rack": "length",
    "rc_front_y": "length",
    "rc_front_z": "length",
    "rc_rear_y": "length",
    "rc_rear_z": "length",
    "arb_front_elastic": "ratio",
    "arb_rear_elastic": "ratio",
    "wheelbase": "length",
    "steer_c_factor": "length",
}
_CORNER_SUFFIX_OWN: dict[str, Dim] = {
    "cp_x": "length",
    "cp_y": "length",
    "cp_z": "length",
    "camber_ground": "angle",
    "steer_ground": "angle",
}

_AXLE: dict[str, Dim] = {
    "track": "length",
    "wheel_rate": "stiffness",
    "arb_wheel_rate_roll": "stiffness",
    "roll_stiffness_springs": "force*length/angle",
    "roll_stiffness_arb": "force*length/angle",
    "roll_stiffness_tyres": "force*length/angle",
    "roll_stiffness": "force*length/angle",
    "rc_height": "length",
    "roll_camber": "ratio",
    "roll_steer": "ratio",
    "rc_migration": "length/angle",
    "rc_height_curvature": "length/angle^2",
}
_SUMMARY_OWN: dict[str, Dim] = {
    "wheelbase": "length",
    "sprung_mass": "mass",
    "sprung_cg_height": "length",
    "roll_axis_height_at_cg": "length",
    "roll_stiffness_total": "force*length/angle",
    "roll_gradient": "angle/gload",
    "ay_ref": "accel",
    "load_transfer_front": "force",
    "load_transfer_rear": "force",
    "tlltd_front": "ratio",
    "anti_dive_front": "percent",
    "anti_lift_rear": "percent",
    "anti_squat_rear": "percent",
    "anti_lift_front_accel": "percent",
    "pitch_centre_x": "length",
    "pitch_centre_z": "length",
    "max_steer_inner": "angle",
    "max_steer_outer": "angle",
    "ackermann_5deg": "percent",
    "ackermann_10deg": "percent",
    "ackermann_15deg": "percent",
    "steering_ratio": "ratio",
    "turning_radius": "length",
}
_SUMMARY: dict[str, Dim] = {
    **_SUMMARY_OWN,
    **{f"{a}.{k}": d for a in ("front", "rear") for k, d in _AXLE.items()},
}


def _lookup(name: str, table: dict[str, Dim], extra: list[str], what: str) -> Dim:
    if name in table:
        return table[name]
    cands = difflib.get_close_matches(name, [*table, *extra], n=3, cutoff=0.6)
    hint = f"; did you mean {', '.join(repr(c) for c in cands)}?" if cands else ""
    raise MetricNameError(f"unknown {what} metric {name!r}{hint}")


def corner_metric_dim(name: str) -> Dim:
    """Dimension of a corner-sweep column (metrics, gradients, state and point columns)."""
    if name in _CORNER:
        return _CORNER[name]
    if _POINT_RE.match(name):
        return "length"
    return _lookup(name, _CORNER, [], "corner")


def corner_metric_names() -> list[str]:
    """Sorted corner metric names (point coordinate columns excluded)."""
    return sorted(_CORNER)


def vehicle_metric_dim(name: str) -> Dim:
    """Dimension of a vehicle-sweep column (``FL_camber_ground``, ``rc_front_z``, ...)."""
    if name in _VEHICLE_OWN:
        return _VEHICLE_OWN[name]
    prefix, _, rest = name.partition("_")
    if prefix in CORNER_IDS and rest:
        if rest in _CORNER_SUFFIX_OWN:
            return _CORNER_SUFFIX_OWN[rest]
        if rest in _CORNER or _POINT_RE.match(rest):
            return corner_metric_dim(rest)
    return _lookup(name, {**_VEHICLE_OWN, **_expanded()}, [], "vehicle")


def _expanded() -> dict[str, Dim]:
    out: dict[str, Dim] = {}
    for cid in CORNER_IDS:
        for k, d in {**_CORNER, **_CORNER_SUFFIX_OWN}.items():
            out[f"{cid}_{k}"] = d
    return out


def vehicle_metric_names() -> list[str]:
    """Sorted vehicle metric names."""
    return sorted({**_VEHICLE_OWN, **_expanded()})


def summary_metric_dim(name: str) -> Dim:
    """Dimension of a ``VehicleSummary`` field (dotted for axle fields)."""
    return _lookup(name, _SUMMARY, [], "summary")


def summary_metric_names() -> list[str]:
    """Sorted summary metric names."""
    return sorted(_SUMMARY)
