"""Display units of the vehicle summary, shared by the figures, report, spec and CLI.

``SUMMARY_ROWS`` drives the summary-table figure, the report comparison table and the
web spec. ``VEHICLE_CLI_UNITS`` drives the ``suspkin vehicle`` table; it keeps its own
unit spellings (``N*m/deg``) and a few extra fields so the CLI output is unchanged.
"""

from __future__ import annotations

import math

# (label, path in the summary, factor from SI, unit, format)
SUMMARY_ROWS: tuple[tuple[str, tuple[str, ...], float, str, str], ...] = (
    ("Wheelbase", ("wheelbase",), 1e3, "mm", "{:.0f}"),
    ("Sprung mass", ("sprung_mass",), 1.0, "kg", "{:.1f}"),
    ("Sprung CG height", ("sprung_cg_height",), 1e3, "mm", "{:.0f}"),
    ("Roll-axis height at CG", ("roll_axis_height_at_cg",), 1e3, "mm", "{:.1f}"),
    ("Front track", ("front", "track"), 1e3, "mm", "{:.0f}"),
    ("Rear track", ("rear", "track"), 1e3, "mm", "{:.0f}"),
    ("Front wheel rate", ("front", "wheel_rate"), 1e-3, "N/mm", "{:.2f}"),
    ("Rear wheel rate", ("rear", "wheel_rate"), 1e-3, "N/mm", "{:.2f}"),
    ("Front RC height", ("front", "rc_height"), 1e3, "mm", "{:.1f}"),
    ("Rear RC height", ("rear", "rc_height"), 1e3, "mm", "{:.1f}"),
    ("Front roll stiffness", ("front", "roll_stiffness"), math.pi / 180, "N m/deg", "{:.0f}"),
    ("Rear roll stiffness", ("rear", "roll_stiffness"), math.pi / 180, "N m/deg", "{:.0f}"),
    ("Total roll stiffness", ("roll_stiffness_total",), math.pi / 180, "N m/deg", "{:.0f}"),
    ("Roll gradient", ("roll_gradient",), 180 / math.pi, "deg/g", "{:.2f}"),
    ("Front TLLTD", ("tlltd_front",), 100.0, "%", "{:.1f}"),
    ("Front roll camber", ("front", "roll_camber"), 1.0, "deg/deg", "{:.3f}"),
    ("Rear roll camber", ("rear", "roll_camber"), 1.0, "deg/deg", "{:.3f}"),
    ("Front roll steer", ("front", "roll_steer"), 1.0, "deg/deg", "{:.3f}"),
    ("Rear roll steer", ("rear", "roll_steer"), 1.0, "deg/deg", "{:.3f}"),
    ("Anti-dive (front)", ("anti_dive_front",), 1.0, "%", "{:.1f}"),
    ("Anti-lift (rear)", ("anti_lift_rear",), 1.0, "%", "{:.1f}"),
    ("Anti-squat (rear)", ("anti_squat_rear",), 1.0, "%", "{:.1f}"),
    ("Pitch centre x", ("pitch_centre_x",), 1e3, "mm", "{:.0f}"),
    ("Pitch centre z", ("pitch_centre_z",), 1e3, "mm", "{:.0f}"),
    ("Max steer (inner)", ("max_steer_inner",), 180 / math.pi, "deg", "{:.1f}"),
    ("Max steer (outer)", ("max_steer_outer",), 180 / math.pi, "deg", "{:.1f}"),
    ("Ackermann @ 5 deg", ("ackermann_5deg",), 1.0, "%", "{:.0f}"),
    ("Ackermann @ 10 deg", ("ackermann_10deg",), 1.0, "%", "{:.0f}"),
    ("Ackermann @ 15 deg", ("ackermann_15deg",), 1.0, "%", "{:.0f}"),
    ("Steering ratio", ("steering_ratio",), 1.0, "-", "{:.2f}"),
    ("Turning radius", ("turning_radius",), 1.0, "m", "{:.2f}"),
)


def format_value(fmt: str, value: float) -> str:
    """``fmt.format(value)`` without a negative zero ("-0.0" becomes "0.0").

    Args:
        fmt: ``str.format`` pattern such as ``"{:.2f}"``.
        value: Number to format.

    Returns:
        The formatted text.
    """
    text = fmt.format(value)
    if "-" in text and not any(c in "123456789" for c in text):
        return (
            text.replace("-", " ", 1).strip() if text == text.strip() else text.replace("-", " ", 1)
        )
    return text


# Leaf field name -> (scale from SI, unit) for the ``vehicle`` CLI table. The CLI spelling
# of units and some factors differ from SUMMARY_ROWS on purpose (output is locked).
_DEG = 180.0 / 3.141592653589793
VEHICLE_CLI_UNITS: dict[str, tuple[float, str]] = {
    "track": (1e3, "mm"),
    "wheel_rate": (1e-3, "N/mm"),
    "arb_wheel_rate_roll": (1e-3, "N/mm"),
    "roll_stiffness_springs": (1 / _DEG, "N*m/deg"),
    "roll_stiffness_arb": (1 / _DEG, "N*m/deg"),
    "roll_stiffness_tyres": (1 / _DEG, "N*m/deg"),
    "roll_stiffness": (1 / _DEG, "N*m/deg"),
    "roll_stiffness_total": (1 / _DEG, "N*m/deg"),
    "rc_height": (1e3, "mm"),
    "roll_camber": (1.0, "deg/deg"),
    "roll_steer": (1.0, "deg/deg"),
    "rc_migration": (
        1e3 / _DEG,
        "mm/deg (+ = toward left of the car centreline, for left-up roll)",
    ),
    "rc_height_curvature": (1e3 / _DEG**2, "mm/deg^2"),
    "wheelbase": (1e3, "mm"),
    "sprung_mass": (1.0, "kg"),
    "sprung_cg_height": (1e3, "mm"),
    "roll_axis_height_at_cg": (1e3, "mm"),
    "roll_gradient": (_DEG, "deg/g"),
    "ay_ref": (1.0, "m/s^2"),
    "load_transfer_front": (1.0, "N"),
    "load_transfer_rear": (1.0, "N"),
    "tlltd_front": (100.0, "%"),
    "anti_dive_front": (1.0, "%"),
    "anti_lift_rear": (1.0, "%"),
    "anti_squat_rear": (1.0, "%"),
    "anti_lift_front_accel": (1.0, "%"),
    "pitch_centre_x": (1e3, "mm"),
    "pitch_centre_z": (1e3, "mm"),
    "max_steer_inner": (_DEG, "deg"),
    "max_steer_outer": (_DEG, "deg"),
    "ackermann_5deg": (1.0, "%"),
    "ackermann_10deg": (1.0, "%"),
    "ackermann_15deg": (1.0, "%"),
    "steering_ratio": (1.0, "deg/deg (handwheel/road wheel)"),
    "turning_radius": (1.0, "m"),
}
