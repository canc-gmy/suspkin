"""Kinematic metrics: corner metrics, metric tables, vehicle formulas and the vehicle summary.

Only the names used outside this package are re-exported here; everything else lives in
:mod:`suspkin.metrics.corner`, :mod:`suspkin.metrics.arb`, :mod:`suspkin.metrics.table`,
:mod:`suspkin.metrics.arrays` and :mod:`suspkin.metrics.vehicle`.

All values are SI (m, rad, N, N/m); signs are defined in ``docs/concepts/conventions.md`` and
every column is explained in ``docs/concepts/metrics.md``.
"""

from suspkin.metrics.corner import joint_angles_frame
from suspkin.metrics.table import add_gradients, metrics_frame, static_summary
from suspkin.metrics.vehicle import pitch_centre, vehicle_summary

__all__ = [
    "add_gradients",
    "joint_angles_frame",
    "metrics_frame",
    "pitch_centre",
    "static_summary",
    "vehicle_summary",
]
