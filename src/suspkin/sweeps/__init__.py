"""Sweep definitions producing tidy result tables."""

from suspkin.sweeps.corner_sweeps import states_to_frame
from suspkin.sweeps.vehicle_sweeps import sweep_vehicle, vehicle_frame

__all__ = ["states_to_frame", "sweep_vehicle", "vehicle_frame"]
