"""Standard study set of one design: corner, steering, heave/roll/pitch sweeps, summary.

Pure: it runs the solver and returns tidy tables (SI units), nothing is written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from suspkin.geometry import Vehicle, build_corners
from suspkin.kinematics.corner import sweep_corner
from suspkin.metrics import add_gradients, vehicle_summary
from suspkin.metrics.vehicle_summary import VehicleSummary
from suspkin.sweeps import states_to_frame, sweep_vehicle, vehicle_frame


@dataclass(frozen=True)
class StudyResults:
    """Result tables of the standard studies of one vehicle (SI: m, rad).

    ``rack`` is empty for a vehicle without steering. Corner frames are
    ``states_to_frame(metrics=True)`` plus gradients, so they carry the state columns
    (``coilover_length``, ``toggle_margin``) next to the metrics.
    """

    vehicle: Vehicle
    front_corner: pd.DataFrame
    rear_corner: pd.DataFrame
    rack: pd.DataFrame
    heave: pd.DataFrame
    roll: pd.DataFrame
    pitch: pd.DataFrame
    summary: VehicleSummary

    def tables(self) -> dict[str, pd.DataFrame]:
        """Non-empty tables by study name (file-name stems of the CSV export)."""
        named = {
            "front_corner": self.front_corner,
            "rear_corner": self.rear_corner,
            "rack": self.rack,
            "heave": self.heave,
            "roll": self.roll,
            "pitch": self.pitch,
        }
        return {k: v for k, v in named.items() if len(v) > 0}


def run_standard_studies(
    vehicle: Vehicle,
    *,
    travel: float = 0.030,
    rack: float | None = None,
    roll_deg: float = 2.0,
    heave: float = 0.025,
    pitch_deg: float = 1.5,
    steps: int = 31,
) -> StudyResults:
    """Run the standard studies of ``vehicle``.

    Parameters
    ----------
    travel : float
        Corner bump/droop range +-``travel`` [m] for the FL and RL corners.
    rack : float, optional
        Rack sweep +-``rack`` [m]; default the full rack travel of the front axle.
    roll_deg, pitch_deg : float
        Body roll / pitch range +-value [deg].
    heave : float
        Body heave range +-``heave`` [m].
    steps : int
        Points per sweep (>= 3; odd values include the static pose).
    """
    if steps < 3:
        raise ValueError("steps must be >= 3")
    corners = build_corners(vehicle)
    tr = np.linspace(-travel, travel, steps)

    def corner_frame(cid: str) -> pd.DataFrame:
        c = corners[cid]  # type: ignore[index]
        return add_gradients(states_to_frame(sweep_corner(c, tr), c, metrics=True))

    steer = vehicle.front.steering
    if steer is None:
        rack_df = pd.DataFrame()
    else:
        lo, hi = steer.rack_travel if rack is None else (-rack, rack)
        rack_df = vehicle_frame(sweep_vehicle(vehicle, rack=np.linspace(lo, hi, steps)), vehicle)
        rack_df["wheelbase"] = vehicle.wheelbase
        rack_df["steer_c_factor"] = steer.c_factor

    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    return StudyResults(
        vehicle=vehicle,
        front_corner=corner_frame("FL"),
        rear_corner=corner_frame("RL"),
        rack=rack_df,
        heave=vehicle_frame(
            sweep_vehicle(vehicle, heave=np.linspace(-heave, heave, steps)), vehicle
        ),
        roll=vehicle_frame(sweep_vehicle(vehicle, roll=np.linspace(-r, r, steps)), vehicle),
        pitch=vehicle_frame(sweep_vehicle(vehicle, pitch=np.linspace(-p, p, steps)), vehicle),
        summary=vehicle_summary(vehicle),
    )
