"""Load the reference car, sweep the front-left corner in bump/droop and print key metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import suspkin

REPO = Path(__file__).resolve().parents[1]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load the vehicle (SI units) and build the four side-resolved corners.
    vehicle = suspkin.load_vehicle(REPO / "configs" / "vehicles" / "fs_reference.yaml")
    corners = suspkin.build_corners(vehicle)
    fl = corners["FL"]

    # 2. Bump sweep +-30 mm (bump positive), 13 points; the table is in SI (m, rad).
    travels = np.linspace(-0.030, 0.030, 13)
    states = suspkin.sweep_corner(fl, travels)
    table = suspkin.add_gradients(suspkin.metrics_frame(states, fl))

    # 3. Read the static values (row at travel == 0) and convert to display units.
    row = table.iloc[int(np.argmin(np.abs(table["travel"])))]
    print(f"Vehicle: {vehicle.vehicle.name}")
    print(f"static camber : {np.degrees(row['camber']):8.3f} deg")
    print(f"static toe    : {np.degrees(row['toe']):8.3f} deg")
    print(f"RC height     : {row['rc_height'] * 1e3:8.1f} mm")
    # camber_gain is rad/m -> deg/mm is x (180/pi)/1000
    print(f"camber gain   : {np.degrees(row['camber_gain']) / 1e3:8.4f} deg/mm")

    # 4. Save the full table as CSV (SI units).
    path = out / "fl_bump_sweep.csv"
    table.to_csv(path, index=False)
    print(f"saved {path} ({len(table)} rows, {table.shape[1]} columns)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "examples" / "01_load_and_sweep")
    main(ap.parse_args().out)
