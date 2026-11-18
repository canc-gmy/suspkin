"""Compare the reference car with a variant (front LCA inner points raised 10 mm)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import suspkin

REPO = Path(__file__).resolve().parents[1]


def raise_front_lca(vehicle: suspkin.Vehicle, dz: float) -> suspkin.Vehicle:
    """Return a copy with both front LCA chassis pivots moved up by ``dz`` metres."""
    data = vehicle.model_dump(by_alias=True)  # plain dict, safe to edit
    hp = data["front"]["hardpoints"]
    for name in ("lca_front", "lca_rear"):
        x, y, z = hp[name]
        hp[name] = (x, y, z + dz)
    return suspkin.Vehicle.model_validate(data)  # re-validates the new geometry


def main(out: Path) -> None:
    base = suspkin.load_vehicle(REPO / "configs" / "vehicles" / "fs_reference.yaml")
    variant = raise_front_lca(base, 0.010)

    # Standard studies (corner, heave, roll, pitch, rack sweeps + summary) for both cars.
    res_base = suspkin.run_standard_studies(base, steps=13)
    res_var = suspkin.run_standard_studies(variant, steps=13)

    print(f"{'design':<12}{'camber gain [deg/mm]':>22}{'RC height [mm]':>16}")
    for label, res in (("base", res_base), ("lca +10 mm", res_var)):
        row = res.front_corner.iloc[int(np.argmin(np.abs(res.front_corner["travel"])))]
        print(
            f"{label:<12}{np.degrees(row['camber_gain']) / 1e3:>22.4f}"
            f"{row['rc_height'] * 1e3:>16.1f}"
        )

    # HTML/Markdown report with overlaid figures, CSV tables and summary.json.
    report = suspkin.build_report(
        [("base", res_base), ("lca_plus_10mm", res_var)], out, formats=("png",), include_3d=False
    )
    print(f"report: {report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=REPO / "results" / "examples" / "03_compare_designs"
    )
    main(ap.parse_args().out)
