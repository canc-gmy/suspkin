"""Draw the car in 3D at a solved pose and add a custom curve from a result table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

import suspkin

REPO = Path(__file__).resolve().parents[1]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vehicle = suspkin.load_vehicle(REPO / "configs" / "vehicles" / "fs_reference.yaml")

    # 1. 3D view at a pose: 1.5 deg roll and 10 mm heave.
    state = suspkin.solve_vehicle(vehicle, roll=np.radians(1.5), heave=0.010)
    fig3d = suspkin.plot_vehicle_3d(vehicle, state=state)
    p3d = out / "vehicle_3d.png"
    fig3d.savefig(p3d, dpi=120)
    plt.close(fig3d)
    print(f"saved {p3d}")

    # 2. Custom curve: plotting only reads result tables, it never calls the solver.
    corner = suspkin.build_corners(vehicle)["FL"]
    table = suspkin.add_gradients(
        suspkin.metrics_frame(suspkin.sweep_corner(corner, np.linspace(-0.03, 0.03, 13)), corner)
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(table["travel"] * 1e3, np.degrees(table["camber"]), marker="o")
    ax.set_xlabel("wheel travel [mm] (bump positive)")
    ax.set_ylabel("camber [deg]")
    ax.set_title("FL camber curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pcurve = out / "camber_curve.png"
    fig.savefig(pcurve, dpi=120)
    plt.close(fig)
    print(f"saved {pcurve}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "examples" / "04_custom_plot")
    main(ap.parse_args().out)
