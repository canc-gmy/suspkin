"""Whole-car roll sweep: roll gradient, roll centres, TLLTD and camber vs roll plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

import suspkin

REPO = Path(__file__).resolve().parents[1]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    vehicle = suspkin.load_vehicle(REPO / "configs" / "vehicles" / "fs_reference.yaml")

    # Roll sweep +-2 deg about the CG (roll positive = left side up), one state per value.
    rolls = np.radians(np.linspace(-2.0, 2.0, 9))
    states = suspkin.sweep_vehicle(vehicle, roll=rolls)
    df = suspkin.vehicle_frame(states, vehicle)  # SI columns, per-corner prefix FL_/FR_/...

    # Static vehicle summary (runs its own roll sweep, SI units).
    s = suspkin.vehicle_summary(vehicle)
    print(f"roll gradient    : {np.degrees(s.roll_gradient):.3f} deg/g")
    print(f"RC height front  : {s.front.rc_height * 1e3:.1f} mm")
    print(f"RC height rear   : {s.rear.rc_height * 1e3:.1f} mm")
    print(f"roll axis at CG  : {s.roll_axis_height_at_cg * 1e3:.1f} mm")
    print(f"TLLTD (front)    : {s.tlltd_front * 100:.1f} %")

    # Plot: camber relative to the ground (not the body) of all four wheels vs body roll.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for cid in ("FL", "FR", "RL", "RR"):
        ax.plot(np.degrees(df["roll"]), np.degrees(df[f"{cid}_camber_ground"]), label=cid)
    ax.set_xlabel("body roll [deg] (positive: left side up)")
    ax.set_ylabel("camber to ground [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out / "camber_vs_roll.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "examples" / "02_vehicle_roll")
    main(ap.parse_args().out)
