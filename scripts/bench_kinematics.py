"""Benchmark the pure kinematics evaluation of suspkin.

Times, on ``configs/vehicles/fs_reference.yaml`` (median of repeats, single process):

1. one cold corner solve at one pose (FL, 10 mm bump, 5 mm rack);
2. corner sweeps (FL travel) and vehicle sweeps (heave, roll, rack) at 31/61/121 points,
   solver only;
3. ``run_standard_studies`` end to end at 31/61/121 points (solve + metric tables).

Run: ``python -m uv run python scripts/bench_kinematics.py [--repeats N] [--profile]
[--no-numba]``. ``--no-numba`` forces the pure-Python path (sets ``SUSPKIN_NO_NUMBA``).
``--profile`` prints a cProfile top list of ``run_standard_studies(steps=61)``.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

if "--no-numba" in sys.argv:  # must be set before suspkin is used
    os.environ["SUSPKIN_NO_NUMBA"] = "1"

from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics import _fast
from suspkin.kinematics.corner import solve_corner, sweep_corner
from suspkin.report.studies import run_standard_studies
from suspkin.sweeps import sweep_vehicle

VEHICLE = Path(__file__).resolve().parents[1] / "configs" / "vehicles" / "fs_reference.yaml"
STEPS = (31, 61, 121)


def _median(fn: Callable[[], object], repeats: int) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-numba", action="store_true", help="force the pure-Python path")
    args = ap.parse_args()

    vehicle = load_vehicle(VEHICLE)
    corners = build_corners(vehicle)
    fl = corners["FL"]
    rack_lo, rack_hi = vehicle.front.steering.rack_travel  # type: ignore[union-attr]

    t0 = time.perf_counter()
    path = "numba" if _fast.available() else "pure Python"  # loads/compiles the kernels
    sweep_vehicle(vehicle, heave=np.linspace(-0.005, 0.005, 3))
    print(f"path: {path}")
    print(f"one-off kernel load + first vehicle sweep: {time.perf_counter() - t0:.3f} s")

    reps = args.repeats
    rows: list[tuple[str, float]] = [
        ("corner solve, 1 pose", _median(partial(solve_corner, fl, 0.010, 0.005), 200))
    ]
    for n in STEPS:
        tr = np.linspace(-0.030, 0.030, n)
        hv = np.linspace(-0.025, 0.025, n)
        rl = np.linspace(-np.radians(2.0), np.radians(2.0), n)
        rk = np.linspace(rack_lo, rack_hi, n)
        rows += [
            (f"corner sweep FL travel, {n}", _median(partial(sweep_corner, fl, tr), reps)),
            (f"vehicle sweep heave, {n}", _median(partial(sweep_vehicle, vehicle, heave=hv), reps)),
            (f"vehicle sweep roll, {n}", _median(partial(sweep_vehicle, vehicle, roll=rl), reps)),
            (f"vehicle sweep rack, {n}", _median(partial(sweep_vehicle, vehicle, rack=rk), reps)),
        ]
    for n in STEPS:
        fn = partial(run_standard_studies, vehicle, steps=n)
        rows.append((f"run_standard_studies, {n}", _median(fn, max(3, reps // 2))))

    width = max(len(k) for k, _ in rows)
    print(f"{'case':<{width}}  time [ms]")
    for k, t in rows:
        print(f"{k:<{width}}  {t * 1e3:9.3f}")

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        run_standard_studies(vehicle, steps=61)
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(30)


if __name__ == "__main__":
    main()
