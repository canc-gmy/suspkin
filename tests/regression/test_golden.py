"""Golden snapshots of the reference car (SI). Regenerate with ``--update-golden``.

See ``golden/README.md``.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _golden import check_frames, check_json, flatten

from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics.corner import sweep_corner, sweep_rack
from suspkin.metrics import add_gradients, vehicle_summary
from suspkin.optimize import DesignSpace, evaluate
from suspkin.optimize.spec import load_study
from suspkin.sweeps import states_to_frame, sweep_vehicle, vehicle_frame

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "vehicles" / "fs_reference.yaml"
STUDIES = ROOT / "configs" / "studies"


@pytest.fixture(scope="module")
def vehicle():  # type: ignore[no-untyped-def]
    return load_vehicle(CONFIG)


@pytest.mark.parametrize("cid", ["FL", "FR", "RL", "RR"])
def test_golden_corner_sweep(vehicle, cid: str, update_golden: bool) -> None:  # type: ignore[no-untyped-def]
    c = build_corners(vehicle)[cid]  # type: ignore[index]
    tr = np.linspace(-0.030, 0.030, 25)
    df = add_gradients(states_to_frame(sweep_corner(c, tr), c, metrics=True))
    check_frames({f"corner_{cid}": df}, update_golden)


def test_golden_rack_sweep(vehicle, update_golden: bool) -> None:  # type: ignore[no-untyped-def]
    assert vehicle.front.steering is not None
    lo, hi = vehicle.front.steering.rack_travel
    c = build_corners(vehicle)["FL"]  # type: ignore[index]
    df = states_to_frame(sweep_rack(c, np.linspace(0.9 * lo, 0.9 * hi, 21)), c, metrics=True)
    check_frames({"rack_FL": df}, update_golden)


def test_golden_vehicle_sweeps(vehicle, update_golden: bool) -> None:  # type: ignore[no-untyped-def]
    assert vehicle.front.steering is not None
    lo, hi = vehicle.front.steering.rack_travel
    lim = 0.9 * min(abs(lo), hi)
    r, p = math.radians(2.0), math.radians(1.5)
    cases = {
        "heave": {"heave": np.linspace(-0.025, 0.025, 11)},
        "roll": {"roll": np.linspace(-r, r, 11)},
        "pitch": {"pitch": np.linspace(-p, p, 11)},
        "rack": {"rack": np.linspace(-lim, lim, 11)},
    }
    frames = {
        f"vehicle_{k}": vehicle_frame(sweep_vehicle(vehicle, **v), vehicle)
        for k, v in cases.items()
    }
    check_frames(frames, update_golden)


def test_golden_vehicle_summary(vehicle, update_golden: bool) -> None:  # type: ignore[no-untyped-def]
    data = flatten(dataclasses.asdict(vehicle_summary(vehicle)))
    check_json("vehicle_summary", data, update_golden)


def test_golden_reference_studies(update_golden: bool) -> None:
    def rows(ev):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            [{"name": r.name, "value": r.value, "g": r.g, "status": r.status} for r in ev.results]
        )

    cons, veh = load_study(STUDIES / "fs_reference_constraints.yaml")
    opt, veh2 = load_study(STUDIES / "fs_reference_optimize.yaml")
    ev2 = evaluate(DesignSpace(opt, veh2))
    obj = pd.DataFrame([{"name": o.name, "raw": o.raw, "f": o.f} for o in ev2.objectives])
    frames = {
        "study_constraints": rows(evaluate(DesignSpace(cons, veh))),
        "study_optimize": rows(ev2),
        "study_objectives": obj,
    }
    check_frames(frames, update_golden)
