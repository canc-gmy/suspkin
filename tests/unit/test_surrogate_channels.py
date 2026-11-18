"""Channel registry."""

import numpy as np
import pytest

from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics.corner import sweep_corner
from suspkin.metrics.joints import joint_angles_frame
from suspkin.metrics.table import metrics_frame
from suspkin.surrogate import CHANNELS, channels_for
from suspkin.surrogate.extract import _metric_arrays

REF = "configs/vehicles/fs_reference.yaml"


def test_names_unique_and_tolerances_positive() -> None:
    names = [c.name for c in CHANNELS]
    assert len(set(names)) == len(names)
    assert all(c.tol > 0 for c in CHANNELS)


def test_cos_sin_pairs_complete() -> None:
    cos = {c.source for c in CHANNELS if c.encoding == "cos"}
    sin = {c.source for c in CHANNELS if c.encoding == "sin"}
    assert cos == sin and len(cos) == 3
    for c in CHANNELS:
        if c.encoding != "value":
            assert c.name.endswith(f"_{c.encoding}")


def test_axle_selection() -> None:
    front = [c.name for c in channels_for("front")]
    rear = [c.name for c in channels_for("rear")]
    assert "steer_angle" in front and "steer_angle" not in rear
    assert len(front) == len(rear) + 1
    with pytest.raises(ValueError):
        channels_for("middle")


def test_sources_exist_in_real_tables() -> None:
    veh = load_vehicle(REF)
    corners = build_corners(veh)
    for cid in ("FL", "RL"):
        st = sweep_corner(corners[cid], np.array([-0.005, 0.0, 0.005]))
        cols = set(metrics_frame(st, corners[cid]).columns) | set(
            joint_angles_frame(st, corners[cid]).columns
        )
        # taken from the solver state by the adapter
        cols |= {"coilover_length", "toggle_margin", "theta_uca", "theta_lca", "psi"}
        arrays = _metric_arrays(st, corners[cid])
        for c in channels_for("front" if cid == "FL" else "rear"):
            assert c.source in cols, c.name
            assert c.source in arrays
