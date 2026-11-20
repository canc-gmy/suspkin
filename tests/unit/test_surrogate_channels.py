"""Channel registry."""

import numpy as np
import pytest

from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics.corner import sweep_corner
from suspkin.metrics.corner import joint_angles_frame
from suspkin.metrics.table import metrics_frame
from suspkin.surrogate import CHANNELS, channels_for
from suspkin.surrogate.channels import wrap_line_angle
from suspkin.surrogate.extract import _metric_arrays

REF = "configs/vehicles/fs_reference.yaml"


def test_names_unique_and_tolerances_positive() -> None:
    names = [c.name for c in CHANNELS]
    assert len(set(names)) == len(names)
    assert all(c.tol > 0 for c in CHANNELS)


def test_line_angle_channels() -> None:
    lines = [c for c in CHANNELS if c.encoding == "line_angle"]
    assert {c.name for c in lines} == {"fv_swing_angle", "sv_swing_angle", "sv_wc_angle"}
    assert all(c.name == c.source and c.tol == pytest.approx(0.05 * np.pi / 180) for c in lines)
    assert {c.encoding for c in CHANNELS} == {"value", "line_angle"}


def test_wrap_line_angle_range_period_and_branch_flip() -> None:
    a = np.linspace(-4 * np.pi, 4 * np.pi, 2001)
    w = wrap_line_angle(a)
    assert np.all(w > -np.pi / 2 - 1e-15) and np.all(w <= np.pi / 2 + 1e-15)
    np.testing.assert_allclose(wrap_line_angle(a + np.pi), w, atol=1e-12)
    np.testing.assert_allclose(np.sin(w - a), 0.0, atol=1e-12)  # differ by a multiple of pi
    # identity inside the range, sign kept
    np.testing.assert_allclose(wrap_line_angle([0.1, -0.1, 1.5]), [0.1, -0.1, 1.5])
    # direction flip (IC from in front to behind): +-180 deg neighbours give a continuous value
    eps = 1e-3
    flip = wrap_line_angle([np.pi - eps, -np.pi + eps])
    np.testing.assert_allclose(flip, [-eps, eps], atol=1e-12)
    assert wrap_line_angle(np.pi) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(wrap_line_angle(np.nan))


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
