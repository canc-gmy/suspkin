"""Tests for metric tables, gradients and static summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from suspkin.kinematics import sweep_corner
from suspkin.metrics import add_gradients, metrics_frame, static_summary
from suspkin.metrics.alignment import alignment_metrics
from suspkin.metrics.centres import centre_metrics
from suspkin.sweeps import states_to_frame
from tests.unit.helpers import fixture_corners

CORNERS = fixture_corners()
TRAVELS = np.linspace(-0.01, 0.01, 11)


def test_metrics_frame_columns_and_rows() -> None:
    c = CORNERS["FL"]
    states = sweep_corner(c, TRAVELS)
    df = metrics_frame(states, c)
    assert len(df) == 11
    expected = {"travel", "rack"} | set(alignment_metrics(states[0], c, states[0]))
    expected |= set(centre_metrics(states[0], c))
    assert set(df.columns) == expected
    np.testing.assert_allclose(df["travel"], TRAVELS, atol=1e-12)
    assert df["half_track_change"].abs().min() < 1e-9  # static row is in the sweep


def test_gradients_analytic_linear() -> None:
    x = np.linspace(-0.01, 0.01, 9)
    df = pd.DataFrame(
        {"travel": x, "camber": 2.0 * x, "toe": -3.0 * x, "caster": 0.5 * x,
         "half_track_change": 0.1 * x, "rc_height": 0.02 + 4.0 * x}
    )  # fmt: skip
    g = add_gradients(df)
    assert "camber_gain" not in df.columns
    for col, val in (("camber_gain", 2.0), ("bump_steer", -3.0), ("caster_change", 0.5),
                     ("track_change_rate", 0.1), ("rc_migration_rate", 4.0)):  # fmt: skip
        np.testing.assert_allclose(g[col], val, atol=1e-9)


def test_gradient_rack_and_validation() -> None:
    x = np.linspace(-0.01, 0.01, 5)
    g = add_gradients(pd.DataFrame({"rack": x, "steer_angle": 7.0 * x}), "rack")
    np.testing.assert_allclose(g["steer_gain"], 7.0, atol=1e-9)
    with pytest.raises(ValueError):
        add_gradients(pd.DataFrame({"travel": [0.0, 1.0], "camber": [0.0, 1.0]}))
    with pytest.raises(ValueError):
        add_gradients(pd.DataFrame({"travel": [0.0, 2.0, 1.0], "camber": [0.0, 1.0, 2.0]}))


def test_static_summary_interpolates() -> None:
    x = np.array([-0.02, -0.01, 0.005, 0.02])
    df = pd.DataFrame({"travel": x, "camber": 3.0 * x + 1.0})
    s = static_summary(df)
    assert s["camber"] == pytest.approx(1.0)
    assert static_summary(df.iloc[::-1].reset_index(drop=True))["camber"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        static_summary(pd.DataFrame({"travel": [0.1, 0.2, 0.3], "camber": [0.0, 0.0, 0.0]}))


def test_fl_fr_summaries_equal() -> None:
    out = {}
    for cid in ("FL", "FR"):
        c = CORNERS[cid]
        out[cid] = static_summary(add_gradients(metrics_frame(sweep_corner(c, TRAVELS), c)))
    for k, v in out["FL"].items():
        if k in ("fvic_y", "contact_patch_dy"):
            continue  # signed by side
        if np.isnan(v) or np.isinf(v):
            continue
        # steer_angle and toe-derived bump steer flip sign between sides only via toe sign
        assert out["FR"][k] == pytest.approx(v, abs=1e-8), k


def test_states_to_frame_metrics_option() -> None:
    c = CORNERS["FL"]
    states = sweep_corner(c, TRAVELS)
    base = states_to_frame(states)
    assert "camber" not in base.columns
    assert "camber" not in states_to_frame(states, c).columns
    df = states_to_frame(states, c, metrics=True)
    assert list(df.columns[: len(base.columns)]) == list(base.columns)
    assert {"camber", "rc_height", "fvic_z"} <= set(df.columns)
    assert len(df) == len(base)
