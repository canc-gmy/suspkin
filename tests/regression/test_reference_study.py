"""Regression: the reference constraint study on the reference vehicle."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from suspkin.optimize import DesignSpace, evaluate
from suspkin.optimize.spec import load_study

STUDY = (
    Path(__file__).resolve().parents[2] / "configs" / "studies" / "fs_reference_constraints.yaml"
)

# g of the baseline, YAML order (normalised: g <= 0 is feasible)
EXPECTED_G = np.array(
    [
        -0.6292069195577398,  # camber_band_f
        -3.65438574725512,  # toe_band_f
        -0.3499630389992174,  # camber_gain_f
        -0.49269035775428277,  # bump_steer_r
        -0.4780313014632173,  # roll_camber_f
        -0.40239538991481183,  # mr_front
        -0.4635614727317609,  # mr_rear
        -10.22443887192806,  # travel_f
        -9.5805941400115,  # travel_r
        -190.68692891664034,  # ubj_angle
        -0.33886622544484774,  # rc_height_f
        -0.47564870601480586,  # roll_gradient
        -0.3534989374291098,  # tlltd_front
        -138.00297616411444,  # uca_front_keepout
        -54.630396206473826,  # tie_inner_tube
        -39.99999999999995,  # uca_front_zone
        -39.99999999999998,  # uca_rear_zone
        -35.018390415853474,  # tie_len
        -60.0,  # uca_base
        -100.0,  # uca_above_lca
    ]
)


def test_reference_study_baseline_passes_and_g_is_pinned() -> None:
    study, vehicle = load_study(STUDY)
    ev = evaluate(DesignSpace(study, vehicle))
    assert [r.status for r in ev.results] == ["pass"] * len(study.constraints)
    assert ev.feasible
    np.testing.assert_allclose(ev.g_vector(), EXPECTED_G, rtol=1e-6)


def test_reference_summary_values_match_reference_car() -> None:
    study, vehicle = load_study(STUDY)
    by_name = {r.name: r for r in evaluate(DesignSpace(study, vehicle)).results}
    assert abs(by_name["rc_height_f"].value - 0.0337) < 5e-4
    assert abs(by_name["tlltd_front"].value - 0.503) < 2e-3
