"""Corner extraction on a fixed grid."""

from typing import Literal

import numpy as np
import pytest

from suspkin.geometry import Vehicle, build_corners, load_vehicle
from suspkin.kinematics.corner import sweep_corner
from suspkin.metrics.joints import joint_angles_frame
from suspkin.metrics.table import metrics_frame
from suspkin.surrogate import AxleGrid, channels_for, extract_axles, extract_corner

REF = "configs/vehicles/fs_reference.yaml"


@pytest.fixture(scope="module")
def veh() -> Vehicle:
    return load_vehicle(REF)


def _grid(
    axle: Literal["front", "rear"], n_t: int = 7, n_r: int = 5, t: float = 0.03, r: float = 0.02
) -> AxleGrid:
    rack = np.linspace(-r, r, n_r) if axle == "front" else np.array([0.0])
    return AxleGrid(axle, np.linspace(-t, t, n_t), rack)


def test_front_matches_tables(veh: Vehicle) -> None:
    g = _grid("front")
    s = extract_corner(veh, "front", g)
    chans = channels_for("front")
    assert s.Y.shape == (len(chans), 5, 7) and s.mask.shape == (5, 7)
    assert s.feasible and s.mask.all() and np.isfinite(s.Y).all()
    assert s.channels == tuple(c.name for c in chans)

    corner = build_corners(veh)["FL"]
    states = [st for r in g.rack for st in sweep_corner(corner, g.travel, rack=float(r))]
    mf = metrics_frame(states, corner)
    jf = joint_angles_frame(states, corner)
    for k, c in enumerate(chans):
        if c.source in mf:
            ref = mf[c.source].to_numpy()
        elif c.source in jf:
            ref = jf[c.source].to_numpy()
        else:
            ref = np.array([getattr(st, c.source) for st in states])
        ref = np.cos(ref) if c.encoding == "cos" else np.sin(ref) if c.encoding == "sin" else ref
        np.testing.assert_allclose(s.Y[k].reshape(-1), ref, atol=1e-12, rtol=0, err_msg=c.name)

    names = {c.name: k for k, c in enumerate(chans)}
    for base in ("fv_swing_angle", "sv_swing_angle", "sv_wc_angle"):
        cs = s.Y[names[f"{base}_cos"]] ** 2 + s.Y[names[f"{base}_sin"]] ** 2
        np.testing.assert_allclose(cs, 1.0, atol=1e-12)

    # static pose (rack 0, travel 0): relative channels vanish
    r0, t0 = 2, 3
    assert g.rack[r0] == 0.0 and g.travel[t0] == 0.0
    for n in ("half_track_change", "contact_patch_dx", "contact_patch_dy", "rocker_angle"):
        assert abs(s.Y[names[n], r0, t0]) < 1e-9
    static = metrics_frame(sweep_corner(corner, [0.0]), corner).iloc[0]
    assert s.Y[names["camber"], r0, t0] == pytest.approx(static["camber"], abs=1e-12)
    assert s.Y[names["steer_angle"], r0, t0] == pytest.approx(0.0, abs=1e-12)


def test_rear_single_rack(veh: Vehicle) -> None:
    s = extract_corner(veh, "rear", _grid("rear"))
    assert s.Y.shape == (len(channels_for("rear")), 1, 7)
    assert s.feasible and "steer_angle" not in s.channels


def test_unsolvable_travel_partial_mask(veh: Vehicle) -> None:
    g = AxleGrid("rear", np.linspace(-0.3, 0.3, 13), np.array([0.0]))
    s = extract_corner(veh, "rear", g)
    assert not s.feasible
    assert s.mask[0, 6] and 0 < s.mask.sum() < s.mask.size
    assert not s.mask[0, 0] and not s.mask[0, -1]
    assert np.isnan(s.Y[:, ~s.mask]).all() and np.isfinite(s.Y[:, s.mask]).all()


def test_stroke_limit_is_not_masked(veh: Vehicle) -> None:
    corner = build_corners(veh)["RL"]
    g = AxleGrid("rear", np.linspace(-0.05, 0.05, 5), np.array([0.0]))
    states = sweep_corner(corner, g.travel)
    assert not all(st.coilover_in_range for st in states)  # ends leave the coilover range
    s = extract_corner(veh, "rear", g)
    assert s.mask.all() and s.feasible
    k = s.channels.index("toggle_margin")
    np.testing.assert_allclose(s.Y[k, 0], [st.toggle_margin for st in states], atol=1e-12)


def test_extract_axles(veh: Vehicle) -> None:
    out = extract_axles(veh, {"front": _grid("front", 3, 3), "rear": _grid("rear", 3)})
    assert set(out) == {"front", "rear"}
    assert out["front"].Y.shape[1:] == (3, 3) and out["rear"].Y.shape[1:] == (1, 3)
