"""Batch metric arrays equal the per-state reference functions (alignment, centres, ...)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from suspkin.geometry import build_corners, load_vehicle
from suspkin.geometry.corners import Corner
from suspkin.kinematics.corner import CornerState, solve_corner, sweep_corner, sweep_rack
from suspkin.metrics import actuation
from suspkin.metrics.alignment import alignment_metrics
from suspkin.metrics.batch import (
    corner_arrays,
    corner_metric_arrays,
    joint_angle_arrays,
    roll_line_arrays,
)
from suspkin.metrics.centres import centre_metrics, roll_centre_pair
from suspkin.metrics.joints import JOINTS, joint_angle, joint_angles_frame
from suspkin.metrics.table import metrics_frame
from suspkin.sweeps.corner_sweeps import states_to_frame
from suspkin.sweeps.vehicle_sweeps import _roll_centre_pair_arrays
from tests.unit.helpers import fixture_vehicle
from tests.unit.test_corner_solver import _with_mount

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"
TOL = 1e-12


def reference_metrics(
    states: Sequence[CornerState], corner: Corner, static: CornerState
) -> dict[str, np.ndarray]:
    """Per-state reference: the original dict-per-row construction."""
    cfg = corner.axle_cfg
    consts = {
        "spring_rate": cfg.coilover.spring_rate,
        "tyre_stiffness": cfg.wheel.tyre_vertical_stiffness,
    }
    rows = []
    for s in states:
        row = {"travel": float(s.travel), "rack": float(s.rack)}
        row.update(alignment_metrics(s, corner, static))
        row.update(centre_metrics(s, corner))
        row.update(actuation.actuation_metrics(s, corner, static))
        row.update(consts)
        rows.append(row)
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def reference_joints(
    states: Sequence[CornerState], corner: Corner, static: CornerState
) -> dict[str, np.ndarray]:
    return {
        f"joint_{j}": np.array([joint_angle(s, static, corner, j) for s in states]) for j in JOINTS
    }


def assert_same(got: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> float:
    """Same keys (order), same nan/inf positions, values within TOL; returns max deviation."""
    assert list(got) == list(ref)
    worst = 0.0
    for k in ref:
        a, b = np.asarray(got[k]), np.asarray(ref[k])
        assert a.shape == b.shape, k
        assert np.array_equal(np.isnan(a), np.isnan(b)), k
        assert np.array_equal(np.isinf(a), np.isinf(b)), k
        fin = np.isfinite(b)
        assert np.array_equal(a[~fin & ~np.isnan(b)], b[~fin & ~np.isnan(b)]), k  # +-inf signs
        if fin.any():
            dev = float(np.max(np.abs(a[fin] - b[fin])))
            assert dev <= TOL, f"{k}: {dev:.2e}"
            worst = max(worst, dev)
    return worst


@pytest.fixture(scope="module")
def corners() -> dict[str, Corner]:
    return dict(build_corners(load_vehicle(CONFIG)))


def _states(corner: Corner, kind: str) -> list[CornerState]:
    travels = np.linspace(-0.03, 0.03, 13)
    if kind == "travel":
        return sweep_corner(corner, travels)
    if kind == "rack":
        return sweep_rack(corner, np.linspace(-0.02, 0.02, 9))
    out: list[CornerState] = []
    for r in np.linspace(-0.02, 0.02, 5):
        out += sweep_corner(corner, travels[::3], rack=float(r))
    return out


@pytest.mark.parametrize("cid", ["FL", "FR", "RL", "RR"])
@pytest.mark.parametrize("kind", ["travel", "rack", "grid"])
def test_batch_matches_reference(corners: dict[str, Corner], cid: str, kind: str) -> None:
    corner = corners[cid]
    if kind != "travel" and corner.axle != "front":
        pytest.skip("no steering on the rear axle")
    states = _states(corner, kind)
    static = solve_corner(corner, 0.0)
    assert_same(corner_metric_arrays(states, corner), reference_metrics(states, corner, static))
    assert_same(joint_angle_arrays(states, corner), reference_joints(states, corner, static))


def test_explicit_static_and_frames(corners: dict[str, Corner]) -> None:
    corner = corners["FL"]
    states = sweep_corner(corner, np.linspace(-0.02, 0.02, 7))
    static = solve_corner(corner, 0.0)
    assert_same(
        corner_metric_arrays(states, corner, static), reference_metrics(states, corner, static)
    )
    df = metrics_frame(states, corner)
    assert list(df.columns) == list(corner_metric_arrays(states, corner))
    assert all(dt == np.float64 for dt in df.dtypes)
    jf = joint_angles_frame(states, corner)
    assert list(jf.columns) == [f"joint_{j}" for j in JOINTS]
    assert len(jf) == len(states)


def test_states_to_frame_columns(corners: dict[str, Corner]) -> None:
    corner = corners["FL"]
    states = sweep_corner(corner, np.linspace(-0.02, 0.02, 5))
    df = states_to_frame(states, corner, metrics=True)
    assert list(df["corner"]) == ["FL"] * 5
    assert df["coilover_in_range"].dtype == bool
    for name, v in states[2].points.items():
        for i, a in enumerate("xyz"):
            assert df[f"{name}_{a}"].iloc[2] == float(v[i])
    assert df["spin_z"].iloc[1] == float(states[1].spin_axis[2])
    assert "camber" in df.columns and "rack" in df.columns
    assert df.columns.is_unique


def test_empty_states(corners: dict[str, Corner]) -> None:
    corner = corners["FL"]
    assert len(metrics_frame([], corner)) == 0
    assert len(joint_angles_frame([], corner)) == 0
    assert len(states_to_frame([])) == 0


# --------------------------------------------------------------------------- pushrod mounts
@pytest.mark.parametrize("mount", ["upright", "lca", "uca"])
@pytest.mark.parametrize("cid", ["FL", "RR"])
def test_batch_matches_reference_per_mount(mount: str, cid: str) -> None:
    corner = build_corners(_with_mount(fixture_vehicle(), mount))[cid]
    states = sweep_corner(corner, np.linspace(-0.025, 0.025, 9))
    if corner.axle == "front":
        states += sweep_corner(corner, np.linspace(-0.02, 0.02, 5), rack=0.01)
    static = solve_corner(corner, 0.0)
    assert_same(corner_metric_arrays(states, corner), reference_metrics(states, corner, static))
    assert_same(joint_angle_arrays(states, corner), reference_joints(states, corner, static))


# Reference values of the pre-vectorisation solver (HEAD before the scalar fast path):
# sweep_corner(corner, [-0.02, 0, 0.025], rack=0.01 (FL) / 0 (RR))[2] on the fixture vehicle.
# Layout: coilover_length, rocker_angle, toggle_margin, wheel_center, pushrod_outer,
# pushrod_inner, contact_patch, spin_axis.
PINS = [
    (
        "upright",
        "FL",
        [
            0.18957178918177,
            0.2206374954296555,
            1.417750973977738,
            -0.0045975902428964386,
            0.5981496786928137,
            0.2282,
            -0.0016117760113222215,
            0.5583407678355978,
            0.2950474048092876,
            -0.00033019803235349214,
            0.20955842680104364,
            0.4233019803235349,
            -0.005306167060572983,
            0.6038129218250049,
            0.025080169367638683,
            -0.12410158357625015,
            0.9918719090850651,
            0.028087593734875457,
        ],
    ),
    (
        "upright",
        "RR",
        [
            0.1892348187631955,
            0.2325163380420867,
            1.408845847338378,
            -1.5504846040375007,
            -0.5782817979043827,
            0.22820000000000001,
            -1.601009876141219,
            -0.5283453429577196,
            0.2946513908522999,
            -1.6003799399883767,
            -0.20909560537967278,
            0.42379939988376575,
            -1.550479759759205,
            -0.5835405420816161,
            0.025068058675619953,
            0.0009208764606601126,
            -0.9996646414238566,
            0.025879657522008388,
        ],
    ),
    (
        "lca",
        "FL",
        [
            0.18086969311008888,
            0.39545655664098645,
            1.2348910385452423,
            -0.0045975902428964386,
            0.5981496786928137,
            0.2282,
            0.0,
            0.5487174426654582,
            0.29333448268443674,
            -0.0004250849501800573,
            0.20092335961900012,
            0.4242508495018006,
            -0.005306167060572983,
            0.6038129218250049,
            0.025080169367638683,
            -0.12410158357625015,
            0.9918719090850651,
            0.028087593734875457,
        ],
    ),
    (
        "lca",
        "RR",
        [
            0.18000374264120364,
            0.4185878051884999,
            1.212910144735866,
            -1.5504846040375007,
            -0.5782817979043827,
            0.22820000000000001,
            -1.6,
            -0.5179133651023249,
            0.2927646575356456,
            -1.6004970423658,
            -0.19991747149222186,
            0.424970423658,
            -1.550479759759205,
            -0.5835405420816161,
            0.025068058675619953,
            0.0009208764606601126,
            -0.9996646414238566,
            0.025879657522008388,
        ],
    ),
    (
        "uca",
        "FL",
        [
            0.1917681211055358,
            0.17533804674520992,
            1.4674220184018834,
            -0.0045975902428964386,
            0.5981496786928137,
            0.2282,
            -0.0013055981206949174,
            0.561108624828797,
            0.2961119624138985,
            -0.0002812468958115272,
            0.21175789512295115,
            0.42281246895811525,
            -0.005306167060572983,
            0.6038129218250049,
            0.025080169367638683,
            -0.12410158357625015,
            0.9918719090850651,
            0.028087593734875457,
        ],
    ),
    (
        "uca",
        "RR",
        [
            0.19138359263162533,
            0.1878885392616496,
            1.458602426042435,
            -1.5504846040375007,
            -0.5782817979043827,
            0.22820000000000001,
            -1.601286811356313,
            -0.5311508843543606,
            0.29573622712625863,
            -1.6003267582982654,
            -0.2112527262296918,
            0.4232675829826523,
            -1.550479759759205,
            -0.5835405420816161,
            0.025068058675619953,
            0.0009208764606601126,
            -0.9996646414238566,
            0.025879657522008388,
        ],
    ),
]


@pytest.mark.parametrize(("mount", "cid", "ref"), PINS)
def test_solver_matches_pre_change_values(mount: str, cid: str, ref: list[float]) -> None:
    corner = build_corners(_with_mount(fixture_vehicle(), mount))[cid]
    s = sweep_corner(corner, [-0.02, 0.0, 0.025], rack=0.01 if cid == "FL" else 0.0)[2]
    got = [
        s.coilover_length,
        s.rocker_angle,
        s.toggle_margin,
        *s.points["wheel_center"],
        *s.points["pushrod_outer"],
        *s.points["pushrod_inner"],
        *s.points["contact_patch"],
        *s.spin_axis,
    ]
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-11)


# --------------------------------------------------------------------------- synthetic cases
def _moved(state: CornerState, **points: np.ndarray) -> CornerState:
    return replace(state, points={**state.points, **points})


def _flat_wishbones(state: CornerState, dz_uca: float = 0.1) -> CornerState:
    """Horizontal wishbones (all points of a wishbone at one height): both planes horizontal."""
    new = {}
    for w, z in (
        ("uca", state.points["lca_outer"][2] + dz_uca),
        ("lca", state.points["lca_outer"][2]),
    ):
        for end in ("front", "rear", "outer"):
            p = np.array(state.points[f"{w}_{end}"])
            p[2] = z
            new[f"{w}_{end}"] = p
    return _moved(state, **new)


def test_fvic_at_infinity_matches(corners: dict[str, Corner]) -> None:
    corner = corners["FL"]
    st = _flat_wishbones(solve_corner(corner, 0.0))
    got = corner_metric_arrays([st], corner, st)
    ref = reference_metrics([st], corner, st)
    assert np.isnan(ref["fvic_y"][0]) and np.isinf(ref["fvsa"][0])  # really at infinity
    assert np.isfinite(ref["fv_swing_angle"][0]) and np.isfinite(ref["rc_height"][0])
    assert_same(got, ref)


def test_degenerate_plane_matches(corners: dict[str, Corner]) -> None:
    corner = corners["FL"]
    base = solve_corner(corner, 0.0)
    f, r = base.points["uca_front"], base.points["uca_rear"]
    st = _moved(base, uca_outer=f + 0.3 * (r - f))  # collinear: no plane
    got = corner_metric_arrays([st], corner, base)
    ref = reference_metrics([st], corner, base)
    assert np.isnan(ref["fvic_y"][0]) and np.isnan(ref["svic_z"][0])
    assert np.isnan(ref["fv_swing_angle"][0]) and np.isnan(ref["rc_height"][0])  # nan direction
    assert_same(got, ref)


def _roll_tuple(state: CornerState, corner: Corner) -> tuple[np.ndarray, ...]:
    _arr, ic = corner_arrays([state], corner, state)
    cp = np.array([state.points["contact_patch"]])
    rd = roll_line_arrays(cp[:, 1], cp[:, 2], ic)
    return (cp[:, 0], cp[:, 1], cp[:, 2], rd[0], rd[1])


@pytest.mark.parametrize("dz_right", [0.0, 0.01])
def test_roll_centre_pair_coincident_and_parallel(
    corners: dict[str, Corner], dz_right: float
) -> None:
    """Horizontal wishbones: both roll lines horizontal; coincident (dz 0) or distinct."""
    left = _flat_wishbones(solve_corner(corners["FL"], 0.0))
    right = _flat_wishbones(solve_corner(corners["FR"], 0.0))
    cp = np.array(right.points["contact_patch"])
    cp[2] += dz_right
    right = _moved(right, contact_patch=cp)
    y, z = _roll_centre_pair_arrays(
        _roll_tuple(left, corners["FL"]), _roll_tuple(right, corners["FR"])
    )
    ry, rz = roll_centre_pair(left, right)
    if dz_right == 0.0:
        assert ry == 0.0 and np.isfinite(rz)  # the coincident branch
    else:
        assert np.isnan(ry) and np.isnan(rz)
    np.testing.assert_allclose([y[0], z[0]], [ry, rz], atol=1e-12, equal_nan=True)


def test_roll_centre_pair_nan_direction(corners: dict[str, Corner]) -> None:
    left = solve_corner(corners["FL"], 0.0)
    f, r = left.points["uca_front"], left.points["uca_rear"]
    bad = _moved(left, uca_outer=f + 0.3 * (r - f))
    right = solve_corner(corners["FR"], 0.0)
    y, z = _roll_centre_pair_arrays(
        _roll_tuple(bad, corners["FL"]), _roll_tuple(right, corners["FR"])
    )
    assert np.isnan(y[0]) and np.isnan(z[0])
    assert all(np.isnan(v) for v in roll_centre_pair(bad, right))


def test_roll_centre_pair_finite_matches(corners: dict[str, Corner]) -> None:
    left = solve_corner(corners["FL"], 0.01)
    right = solve_corner(corners["FR"], -0.01)
    y, z = _roll_centre_pair_arrays(
        _roll_tuple(left, corners["FL"]), _roll_tuple(right, corners["FR"])
    )
    np.testing.assert_allclose([y[0], z[0]], roll_centre_pair(left, right), atol=1e-12)
