"""Vectorised tables equal the per-state reference on the reference car and random designs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from suspkin.geometry import Vehicle, build_corners, load_vehicle
from suspkin.geometry.model import HARDPOINT_NAMES
from suspkin.geometry.vectors import body_rotation, plane_from_points, plane_section_2d
from suspkin.kinematics.corner import CornerState, solve_corner, sweep_corner
from suspkin.kinematics.rocker import KinematicsError
from suspkin.kinematics.vehicle import VehicleState
from suspkin.metrics.arrays import corner_metric_arrays, joint_angle_arrays
from suspkin.metrics.corner import (
    JOINTS,
    joint_angle,
    roll_centre_pair,
)
from suspkin.sweeps import sweep_vehicle, vehicle_frame
from tests.unit._scalar_oracles import actuation_metrics, alignment_metrics, centre_metrics

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"
_IDS = ("FL", "FR", "RL", "RR")
TOL = 1e-12


def reference_metrics(states, corner, static):  # type: ignore[no-untyped-def]
    """Per-state reference: the original dict-per-row construction of ``metrics_frame``."""
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
        row.update(actuation_metrics(s, corner, static))
        row.update(consts)
        rows.append(row)
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def reference_joints(states, corner, static):  # type: ignore[no-untyped-def]
    return {
        f"joint_{j}": np.array([joint_angle(s, static, corner, j) for s in states]) for j in JOINTS
    }


def assert_same(got, ref, scaled: bool = False) -> None:  # type: ignore[no-untyped-def]
    """Same keys (order), same nan/inf positions, values within TOL.

    ``scaled`` allows ``TOL * max(1, |value|)``: far-away instant centres of random designs
    are ill-conditioned (near-parallel section lines), so only the relative error is O(eps).
    """
    assert list(got) == list(ref)
    for k in ref:
        a, b = np.asarray(got[k]), np.asarray(ref[k])
        assert np.array_equal(np.isnan(a), np.isnan(b)), k
        assert np.array_equal(np.isinf(a), np.isinf(b)), k
        fin = np.isfinite(b)
        tol = TOL * np.maximum(1.0, np.abs(b[fin])) if scaled else TOL
        assert np.all(np.abs(a[fin] - b[fin]) <= tol), k


@pytest.fixture(scope="module")
def vehicle() -> Vehicle:
    return load_vehicle(CONFIG)


def _grounded(s: VehicleState, cid: str) -> CornerState:
    """Corner state whose contact patch is the vehicle (tilted-ground) contact point."""
    st_ = s.corners[cid]  # type: ignore[index]
    return replace(st_, points={**st_.points, "contact_patch": s.contact_points[cid]})  # type: ignore[index]


def reference_vehicle_columns(
    states: list[VehicleState], vehicle: Vehicle
) -> dict[str, np.ndarray]:
    """Per-state reference of the vehicle table (the pre-vectorisation construction)."""
    corners = build_corners(vehicle)
    cols: dict[str, np.ndarray] = {}
    for cid in _IDS:
        grounded = [_grounded(s, cid) for s in states]
        ref = reference_metrics(grounded, corners[cid], solve_corner(corners[cid], 0.0))
        for k, v in ref.items():
            if k != "rack":
                cols[f"{cid}_{k}"] = v
    for cid in _IDS:
        cp = np.array([s.contact_points[cid] for s in states])
        for i, a in enumerate("xyz"):
            cols[f"{cid}_cp_{a}"] = cp[:, i]
    for cid in _IDS:
        side = 1.0 if corners[cid].side == "left" else -1.0
        sw = np.array([body_rotation(s.roll, s.pitch) @ s.corners[cid].spin_axis for s in states])
        cols[f"{cid}_camber_ground"] = -np.arcsin(np.clip(sw[:, 2], -1, 1))
        cols[f"{cid}_steer_ground"] = np.arctan2(-side * sw[:, 0], side * sw[:, 1])
    for axle, (lc, rc) in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        ys, zs = [], []
        for s in states:
            y_b, z_b = roll_centre_pair(_grounded(s, lc), _grounded(s, rc))
            if np.isfinite(y_b) and np.isfinite(z_b):
                rot = body_rotation(s.roll, s.pitch)
                x_cp = 0.5 * (s.contact_points[lc][0] + s.contact_points[rc][0])  # type: ignore[index]
                w = rot @ (np.array([x_cp, y_b, z_b]) - s.pivot) + s.pivot + [0.0, 0.0, s.heave]
                ys.append(y_b)
                zs.append(w[2])
            else:
                ys.append(np.nan)
                zs.append(np.nan)
        cols[f"rc_{axle}_y"] = np.array(ys)
        cols[f"rc_{axle}_z"] = np.array(zs)
    return cols


@pytest.mark.parametrize(
    "kwargs",
    [
        {"roll": np.radians(np.linspace(-2.0, 2.0, 9))},
        {"heave": np.linspace(-0.02, 0.02, 7)},
        {"pitch": np.radians(np.linspace(-1.0, 1.0, 5))},
        {"rack": np.linspace(-0.02, 0.02, 7)},
    ],
    ids=["roll", "heave", "pitch", "rack"],
)
def test_vehicle_frame_matches_reference(vehicle: Vehicle, kwargs: dict[str, np.ndarray]) -> None:
    states = sweep_vehicle(vehicle, **kwargs)
    df = vehicle_frame(states, vehicle)
    ref = reference_vehicle_columns(states, vehicle)
    for k, v in ref.items():
        got = df[k].to_numpy()
        assert np.array_equal(np.isnan(got), np.isnan(v)), k
        assert np.array_equal(np.isinf(got), np.isinf(v)), k
        fin = np.isfinite(v)
        assert np.max(np.abs(got[fin] - v[fin]), initial=0.0) <= TOL, k
    assert df["arb_front_elastic"].notna().all()


_PERTURBED = [n for n in HARDPOINT_NAMES if n != "rocker_axis"]


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    noise=st.lists(
        st.floats(-4e-3, 4e-3), min_size=3 * len(_PERTURBED), max_size=3 * len(_PERTURBED)
    ),
    side=st.sampled_from(["front", "rear"]),
    mount=st.sampled_from(["upright", "lca", "uca"]),
)
def test_random_designs_match_reference(noise: list[float], side: str, mount: str) -> None:
    v = load_vehicle(CONFIG)
    axle = getattr(v, side).model_copy(update={"pushrod_mount": mount})
    hp = axle.hardpoints
    upd = {}
    for i, name in enumerate(_PERTURBED):
        base = getattr(hp, name)
        upd[name] = tuple(float(b + noise[3 * i + j]) for j, b in enumerate(base))
    v2 = v.model_copy(
        update={side: axle.model_copy(update={"hardpoints": hp.model_copy(update=upd)})}
    )
    try:
        corners = build_corners(v2)
        cid = "FL" if side == "front" else "RL"
        corner = corners[cid]
        states = sweep_corner(corner, np.linspace(-0.02, 0.02, 7))
        static = solve_corner(corner, 0.0)
        ref_m = reference_metrics(states, corner, static)
        ref_j = reference_joints(states, corner, static)
        # skip designs whose wishbone section lines are within 10 % of the parallel threshold
        # (the nan/inf decision flips on round-off there)
        for s_ in states:
            for axis in (0, 1):
                n2 = []
                for w in ("uca", "lca"):
                    pt, nrm = plane_from_points(
                        s_.points[f"{w}_front"], s_.points[f"{w}_rear"], s_.points[f"{w}_outer"]
                    )
                    n2.append(
                        plane_section_2d(pt, nrm, axis, float(s_.points["contact_patch"][axis]))[0]
                    )
                det = abs(float(n2[0][0] * n2[1][1] - n2[0][1] * n2[1][0]))
                assume(abs(det - 1e-9) > 1e-10)
    except (KinematicsError, ValueError, FloatingPointError):
        assume(False)
        return
    assert_same(corner_metric_arrays(states, corner), ref_m, scaled=True)
    assert_same(joint_angle_arrays(states, corner), ref_j, scaled=True)
