"""Rotation centres of the 3D scene: values match the metrics, frame matches the scene."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from suspkin.app.centres import scene_centres
from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics import solve_vehicle
from suspkin.metrics.centres import centre_metrics
from suspkin.sweeps.vehicle_sweeps import vehicle_frame

REF = Path(__file__).parents[2] / "configs" / "vehicles" / "fs_reference.yaml"
IDS = ("FL", "FR", "RL", "RR")


@pytest.fixture(scope="module")
def vehicle():  # type: ignore[no-untyped-def]
    return load_vehicle(REF)


def test_static_matches_centre_metrics(vehicle) -> None:  # type: ignore[no-untyped-def]
    c = scene_centres(vehicle)
    st = solve_vehicle(vehicle)
    corners = build_corners(vehicle)
    for cid in IDS:
        m = centre_metrics(st.corners[cid], corners[cid])
        cp = getattr(c.contact, cid)
        assert cp == pytest.approx(tuple(st.contact_points[cid]), abs=1e-12)
        fv, sv = getattr(c.fv_ic, cid), getattr(c.sv_ic, cid)
        if fv is None:
            assert math.isnan(m["fvic_y"])
        else:
            assert fv[1] == pytest.approx(m["fvic_y"], abs=1e-6)
            assert fv[2] == pytest.approx(m["fvic_z"], abs=1e-6)
            assert getattr(c.fvsa, cid) == pytest.approx(m["fvsa"], abs=1e-6)
        if sv is None:
            assert math.isnan(m["svic_x"])
        else:
            assert sv[0] == pytest.approx(m["svic_x"], abs=1e-6)
            assert sv[2] == pytest.approx(m["svic_z"], abs=1e-6)
            assert getattr(c.svsa, cid) == pytest.approx(m["svsa"], abs=1e-6)
    assert c.roll.front is not None and c.roll.front[1] == pytest.approx(0.0, abs=1e-9)
    assert c.roll_height.front == pytest.approx(c.roll.front[2], abs=1e-9)  # static: body = world


def test_roll_height_matches_vehicle_frame(vehicle) -> None:  # type: ignore[no-untyped-def]
    st = solve_vehicle(vehicle, roll=0.03, pitch=0.01, heave=0.005)
    c = scene_centres(vehicle, st)
    row = vehicle_frame([st], vehicle).iloc[0]
    assert c.roll_height.front == pytest.approx(row["rc_front_z"], abs=1e-9)
    assert c.roll_height.rear == pytest.approx(row["rc_rear_z"], abs=1e-9)
    assert c.roll.front is not None and c.roll.front[1] == pytest.approx(
        row["rc_front_y"], abs=1e-9
    )
    # contact patches are the scene's markers (body frame)
    from suspkin.plotting.geometry3d import vehicle_segments

    scene = vehicle_segments(vehicle, st)
    for i, cid in enumerate(IDS):
        assert getattr(c.contact, cid) == pytest.approx(tuple(scene.contacts[i]), abs=1e-12)


def test_pitch_centre_on_centreline(vehicle) -> None:  # type: ignore[no-untyped-def]
    c = scene_centres(vehicle)
    if c.pitch is not None:
        assert c.pitch[1] == 0.0 and all(math.isfinite(v) for v in c.pitch)


def test_infinite_centres_are_null(vehicle, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import suspkin.app.centres as mod
    from suspkin.metrics.centres import InstantCentre

    nan = float("nan")
    inf_ic = InstantCentre(nan, nan, False, np.array([1.0, 0.0]))
    monkeypatch.setattr(mod, "front_view_ic", lambda s, c: inf_ic)
    monkeypatch.setattr(mod, "side_view_ic", lambda s, c: inf_ic)
    monkeypatch.setattr(mod, "roll_centre_pair", lambda a, b: (nan, nan))
    c = scene_centres(vehicle)
    for cid in IDS:
        assert getattr(c.fv_ic, cid) is None and getattr(c.sv_ic, cid) is None
        assert getattr(c.fvsa, cid) is None and getattr(c.svsa, cid) is None
        assert getattr(c.contact, cid) is not None
    assert c.roll.front is None and c.roll_height.rear is None
    assert c.model_dump(mode="json")["roll"] == {"front": None, "rear": None}
