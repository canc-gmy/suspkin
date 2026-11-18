"""Analytic vehicle layer: statics, interpolation, summary and sweeps against the FOM."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from suspkin.geometry import Vehicle, build_corners, load_vehicle
from suspkin.kinematics.corner import solve_corner
from suspkin.kinematics.vehicle import cg_pivot
from suspkin.metrics.arb import arb_rates
from suspkin.metrics.vehicle_summary import AxleSummary, VehicleSummary, vehicle_summary
from suspkin.surrogate import (
    AxleBatch,
    AxleGrid,
    UnsupportedColumnError,
    extract_axles,
    stack_samples,
    summary_fields,
    supported_sweep_columns,
    vehicle_layer,
    vehicle_statics,
)
from suspkin.surrogate.vehicle_layer import _Hermite
from suspkin.sweeps.vehicle_sweeps import sweep_vehicle, vehicle_frame

REF = "configs/vehicles/fs_reference.yaml"
GRIDS = {
    "front": AxleGrid("front", np.linspace(-0.03, 0.03, 31), np.linspace(-0.03, 0.03, 13)),
    "rear": AxleGrid("rear", np.linspace(-0.03, 0.03, 31), np.array([0.0])),
}


def _batches(vehicles: list[Vehicle], grids: dict[str, AxleGrid] = GRIDS) -> dict[str, AxleBatch]:
    per = [extract_axles(v, grids) for v in vehicles]
    return {a: stack_samples([p[a] for p in per], grids[a]) for a in grids}


def _edit(veh: Vehicle, fn: Callable[[dict[str, Any]], None]) -> Vehicle:
    d = veh.model_dump()
    fn(d)
    return Vehicle.model_validate(d)


@pytest.fixture(scope="module")
def veh() -> Vehicle:
    return load_vehicle(REF)


@pytest.fixture(scope="module")
def baseline(veh: Vehicle) -> tuple[dict[str, np.ndarray], VehicleSummary]:
    res = vehicle_layer(_batches([veh]), vehicle_statics([veh]))
    return res.summary, vehicle_summary(veh)


def _flat(s: VehicleSummary) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in fields(VehicleSummary):
        v = getattr(s, f.name)
        if f.name in ("front", "rear"):
            out.update(
                {f"{f.name}.{g.name}": float(getattr(v, g.name)) for g in fields(AxleSummary)}
            )
        else:
            out[f.name] = float("nan") if v is None else float(v)
    return out


def test_stack_samples_shapes_and_errors(veh: Vehicle) -> None:
    per = [extract_axles(veh, GRIDS)] * 3
    b = stack_samples([p["front"] for p in per], GRIDS["front"])
    assert b.n == 3 and b.Y.shape[0] == 3 and b.mask.shape == (3, 13, 31)
    with pytest.raises(ValueError):
        stack_samples([], GRIDS["front"])
    with pytest.raises(ValueError):  # grid does not match the tensor
        stack_samples([p["front"] for p in per], GRIDS["rear"])
    with pytest.raises(ValueError):  # mixed channel sets
        stack_samples([per[0]["front"], per[0]["rear"]], GRIDS["front"])


def test_statics_match_solver(veh: Vehicle) -> None:
    st = vehicle_statics([veh, veh])
    corners = build_corners(veh)
    for axle, cid, ax in (("front", "FL", st.front), ("rear", "RL", st.rear)):
        state = solve_corner(corners[cid], 0.0)
        np.testing.assert_allclose(ax.scp[0], state.points["contact_patch"], atol=1e-12)
        np.testing.assert_allclose(ax.swc[0], state.points["wheel_center"], atol=1e-12)
        assert ax.track[0] == pytest.approx(getattr(veh, f"{axle}_track"))
        assert ax.arb_type is not None
    np.testing.assert_allclose(st.pivot[1], cg_pivot(veh))
    assert st.n == 2 and st.rack_travel.shape == (2, 2)
    assert st.wheelbase[0] == pytest.approx(veh.wheelbase)


def test_mixed_arb_types_rejected(veh: Vehicle) -> None:
    other = _edit(veh, lambda d: d["rear"].update(arb=None))
    with pytest.raises(ValueError, match="same ARB type"):
        vehicle_statics([veh, other])


def test_hermite_reproduces_quadratics_and_flags_range() -> None:
    x = np.linspace(-1.0, 1.0, 9)
    y = (0.3 + 2.0 * x - 1.5 * x**2)[None, None, :]
    q = np.array([-0.55, 0.0, 0.123, 0.5, 0.7])  # interior: central-difference slopes
    out = _Hermite(y, x)(q)[0, 0]
    np.testing.assert_allclose(out, 0.3 + 2.0 * q - 1.5 * q**2, atol=1e-12)
    far = _Hermite(y, x)(np.array([-1.01, 1.2, np.nan]))
    assert np.isnan(far).all()


def test_summary_complete_and_matches_fom(
    baseline: tuple[dict[str, np.ndarray], VehicleSummary],
) -> None:
    layer, fom = baseline
    flat = _flat(fom)
    assert list(layer) == list(summary_fields()) == list(flat)
    assert all(v.shape == (1,) for v in layer.values())
    assert layer["front.rc_height"][0] == pytest.approx(flat["front.rc_height"], abs=1e-6)
    assert layer["rear.rc_height"][0] == pytest.approx(flat["rear.rc_height"], abs=1e-6)
    assert np.isnan(layer["pitch_centre_x"][0]) == np.isnan(flat["pitch_centre_x"])
    for k in ("anti_dive_front", "anti_lift_rear", "anti_squat_rear"):
        assert layer[k][0] == pytest.approx(flat[k], abs=1e-6), k
    assert layer["tlltd_front"][0] == pytest.approx(flat["tlltd_front"], abs=1e-4)
    assert layer["roll_gradient"][0] == pytest.approx(flat["roll_gradient"], rel=1e-3)
    assert layer["front.wheel_rate"][0] == pytest.approx(flat["front.wheel_rate"], rel=1e-3)
    assert layer["rear.arb_wheel_rate_roll"][0] == pytest.approx(
        flat["rear.arb_wheel_rate_roll"], rel=1e-3
    )
    assert layer["max_steer_outer"][0] == pytest.approx(flat["max_steer_outer"], abs=1e-3)
    assert layer["ackermann_10deg"][0] == pytest.approx(flat["ackermann_10deg"], abs=1.0)
    assert layer["front.roll_camber"][0] == pytest.approx(flat["front.roll_camber"], abs=1e-4)


def test_designs_are_independent_in_a_batch(veh: Vehicle) -> None:
    def move(d: dict[str, Any]) -> None:
        x, y, z = d["rear"]["hardpoints"]["pushrod_outer"]
        d["rear"]["hardpoints"]["pushrod_outer"] = (x + 0.006, y, z - 0.004)

    other = _edit(veh, move)
    both = vehicle_layer(_batches([veh, other]), vehicle_statics([veh, other])).summary
    for i, v in enumerate((veh, other)):
        one = vehicle_layer(_batches([v]), vehicle_statics([v])).summary
        for k, arr in both.items():
            np.testing.assert_allclose(arr[i], one[k][0], rtol=1e-12, atol=1e-12, err_msg=k)
    assert both["rear.wheel_rate"][0] != both["rear.wheel_rate"][1]


def test_outside_grid_gives_nan_for_affected_outputs_only(veh: Vehicle) -> None:
    short = {
        "front": AxleGrid("front", np.linspace(-0.012, 0.012, 9), np.linspace(-0.03, 0.03, 13)),
        "rear": AxleGrid("rear", np.linspace(-0.012, 0.012, 9), np.array([0.0])),
    }
    s = vehicle_layer(_batches([veh], short), vehicle_statics([veh])).summary
    assert np.isnan(s["front.roll_camber"]).all() and np.isnan(s["rear.rc_migration"]).all()
    assert np.isfinite(s["front.rc_height"]).all() and np.isfinite(s["front.wheel_rate"]).all()
    assert np.isfinite(s["anti_dive_front"]).all()
    assert np.isfinite(s["max_steer_outer"]).all()


def test_masked_node_gives_nan_locally(veh: Vehicle) -> None:
    b = _batches([veh])
    rear = b["rear"]
    t0 = int(np.argmin(np.abs(rear.grid.travel)))
    mask = rear.mask.copy()
    mask[:, :, t0] = False
    masked = {"front": b["front"], "rear": AxleBatch(rear.Y, mask, rear.channels, rear.grid)}
    s = vehicle_layer(masked, vehicle_statics([veh])).summary
    assert np.isnan(s["rear.rc_height"]).all() and np.isnan(s["rear.wheel_rate"]).all()
    assert np.isnan(s["anti_lift_rear"]).all() and np.isnan(s["tlltd_front"]).all()
    assert np.isfinite(s["front.rc_height"]).all() and np.isfinite(s["front.wheel_rate"]).all()
    assert np.isfinite(s["anti_dive_front"]).all()


def test_roll_sweep_matches_vehicle_frame(veh: Vehicle) -> None:
    rolls = np.linspace(-0.035, 0.035, 7)
    res = vehicle_layer(
        _batches([veh]),
        vehicle_statics([veh]),
        sweeps={"roll": rolls},
        columns={"roll": ["FL_camber_ground", "RR_steer_ground", "rc_front_z", "FL_travel"]},
    )
    df = vehicle_frame(sweep_vehicle(veh, roll=rolls), veh)
    assert set(res.sweeps["roll"]) == {
        "FL_camber_ground",
        "RR_steer_ground",
        "rc_front_z",
        "FL_travel",
    }
    for col, arr in res.sweeps["roll"].items():
        assert arr.shape == (1, 7)
        np.testing.assert_allclose(arr[0], df[col].to_numpy(), atol=2e-6, err_msg=col)


def test_heave_and_pitch_sweeps_match_vehicle_frame(veh: Vehicle) -> None:
    sweeps = {"heave": np.linspace(-0.02, 0.02, 5), "pitch": np.linspace(-0.015, 0.015, 5)}
    res = vehicle_layer(_batches([veh]), vehicle_statics([veh]), sweeps=sweeps)
    for kind, vals in sweeps.items():
        df = vehicle_frame(sweep_vehicle(veh, **{kind: vals}), veh)  # type: ignore[arg-type]
        for col, arr in res.sweeps[kind].items():
            np.testing.assert_allclose(arr[0], df[col].to_numpy(), atol=2e-4, err_msg=col)


def test_rack_sweep_steer_angle(veh: Vehicle) -> None:
    racks = np.linspace(-0.03, 0.03, 7)
    res = vehicle_layer(_batches([veh]), vehicle_statics([veh]), sweeps={"rack": racks}).sweeps[
        "rack"
    ]
    df = vehicle_frame(sweep_vehicle(veh, rack=racks), veh)
    for col in ("FL_steer_angle", "FR_steer_angle", "FL_camber"):
        np.testing.assert_allclose(res[col][0], df[col].to_numpy(), atol=2e-4, err_msg=col)


def test_unsupported_columns_raise(veh: Vehicle) -> None:
    args = (_batches([veh]), vehicle_statics([veh]))
    with pytest.raises(UnsupportedColumnError, match="FL_scrub_radius"):
        vehicle_layer(*args, sweeps={"roll": [0.0]}, columns={"roll": ["FL_scrub_radius"]})
    with pytest.raises(UnsupportedColumnError):
        vehicle_layer(*args, sweeps={"roll": [0.0]}, columns={"roll": ["arb_front_elastic"]})
    with pytest.raises(UnsupportedColumnError):  # rack sweeps carry no rear columns
        vehicle_layer(*args, sweeps={"rack": [0.0]}, columns={"rack": ["RL_camber"]})
    with pytest.raises(ValueError, match="sweep kind"):
        supported_sweep_columns("yaw")
    assert "FL_camber_ground" in supported_sweep_columns("roll")


def test_arb_rate_matches_scalar_model_for_all_bar_types(veh: Vehicle) -> None:
    beam = {
        "type": "beam",
        "beam_center": (-1.64, 0.0, 0.579),
        "beam_axis": (0.0, 1.0, 0.0),
        "arm_end": (-1.64, 0.26, 0.579),
        "beam_width": 0.02,
        "beam_thickness": 0.003,
        "beam_youngs_modulus": 2.1e11,
        "beam_angle": 0.0,
    }
    variants = [veh, _edit(veh, lambda d: d["rear"].update(arb=beam))]
    for i, v in enumerate(variants):
        s = vehicle_layer(_batches([v]), vehicle_statics([v])).summary
        corners = build_corners(v)
        for axle, (cl, cr) in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
            ref = arb_rates(corners[cl], corners[cr], getattr(v, axle)).wheel_rate_roll  # type: ignore[index]
            assert s[f"{axle}.arb_wheel_rate_roll"][0] == pytest.approx(ref, rel=2e-3), (i, axle)


def test_axle_without_arb(veh: Vehicle) -> None:
    bare = _edit(
        veh,
        lambda d: (
            d["rear"].update(arb=None),
            d["rear"]["hardpoints"].update(rocker_arb=None),
        ),
    )
    s = vehicle_layer(_batches([bare]), vehicle_statics([bare])).summary
    fom = _flat(vehicle_summary(bare))
    assert s["rear.arb_wheel_rate_roll"][0] == 0.0 and s["rear.roll_stiffness_arb"][0] == 0.0
    assert s["rear.roll_stiffness"][0] == pytest.approx(fom["rear.roll_stiffness"], rel=1e-3)
    assert s["tlltd_front"][0] == pytest.approx(fom["tlltd_front"], abs=1e-4)


@pytest.mark.parametrize(
    ("front_brakes", "rear_brakes", "drive", "split"),
    [
        ("inboard", "inboard", "awd", None),
        ("outboard", "inboard", "fwd", None),
        ("inboard", "outboard", "rwd", 0.3),
    ],
)
def test_brake_and_drive_layouts(
    veh: Vehicle, front_brakes: str, rear_brakes: str, drive: str, split: float | None
) -> None:
    def edit(d: dict[str, Any]) -> None:
        d["front"]["brakes"] = front_brakes
        d["rear"]["brakes"] = rear_brakes
        d["vehicle"]["drive"] = drive
        d["vehicle"]["drive_split_front"] = split
        for axle, dz in (("front", 0.03), ("rear", -0.03)):  # tilt the side-view arm axes
            x, y, z = d[axle]["hardpoints"]["uca_front"]
            d[axle]["hardpoints"]["uca_front"] = (x, y, z + dz)

    v = _edit(veh, edit)
    s = vehicle_layer(_batches([v]), vehicle_statics([v])).summary
    fom = _flat(vehicle_summary(v))
    assert fom["anti_lift_front_accel"] == fom["anti_lift_front_accel"]  # split > 0: not NaN
    for k in ("anti_dive_front", "anti_lift_rear", "anti_squat_rear", "anti_lift_front_accel"):
        assert s[k][0] == pytest.approx(fom[k], abs=1e-6), k
    # non-trivial side-view geometry, otherwise the comparison would be 0 == 0
    assert min(abs(fom[k]) for k in ("anti_dive_front", "anti_lift_rear")) > 1.0
    assert abs(fom["anti_lift_front_accel"]) > 1.0


def test_pose_residual_failure_gives_nan_travel(
    veh: Vehicle, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = (_batches([veh]), vehicle_statics([veh]))
    kw: dict[str, Any] = {"sweeps": {"roll": [0.0, 0.03]}, "columns": {"roll": ["FL_travel"]}}
    ok = vehicle_layer(*args, **kw).sweeps["roll"]["FL_travel"]
    assert np.isfinite(ok).all()
    monkeypatch.setattr(
        importlib.import_module("suspkin.surrogate.vehicle_layer"), "_POSE_ITERS", 0
    )
    bad = vehicle_layer(*args, **kw).sweeps["roll"]["FL_travel"]
    assert np.isnan(bad[0, 1])  # not converged: no silent last iterate
    # a pose far beyond the grid is NaN as well (default iterations)
    monkeypatch.undo()
    far = vehicle_layer(*args, sweeps={"roll": [0.2]}, columns={"roll": ["FL_travel"]})
    assert np.isnan(far.sweeps["roll"]["FL_travel"]).all()


def test_zero_must_be_a_grid_node(veh: Vehicle) -> None:
    shifted = {
        "front": AxleGrid("front", np.linspace(-0.03, 0.03, 30), np.linspace(-0.03, 0.03, 13)),
        "rear": AxleGrid("rear", np.linspace(-0.03, 0.03, 31), np.array([0.0])),
    }
    st = vehicle_statics([veh])
    with pytest.raises(ValueError, match="travel grid must contain 0"):
        vehicle_layer(_batches([veh], shifted), st)
    no_rack0 = {
        "front": AxleGrid("front", np.linspace(-0.03, 0.03, 31), np.linspace(-0.03, 0.03, 12)),
        "rear": GRIDS["rear"],
    }
    with pytest.raises(ValueError, match="rack grid must contain 0"):
        vehicle_layer(_batches([veh], no_rack0), st)
