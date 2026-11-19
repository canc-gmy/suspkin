"""Layer error: FOM corner curves through the vehicle layer versus ``vehicle_summary``.

Designs: the baseline of ``configs/studies/fs_reference_optimize.yaml`` and 8 seeded random
designs from its DesignSpace; grid +-30 mm travel (41 points) and the full rack travel
(21 points). Tolerances follow DESIGN_PLAN section 13.5; the other fields get loose bounds
(roughly 10-100x the error measured when the layer was written) that only guard against
a broken formula, not against interpolation noise.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from suspkin.geometry import Vehicle
from suspkin.optimize.design import DesignSpace, InfeasibleDesign
from suspkin.optimize.spec import load_study
from suspkin.surrogate import AxleGrid, extract_axles, layer_error

STUDY = Path("configs/studies/fs_reference_optimize.yaml")
DEG = math.pi / 180.0
GRIDS = {
    "front": AxleGrid("front", np.linspace(-0.03, 0.03, 41), np.linspace(-0.03, 0.03, 21)),
    "rear": AxleGrid("rear", np.linspace(-0.03, 0.03, 41), np.array([0.0])),
}

# (field, absolute tolerance in SI / field units)
TOLERANCES = {
    "front.rc_height": 1e-3,  # 1 mm
    "rear.rc_height": 1e-3,
    "anti_dive_front": 1.0,  # field is in percent: 1 %-point
    "anti_lift_rear": 1.0,
    "anti_squat_rear": 1.0,
    "tlltd_front": 0.005,  # field is a fraction: 0.5 %-point
    "roll_gradient": 0.01 * DEG,  # rad per g
}


def _draw(study_path: Path, n: int, seed: int) -> list[Vehicle]:
    """Baseline plus ``n`` feasible seeded random designs of the study's DesignSpace."""
    study, baseline = load_study(study_path)
    space = DesignSpace(study, baseline)
    rng = np.random.default_rng(seed)
    out = [baseline]
    for _ in range(200):
        if len(out) == n + 1:
            break
        try:
            cand = space.to_vehicle(rng.random(space.n))
        except InfeasibleDesign:
            continue
        if all(s.feasible for s in extract_axles(cand, GRIDS).values()):
            out.append(cand)
    assert len(out) == n + 1, f"could not draw {n} feasible random designs"
    return out


def _designs() -> list[Vehicle]:
    return _draw(STUDY, 8, 20261118)


_BOXES = {"default": 0.005, "wheel_center": 0.002}
_POINTS = (
    "uca_front",
    "uca_rear",
    "uca_outer",
    "lca_front",
    "lca_rear",
    "lca_outer",
    "tie_inner",
    "tie_outer",
    "pushrod_outer",
    "pushrod_inner",
    "rocker_pivot",
    "rocker_spring",
    "spring_chassis",
    "wheel_center",
    "rocker_arb",
)


def _broad_study(tmp_path: Path) -> Path:
    """Study with every front and rear hardpoint free in a small offset box (+-5 mm)."""
    vehicle = Path("configs/vehicles/fs_reference.yaml").resolve().as_posix()
    lines = ["schema: 1", f"vehicle: {vehicle}", "", "variables:"]
    for axle in ("front", "rear"):
        for name in _POINTS:
            r = _BOXES.get(name, _BOXES["default"]) * 1e3
            ax = f"[-{r} mm, {r} mm]"
            dom = f"{{offset: {{x: {ax}, y: {ax}, z: {ax}}}}}"
            lines.append(f"  - {{point: {axle}.{name}, domain: {dom}}}")
    path = tmp_path / "broad.yaml"
    path.write_text(chr(10).join(lines) + chr(10))
    return path


def _arb_variants(base: Vehicle) -> dict[str, Vehicle]:
    """One design per ARB type: baseline (torsional front, blade rear), beam, none."""
    d = base.model_dump()
    out = {"torsional+blade": base}
    for tag, mk in (("beam", True), ("none", False)):
        dd = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}
        for axle in ("front", "rear"):
            ax = dict(d[axle])
            hp = dict(ax["hardpoints"])
            if mk:
                x, y, z = ax["arb"]["arm_end"]
                ax["arb"] = {
                    "type": "beam",
                    "beam_center": (x, 0.0, z),
                    "beam_axis": (0.0, 1.0, 0.0),
                    "arm_end": (x, y, z),
                    "beam_width": 0.02,
                    "beam_thickness": 0.003,
                    "beam_youngs_modulus": 2.1e11,
                }
            else:
                ax["arb"] = None
                hp["rocker_arb"] = None
            ax["hardpoints"] = hp
            dd[axle] = ax
        out[tag] = Vehicle.model_validate(dd)
    return out


def _bound(name: str) -> float:
    """Loose sanity bound (SI or field units) for fields without a plan tolerance."""
    col = name.split(".")[-1]
    if name.startswith("sweep."):
        return 1e-3  # rad or m: sweeps are compared point by point
    if col in ("max_steer_inner", "max_steer_outer"):
        return 5e-4  # rad (0.03 deg)
    if col in ("wheel_rate", "arb_wheel_rate_roll"):
        return 30.0  # N/m, wheel rates are ~3e4
    if col.startswith("roll_stiffness"):
        return 30.0  # N m/rad, ~3e4
    if col.startswith("load_transfer"):
        return 5.0  # N, ~1e3
    if col.startswith("ackermann"):
        return 1.0  # %-points
    if col in ("roll_camber", "roll_steer", "rc_migration"):
        return 1e-3
    if col == "rc_height_curvature":
        return 5e-3  # m/rad^2
    if col == "steering_ratio":
        return 0.05
    if col == "turning_radius":
        return 0.01  # m
    return 1e-4  # lengths, masses, angles, heights, pitch centre


@pytest.fixture(scope="module")
def errors(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    """Layer-error tables: reference study, broad study (all hardpoints), ARB variants."""
    out = {
        "study": layer_error(_designs(), GRIDS),
        "broad": layer_error(_draw(_broad_study(tmp_path_factory.mktemp("broad")), 8, 7), GRIDS),
    }
    for tag, v in _arb_variants(load_study(STUDY)[1]).items():
        out[f"arb:{tag}"] = layer_error([v], GRIDS)
    return out


def test_layer_error_has_every_summary_field(errors: dict[str, pd.DataFrame]) -> None:
    from suspkin.surrogate import summary_fields

    assert set(errors) == {"study", "broad", "arb:torsional+blade", "arb:beam", "arb:none"}
    assert len(errors["study"]) == 9 and len(errors["broad"]) == 9
    for df in errors.values():
        assert set(summary_fields()) <= set(df.columns)
        assert {"sweep.roll.FL_camber_ground", "sweep.heave.rc_rear_z"} <= set(df.columns)


@pytest.mark.parametrize("field", list(TOLERANCES))
def test_plan_tolerances(errors: dict[str, pd.DataFrame], field: str) -> None:
    for tag, df in errors.items():
        worst = float(df[field].max(skipna=False))
        assert worst <= TOLERANCES[field], f"{tag} {field}: {worst:.3e} > {TOLERANCES[field]:.3e}"


def test_other_fields_within_sanity_bounds(errors: dict[str, pd.DataFrame]) -> None:
    for tag, df in errors.items():
        worst = df.max(skipna=False)
        bad = {
            k: float(v)
            for k, v in worst.items()
            if k not in TOLERANCES and not (float(v) <= _bound(str(k)))
        }
        assert not bad, f"{tag}: layer error above sanity bound (or NaN): {bad}"


def _ic_behind_designs(n: int) -> list[Vehicle]:
    """Seeded designs of the surrogate study whose rear side-view IC is behind the patch.

    In the old direction encoding these had a rear swing angle near +-180 deg (cos -> -1).
    """
    from suspkin.geometry.corners import build_corners
    from suspkin.kinematics.corner import solve_corner
    from suspkin.kinematics.rocker import KinematicsError
    from suspkin.metrics.batch import corner_metric_arrays

    study, baseline = load_study(Path("configs/studies/fs_reference_surrogate.yaml"))
    space = DesignSpace(study, baseline)
    rng = np.random.default_rng(3)
    out: list[Vehicle] = []
    for _ in range(300):
        if len(out) == n:
            break
        try:
            cand = space.to_vehicle(rng.random(space.n))
            corner = build_corners(cand)["RL"]
            raw = float(
                corner_metric_arrays([solve_corner(corner, 0.0)], corner)["sv_swing_angle"][0]
            )
        except (InfeasibleDesign, KinematicsError):
            continue
        if abs(raw) > math.pi / 2 and all(s.feasible for s in extract_axles(cand, GRIDS).values()):
            out.append(cand)
    assert len(out) == n, "could not draw designs with the IC behind the patch"
    return out


def test_layer_error_with_instant_centre_behind_the_patch() -> None:
    df = layer_error(_ic_behind_designs(3), GRIDS)
    for field, tol in TOLERANCES.items():
        worst = float(df[field].max(skipna=False))
        assert worst <= tol, f"{field}: {worst:.3e} > {tol:.3e}"
