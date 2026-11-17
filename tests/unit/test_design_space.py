"""Design space: mapping, mirroring, asymmetry, infeasibility."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from suspkin.geometry import Vehicle, build_corners
from suspkin.optimize import DesignSpace, InfeasibleDesign
from suspkin.optimize.spec import Study, load_study
from tests.unit.helpers import fixture_vehicle

STUDY = Path(__file__).parents[1] / "fixtures" / "studies" / "design_study.yaml"
BAD_BOX = {"box": {"min": [-10, 540, 50], "max": [10, 560, 100], "units": "mm"}}


def _space() -> DesignSpace:
    study, vehicle = load_study(STUDY)
    return DesignSpace(study, vehicle)


def _study(variables: list[dict[str, Any]]) -> Study:
    return Study.model_validate({"schema": 1, "vehicle": "x", "variables": variables})


def _mirror(p: np.ndarray) -> np.ndarray:
    return p * np.array([1.0, -1.0, 1.0])


def test_dimensions_and_names() -> None:
    sp = _space()
    # hex 3 + box 3 + offset (x, z) 2 + two asymmetric boxes 3 + 3
    assert sp.n == 14
    assert len(sp.names) == 14
    assert sp.names[0] == "front.uca_front.u"
    assert "rear.pushrod_outer.x" in sp.names
    assert "rear.pushrod_outer.y" not in sp.names
    assert "front.lca_rear@right.u" in sp.names
    assert sp.bounds.shape == (14, 2)
    assert np.all(sp.bounds[:, 0] == 0.0) and np.all(sp.bounds[:, 1] == 1.0)


def test_baseline_roundtrip_reproduces_hardpoints() -> None:
    sp = _space()
    x0 = sp.from_vehicle()
    assert np.all((x0 >= 0.0) & (x0 <= 1.0))
    v = sp.to_vehicle(x0)
    base_c, new_c = build_corners(sp.baseline), build_corners(v)
    for cid, c in base_c.items():
        for name, p in c.points.items():
            np.testing.assert_allclose(new_c[cid].points[name], p, atol=1e-12, rtol=0)


def test_points_match_vehicle() -> None:
    sp = _space()
    x = np.linspace(0.1, 0.9, sp.n)
    pts = sp.points(x)
    v = sp.to_vehicle(x)
    np.testing.assert_allclose(v.front.hardpoints.uca_front, pts["front.uca_front"], atol=1e-15)
    np.testing.assert_allclose(v.front.right_overrides["lca_rear"], pts["front.lca_rear@right"])
    np.testing.assert_allclose(v.rear.hardpoints.lca_front, pts["rear.lca_front"])
    # offset: y of rear.pushrod_outer stays at baseline
    assert v.rear.hardpoints.pushrod_outer[1] == sp.baseline.rear.hardpoints.pushrod_outer[1]


def test_symmetric_change_mirrors() -> None:
    sp = _space()
    x = sp.from_vehicle()
    x[:3] = [0.2, 0.7, 0.4]  # front.uca_front
    x[3:6] = [0.9, 0.1, 0.5]  # front.tie_inner
    corners = build_corners(sp.to_vehicle(x))
    for name in ("uca_front", "tie_inner"):
        np.testing.assert_allclose(corners["FR"].points[name], _mirror(corners["FL"].points[name]))
    base = build_corners(sp.baseline)
    assert not np.allclose(corners["FL"].points["uca_front"], base["FL"].points["uca_front"])


def test_asymmetric_right_moves_only_right() -> None:
    sp = _space()
    x = sp.from_vehicle()
    i = sp.names.index("front.lca_rear@right.u")
    x[i : i + 3] = [0.0, 1.0, 0.5]
    corners = build_corners(sp.to_vehicle(x))
    base = build_corners(sp.baseline)
    np.testing.assert_allclose(corners["FR"].points["lca_rear"], [-0.16, -0.21, 0.12])
    np.testing.assert_allclose(corners["FL"].points["lca_rear"], base["FL"].points["lca_rear"])


def test_asymmetric_left_pins_right() -> None:
    sp = _space()
    x = sp.from_vehicle()
    i = sp.names.index("rear.lca_front.u")
    x[i : i + 3] = [1.0, 1.0, 1.0]
    v = sp.to_vehicle(x)
    corners = build_corners(v)
    base = build_corners(sp.baseline)
    np.testing.assert_allclose(corners["RL"].points["lca_front"], [-1.44, 0.23, 0.13])
    np.testing.assert_allclose(corners["RR"].points["lca_front"], base["RR"].points["lca_front"])
    assert "lca_front" in v.rear.right_overrides


def test_asymmetric_left_keeps_existing_override() -> None:
    study = _study(
        [
            {
                "point": "front.lca_rear",
                "asymmetric": True,
                "domain": {"offset": {"x": [-0.01, 0.01]}},
            }
        ]
    )
    d = fixture_vehicle().model_dump()
    d["front"]["right_overrides"] = {"lca_rear": (-0.15, -0.23, 0.12)}
    sp = DesignSpace(study, Vehicle.model_validate(d))
    out = sp.to_vehicle(np.array([1.0]))
    assert out.front.right_overrides["lca_rear"] == (-0.15, -0.23, 0.12)
    assert out.front.hardpoints.lca_rear[0] == pytest.approx(-0.14)


def test_symmetric_variable_on_override_rejected() -> None:
    d = fixture_vehicle().model_dump()
    d["front"]["right_overrides"] = {"uca_front": (0.10, -0.28, 0.30)}
    study = _study([{"point": "front.uca_front", "domain": {"offset": {"x": [-0.01, 0.01]}}}])
    with pytest.raises(ValueError, match="symmetric variable on point with right override"):
        DesignSpace(study, Vehicle.model_validate(d))


def test_invalid_geometry_is_infeasible() -> None:
    sp = DesignSpace(_study([{"point": "front.uca_outer", "domain": BAD_BOX}]), fixture_vehicle())
    with pytest.raises(InfeasibleDesign, match="uca_outer z must be above lca_outer z"):
        sp.to_vehicle(np.array([0.5, 0.5, 0.5]))


def test_baseline_outside_domain() -> None:
    sp = DesignSpace(_study([{"point": "front.uca_outer", "domain": BAD_BOX}]), fixture_vehicle())
    with pytest.raises(InfeasibleDesign, match="outside its domain"):
        sp.from_vehicle()
    x = sp.from_vehicle(clip=True)
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_baseline_outside_hexahedron_and_offset() -> None:
    veh = fixture_vehicle()
    hexv = [
        [0.5, 0.2, 0.2],
        [0.6, 0.2, 0.2],
        [0.6, 0.3, 0.2],
        [0.5, 0.3, 0.2],
        [0.5, 0.2, 0.3],
        [0.6, 0.2, 0.3],
        [0.6, 0.3, 0.3],
        [0.5, 0.3, 0.3],
    ]
    hex_space = DesignSpace(
        _study([{"point": "front.uca_front", "domain": {"hexahedron": hexv}}]), veh
    )
    with pytest.raises(InfeasibleDesign):
        hex_space.from_vehicle()
    off = _study([{"point": "front.uca_front", "domain": {"offset": {"x": [0.01, 0.02]}}}])
    with pytest.raises(InfeasibleDesign):
        DesignSpace(off, veh).from_vehicle()


@pytest.mark.parametrize(
    "x",
    [
        np.full(13, 0.5),
        np.full(14, 1.1),
        np.full(14, -0.1),
        np.r_[np.full(13, 0.5), np.nan],
        np.r_[np.full(13, 0.5), np.inf],
    ],
)
def test_bad_x_raises(x: np.ndarray) -> None:
    sp = _space()
    with pytest.raises(ValueError):
        sp.to_vehicle(x)
    with pytest.raises(ValueError):
        sp.points(x)


def test_x_tolerance_accepted() -> None:
    sp = _space()
    x = np.full(sp.n, 0.5)
    x[0], x[1] = -1e-13, 1.0 + 1e-13
    sp.to_vehicle(x)
