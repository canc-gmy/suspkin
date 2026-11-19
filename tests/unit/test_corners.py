from pathlib import Path

import numpy as np
import pytest

from suspkin.geometry import arb_points, build_corners, load_vehicle
from suspkin.geometry.model import Vehicle
from suspkin.geometry.vectors import distance, mirror_y

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"
LINKS = [
    ("uca_front", "uca_outer"),
    ("uca_rear", "uca_outer"),
    ("lca_front", "lca_outer"),
    ("lca_rear", "lca_outer"),
    ("tie_inner", "tie_outer"),
    ("pushrod_inner", "pushrod_outer"),
]


def test_four_corners() -> None:
    c = build_corners(load_vehicle(FIXTURE))
    assert set(c) == {"FL", "FR", "RL", "RR"}
    assert c["FL"].side == "left"
    assert c["RR"].side == "right"
    assert c["RL"].axle == "rear"


def test_mirror_and_link_lengths() -> None:
    c = build_corners(load_vehicle(FIXTURE))
    for left, right in (("FL", "FR"), ("RL", "RR")):
        assert c[left].points.keys() == c[right].points.keys()
        for k, p in c[left].points.items():
            np.testing.assert_allclose(c[right].points[k], mirror_y(p))
        for a, b in LINKS:
            assert distance(c[left].points[a], c[left].points[b]) == pytest.approx(
                distance(c[right].points[a], c[right].points[b])
            )
        np.testing.assert_allclose(c[right].rocker_axis, mirror_y(c[left].rocker_axis))


def test_override_applied() -> None:
    v = load_vehicle(FIXTURE)
    new = (0.0, -0.57, 0.31)
    front = v.front.model_copy(update={"right_overrides": {"uca_outer": new}})
    v2 = Vehicle(vehicle=v.vehicle, front=front, rear=v.rear)
    c = build_corners(v2)
    np.testing.assert_allclose(c["FR"].points["uca_outer"], new)
    np.testing.assert_allclose(c["FL"].points["uca_outer"], v.front.hardpoints.uca_outer)
    for k, p in c["FL"].points.items():
        if k != "uca_outer":
            np.testing.assert_allclose(c["FR"].points[k], mirror_y(p))


def test_points_read_only() -> None:
    c = build_corners(load_vehicle(FIXTURE))
    for corner in c.values():
        assert not corner.rocker_axis.flags.writeable
        for p in corner.points.values():
            assert p.shape == (3,)
            assert not p.flags.writeable
    with pytest.raises(ValueError):
        c["FL"].points["uca_front"][0] = 1.0


def test_arb_points_mirrored() -> None:
    c = build_corners(load_vehicle(FIXTURE))
    fl, fr = arb_points(c["FL"]), arb_points(c["FR"])
    assert set(fl) == {"arm_end", "bar_pivot"}
    assert set(arb_points(c["RL"])) == {"arm_end", "beam_center"}
    for k in fl:
        np.testing.assert_allclose(fr[k], mirror_y(fl[k]))


def test_no_arb_gives_empty() -> None:
    v = load_vehicle(FIXTURE)
    hp = v.front.hardpoints.model_copy(update={"rocker_arb": None})
    front = v.front.model_copy(update={"arb": None, "hardpoints": hp})
    c = build_corners(Vehicle(vehicle=v.vehicle, front=front, rear=v.rear))
    assert arb_points(c["FL"]) == {}
    assert "rocker_arb" not in c["FL"].points
