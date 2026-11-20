"""LHS and the per-axle sub design space."""

from pathlib import Path

import numpy as np
import pytest

from suspkin.optimize.design import DesignSpace
from suspkin.optimize.spec import StudyConfigError, load_study
from suspkin.surrogate.data import AxleSpace, lhs

VEHICLE = Path("configs/vehicles/fs_reference.yaml").resolve().as_posix()

STUDY = f"""
schema: 1
vehicle: {VEHICLE}
variables:
  - point: front.tie_inner
    domain: {{offset: {{x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}}}}
  - point: front.uca_outer
    domain: {{offset: {{x: [-4 mm, 4 mm], y: [-4 mm, 4 mm], z: [-4 mm, 4 mm]}}}}
  - point: rear.pushrod_outer
    domain: {{offset: {{x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}}}}
surrogate:
  grid:
    travel: {{min: -20 mm, max: 20 mm, points: 7}}
    rack: {{min: -15 mm, max: 15 mm, points: 5}}
  dataset: {{n_initial: 12, seed: 3, shard_size: 4}}
"""

REAR_VAR = (
    "  - point: rear.pushrod_outer\n    domain: {offset: {x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}}\n"
)


def _space(tmp_path: Path, text: str = STUDY) -> DesignSpace:
    p = tmp_path / "s.yaml"
    p.write_text(text, encoding="utf-8")
    study, veh = load_study(p)
    return DesignSpace(study, veh)


def test_lhs_deterministic_and_stratified() -> None:
    a, b = lhs(16, 3, seed=5), lhs(16, 3, seed=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, lhs(16, 3, seed=6))
    assert a.shape == (16, 3) and a.min() >= 0.0 and a.max() <= 1.0
    for j in range(3):  # one point per stratum
        assert sorted(np.floor(a[:, j] * 16).astype(int)) == list(range(16))


def test_axle_split(tmp_path: Path) -> None:
    space = _space(tmp_path)
    f, r = AxleSpace(space, "front"), AxleSpace(space, "rear")
    assert f.n == 5 and r.n == 2 and f.n + r.n == space.n
    assert all(nm.startswith("front.") for nm in f.names)
    assert all(nm.startswith("rear.") for nm in r.names)
    assert set(f.idx).isdisjoint(r.idx)
    base = space.from_vehicle(clip=True)
    assert np.allclose(f.baseline(), base[f.idx])
    full = f.to_full(np.full(f.n, 0.3))
    assert np.allclose(full[f.idx], 0.3)
    assert np.allclose(full[r.idx], base[r.idx])  # rear at baseline
    full_r = r.to_full(np.array([0.9, 0.1]))
    assert np.allclose(full_r[f.idx], base[f.idx])
    with pytest.raises(ValueError):
        f.to_full(np.zeros(2))


def test_asymmetric_rejected(tmp_path: Path) -> None:
    text = STUDY.replace(REAR_VAR, REAR_VAR + "    asymmetric: true\n")
    space = _space(tmp_path, text)
    AxleSpace(space, "front")  # the other axle is fine
    with pytest.raises(StudyConfigError, match="asymmetric"):
        AxleSpace(space, "rear")
