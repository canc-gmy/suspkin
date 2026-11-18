"""Oracle regression: the surrogate evaluator fed with exact FOM curves reproduces the FOM.

The "network" is an oracle that returns the channels of ``extract_corner`` (validity = the
kinematic mask), so every deviation measured here comes from the surrogate path itself:
Hermite interpolation of the solver coordinates on the grid, the vectorised corner rebuild,
the metric columns and the analytic vehicle layer. Grid: travel +-30 mm (41 points), rack
+-30 mm (21 points) = the full rack travel of the reference car. Tolerance: 1e-3 on every
normalised ``g`` and ``f``, identical statuses.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from suspkin.optimize import DesignSpace, evaluate
from suspkin.optimize.spec import load_study
from suspkin.surrogate import axle_grids
from suspkin.surrogate.evaluate import OraclePredictor
from suspkin.surrogate.problem import SurrogateProblem

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ROOT / "configs" / "studies"
VEHICLE = ROOT / "configs" / "vehicles" / "fs_reference.yaml"
TOL = 1e-3
GRID_BLOCK = """
surrogate:
  grid:
    travel: {min: -30 mm, max: 30 mm, points: 41}
    rack: {min: -30 mm, max: 30 mm, points: 21}
"""


class CachedOracle:
    """``OraclePredictor`` that memoises rows (the studies share their design space)."""

    cache: ClassVar[dict[tuple[str, tuple[float, ...]], tuple[np.ndarray, np.ndarray]]] = {}

    def __init__(self, inner: OraclePredictor) -> None:
        self.inner = inner
        self.channels, self.travel, self.rack = inner.channels, inner.travel, inner.rack

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys, ps = [], []
        for row in np.atleast_2d(x):
            key = (self.inner.axle, tuple(np.round(row, 12)))
            if key not in self.cache:
                y, p = self.inner.predict(row[None])
                self.cache[key] = (y[0], p[0])
            ys.append(self.cache[key][0])
            ps.append(self.cache[key][1])
        return np.stack(ys), np.stack(ps)


@pytest.mark.parametrize("name", ["fs_reference_optimize", "fs_reference_constraints"])
def test_surrogate_matches_fom(name: str, tmp_path: Path) -> None:
    txt = (STUDIES / f"{name}.yaml").read_text(encoding="utf-8")
    txt = txt.replace("vehicle: ../vehicles/fs_reference.yaml", f"vehicle: {VEHICLE.as_posix()}")
    path = tmp_path / f"{name}.yaml"
    path.write_text(txt + GRID_BLOCK, encoding="utf-8")
    study, veh = load_study(path)
    space = DesignSpace(study, veh)
    grids = axle_grids(study.surrogate.grid)  # type: ignore[union-attr]
    preds = {a: CachedOracle(OraclePredictor(space, a, grids[a])) for a in ("front", "rear")}

    rng = np.random.default_rng(11)
    xs = np.vstack([space.from_vehicle(clip=True), rng.random((4, space.n))])
    with SurrogateProblem(path, None, tmp_path / "run", predictors=preds) as problem:  # type: ignore[arg-type]
        f_sur, g_sur, _, _, _ = problem.evaluate_batch(xs)
        evals = problem.evaluator.evaluate(xs)

    max_g = max_f = 0.0
    for i, x in enumerate(xs):
        fom = evaluate(space, x)
        got = evals[i]
        assert [r.status for r in fom.results] == [r.status for r in got.results], i
        assert [o.status for o in fom.objectives] == [o.status for o in got.objectives], i
        g_fom, f_fom = fom.g_vector(), fom.f_vector()
        assert np.allclose(g_sur[i], got.g_vector())
        if g_fom.size:
            max_g = max(max_g, float(np.max(np.abs(g_fom - g_sur[i]))))
        if f_fom.size:
            max_f = max(max_f, float(np.max(np.abs(f_fom - f_sur[i]))))
    print(f"\n{name}: max |dg| = {max_g:.2e}, max |df| = {max_f:.2e}")
    assert max_g < TOL and max_f < TOL
    # the baseline passes every constraint in the FOM, so it must in the surrogate too
    assert evaluate(space, xs[0]).feasible and np.all(g_sur[0] <= 0.0)
