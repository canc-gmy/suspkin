"""Behaviour lock of the surrogate layer: oracle evaluation and ``surrogate sample``.

``test_oracle_evaluation_lock`` evaluates the surrogate study with the exact-FOM
``OraclePredictor`` on 16 random designs (``default_rng(0)`` in the unit design cube) and
snapshots the evaluator output (statuses, values, ``g``, ``f``) and the batch arrays of the
problem. ``test_surrogate_sample_lock`` runs ``suspkin surrogate sample`` and snapshots the
dataset folder; binary ``.npy`` / ``.npz`` files get a JSON summary (shape, dtype, sums,
first values) next to them, since only that summary is compared by value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from typer.testing import CliRunner

from behaviour._lock import REPO, snapshot_dir
from suspkin.cli import app
from suspkin.kinematics import _fast
from suspkin.optimize import DesignSpace
from suspkin.optimize.spec import load_study
from suspkin.surrogate import axle_grids
from suspkin.surrogate.evaluate import OraclePredictor
from suspkin.surrogate.problem import SurrogateProblem

pytestmark = pytest.mark.lock

STUDY = REPO / "configs" / "studies" / "fs_reference_surrogate.yaml"
N_DESIGNS = 16


@pytest.fixture(autouse=True)
def _no_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUSPKIN_NO_NUMBA", "1")
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("NO_COLOR", "1")
    _fast.reset()


def _arr(x: Any) -> Any:
    return np.asarray(x, dtype=float).tolist()


def test_oracle_evaluation_lock(tmp_path: Path, update_lock: bool) -> None:
    study, veh = load_study(STUDY)
    space = DesignSpace(study, veh)
    grids = axle_grids(study.surrogate.grid)  # type: ignore[union-attr]
    preds = {a: OraclePredictor(space, a, grids[a]) for a in ("front", "rear")}
    xs = np.random.default_rng(0).random((N_DESIGNS, space.n))

    with SurrogateProblem(STUDY, None, tmp_path / "run", predictors=preds) as problem:  # type: ignore[arg-type]
        f, g, f_total, max_g, feasible = problem.evaluate_batch(xs)
        evals = problem.evaluator.evaluate(xs)

    designs: list[dict[str, Any]] = []
    for ev in evals:
        designs.append(
            {
                "feasible": bool(ev.feasible),
                "f_total": float(ev.f_total),
                "constraints": [
                    {"name": r.name, "status": r.status, "value": r.value, "g": r.g}
                    for r in ev.results
                ],
                "objectives": [
                    {"name": o.name, "status": o.status, "raw": o.raw, "f": o.f}
                    for o in ev.objectives
                ],
            }
        )
    doc = {
        "n_variables": int(space.n),
        "designs": designs,
        "batch": {
            "F": _arr(f),
            "G": _arr(g),
            "f_total": _arr(f_total),
            "max_g": _arr(max_g),
            "feasible": [bool(v) for v in feasible],
        },
    }
    out = tmp_path / "out"
    out.mkdir()
    (out / "oracle_eval.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    snapshot_dir("surrogate_oracle", out, "", 0, update=update_lock, tmp=tmp_path)


def _summarise_arrays(out: Path) -> None:
    for p in sorted([*out.rglob("*.npy"), *out.rglob("*.npz")]):
        arrays: dict[str, Any]
        if p.suffix == ".npy":
            arrays = {"array": np.load(p, allow_pickle=False)}
        else:
            with np.load(p, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        summary: dict[str, Any] = {}
        for k, a in arrays.items():
            entry: dict[str, Any] = {"shape": list(a.shape), "dtype": str(a.dtype)}
            if a.dtype.kind in "fiu" and a.size:
                af = a.astype(float)
                entry.update(
                    nan_count=int(np.isnan(af).sum()),
                    nansum=float(np.nansum(af)),
                    nanabssum=float(np.nansum(np.abs(af))),
                    nanmin=float(np.nanmin(af)) if not np.isnan(af).all() else None,
                    nanmax=float(np.nanmax(af)) if not np.isnan(af).all() else None,
                    head=np.nan_to_num(af.ravel()[:16], nan=-9999.0).tolist(),
                )
            summary[k] = entry
        p.with_name(p.name + ".summary.json").write_text(
            json.dumps(summary, indent=1), encoding="utf-8"
        )


def test_surrogate_sample_lock(tmp_path: Path, update_lock: bool) -> None:
    out = tmp_path / "out"
    out.mkdir()
    argv = ["surrogate", "sample", STUDY.as_posix(), "--root", out.as_posix()]
    argv += ["--axle", "front", "--n", "8", "--workers", "1"]
    res = CliRunner().invoke(app, argv)
    assert res.exit_code == 0, res.output
    _summarise_arrays(out)
    snapshot_dir(
        "surrogate_sample", out, res.output, res.exit_code, update=update_lock, tmp=tmp_path
    )
