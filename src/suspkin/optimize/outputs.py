"""Run folder writer: best / Pareto vehicles, result.json, meta.json, figures, report."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import suspkin
from suspkin.geometry.io import save_vehicle
from suspkin.optimize.design import DesignSpace
from suspkin.optimize.evaluate import Evaluation
from suspkin.optimize.history import objective_names
from suspkin.optimize.spec import Study
from suspkin.plotting.optimization import knee_index, pareto_mask

F64 = NDArray[np.float64]

__all__ = [
    "best_row",
    "git_hash",
    "pareto_rows",
    "ranked_rows",
    "write_outputs",
]


# --------------------------------------------------------------------------- selection


def ranked_rows(history: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Row positions ordered best first: feasible by ``f_total``, then infeasible by ``max_g``."""
    feas = history["feasible"].to_numpy(dtype=bool)
    ft = np.nan_to_num(history["f_total"].to_numpy(dtype=float), nan=np.inf)
    mg = np.nan_to_num(history["max_g"].to_numpy(dtype=float), nan=np.inf)
    # lexsort: last key is primary
    return np.lexsort((np.where(feas, ft, mg), ~feas))


def best_row(history: pd.DataFrame) -> int:
    """Row position of the best point: feasible with minimal ``f_total``, else minimal ``max_g``."""
    if len(history) == 0:
        raise ValueError("empty history")
    return int(ranked_rows(history)[0])


def pareto_rows(history: pd.DataFrame, study: Study) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Row positions of the feasible non-dominated, x-unique designs (objective columns ``f_*``)."""
    cols = [f"f_{n}" for n in objective_names(study)]
    xcols = [c for c in history.columns if c.startswith("x_")]
    feas = np.flatnonzero(history["feasible"].to_numpy(dtype=bool))
    if feas.size == 0:
        return np.array([], dtype=np.intp)
    objs = history[cols].to_numpy(dtype=float)[feas]
    keep = feas[pareto_mask(objs)]
    sub = history.iloc[keep]
    uniq = ~sub.duplicated(subset=xcols).to_numpy()
    return np.asarray(keep[uniq], dtype=np.intp)


# --------------------------------------------------------------------------- helpers


def git_hash(cwd: Path) -> str | None:
    """``git rev-parse HEAD`` in ``cwd``, or None when git or the repository is missing."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = res.stdout.strip()
    return out if res.returncode == 0 and out else None


def _clean(v: Any) -> Any:
    """Make a structure JSON safe: numpy -> python, nan/inf -> None."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, np.generic):
        return _clean(v.item())
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return v


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_clean(r) for r in df.to_dict("records")]


def _eval_summary(ev: Evaluation, space: DesignSpace, x: F64 | None) -> dict[str, Any]:
    return {
        "x": None if x is None else dict(zip(space.names, map(float, x), strict=True)),
        "feasible": ev.feasible,
        "f_total": ev.f_total,
        "f": {o.name: o.f for o in ev.objectives},
        "g": {r.name: r.g for r in ev.results if r.status != "disabled"},
        "constraints": _records(ev.frame()),
        "objectives": _records(ev.objectives_frame()),
    }


# --------------------------------------------------------------------------- writer


def write_outputs(
    out_dir: Path,
    *,
    study: Study,
    space: DesignSpace,
    history: pd.DataFrame,
    baseline: Evaluation,
    best_x: F64,
    best_eval: Evaluation,
    best_eval_id: int | None,
    pareto_idx: np.ndarray[Any, np.dtype[np.intp]] | None,
    meta: dict[str, Any],
    report: bool,
) -> Path | None:
    """Write ``best.yaml``, Pareto files, ``result.json``, ``meta.json``, figures and report.

    ``pareto_idx`` is None outside Pareto mode. Returns the path of ``report.html`` (or None).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    v0 = space.baseline
    v_best = space.to_vehicle(best_x)
    save_vehicle(v_best, out_dir / "best.yaml")

    # ---- Pareto set ------------------------------------------------------------------
    knee_id: int | None = None
    if pareto_idx is not None and pareto_idx.size:
        pdir = out_dir / "pareto"
        pdir.mkdir(exist_ok=True)
        cols = [f"f_{n}" for n in objective_names(study)]
        sub = history.iloc[pareto_idx].reset_index(drop=True)
        k = knee_index(sub[cols].to_numpy(dtype=float))
        knee_id = int(sub["eval_id"].iloc[k])
        xcols = [f"x_{n}" for n in space.names]
        for i in range(len(sub)):
            save_vehicle(
                space.to_vehicle(sub.loc[i, xcols].to_numpy(dtype=float)), pdir / f"pareto_{i}.yaml"
            )
        tab = sub[["eval_id", *xcols, *cols, "f_total"]].copy()
        tab.insert(0, "k", np.arange(len(sub)))
        tab["knee"] = np.arange(len(sub)) == k
        tab.to_csv(out_dir / "pareto.csv", index=False)

    # ---- result.json / meta.json -----------------------------------------------------
    result: dict[str, Any] = {
        "mode": meta.get("mode"),
        "best_eval_id": best_eval_id,
        "knee_eval_id": knee_id,
        "n_pareto": 0 if pareto_idx is None else int(pareto_idx.size),
        "baseline": _eval_summary(baseline, space, None),
        "best": _eval_summary(best_eval, space, best_x),
    }
    (out_dir / "result.json").write_text(json.dumps(_clean(result), indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(_clean(meta), indent=2), encoding="utf-8")

    # ---- figures ---------------------------------------------------------------------
    from suspkin.plotting import save_figure
    from suspkin.plotting.optimization import (
        plot_constraint_margins,
        plot_convergence,
        plot_parallel_coordinates,
        plot_pareto,
    )

    fdir = out_dir / "figures"
    fmt = ("png",)
    if len(history):
        save_figure(plot_convergence(history), fdir / "convergence", fmt)
        if pareto_idx is not None and len(study.objectives) >= 2:
            mask = np.zeros(len(history), dtype=bool)
            mask[pareto_idx] = True
            save_figure(plot_pareto(history, front_mask=mask), fdir / "pareto", fmt)
        hl = None if best_eval_id is None else int(best_eval_id)
        save_figure(plot_parallel_coordinates(history, highlight=hl), fdir / "parallel", fmt)
    if len(baseline.results):
        save_figure(
            plot_constraint_margins(baseline.frame(), best_eval.frame()),
            fdir / "constraint_margins",
            fmt,
        )

    # ---- report ----------------------------------------------------------------------
    if not report:
        return None
    from suspkin.report.builder import build_report
    from suspkin.report.studies import run_standard_studies

    return build_report(
        [
            ("baseline", run_standard_studies(v0)),
            ("optimized", run_standard_studies(v_best)),
        ],
        out_dir / "report",
        include_3d=False,
    )


def copy_inputs(study_path: Path, vehicle_path: Path, out_dir: Path) -> None:
    """Copy the study and vehicle into the run folder; the study then points at ``vehicle.yaml``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_study, dst_vehicle = out_dir / "study.yaml", out_dir / "vehicle.yaml"
    text = study_path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^vehicle:.*$", "vehicle: vehicle.yaml", text, count=1)
    if vehicle_path.resolve() != dst_vehicle.resolve():
        shutil.copyfile(vehicle_path, dst_vehicle)
    if study_path.resolve() != dst_study.resolve():
        dst_study.write_text(text, encoding="utf-8")


def make_meta(
    *,
    study_path: Path,
    seed: int,
    method: str,
    mode: str,
    started: datetime,
    ended: datetime,
    n_evals: int,
    n_total: int,
    workers: int,
    resumed: bool,
) -> dict[str, Any]:
    """Assemble ``meta.json`` content."""
    return {
        "suspkin_version": suspkin.__version__,
        "git_hash": git_hash(study_path.resolve().parent),
        "seed": seed,
        "method": method,
        "mode": mode,
        "start": started.isoformat(timespec="seconds"),
        "end": ended.isoformat(timespec="seconds"),
        "n_evals": n_evals,
        "n_evals_total": n_total,
        "workers": workers,
        "resumed": resumed,
    }
