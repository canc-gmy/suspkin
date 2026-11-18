"""``/api/v1`` routes. Handlers are sync, so FastAPI runs them in its threadpool."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from suspkin import __version__
from suspkin.app.datasets import Dataset, DatasetManager, RunInfo, clean_json
from suspkin.app.models import (
    BrowseEntry,
    BrowseOut,
    DatasetOut,
    DatasetPatch,
    ExportOut,
    ExportRequest,
    HealthOut,
    HistoryOut,
    OpenRequest,
    PoseInput,
    TableOut,
)
from suspkin.kinematics import KinematicsError, solve_vehicle
from suspkin.plotting.optimization import best_so_far, knee_index, pareto_mask
from suspkin.plotting.spec import spec_json
from suspkin.units import display_factor, display_unit

__all__ = ["make_router"]


def _dataset_out(ds: Dataset) -> DatasetOut:
    return DatasetOut(
        id=ds.id,
        label=ds.label,
        kind=ds.kind,
        source=ds.source,
        color=ds.color,
        studies=list(ds.tables),
        run_id=ds.run_id,
        params=ds.params,
        steps=ds.steps,
        resolution_fixed=ds.resolution_fixed,
    )


def _column(s: pd.Series[Any]) -> list[Any]:
    """JSON list of a column: non-finite floats become None."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return [v if math.isfinite(v) else None for v in s.astype(float).tolist()]
    return [None if (isinstance(v, float) and math.isnan(v)) else v for v in s.tolist()]


def analyse_history(
    history: pd.DataFrame,
) -> tuple[list[str], list[float | None], list[bool], dict[str, list[bool]], int | None]:
    """Objectives, best-so-far, Pareto masks (all objectives, per pair) and knee ``eval_id``.

    Masks are per history row over the feasible rows with finite objectives. The knee is the
    knee of the all-objective front.
    """
    n = len(history)
    objs = [str(c) for c in history.columns if str(c).startswith("f_") and c != "f_total"]
    if n == 0 or "f_total" not in history.columns or "feasible" not in history.columns:
        return objs, [None] * n, [False] * n, {}, None
    feas = history["feasible"].to_numpy(dtype=bool)
    order = np.argsort(history["eval_id"].to_numpy(dtype=float), kind="stable")
    ft = history["f_total"].to_numpy(dtype=float)
    best = np.full(n, np.nan)
    best[order] = best_so_far(ft[order], feas[order])
    best_out = [float(v) if math.isfinite(v) else None for v in best]
    mask = np.zeros(n, dtype=bool)
    pairs: dict[str, list[bool]] = {}
    knee: int | None = None
    if len(objs) >= 2:
        vals = history[objs].to_numpy(dtype=float)
        ok = feas & np.all(np.isfinite(vals), axis=1)
        idx = np.flatnonzero(ok)
        if idx.size:
            mask[idx[pareto_mask(vals[idx])]] = True
            front = np.flatnonzero(mask)
            knee = int(history["eval_id"].iloc[int(front[knee_index(vals[front])])])
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                m2 = np.zeros(n, dtype=bool)
                if idx.size:
                    m2[idx[pareto_mask(vals[idx][:, [i, j]])]] = True
                pairs[f"{objs[i]}|{objs[j]}"] = [bool(v) for v in m2]
    return objs, best_out, [bool(v) for v in mask], pairs, knee


def pose_inputs() -> list[PoseInput]:
    """Pose input metadata: names, units and conversions come from ``suspkin.units``."""
    length = display_factor("length")
    angle = display_factor("angle")
    lu, au = display_unit("length"), display_unit("angle")
    return [
        PoseInput(name="heave", label="Heave", si_unit="m", display_unit=lu, factor=length,
                  param="heave", param_factor=length, default_max=25.0, step=0.5),
        PoseInput(name="roll", label="Roll", si_unit="rad", display_unit=au, factor=angle,
                  param="roll_deg", param_factor=1.0, default_max=2.0, step=0.1),
        PoseInput(name="pitch", label="Pitch", si_unit="rad", display_unit=au, factor=angle,
                  param="pitch_deg", param_factor=1.0, default_max=1.5, step=0.1),
        PoseInput(name="rack", label="Rack", si_unit="m", display_unit=lu, factor=length,
                  param="rack", param_factor=length, default_max=30.0, step=0.5),
    ]  # fmt: skip


def health_out(manager: DatasetManager) -> HealthOut:
    """The ``/health`` payload (also embedded in snapshots)."""
    return HealthOut(version=__version__, workspace=str(manager.workspace), pose=pose_inputs())


def history_out(run: RunInfo) -> HistoryOut:
    """The history response of an opened optimizer run (columns plus the analysis)."""
    objectives, best, front, pairs, knee = analyse_history(run.history)
    return HistoryOut(
        run_id=run.run_id,
        columns={str(c): _column(run.history[c]) for c in run.history.columns},
        meta=clean_json(run.meta),
        result=clean_json(run.result),
        objectives=objectives,
        best_so_far=best,
        pareto_all=front,
        pareto_pairs=pairs,
        knee_eval_id=knee,
    )


def scene_json(ds: Dataset) -> dict[str, Any]:
    """Plotly 3D figure JSON of a dataset at the static pose (raises ImportError without plotly)."""
    from suspkin.plotting.interactive import interactive_figure

    out: dict[str, Any] = json.loads(interactive_figure(ds.vehicle).to_json())
    return out


def make_router(manager: DatasetManager) -> APIRouter:
    """Build the ``/api/v1`` router bound to ``manager``."""
    r = APIRouter(prefix="/api/v1")

    @r.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        return health_out(manager)

    @r.get("/figures", response_model=dict[str, Any])
    def figures() -> dict[str, Any]:
        return spec_json()

    @r.get("/datasets", response_model=list[DatasetOut])
    def list_datasets() -> list[DatasetOut]:
        return [_dataset_out(d) for d in manager.list()]

    @r.post("/datasets", response_model=list[DatasetOut])
    def open_dataset(req: OpenRequest) -> list[DatasetOut]:
        return [_dataset_out(d) for d in manager.open(req.path)]

    @r.patch("/datasets/{dataset_id}", response_model=DatasetOut)
    def patch_dataset(dataset_id: str, patch: DatasetPatch) -> DatasetOut:
        ds = manager.get(dataset_id)
        if patch.steps is not None and ds.resolution_fixed:
            raise HTTPException(
                400, "the resolution of bundles and runs is stored; it cannot change"
            )
        if patch.label is not None:
            manager.rename(dataset_id, patch.label)
        if patch.color is not None:
            manager.recolor(dataset_id, patch.color)
        if patch.steps is not None:
            manager.set_steps(dataset_id, patch.steps)
        return _dataset_out(manager.get(dataset_id))

    @r.delete("/datasets/{dataset_id}", status_code=204)
    def delete_dataset(dataset_id: str) -> None:
        manager.close(dataset_id)

    @r.get("/datasets/{dataset_id}/tables/{study}", response_model=TableOut)
    def table(dataset_id: str, study: str) -> TableOut:
        ds = manager.get(dataset_id)
        df = ds.tables.get(study)
        if df is None:
            raise HTTPException(404, f"dataset {ds.label!r} has no study {study!r}")
        return TableOut(columns={str(c): _column(df[c]) for c in df.columns})

    @r.get("/datasets/{dataset_id}/summary", response_model=dict[str, Any])
    def summary(dataset_id: str) -> dict[str, Any]:
        return manager.get(dataset_id).summary

    @r.get("/datasets/{dataset_id}/scene", response_model=dict[str, Any])
    def scene(
        dataset_id: str,
        heave: Annotated[float, Query(ge=-0.2, le=0.2, allow_inf_nan=False)] = 0.0,
        roll: Annotated[float, Query(ge=-0.35, le=0.35, allow_inf_nan=False)] = 0.0,
        pitch: Annotated[float, Query(ge=-0.35, le=0.35, allow_inf_nan=False)] = 0.0,
        rack: Annotated[float, Query(ge=-0.2, le=0.2, allow_inf_nan=False)] = 0.0,
    ) -> dict[str, Any]:
        """Plotly 3D figure at the solved body pose (SI: m, rad; rotations about the CG)."""
        from suspkin.plotting.interactive import interactive_figure

        ds = manager.get(dataset_id)
        state = None
        if any((heave, roll, pitch, rack)):
            try:
                state = solve_vehicle(ds.vehicle, heave=heave, roll=roll, pitch=pitch, rack=rack)
            except (KinematicsError, ValueError, ArithmeticError) as exc:
                raise HTTPException(422, f"pose not solvable: {exc}") from exc
        try:
            fig = interactive_figure(ds.vehicle, state)
        except ImportError as exc:
            raise HTTPException(501, str(exc)) from exc
        out: dict[str, Any] = json.loads(fig.to_json())
        return out

    @r.get("/runs/{run_id}/history", response_model=HistoryOut)
    def history(run_id: str) -> HistoryOut:
        return history_out(manager.run(run_id))

    @r.post("/export", response_model=ExportOut)
    def export(req: ExportRequest | None = None) -> ExportOut:
        """Write a snapshot of the open datasets into the workspace."""
        import time

        from suspkin.app.snapshot import export_dataset_snapshot, is_snapshot_file

        out = (req.out if req else None) or time.strftime("suspkin_snapshot_%Y%m%d_%H%M%S.html")
        target = manager.resolve(out)
        if target.suffix.lower() != ".html":
            raise HTTPException(400, "the snapshot file name must end in .html")
        if not manager.list():
            raise HTTPException(400, "open at least one revision before exporting")
        if target.exists():
            if not target.is_file() or not is_snapshot_file(target):
                raise HTTPException(
                    400,
                    f"{manager.rel(target)} exists and is not a suspkin snapshot; "
                    "choose another file name",
                )
            if not (req and req.overwrite):
                raise HTTPException(409, f"already exists: {manager.rel(target)}")
        size = export_dataset_snapshot(manager, target)
        return ExportOut(path=manager.rel(target), size=size)

    @r.get("/browse", response_model=BrowseOut)
    def browse(dir: str = "") -> BrowseOut:
        rel, entries = manager.browse(dir)
        parent = None if rel == "" else rel.rpartition("/")[0]
        return BrowseOut(
            dir=rel,
            parent=parent,
            entries=[BrowseEntry(name=n, path=p, kind=k) for n, p, k in entries],
        )

    return r
