"""``/api/v1`` routes. Handlers are sync, so FastAPI runs them in its threadpool."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from suspkin import __version__
from suspkin.app.datasets import Dataset, DatasetManager
from suspkin.app.models import (
    BrowseEntry,
    BrowseOut,
    DatasetOut,
    DatasetPatch,
    HealthOut,
    HistoryOut,
    OpenRequest,
    TableOut,
)
from suspkin.kinematics import KinematicsError, solve_vehicle
from suspkin.plotting.spec import spec_json

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
    )


def _column(s: pd.Series[Any]) -> list[Any]:
    """JSON list of a column: non-finite floats become None."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return [v if math.isfinite(v) else None for v in s.astype(float).tolist()]
    return [None if (isinstance(v, float) and math.isnan(v)) else v for v in s.tolist()]


def make_router(manager: DatasetManager) -> APIRouter:
    """Build the ``/api/v1`` router bound to ``manager``."""
    r = APIRouter(prefix="/api/v1")

    @r.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        return HealthOut(version=__version__)

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
        manager.get(dataset_id)
        if patch.label is not None:
            manager.rename(dataset_id, patch.label)
        if patch.color is not None:
            manager.recolor(dataset_id, patch.color)
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
        heave: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        rack: float = 0.0,
    ) -> dict[str, Any]:
        """Plotly 3D figure at the solved body pose (SI: m, rad; rotations about the CG)."""
        from suspkin.plotting.interactive import interactive_figure

        ds = manager.get(dataset_id)
        state = None
        if any((heave, roll, pitch, rack)):
            try:
                state = solve_vehicle(ds.vehicle, heave=heave, roll=roll, pitch=pitch, rack=rack)
            except KinematicsError as exc:
                raise HTTPException(422, f"pose not solvable: {exc}") from exc
        try:
            fig = interactive_figure(ds.vehicle, state)
        except ImportError as exc:
            raise HTTPException(501, str(exc)) from exc
        out: dict[str, Any] = json.loads(fig.to_json())
        return out

    @r.get("/runs/{run_id}/history", response_model=HistoryOut)
    def history(run_id: str) -> HistoryOut:
        run = manager.run(run_id)
        cols = {str(c): _column(run.history[c]) for c in run.history.columns}
        return HistoryOut(run_id=run_id, columns=cols, meta=run.meta, result=run.result)

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
