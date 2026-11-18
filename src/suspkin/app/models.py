"""Pydantic response/request models of the ``/api/v1`` API (precise OpenAPI schema)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DatasetKind = Literal["vehicle", "bundle", "run"]
EntryKind = Literal["dir", "vehicle", "bundle", "report", "run", "other"]


class PoseInput(BaseModel):
    """One body-pose input of the 3D view: SI unit, display unit and how to convert."""

    name: Literal["heave", "roll", "pitch", "rack"]
    label: str
    si_unit: str
    display_unit: str
    factor: float = Field(description="Display value = SI value x factor.")
    param: str = Field(description="Dataset study parameter that sets the slider range.")
    param_factor: float = Field(description="Display value of the parameter = parameter x this.")
    default_max: float = Field(description="Range (+-, display unit) when the parameter is unset.")
    step: float = Field(description="Slider step in display units.")


class HealthOut(BaseModel):
    """Server liveness, package version and workspace root."""

    status: str = "ok"
    version: str
    workspace: str = Field(description="Absolute path of the workspace root.")
    pose: list[PoseInput] = Field(default_factory=list, description="3D pose inputs.")


class DatasetOut(BaseModel):
    """Summary metadata of one open dataset (no tables)."""

    id: str
    label: str
    kind: DatasetKind
    source: str = Field(description="Workspace-relative path of the opened file or folder.")
    color: int = Field(description="Colour index into the dataset palette.")
    studies: list[str] = Field(description="Names of the tables this dataset has.")
    run_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict, description="Study parameters (SI).")
    steps: int = Field(default=31, description="Points per sweep of the solved tables.")
    resolution_fixed: bool = Field(
        default=True,
        description="True for bundles and runs: the stored resolution cannot be changed.",
    )


class OpenRequest(BaseModel):
    """Open a path (absolute, or relative to the workspace)."""

    path: str


class DatasetPatch(BaseModel):
    """Editable dataset attributes."""

    label: str | None = None
    color: int | None = Field(default=None, ge=0)
    steps: int | None = Field(
        default=None,
        ge=3,
        le=401,
        description="Re-solve with this many points per sweep (vehicle files only).",
    )


class TableOut(BaseModel):
    """One study table, columnar, SI units, NaN as null (a few label columns are strings)."""

    columns: dict[str, list[float | str | None]]


class HistoryOut(BaseModel):
    """Optimizer history (columnar) with the run's ``meta.json`` and ``result.json``."""

    run_id: str
    columns: dict[str, list[float | str | bool | None]]
    meta: dict[str, Any]
    result: dict[str, Any]
    objectives: list[str] = Field(
        default_factory=list, description="History columns of the objectives (``f_*``, no total)."
    )
    best_so_far: list[float | None] = Field(
        default_factory=list, description="Running best feasible ``f_total`` per history row."
    )
    pareto_all: list[bool] = Field(
        default_factory=list,
        description="Per history row: feasible and non-dominated in all objectives "
        "(needs >= 2 objectives).",
    )
    pareto_pairs: dict[str, list[bool]] = Field(
        default_factory=dict,
        description="Per objective pair ``'f_a|f_b'`` (objective order): per history row, "
        "feasible and non-dominated in those two objectives only.",
    )
    knee_eval_id: int | None = Field(default=None, description="``eval_id`` of the front knee.")


class ExportRequest(BaseModel):
    """Export the open datasets as a self-contained HTML snapshot."""

    out: str | None = Field(
        default=None, description="Workspace-relative output file (default: timestamped name)."
    )
    overwrite: bool = Field(
        default=False, description="Replace an existing snapshot file (never any other file)."
    )


class ExportOut(BaseModel):
    """Result of a snapshot export."""

    path: str = Field(description="Workspace-relative path of the written file.")
    size: int = Field(description="File size in bytes.")


class BrowseEntry(BaseModel):
    """One directory entry of the file picker."""

    name: str
    path: str = Field(description="Workspace-relative POSIX path.")
    kind: EntryKind


class BrowseOut(BaseModel):
    """Listing of a workspace directory."""

    dir: str = Field(description="Workspace-relative path of the listed directory ('' = root).")
    parent: str | None
    entries: list[BrowseEntry]
