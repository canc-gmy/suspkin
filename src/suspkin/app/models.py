"""Pydantic response/request models of the ``/api/v1`` API (precise OpenAPI schema)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DatasetKind = Literal["vehicle", "bundle", "report", "run"]
EntryKind = Literal["dir", "vehicle", "bundle", "report", "run", "other"]


class HealthOut(BaseModel):
    """Server liveness, package version and workspace root."""

    status: str = "ok"
    version: str
    workspace: str = Field(description="Absolute path of the workspace root.")


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


class OpenRequest(BaseModel):
    """Open a path (absolute, or relative to the workspace)."""

    path: str


class DatasetPatch(BaseModel):
    """Editable dataset attributes."""

    label: str | None = None
    color: int | None = Field(default=None, ge=0)


class TableOut(BaseModel):
    """One study table, columnar, SI units, NaN as null (a few label columns are strings)."""

    columns: dict[str, list[float | str | None]]


class HistoryOut(BaseModel):
    """Optimizer history (columnar) with the run's ``meta.json`` and ``result.json``."""

    run_id: str
    columns: dict[str, list[float | str | bool | None]]
    meta: dict[str, Any]
    result: dict[str, Any]


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
