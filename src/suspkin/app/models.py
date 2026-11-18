"""Pydantic response/request models of the ``/api/v1`` API (precise OpenAPI schema)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DatasetKind = Literal["vehicle", "bundle", "run", "edited"]
EntryKind = Literal["dir", "vehicle", "study", "bundle", "report", "run", "other"]


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


class HardpointEdit(BaseModel):
    """Move one hardpoint (SI). Missing coordinates keep their current value."""

    axle: Literal["front", "rear"]
    point: str
    side: Literal["left", "right", "both"] = Field(
        description="Left point (right mirrors it, a right override is kept), right point "
        "only (written to right_overrides, right-side coordinates), or both (mirrored; a "
        "right override of the point is dropped)."
    )
    x: float | None = Field(default=None, allow_inf_nan=False, description="Metres.")
    y: float | None = Field(default=None, allow_inf_nan=False, description="Metres.")
    z: float | None = Field(default=None, allow_inf_nan=False, description="Metres.")


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
    parent_id: str | None = Field(
        default=None, description="Revision an edited revision was derived from."
    )
    edits: list[HardpointEdit] = Field(
        default_factory=list, description="Hardpoint edits of an edited revision."
    )
    has_study: bool = Field(
        default=False,
        description="True when an optimization study is attached: ``/regions`` has content.",
    )


class HardpointOut(BaseModel):
    """One hardpoint of one corner after mirroring (SI, ISO 8855 vehicle frame)."""

    axle: Literal["front", "rear"]
    point: str
    side: Literal["left", "right"]
    x: float = Field(description="Metres.")
    y: float = Field(description="Metres.")
    z: float = Field(description="Metres.")
    overridden: bool = Field(
        default=False, description="True for a right point taken from ``right_overrides``."
    )


Vec3 = tuple[float, float, float]


class AxleCentres(BaseModel):
    """A point per axle (SI, scene frame); ``None`` when undefined."""

    front: Vec3 | None
    rear: Vec3 | None


class AxleHeights(BaseModel):
    """A ground height per axle in metres; ``None`` when undefined."""

    front: float | None
    rear: float | None


class CornerCentres(BaseModel):
    """A point per corner (SI, scene frame); ``None`` when undefined (centre at infinity)."""

    FL: Vec3 | None
    FR: Vec3 | None
    RL: Vec3 | None
    RR: Vec3 | None


class CornerLengths(BaseModel):
    """A length per corner in metres; ``None`` when undefined."""

    FL: float | None
    FR: float | None
    RL: float | None
    RR: float | None


class SceneCentres(BaseModel):
    """Rotation centres at the scene pose, in the plotted (body) frame, SI.

    Instant centres are ``None`` when at infinity. ``fvsa`` / ``svsa`` are the finite
    distances from the contact patch to the front / side-view instant centre.
    """

    roll: AxleCentres = Field(description="Front-view roll centre per axle (x = mean patch x).")
    roll_height: AxleHeights = Field(description="Ground height of the roll centres.")
    fv_ic: CornerCentres = Field(
        description="Front-view instant centres (in the plane x = patch x)."
    )
    sv_ic: CornerCentres = Field(
        description="Side-view instant centres (in the plane y = patch y)."
    )
    contact: CornerCentres = Field(description="Contact patches, equal to the scene markers.")
    fvsa: CornerLengths
    svsa: CornerLengths
    pitch: Vec3 | None = Field(description="Pitch centre on the centreline (y = 0).")


class RegionOut(BaseModel):
    """One optimization region as wireframe polylines (SI, body frame of the 3D scene).

    Domains are the variable domains (a symmetric variable gives the left one and a
    ``mirrored`` right copy); ``inside`` / ``outside`` are constraint regions, given in the
    vehicle frame and not mirrored. Regions do not move with the body pose.
    """

    name: str = Field(description="Variable point or constraint name.")
    role: Literal["domain", "inside", "outside"]
    point: str = Field(description="Hardpoint reference, e.g. ``front.uca_front``.")
    side: Literal["left", "right"]
    shape: Literal["hexahedron", "box", "offset", "sphere", "cylinder"]
    mirrored: bool = Field(default=False, description="Right copy of a symmetric domain.")
    lines: list[list[Vec3]] = Field(description="Wireframe polylines.")
    over: str | None = Field(default=None, description="Sweep a constraint is evaluated over.")
    note: str | None = None


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


class DeriveRequest(BaseModel):
    """Derive an edited revision from a vehicle-backed one."""

    edits: list[HardpointEdit] = Field(min_length=1)
    label: str | None = Field(default=None, description="Default: '<source label> edited'.")
    steps: int | None = Field(
        default=None, ge=3, le=401, description="Points per sweep (default: the source's)."
    )


class SaveRequest(BaseModel):
    """Write a revision's vehicle to a workspace YAML file."""

    path: str = Field(description="Workspace-relative file name ending in .yaml or .yml.")
    overwrite: bool = False


class SaveOut(BaseModel):
    """Result of saving a vehicle YAML."""

    path: str = Field(description="Workspace-relative path of the written file.")
    size: int = Field(description="File size in bytes.")


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
