"""Dataset manager: opens vehicle YAMLs, bundles, reports and optimizer runs.

All paths are confined to the workspace. Solving a vehicle YAML is cached in memory by
(sha256 of the file bytes, study parameters). Thread-safe: state is guarded by a lock and
solving happens outside it.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from suspkin.geometry import Vehicle, VehicleConfigError, load_vehicle
from suspkin.optimize.history import read_history
from suspkin.optimize.outputs import read_meta
from suspkin.plotting.spec import derived_columns
from suspkin.report.bundle import BundleError, is_bundle, load_bundle
from suspkin.report.studies import run_standard_studies

__all__ = [
    "Dataset",
    "DatasetError",
    "DatasetManager",
    "DatasetNotFoundError",
    "RunInfo",
    "WorkspaceError",
    "classify",
    "clean_json",
]


class WorkspaceError(PermissionError):
    """A path outside the workspace."""


class DatasetError(ValueError):
    """A path that cannot be opened as a dataset."""


class DatasetNotFoundError(LookupError):
    """Unknown dataset or run id."""


@dataclass
class Dataset:
    """One open dataset. ``tables`` are SI and include the derived columns."""

    id: str
    label: str
    kind: str
    source: str
    color: int
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    vehicle: Vehicle
    params: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None


@dataclass(frozen=True)
class RunInfo:
    """An opened optimizer run folder."""

    run_id: str
    history: pd.DataFrame
    meta: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class _Solved:
    vehicle: Vehicle
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    params: dict[str, Any]


def clean_json(obj: Any) -> Any:
    """JSON-safe copy of nested dicts/lists: non-finite floats become None."""
    if isinstance(obj, Mapping):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _short(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


def _is_run(p: Path) -> bool:
    return (
        p.is_dir()
        and (p / "history.csv").is_file()
        and ((p / "meta.json").is_file() or (p / "result.json").is_file())
    )


def _is_report(p: Path) -> bool:
    return p.is_dir() and (p / "bundles").is_dir()


def _is_vehicle_yaml(p: Path) -> bool:
    if p.suffix.lower() not in (".yaml", ".yml") or not p.is_file():
        return False
    try:
        if p.stat().st_size > 1_000_000:
            return False
        load_vehicle(p)
    except Exception:
        return False
    return True


def classify(p: Path) -> str:
    """Entry kind of ``p``: dir | vehicle | bundle | report | run | other."""
    if p.is_dir():
        if is_bundle(p):
            return "bundle"
        if _is_run(p):
            return "run"
        if _is_report(p):
            return "report"
        return "dir"
    return "vehicle" if _is_vehicle_yaml(p) else "other"


class DatasetManager:
    """Registry of open datasets and optimizer runs."""

    def __init__(self, workspace: Path, *, study_params: Mapping[str, Any] | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.study_params: dict[str, Any] = dict(study_params or {})
        self._lock = threading.RLock()
        self._datasets: dict[str, Dataset] = {}
        self._runs: dict[str, RunInfo] = {}
        self._cache: dict[tuple[str, tuple[tuple[str, Any], ...]], _Solved] = {}

    # ----------------------------------------------------------------------- paths
    def resolve(self, path: str | Path) -> Path:
        """Resolve ``path`` (absolute or workspace-relative); it must be inside the workspace."""
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        p = p.resolve()
        if not p.is_relative_to(self.workspace):
            raise WorkspaceError(f"path is outside the workspace: {path}")
        return p

    def rel(self, p: Path) -> str:
        """Workspace-relative POSIX path ('' for the root)."""
        r = p.resolve().relative_to(self.workspace).as_posix()
        return "" if r == "." else r

    # ------------------------------------------------------------------------ open
    def open(self, path: str | Path) -> list[Dataset]:
        """Open ``path`` (auto-detected) and return its datasets (new or already open)."""
        p = self.resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"no such file or folder: {self.rel(p)}")
        if p.is_dir():
            if is_bundle(p):
                return [self._open_bundle(p)]
            if _is_run(p):
                return self._open_run(p)
            if _is_report(p):
                out = [
                    self._open_bundle(b) for b in sorted((p / "bundles").iterdir()) if is_bundle(b)
                ]
                if not out:
                    raise DatasetError(f"report folder has no bundles: {self.rel(p)}")
                return out
            raise DatasetError(f"not a bundle, report or optimizer run folder: {self.rel(p)}")
        return [self._open_vehicle(p)]

    def _solve(self, vehicle_path: Path) -> _Solved:
        data = vehicle_path.read_bytes()
        key = (hashlib.sha256(data).hexdigest(), tuple(sorted(self.study_params.items())))
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        try:
            veh = load_vehicle(vehicle_path)
        except VehicleConfigError as exc:
            raise DatasetError(f"{self.rel(vehicle_path)} is not a vehicle: {exc}") from exc
        res = run_standard_studies(veh, **self.study_params)
        solved = _Solved(
            vehicle=veh,
            tables={k: derived_columns(k, v) for k, v in res.tables().items()},
            summary=clean_json(res.summary.to_dict()),
            params=clean_json(dict(res.params)),
        )
        with self._lock:
            self._cache[key] = solved
        return solved

    def _register(
        self,
        *,
        kind: str,
        path: Path,
        label: str,
        solved: _Solved,
        run_id: str | None = None,
        content: str = "",
    ) -> Dataset:
        did = _short(kind, str(path), content)
        with self._lock:
            if did in self._datasets:
                return self._datasets[did]
            used_labels = {d.label for d in self._datasets.values()}
            lab, k = label, 1
            while lab in used_labels:
                k += 1
                lab = f"{label} ({k})"
            used_colors = {d.color for d in self._datasets.values()}
            color = next(i for i in range(len(used_colors) + 1) if i not in used_colors)
            ds = Dataset(
                id=did,
                label=lab,
                kind=kind,
                source=self.rel(path),
                color=color,
                tables=solved.tables,
                summary=solved.summary,
                vehicle=solved.vehicle,
                params=solved.params,
                run_id=run_id,
            )
            self._datasets[did] = ds
            return ds

    def _open_vehicle(
        self, p: Path, *, label: str | None = None, kind: str = "vehicle", run_id: str | None = None
    ) -> Dataset:
        solved = self._solve(p)
        content = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        return self._register(
            kind=kind, path=p, label=label or p.stem, solved=solved, run_id=run_id, content=content
        )

    def _open_bundle(self, p: Path) -> Dataset:
        try:
            b = load_bundle(p)
        except (BundleError, OSError, ValueError, VehicleConfigError) as exc:
            raise DatasetError(f"cannot read bundle {self.rel(p)}: {exc}") from exc
        solved = _Solved(
            vehicle=b.vehicle,
            tables={k: derived_columns(k, v) for k, v in b.tables.items()},
            summary=clean_json(b.summary),
            params=clean_json(dict(b.manifest.get("params") or {})),
        )
        mtime = str((p / "manifest.json").stat().st_mtime_ns)
        return self._register(kind="bundle", path=p, label=b.label, solved=solved, content=mtime)

    def _open_run(self, p: Path) -> list[Dataset]:
        run_id = _short("run", str(p))
        meta = read_meta(p)
        try:
            result = json.loads((p / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = {}
        if not isinstance(result, dict):
            result = {}
        history = read_history(p / "history.csv")
        items: list[tuple[str, Path]] = []
        if (p / "vehicle.yaml").is_file():
            items.append(("baseline", p / "vehicle.yaml"))
        if (p / "best.yaml").is_file():
            items.append(("best", p / "best.yaml"))
        knee = self._knee_path(p, result)
        if knee is not None:
            items.append(("knee", knee))
        if not items:
            raise DatasetError(f"run folder has no vehicle.yaml/best.yaml: {self.rel(p)}")
        with self._lock:
            self._runs[run_id] = RunInfo(run_id, history, meta, result)
        return [
            self._open_vehicle(q, label=f"{p.name} {lab}", kind="run", run_id=run_id)
            for lab, q in items
        ]

    @staticmethod
    def _knee_path(p: Path, result: dict[str, Any]) -> Path | None:
        if result.get("knee_eval_id") is None:
            return None
        tab = p / "pareto.csv"
        if not tab.is_file():
            return None
        try:
            df = pd.read_csv(tab)
            k = int(df.loc[df["knee"].astype(bool), "k"].iloc[0])
        except (OSError, ValueError, KeyError, IndexError):
            return None
        q = p / "pareto" / f"pareto_{k}.yaml"
        return q if q.is_file() else None

    # ------------------------------------------------------------------- registry
    def list(self) -> list[Dataset]:
        """All open datasets in opening order."""
        with self._lock:
            return list(self._datasets.values())

    def get(self, dataset_id: str) -> Dataset:
        """Dataset by id (raises ``DatasetNotFoundError``)."""
        with self._lock:
            try:
                return self._datasets[dataset_id]
            except KeyError:
                raise DatasetNotFoundError(f"unknown dataset: {dataset_id}") from None

    def rename(self, dataset_id: str, label: str) -> Dataset:
        """Set a unique, non-empty label."""
        label = label.strip()
        if not label:
            raise DatasetError("label must not be empty")
        with self._lock:
            ds = self.get(dataset_id)
            if any(d.label == label for d in self._datasets.values() if d.id != dataset_id):
                raise DatasetError(f"label already in use: {label}")
            ds.label = label
            return ds

    def recolor(self, dataset_id: str, color: int) -> Dataset:
        """Set the colour index."""
        with self._lock:
            ds = self.get(dataset_id)
            ds.color = color
            return ds

    def close(self, dataset_id: str) -> None:
        """Close a dataset (and its run once no dataset uses it)."""
        with self._lock:
            self.get(dataset_id)
            ds = self._datasets.pop(dataset_id)
            if ds.run_id and not any(d.run_id == ds.run_id for d in self._datasets.values()):
                self._runs.pop(ds.run_id, None)

    def run(self, run_id: str) -> RunInfo:
        """Opened optimizer run by id."""
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError:
                raise DatasetNotFoundError(f"unknown run: {run_id}") from None

    # --------------------------------------------------------------------- browse
    def browse(self, directory: str | Path = "") -> tuple[str, builtins.list[tuple[str, str, str]]]:
        """List a workspace directory: ``(rel dir, [(name, rel path, kind)])``."""
        d = self.resolve(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"not a directory: {self.rel(d)}")
        entries = []
        for q in sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if q.name.startswith("."):
                continue
            try:
                inside = q.resolve().is_relative_to(self.workspace)
            except OSError:  # pragma: no cover
                continue
            if inside:
                entries.append((q.name, self.rel(q), classify(q)))
        return self.rel(d), entries
