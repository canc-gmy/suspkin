"""Result bundle: a portable, versioned folder of one design's study results.

A bundle is the data contract between the Python package and the web app. Layout::

    <bundle>/
        manifest.json   schema, kind, label, suspkin_version, created, source, params
        vehicle.yaml    the vehicle (``save_vehicle``)
        tables/<study>.csv   SI units, floats as ``%.17g`` (lossless), LF line endings
        summary.json    ``VehicleSummary`` as a nested dict (SI, nan -> null)

All files use LF line endings. nan cells of a table are empty CSV fields (read back as
nan).

``params`` are the arguments of ``run_standard_studies`` (``travel``, ``rack``,
``roll_deg``, ``heave``, ``pitch_deg``, ``steps``). Tables of an empty study (the rack of
a vehicle without steering) are not written. :func:`load_bundle` returns a
``Bundle`` with the tables, the summary dict (null values stay ``None``) and the vehicle.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from suspkin import __version__
from suspkin.geometry import Vehicle
from suspkin.geometry.io import VehicleConfigError, load_vehicle, save_vehicle
from suspkin.metrics.vehicle_summary import AxleSummary, VehicleSummary
from suspkin.report.studies import StudyResults

__all__ = [
    "BUNDLE_KIND",
    "BUNDLE_SCHEMA",
    "Bundle",
    "BundleError",
    "is_bundle",
    "json_safe",
    "load_bundle",
    "save_bundle",
]

BUNDLE_SCHEMA = 1
BUNDLE_KIND = "suspkin-bundle"
_STUDIES = ("front_corner", "rear_corner", "rack", "heave", "roll", "pitch")


class BundleError(ValueError):
    """Not a bundle, or a bundle of an unsupported schema."""


@dataclass(frozen=True)
class Bundle:
    """A loaded result bundle (tables and summary in SI)."""

    label: str
    manifest: dict[str, Any]
    vehicle: Vehicle
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]


def json_safe(obj: Any) -> Any:
    """JSON-safe copy: recurses into mappings, lists, tuples and arrays; numpy scalars
    become Python scalars; nan/inf become null."""
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def save_bundle(
    results: StudyResults,
    out_dir: str | Path,
    *,
    label: str,
    source: str | None = None,
    params: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    created: str | None = None,
) -> Path:
    """Write ``results`` as a bundle into ``out_dir`` (created) and return ``out_dir``.

    ``params`` defaults to ``results.params``; ``source`` is free text (e.g. the vehicle
    YAML path) kept in the manifest; ``created`` is the ISO timestamp of the manifest
    (default: now, UTC; inject it for reproducible bundles).

    The bundle is written into a sibling temporary folder and then moved into place, so a
    failure never leaves a half-written bundle (nor stale tables of a previous one).
    An existing non-empty ``out_dir`` raises :class:`BundleError` unless ``overwrite``;
    even then only an existing bundle is replaced, never an unrelated folder. Replacing is
    "rename old aside, rename new into place, delete old": on Windows (no atomic directory
    replace) a crash between the two renames leaves the old bundle in a ``.<name>.old-*``
    sibling folder.
    """
    out = Path(out_dir)
    if out.exists():
        if not out.is_dir():
            raise BundleError(f"{out} exists and is not a folder")
        if any(out.iterdir()):
            if not overwrite:
                raise BundleError(f"{out} is not empty (pass overwrite=True to replace it)")
            if not is_bundle(out):
                raise BundleError(f"refusing to overwrite {out}: not a suspkin bundle")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.new-", dir=out.parent))
    old: Path | None = None
    try:
        _write_bundle(results, tmp, label, source, params, created)
        if out.exists():
            old = Path(tempfile.mkdtemp(prefix=f".{out.name}.old-", dir=out.parent))
            os.rmdir(old)  # only the unique name is wanted
            os.replace(out, old)
        try:
            os.replace(tmp, out)
        except BaseException:
            if old is not None:
                os.replace(old, out)  # restore the previous bundle
                old = None
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    return out


def _write_bundle(
    results: StudyResults,
    out: Path,
    label: str,
    source: str | None,
    params: Mapping[str, Any] | None,
    created: str | None,
) -> None:
    (out / "tables").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "kind": BUNDLE_KIND,
        "label": label,
        "suspkin_version": __version__,
        "created": created or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source": source,
        "params": json_safe(dict(results.params if params is None else params)),
    }
    save_vehicle(results.vehicle, out / "vehicle.yaml")
    for name, df in results.tables().items():
        df.to_csv(
            out / "tables" / f"{name}.csv",
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
    _write_json(out / "summary.json", json_safe(results.summary.to_dict()))
    _write_json(out / "manifest.json", manifest)


def is_bundle(path: str | Path) -> bool:
    """True if ``path`` is a folder with a suspkin bundle manifest."""
    mf = Path(path) / "manifest.json"
    if not mf.is_file():
        return False
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("kind") == BUNDLE_KIND


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"{path}: cannot read {what}: {exc}") from exc


def load_bundle(path: str | Path) -> Bundle:
    """Read a bundle folder.

    Raises
    ------
    BundleError
        If ``path`` is not a bundle, its schema is not :data:`BUNDLE_SCHEMA`, or one of its
        files (``manifest.json``, ``vehicle.yaml``, ``summary.json``, a table CSV) is
        missing or malformed; the message includes the offending path.
    """
    root = Path(path)
    if not is_bundle(root):
        raise BundleError(f"{root} is not a suspkin bundle (no valid manifest.json)")
    manifest = _read_json(root / "manifest.json", "manifest")
    schema = manifest.get("schema")
    if schema != BUNDLE_SCHEMA:
        raise BundleError(
            f"unsupported bundle schema {schema!r} in {root} (this suspkin reads {BUNDLE_SCHEMA})"
        )
    tables: dict[str, pd.DataFrame] = {}
    for p in sorted((root / "tables").glob("*.csv")):
        if p.stem not in _STUDIES:
            continue
        try:
            tables[p.stem] = pd.read_csv(p)
        except (OSError, ValueError, pd.errors.ParserError) as exc:  # EmptyDataError is ValueError
            raise BundleError(f"{p}: cannot read table: {exc}") from exc
    try:
        vehicle = load_vehicle(root / "vehicle.yaml")
    except VehicleConfigError as exc:
        raise BundleError(f"{root / 'vehicle.yaml'}: invalid vehicle: {exc}") from exc
    summary = _read_json(root / "summary.json", "summary")
    try:
        _summary_from_dict(summary)
    except (AttributeError, KeyError, TypeError) as exc:
        raise BundleError(f"{root / 'summary.json'}: malformed summary: {exc!r}") from exc
    return Bundle(
        label=str(manifest.get("label", root.name)),
        manifest=manifest,
        vehicle=vehicle,
        tables=tables,
        summary=summary,
    )


def _nan(v: Any) -> Any:
    return float("nan") if v is None else v


def _summary_from_dict(d: Mapping[str, Any]) -> VehicleSummary:
    own = {
        k: (v if k == "anti_lift_front_accel" else _nan(v))  # the one optional field
        for k, v in d.items()
        if k not in ("front", "rear")
    }
    axles = {ax: AxleSummary(**{k: _nan(v) for k, v in d[ax].items()}) for ax in ("front", "rear")}
    return VehicleSummary(front=axles["front"], rear=axles["rear"], **own)
