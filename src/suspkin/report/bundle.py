"""Result bundle: a portable, versioned folder of one design's study results.

A bundle is the data contract between the Python package and the web app. Layout::

    <bundle>/
        manifest.json   schema, kind, label, suspkin_version, created, source, params
        vehicle.yaml    the vehicle (``save_vehicle``)
        tables/<study>.csv   SI units, floats as ``%.12g``, LF line endings
        summary.json    ``VehicleSummary`` as a nested dict (SI, nan -> null)

``params`` are the arguments of ``run_standard_studies`` (``travel``, ``rack``,
``roll_deg``, ``heave``, ``pitch_deg``, ``steps``). Tables of an empty study (the rack of
a vehicle without steering) are not written. :func:`bundle_to_results` rebuilds a
``StudyResults`` (missing tables become empty frames; ``summary`` is rebuilt as a
``VehicleSummary``, null values becoming nan).
"""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from suspkin import __version__
from suspkin.geometry import Vehicle
from suspkin.geometry.io import load_vehicle, save_vehicle
from suspkin.metrics.vehicle_summary import AxleSummary, VehicleSummary
from suspkin.report.studies import StudyResults

__all__ = [
    "BUNDLE_KIND",
    "BUNDLE_SCHEMA",
    "Bundle",
    "BundleError",
    "bundle_to_results",
    "is_bundle",
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


def _clean(obj: Any) -> Any:
    """JSON-safe copy: nan/inf become null."""
    if isinstance(obj, Mapping):
        return {k: _clean(v) for k, v in obj.items()}
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
) -> Path:
    """Write ``results`` as a bundle into ``out_dir`` (created) and return ``out_dir``.

    ``params`` defaults to ``results.params``; ``source`` is free text (e.g. the vehicle
    YAML path) kept in the manifest.
    """
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "kind": BUNDLE_KIND,
        "label": label,
        "suspkin_version": __version__,
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source": source,
        "params": _clean(dict(results.params if params is None else params)),
    }
    save_vehicle(results.vehicle, out / "vehicle.yaml")
    for name, df in results.tables().items():
        df.to_csv(
            out / "tables" / f"{name}.csv",
            index=False,
            float_format="%.12g",
            lineterminator="\n",
        )
    _write_json(out / "summary.json", _clean(results.summary.to_dict()))
    _write_json(out / "manifest.json", manifest)
    return out


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


def load_bundle(path: str | Path) -> Bundle:
    """Read a bundle folder.

    Raises
    ------
    BundleError
        If ``path`` is not a bundle or its schema is not :data:`BUNDLE_SCHEMA`.
    """
    root = Path(path)
    if not is_bundle(root):
        raise BundleError(f"{root} is not a suspkin bundle (no valid manifest.json)")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema != BUNDLE_SCHEMA:
        raise BundleError(
            f"unsupported bundle schema {schema!r} in {root} (this suspkin reads {BUNDLE_SCHEMA})"
        )
    tables = {
        p.stem: pd.read_csv(p)
        for p in sorted((root / "tables").glob("*.csv"))
        if p.stem in _STUDIES
    }
    return Bundle(
        label=str(manifest.get("label", root.name)),
        manifest=manifest,
        vehicle=load_vehicle(root / "vehicle.yaml"),
        tables=tables,
        summary=json.loads((root / "summary.json").read_text(encoding="utf-8")),
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


def bundle_to_results(bundle: Bundle) -> StudyResults:
    """Rebuild a ``StudyResults`` from a bundle (null summary values become nan)."""
    empty = pd.DataFrame()
    t = bundle.tables
    return StudyResults(
        vehicle=bundle.vehicle,
        front_corner=t.get("front_corner", empty),
        rear_corner=t.get("rear_corner", empty),
        rack=t.get("rack", empty),
        heave=t.get("heave", empty),
        roll=t.get("roll", empty),
        pitch=t.get("pitch", empty),
        summary=_summary_from_dict(bundle.summary),
        params=dict(bundle.manifest.get("params", {})),
    )
