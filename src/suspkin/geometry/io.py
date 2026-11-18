"""YAML load/save for :class:`~suspkin.geometry.models.Vehicle`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from suspkin.geometry.models import Vehicle


class VehicleConfigError(ValueError):
    """Raised when a vehicle file cannot be parsed or validated."""


def load_vehicle(path: str | Path) -> Vehicle:
    """Load and validate a vehicle YAML file.

    Raises
    ------
    VehicleConfigError
        On I/O, YAML or validation errors; the message includes the file path.
    """
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return Vehicle.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise VehicleConfigError(f"{p}: {exc}") from exc


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def save_vehicle(vehicle: Vehicle, path: str | Path) -> None:
    """Write a vehicle to YAML (tuples as lists, model key order)."""
    data = _plain(vehicle.model_dump(mode="python", by_alias=True))
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
