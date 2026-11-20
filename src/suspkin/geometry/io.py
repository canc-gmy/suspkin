"""YAML load/save for :class:`~suspkin.geometry.model.Vehicle`.

Vehicle files are SI (m, rad, N, N/m); see ``docs/concepts/vehicle-model.md`` for the schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from suspkin.geometry.model import Vehicle


class VehicleConfigError(ValueError):
    """Raised when a vehicle file cannot be parsed or validated."""


def load_vehicle(path: str | Path) -> Vehicle:
    """Load and validate a vehicle YAML file.

    Args:
        path: Path of the YAML file (UTF-8). All lengths in m, angles in rad.

    Returns:
        The validated, immutable vehicle model.

    Raises:
        VehicleConfigError: On I/O, YAML or validation errors; the message includes the
            file path.
    """
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return Vehicle.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise VehicleConfigError(f"{p}: {exc}") from exc


def _plain(obj: Any) -> Any:
    """Recursively turn tuples into lists so ``yaml.safe_dump`` writes plain sequences."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def vehicle_yaml(vehicle: Vehicle) -> str:
    """YAML text of a vehicle (tuples as lists, model key order, ``schema`` key included).

    Args:
        vehicle: Vehicle to serialise.

    Returns:
        YAML document (SI units) that :func:`load_vehicle` reads back to an equal model.
    """
    data = _plain(vehicle.model_dump(mode="python", by_alias=True))
    return yaml.safe_dump(data, sort_keys=False)


def save_vehicle(vehicle: Vehicle, path: str | Path) -> None:
    """Write a vehicle to YAML (tuples as lists, model key order, LF line endings).

    Args:
        vehicle: Vehicle to serialise.
        path: Destination file; overwritten if it exists.

    Raises:
        OSError: If the file cannot be written.
    """
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        f.write(vehicle_yaml(vehicle))
