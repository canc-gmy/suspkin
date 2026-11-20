"""Public API snapshot, schema versioning."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import suspkin
from suspkin.geometry.io import VehicleConfigError, load_vehicle, save_vehicle

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "minimal_vehicle.yaml"

EXPECTED_ALL = sorted(
    [
        "KinematicsError",
        "Vehicle",
        "VehicleConfigError",
        "__version__",
        "add_gradients",
        "build_corners",
        "build_report",
        "load_vehicle",
        "metrics_frame",
        "plot_vehicle_3d",
        "run_standard_studies",
        "save_vehicle",
        "solve_corner",
        "solve_vehicle",
        "states_to_frame",
        "sweep_corner",
        "sweep_rack",
        "sweep_vehicle",
        "vehicle_frame",
        "vehicle_summary",
    ]
)


def test_all_snapshot() -> None:
    assert sorted(suspkin.__all__) == EXPECTED_ALL


@pytest.mark.parametrize("name", EXPECTED_ALL)
def test_names_importable_with_docstring(name: str) -> None:
    obj = getattr(suspkin, name)
    if name != "__version__":
        assert obj.__doc__, f"{name} has no docstring"


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError):
        _ = suspkin.nonexistent  # type: ignore[attr-defined]


def _data() -> dict:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "v.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def test_schema_missing_defaults_to_1(tmp_path: Path) -> None:
    data = _data()
    data.pop("schema", None)
    assert load_vehicle(_write(tmp_path, data)).schema_version == 1


def test_schema_1_ok(tmp_path: Path) -> None:
    data = _data()
    data["schema"] = 1
    assert load_vehicle(_write(tmp_path, data)).schema_version == 1


def test_schema_2_rejected(tmp_path: Path) -> None:
    data = _data()
    data["schema"] = 2
    with pytest.raises(VehicleConfigError, match="unsupported vehicle schema 2"):
        load_vehicle(_write(tmp_path, data))


def test_save_writes_schema_first_and_roundtrips(tmp_path: Path) -> None:
    v = load_vehicle(FIXTURE)
    out = tmp_path / "out.yaml"
    save_vehicle(v, out)
    saved = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert next(iter(saved)) == "schema"
    assert saved["schema"] == 1
    assert load_vehicle(out) == v


def test_light_import() -> None:
    code = (
        "import sys, suspkin, suspkin.cli; "
        "heavy = [m for m in ('torch', 'optuna', 'fastapi', 'plotly', 'pymoo') "
        "if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
