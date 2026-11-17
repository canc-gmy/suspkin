from pathlib import Path

import pytest

from suspkin.geometry import VehicleConfigError, load_vehicle, save_vehicle

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"


def test_round_trip(tmp_path: Path) -> None:
    v = load_vehicle(FIXTURE)
    out = tmp_path / "v.yaml"
    save_vehicle(v, out)
    assert load_vehicle(out) == v
    assert "!!python" not in out.read_text(encoding="utf-8")


def test_error_includes_path(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("vehicle: {name: x}\n", encoding="utf-8")
    with pytest.raises(VehicleConfigError, match=r"bad.yaml"):
        load_vehicle(bad)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VehicleConfigError, match=r"nope.yaml"):
        load_vehicle(tmp_path / "nope.yaml")
