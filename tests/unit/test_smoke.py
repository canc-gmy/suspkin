"""Smoke tests: version, units and CLI stub."""

import tomllib
from pathlib import Path

import numpy as np
import numpy.testing as npt
from typer.testing import CliRunner

import suspkin
from suspkin import units
from suspkin.cli import app


def test_version() -> None:
    assert suspkin.__version__ == "1.0.0"


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as f:
        assert tomllib.load(f)["project"]["version"] == suspkin.__version__


def test_mm_m_roundtrip_scalar() -> None:
    npt.assert_allclose(units.mm_to_m(units.m_to_mm(1.5)), 1.5)


def test_mm_m_roundtrip_array() -> None:
    arr = np.array([0.1, 2.0, -3.5])
    npt.assert_allclose(units.mm_to_m(units.m_to_mm(arr)), arr)


def test_deg_rad_roundtrip_scalar() -> None:
    npt.assert_allclose(units.rad_to_deg(units.deg_to_rad(45.0)), 45.0)


def test_deg_rad_roundtrip_array() -> None:
    arr = np.array([0.0, 90.0, -30.0])
    npt.assert_allclose(units.rad_to_deg(units.deg_to_rad(arr)), arr)


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert suspkin.__version__ in result.output
