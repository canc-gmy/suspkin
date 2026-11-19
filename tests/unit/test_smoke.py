"""Smoke tests: version, units and CLI stub."""

import tomllib
from pathlib import Path

from typer.testing import CliRunner

import suspkin
from suspkin.cli import app


def test_version() -> None:
    assert suspkin.__version__ == "1.1.0"


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as f:
        assert tomllib.load(f)["project"]["version"] == suspkin.__version__


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert suspkin.__version__ in result.output
