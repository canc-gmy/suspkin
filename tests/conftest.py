"""Shared pytest configuration."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite the golden snapshots in tests/regression/golden instead of comparing.",
    )


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "regression: golden-snapshot regression test")
