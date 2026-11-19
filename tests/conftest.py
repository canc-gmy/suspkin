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
    _add_lock_option(parser)


def _add_lock_option(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-lock",
        action="store_true",
        default=False,
        help="Rewrite the behaviour-lock snapshots in tests/regression/behaviour/snapshots.",
    )


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))


@pytest.fixture
def update_lock(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-lock"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "regression: golden-snapshot regression test")
    config.addinivalue_line("markers", "lock: behaviour-lock snapshot test (refactor safety net)")
