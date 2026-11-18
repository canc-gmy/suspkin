"""Run every script in ``examples/`` end to end into a temporary folder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = sorted(EXAMPLES.glob("0*.py"))

pytestmark = pytest.mark.examples


def test_scripts_found() -> None:
    assert len(SCRIPTS) == 5


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_runs(script: Path, tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip()
    assert any(tmp_path.iterdir())
