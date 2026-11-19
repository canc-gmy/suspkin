"""Behaviour lock of the ``suspkin`` CLI: stdout, exit code and every written file.

Each case runs the real Typer app with ``SUSPKIN_NO_NUMBA=1`` and snapshots the result under
``snapshots/cli_<case>/``. Argv follow the lock brief with these adaptations to the real CLI:
``sweep``/``vsweep``/``show``/``animate`` take ``--out`` as a *file* (``{O}/<file>``), not a
folder; ``report`` takes a folder. ``vsweep`` ranges are mm / deg. ``show --html`` writes to
``--out`` (HTML file). ``--json`` takes an output file (``{O}/...json``). ``--metrics`` needs
``--travel``; ``--rack-sweep`` takes MIN MAX. ``app-export`` runs against a stub single-file
template (the real one is a git-ignored frontend build). ``--compare`` of ``report`` is
repeatable and takes a path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from behaviour._lock import REPO, snapshot_dir
from suspkin.cli import app
from suspkin.kinematics import _numba

pytestmark = pytest.mark.lock

V = (REPO / "configs" / "vehicles" / "fs_reference.yaml").as_posix()
STUDY = REPO / "configs" / "studies"
CASES = Path(__file__).resolve().parent / "cases"
COMMANDS = [
    "version",
    "show",
    "animate",
    "sweep",
    "summary",
    "vehicle",
    "vsweep",
    "report",
    "check",
    "optimize",
    "app",
    "app-export",
    "surrogate",
]
SURROGATE = ["sample", "train", "validate"]

# id -> list of argv (placeholders {V} vehicle, {O} output folder, {S} studies folder,
# {C} cases folder). Several argv of one id run in order, each snapshotted as <id>_<k>.
CASE_ARGV: dict[str, list[list[str]]] = {
    "version": [["version"]],
    "summary": [["summary", "{V}"]],
    "summary_json": [["summary", "{V}", "--json", "{O}/summary.json"]],
    "vehicle": [["vehicle", "{V}"]],
    "vehicle_json": [["vehicle", "{V}", "--json", "{O}/vehicle.json"]],
    "sweep_metrics": [
        ["sweep", "{V}", "--out", "{O}/sweep.csv", "--travel", "-30", "30", "--metrics"]
    ],
    "sweep_rack": [["sweep", "{V}", "--out", "{O}/sweep.csv", "--rack-sweep", "-30", "30"]],
    "sweep_rl": [
        [
            *("sweep", "{V}", "--out", "{O}/sweep.csv", "--corner", "RL"),
            *("--travel", "-30", "30", "--steps", "7"),
        ]
    ],
    "sweep_bad": [["sweep", "{V}", "--out", "{O}/sweep.csv", "--travel", "-500", "500"]],
    "vsweep_heave": [
        ["vsweep", "{V}", "--out", "{O}/v.csv", "--heave", "-20", "20", "--steps", "5"]
    ],
    "vsweep_roll": [["vsweep", "{V}", "--out", "{O}/v.csv", "--roll", "-2", "2", "--steps", "5"]],
    "vsweep_pitch": [["vsweep", "{V}", "--out", "{O}/v.csv", "--pitch", "-1", "1", "--steps", "5"]],
    "vsweep_rack": [["vsweep", "{V}", "--out", "{O}/v.csv", "--rack", "-20", "20", "--steps", "5"]],
    "show": [["show", "{V}", "--out", "{O}/show.png"]],
    "show_html": [["show", "{V}", "--out", "{O}/show.html", "--html"]],
    "animate": [
        ["animate", "{V}", "--out", "{O}/anim.gif", "--heave", "-20", "20", "--steps", "3"]
    ],
    "report": [["report", "{V}", "--out", "{O}/rep", "--steps", "5"]],
    "report_compare": [
        ["report", "{V}", "--out", "{O}/rep", "--compare", "{V}", "--no-3d", "--steps", "5"]
    ],
    "check": [["check", "{S}/fs_reference_constraints.yaml", "--csv", "{O}/c.csv", "--strict"]],
    "optimize": [
        [
            "optimize",
            "{S}/fs_reference_optimize.yaml",
            "--out",
            "{O}/run",
            "--workers",
            "1",
            "--evaluations",
            "30",
            "--no-report",
        ],
        [
            "optimize",
            "{S}/fs_reference_optimize.yaml",
            "--out",
            "{O}/run",
            "--workers",
            "1",
            "--evaluations",
            "45",
            "--no-report",
            "--resume",
        ],
    ],
    "app_export": [["app-export", "{V}", "--out", "{O}/snap.html", "--no-scene"]],
    "bad_vehicle": [["summary", "{C}/bad_vehicle.yaml"]],
    "bad_study": [["check", "{C}/bad_study.yaml"]],
}
# Help of the app server needs no extra; run them all through the same machinery.
HELP_ARGV: dict[str, list[str]] = {"help_root": ["--help"]}
HELP_ARGV.update({f"help_{c.replace('-', '_')}": [c, "--help"] for c in COMMANDS})
HELP_ARGV.update({f"help_surrogate_{c}": ["surrogate", c, "--help"] for c in SURROGATE})


# The DE+SLSQP optimizer run (evaluation count, history values) can differ between platforms.
# Locally on Windows (where the snapshots were recorded and the refactor runs) it stays strict;
# under CI or on another platform only layout, CSV column names and exit codes are compared.
OPTIMIZE_RELAXED = bool(os.environ.get("CI")) or sys.platform != "win32"


@pytest.fixture(autouse=True)
def _no_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUSPKIN_NO_NUMBA", "1")
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    _numba.reset()


def _run(argv: list[str], out: Path) -> tuple[str, int]:
    fill = {"{V}": V, "{O}": out.as_posix(), "{S}": STUDY.as_posix(), "{C}": CASES.as_posix()}
    full = []
    for a in argv:
        for k, v in fill.items():
            a = a.replace(k, v)
        full.append(a)
    res = CliRunner().invoke(app, full)
    return res.output, res.exit_code


def _needs(case: str) -> None:
    if case == "show_html":
        pytest.importorskip("plotly")
    if case == "optimize":
        pytest.importorskip("pymoo")
    if case == "app_export":
        pytest.importorskip("fastapi")


@pytest.mark.parametrize("case", list(CASE_ARGV))
def test_cli_lock(
    case: str, tmp_path: Path, update_lock: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _needs(case)
    if case == "app_export":
        import suspkin.app.snapshot as snap

        stub = tmp_path / "single"
        stub.mkdir()
        (stub / "index.html").write_text(
            "<!doctype html><html><head><title>stub</title></head><body></body></html>\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(snap, "DEFAULT_SINGLE_DIR", stub)
    out = tmp_path / "out"
    out.mkdir()
    steps = CASE_ARGV[case]
    for k, argv in enumerate(steps, start=1):
        stdout, code = _run(argv, out)
        name = f"cli_{case}" if len(steps) == 1 else f"cli_{case}_{k}"
        snapshot_dir(
            name,
            out,
            stdout,
            code,
            update=update_lock,
            tmp=tmp_path,
            relaxed=case == "optimize" and OPTIMIZE_RELAXED,
        )
    expect_fail = case in {"sweep_bad", "bad_vehicle", "bad_study"}
    assert (code != 0) == expect_fail, stdout


@pytest.mark.parametrize("case", list(HELP_ARGV))
def test_cli_help_lock(case: str, tmp_path: Path, update_lock: bool) -> None:
    out = tmp_path / "out"
    out.mkdir()
    stdout, code = _run(HELP_ARGV[case], out)
    assert code == 0, stdout
    snapshot_dir(f"cli_{case}", out, stdout, code, update=update_lock, tmp=tmp_path)
