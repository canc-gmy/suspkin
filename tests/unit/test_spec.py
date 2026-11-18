"""Figure spec: columns exist in the study tables, parity with figures.py, JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from suspkin.geometry import load_vehicle
from suspkin.plotting import figures as f
from suspkin.plotting.figures import FIGURE_SET
from suspkin.plotting.spec import (
    FIGURES,
    SUMMARY_ROWS,
    derived_columns,
    spec_json,
)
from suspkin.report import StudyResults, run_standard_studies
from suspkin.units import display_factor

REF = Path(__file__).parents[2] / "configs" / "vehicles" / "fs_reference.yaml"


def _all_panels() -> list[f._Panel]:
    flat = [p for ps in f._PANELS.values() for p in ps]
    flat += [*f._ROLL_PANELS, *f._HEAVE_PANELS, *f._PITCH_PANELS]
    flat += [t[0] for t in (*f._STEERING_PANELS, f._STEER_RATIO)]
    return flat


@pytest.fixture(scope="module")
def results() -> StudyResults:
    return run_standard_studies(load_vehicle(REF), steps=5)


def test_every_column_exists(results: StudyResults) -> None:
    tables = results.tables()
    for fig in FIGURES:
        df = derived_columns(fig.study, tables[fig.study])
        assert fig.x.column in df.columns, fig.id
        for p in fig.panels:
            if p.x is not None:
                assert p.x.column in df.columns, (fig.id, p.id)
            for s in p.series:
                assert s.column in df.columns, (fig.id, p.id, s.column)


def test_ids_unique() -> None:
    assert len({fig.id for fig in FIGURES}) == len(FIGURES)
    for fig in FIGURES:
        assert len({p.id for p in fig.panels}) == len(fig.panels)


def test_parity_with_figures() -> None:
    flat = _all_panels()
    expected = {s.col for p in flat for s in p.series}
    spec_cols = {s.column for fig in FIGURES for p in fig.panels for s in p.series}
    assert spec_cols == expected
    assert {p.title for fig in FIGURES for p in fig.panels} == {p.title for p in flat}
    assert {fig.group for fig in FIGURES} == set(FIGURE_SET) - {"summary_table"}
    assert len(SUMMARY_ROWS) == len(f._SUMMARY_ROWS)


def test_factors_match_figures() -> None:
    """Display factors in the spec equal what figures.py draws."""
    data = spec_json()
    kinds = {s.col: f._KINDS[s.kind] for p in _all_panels() for s in p.series}
    for fig in data["figures"]:
        for p in fig["panels"]:
            for s in p["series"]:
                assert s["factor"] == pytest.approx(kinds[s["column"]][0]), s["column"]
    mr = next(
        s
        for fig in data["figures"]
        for p in fig["panels"]
        for s in p["series"]
        if s["column"] == "mr_progression"
    )
    assert mr["unit"] == "1/mm"
    assert mr["factor"] == pytest.approx(1e-3)


def test_summary_rows_factors() -> None:
    """Row factor/unit follow the registry dimension, except deliberate display choices."""
    custom = {"Front TLLTD", "Turning radius", "Front roll stiffness", "Rear roll stiffness"}
    custom.add("Total roll stiffness")  # N m/deg instead of the registry's mm*N/deg
    for r in SUMMARY_ROWS:
        assert r.dim is not None
        if r.label not in custom:
            assert r.factor == pytest.approx(display_factor(r.dim)), r.label


def test_spec_json_serialisable() -> None:
    data = spec_json()
    assert json.loads(json.dumps(data))["schema"] == 1
    assert len(data["figures"]) == len(FIGURES)
    assert len(data["summary_rows"]) == len(SUMMARY_ROWS)


def test_derived_rack_columns(results: StudyResults) -> None:
    df = derived_columns("rack", results.rack)
    for c in ("d_left", "d_right", "outer", "ackermann", "ratio"):
        assert c in df.columns
    assert len(df) == len(results.rack)
    assert derived_columns("roll", results.roll) is results.roll
