"""Result bundle: save/load round trip, schema check, build_report integration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.report import (
    BUNDLE_SCHEMA,
    BundleError,
    StudyResults,
    build_report,
    bundle_to_results,
    is_bundle,
    load_bundle,
    run_standard_studies,
    save_bundle,
)
from tests.unit.helpers import FIXTURE


@pytest.fixture(scope="module")
def results() -> StudyResults:
    return run_standard_studies(load_vehicle(FIXTURE), steps=5, travel=0.02, roll_deg=1.0)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def test_round_trip(results: StudyResults, tmp_path: Path) -> None:
    out = save_bundle(results, tmp_path / "b", label="ref", source="fixture.yaml")
    assert is_bundle(out)
    b = load_bundle(out)
    assert b.label == "ref"
    assert b.vehicle == results.vehicle
    assert set(b.tables) == set(results.tables())
    for name, df in results.tables().items():
        assert list(b.tables[name].columns) == list(df.columns)
        for col in df.columns:
            got = b.tables[name][col]
            if df[col].dtype.kind in "fi":
                np.testing.assert_allclose(
                    got.to_numpy(float), df[col].to_numpy(float), rtol=1e-11, atol=1e-14,
                    equal_nan=True,
                )  # fmt: skip
            else:
                assert list(got) == list(df[col]), (name, col)
    m = b.manifest
    assert m["schema"] == BUNDLE_SCHEMA == 1
    assert m["kind"] == "suspkin-bundle"
    assert m["source"] == "fixture.yaml"
    assert m["created"] and m["suspkin_version"]
    assert m["params"]["steps"] == 5 and m["params"]["travel"] == 0.02
    assert m["params"]["roll_deg"] == 1.0
    assert set(m["params"]) == {"travel", "rack", "roll_deg", "heave", "pitch_deg", "steps"}
    json.dumps(b.summary)  # JSON-safe: nan became null
    back = bundle_to_results(b)
    assert _same(back.summary.to_dict(), results.summary.to_dict())
    assert back.params == results.params
    assert len(back.roll) == 5


def test_files_use_lf(results: StudyResults, tmp_path: Path) -> None:
    out = save_bundle(results, tmp_path / "b", label="x")
    for p in (out / "manifest.json", out / "summary.json", out / "tables" / "roll.csv"):
        assert b"\r" not in p.read_bytes()


def test_bad_schema_and_not_a_bundle(results: StudyResults, tmp_path: Path) -> None:
    out = save_bundle(results, tmp_path / "b", label="x")
    mf = out / "manifest.json"
    data = json.loads(mf.read_text(encoding="utf-8"))
    data["schema"] = 99
    mf.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BundleError, match="schema"):
        load_bundle(out)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not is_bundle(empty)
    with pytest.raises(BundleError, match="not a suspkin bundle"):
        load_bundle(empty)


def test_build_report_writes_bundles(results: StudyResults, tmp_path: Path) -> None:
    build_report([("My design", results)], tmp_path / "rep", include_3d=False, formats=("png",))
    bdir = tmp_path / "rep" / "bundles" / "My_design"
    assert is_bundle(bdir)
    assert load_bundle(bdir).label == "My design"
