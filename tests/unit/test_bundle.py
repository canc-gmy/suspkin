"""Result bundle: save/load round trip, schema check, build_report integration."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from suspkin.report.bundle import json_safe
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
    for p in (
        out / "manifest.json",
        out / "summary.json",
        out / "vehicle.yaml",
        out / "tables" / "roll.csv",
    ):
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


def test_lossless_floats_nan_int_and_empty_rack(results: StudyResults, tmp_path: Path) -> None:
    roll = results.roll.copy()
    roll["n_int"] = np.arange(len(roll))
    roll.loc[roll.index[1], "roll"] = float("nan")
    roll["third"] = 1.0 / 3.0 + np.arange(len(roll)) * 1e-13
    res = dataclasses.replace(results, roll=roll, rack=pd.DataFrame())
    b = load_bundle(save_bundle(res, tmp_path / "b", label="x"))
    assert "rack" not in b.tables
    got = b.tables["roll"]
    assert got["n_int"].dtype.kind == "i" and list(got["n_int"]) == list(roll["n_int"])
    assert np.isnan(got["roll"].iloc[1])
    np.testing.assert_array_equal(got["third"].to_numpy(), roll["third"].to_numpy())  # exact
    for name, df in res.tables().items():
        for col in df.columns:
            if df[col].dtype.kind == "f":
                np.testing.assert_array_equal(
                    b.tables[name][col].to_numpy(), df[col].to_numpy()
                )  # bitwise, nan == nan
    back = bundle_to_results(b)
    assert len(back.rack) == 0 and len(back.roll) == len(roll)


def test_save_overwrite_and_stale_tables(results: StudyResults, tmp_path: Path) -> None:
    out = tmp_path / "b"
    save_bundle(results, out, label="one")
    assert (out / "tables" / "rack.csv").is_file()
    with pytest.raises(BundleError, match="overwrite"):
        save_bundle(results, out, label="two")
    assert load_bundle(out).label == "one"  # untouched
    save_bundle(dataclasses.replace(results, rack=pd.DataFrame()), out, label="two", overwrite=True)
    assert not (out / "tables" / "rack.csv").exists()  # stale table removed
    assert load_bundle(out).label == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["b"]  # no temp dirs left behind
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BundleError, match="not a suspkin bundle"):
        save_bundle(results, other, label="x", overwrite=True)
    assert (other / "keep.txt").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    save_bundle(results, empty, label="x")  # an empty folder is fine


def test_created_injectable(results: StudyResults, tmp_path: Path) -> None:
    out = save_bundle(results, tmp_path / "b", label="x", created="2020-01-02T03:04:05+00:00")
    assert load_bundle(out).manifest["created"] == "2020-01-02T03:04:05+00:00"


def test_json_safe() -> None:
    data = {
        "a": [np.float64(1.5), float("nan"), (np.int64(3), float("inf"))],
        "b": np.array([1.0, np.nan]),
    }
    assert json_safe(data) == {"a": [1.5, None, [3, None]], "b": [1.0, None]}
    assert type(json_safe(np.int64(3))) is int
    json.dumps(json_safe(data), allow_nan=False)


@pytest.mark.parametrize(
    "victim", ["vehicle.yaml", "summary.json", "manifest.json", "tables/roll.csv"]
)
@pytest.mark.parametrize("mode", ["missing", "malformed"])
def test_load_errors_name_path(
    results: StudyResults, tmp_path: Path, victim: str, mode: str
) -> None:
    if victim.startswith("tables/") and mode == "missing":
        pytest.skip("a missing table is a valid (empty) study")
    out = save_bundle(results, tmp_path / "b", label="x")
    target = out / victim
    if mode == "missing":
        target.unlink()
    else:
        target.write_text("{[ not valid", encoding="utf-8")
        if victim == "tables/roll.csv":
            target.write_bytes(b"")  # empty CSV
    with pytest.raises(BundleError) as ei:
        load_bundle(out)
    if victim == "manifest.json":
        assert str(out) in str(ei.value)
    else:
        assert victim.split("/")[-1] in str(ei.value)
        assert str(out) in str(ei.value)


def test_duplicate_slugs_rejected(results: StudyResults, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"'a b'.*'a_b'"):
        build_report(
            [("a b", results), ("a_b", results)], tmp_path / "r", include_3d=False, formats=("png",)
        )
    assert not (tmp_path / "r").exists()
