"""Figure spec: columns exist in the study tables, parity with figures.py, JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from suspkin.geometry import load_vehicle
from suspkin.plotting import figures as f
from suspkin.plotting import units as display_units
from suspkin.plotting.figures import FIGURE_SET
from suspkin.plotting.spec import (
    FIGURES,
    SUMMARY_ROWS,
    FigureSpec,
    Panel,
    Series,
    derived_columns,
    spec_json,
)
from suspkin.report import StudyResults, run_standard_studies
from suspkin.units import display_factor

REF = Path(__file__).parents[2] / "configs" / "vehicles" / "fs_reference.yaml"


def _all_panels() -> list[f._Panel]:
    flat = [p for ps in f._PANELS.values() for p in ps]
    flat += [*f._ROLL_PANELS, *f._HEAVE_PANELS, *f._PITCH_PANELS]
    flat += [t[0] for t in (*f._STEERING_PANELS, f._STEER_RATIO, f._STEER_GAIN)]
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
            if not set(p.requires) <= set(df.columns):
                continue  # alternative panel not drawn for this table (see requires)
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
    assert len(SUMMARY_ROWS) == len(display_units.SUMMARY_ROWS)


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


def test_spec_json_conventions() -> None:
    data = spec_json()
    assert data["palette"][0] == "#0072B2" and data["linestyles"][:2] == ["-", "--"]
    text = json.dumps(data)
    for fig in data["figures"]:
        for ax in (fig["x"], *(p["x"] for p in fig["panels"] if p["x"])):
            assert ax["unit"] != "-"
        for p in fig["panels"]:
            assert all(s["unit"] != "-" for s in p["series"])
            assert set(p) >= {"requires", "alternative_to"}
    assert all(r["unit"] != "-" and isinstance(r["decimals"], int) for r in data["summary_rows"])
    assert '"fmt"' not in text
    sr = next(r for r in data["summary_rows"] if r["label"] == "Steering ratio")
    assert sr["unit"] == "" and sr["decimals"] == 2
    steer = next(fig for fig in data["figures"] if fig["id"] == "steering")
    last = {p["id"]: p for p in steer["panels"]}
    assert last["steering_ratio"]["requires"] == ["ratio"]
    assert last["steer_gain"]["alternative_to"] == "steering_ratio"
    assert last["steer_gain"]["requires"] == ["gain"]


# ------------------------------------------------------------- matplotlib parity (renderer)
def _drawn(fig: FigureSpec, df: pd.DataFrame) -> list[tuple[Panel, list[Series]]]:
    """Reference implementation of the renderer rule in the spec module docstring."""
    cols = set(df.columns)
    out: list[tuple[Panel, list[Series]]] = []
    drawn: set[str] = set()
    for p in fig.panels:
        series = [s for s in p.series if s.column in cols]
        ok = bool(series) and (p.x or fig.x).column in cols and set(p.requires) <= cols
        if p.alternative_to in drawn:
            ok = False
        if ok:
            drawn.add(p.id)
            out.append((p, series))
    return out


def _render(fig: FigureSpec, df: pd.DataFrame) -> list[object]:
    ds = [("a", df)]
    if fig.group == "steering":
        mpl_fig = FIGURE_SET["steering"](ds)
    elif fig.group == "roll":
        mpl_fig = FIGURE_SET["roll"](ds)
    elif fig.group == "heave_pitch":
        mpl_fig = FIGURE_SET["heave_pitch"](ds if fig.study == "heave" else [],
                                            ds if fig.study == "pitch" else [])  # fmt: skip
    else:
        mpl_fig = FIGURE_SET[fig.group](ds, axle=fig.study.split("_")[0])
    return list(mpl_fig.axes)


def _check_parity(fig: FigureSpec, df: pd.DataFrame) -> list[str]:
    """Compare every axes of figures.py with the spec; return the drawn panel ids."""
    js = next(x for x in spec_json()["figures"] if x["id"] == fig.id)
    pjs = {p["id"]: p for p in js["panels"]}
    df = derived_columns(fig.study, df)
    drawn = _drawn(fig, df)
    axes = _render(fig, df)
    assert len(axes) == len(drawn), fig.id
    for ax, (p, series) in zip(axes, drawn, strict=True):
        pj = pjs[p.id]
        assert ax.get_title() == p.title == pj["title"], (fig.id, p.id)
        assert ax.get_ylabel() == pj["y_label"], (fig.id, p.id)
        lines = [ln for ln in ax.lines if ln.get_gid() != "zero"]
        assert len(lines) == len(series), (fig.id, p.id)
        xax = pj["x"] or js["x"]
        sjs = {s["column"]: s for s in pj["series"]}
        for ln, s in zip(lines, series, strict=True):
            for got, col, fac in (
                (ln.get_ydata(), s.column, sjs[s.column]["factor"]),
                (ln.get_xdata(), xax["column"], xax["factor"]),
            ):
                want = df[col].to_numpy(float) * fac
                np.testing.assert_allclose(
                    np.asarray(got, float), np.where(np.isfinite(want), want, np.nan),
                    rtol=1e-12, equal_nan=True,
                    err_msg=f"{fig.id}/{p.id}/{col}",
                )  # fmt: skip
    return [p.id for p, _ in drawn]


@pytest.mark.parametrize("fig", FIGURES, ids=lambda x: x.id)
def test_matplotlib_parity(fig: FigureSpec, results: StudyResults) -> None:
    ids = _check_parity(fig, results.tables()[fig.study])
    if fig.id == "steering":
        assert ids[-1] == "steering_ratio" and "steer_gain" not in ids


def test_steering_gain_fallback(results: StudyResults) -> None:
    """Without ``steer_c_factor`` the last steering panel is the gain, not the ratio."""
    fig = next(x for x in FIGURES if x.id == "steering")
    rack = results.rack.drop(columns="steer_c_factor")
    ids = _check_parity(fig, rack)
    assert ids[-1] == "steer_gain" and "steering_ratio" not in ids
    assert len(ids) == len(fig.panels) - 1
