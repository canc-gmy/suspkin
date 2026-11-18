"""Web app: history analysis, snapshot export and the /export route."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import suspkin.app.snapshot as snap_mod
from suspkin.app.datasets import DatasetManager
from suspkin.app.routes import analyse_history
from suspkin.app.server import create_app
from suspkin.app.snapshot import build_snapshot, export_dataset_snapshot, export_snapshot
from suspkin.cli import app as cli_app
from tests.unit.helpers import FIXTURE

API = "/api/v1"
PARAMS = {"steps": 5, "travel": 0.02, "roll_deg": 1.0}
TEMPLATE = (
    "<!doctype html><html><head><title>t</title></head><body><script>1</script></body></html>"
)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    shutil.copyfile(FIXTURE, tmp_path / "car.yaml")
    shutil.copyfile(FIXTURE, tmp_path / "other.yaml")
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "index.html").write_text(TEMPLATE, encoding="utf-8")
    return tmp_path


def _history() -> pd.DataFrame:
    # eval 3 is infeasible but would dominate; eval 4 is dominated by eval 1
    return pd.DataFrame(
        {
            "eval_id": [0, 1, 2, 3, 4],
            "f_a": [3.0, 1.0, 2.0, 0.1, 2.0],
            "f_b": [3.0, 3.0, 1.0, 0.1, 4.0],
            "f_total": [6.0, 4.0, 3.0, 0.2, 6.0],
            "feasible": [True, True, True, False, True],
        }
    )


def test_analyse_history_best_front_knee() -> None:
    objs, best, front, pairs, knee = analyse_history(_history())
    assert objs == ["f_a", "f_b"]
    assert best == [6.0, 4.0, 3.0, 3.0, 3.0]  # infeasible eval 3 never improves the best
    assert front == [False, True, True, False, False]
    assert knee in (1, 2)
    # in the pair (f_a, f_b) eval 4 is dominated by eval 1; eval 0 by eval 2 and by eval 1
    assert pairs == {"f_a|f_b": front}


def test_pair_front_differs_from_all_objective_front() -> None:
    h = pd.DataFrame(
        {
            "eval_id": [0, 1, 2],
            "f_a": [1.0, 2.0, 3.0],
            "f_b": [3.0, 2.0, 1.0],
            "f_c": [5.0, 5.0, 0.0],
            "f_total": [1.0, 1.0, 1.0],
            "feasible": [True, True, True],
        }
    )
    _, _, front, pairs, _ = analyse_history(h)
    assert front == [True, True, True]  # nobody dominates anybody in 3 objectives
    assert pairs["f_a|f_b"] == [True, True, True]
    assert pairs["f_a|f_c"] == [True, False, True]  # eval 0 dominates eval 1 on (a, c)
    assert pairs["f_b|f_c"] == [False, False, True]
    assert set(pairs) == {"f_a|f_b", "f_a|f_c", "f_b|f_c"}


def test_analyse_history_order_and_degenerate_cases() -> None:
    h = _history().iloc[::-1].reset_index(drop=True)  # rows in reverse eval order
    _, best, front, _, _ = analyse_history(h)
    by_eval = dict(zip(h["eval_id"], best, strict=True))
    assert by_eval == {0: 6.0, 1: 4.0, 2: 3.0, 3: 3.0, 4: 3.0}
    assert sum(front) == 2
    one = _history().drop(columns="f_b")
    assert analyse_history(one)[2] == [False] * 5  # a single objective has no front
    assert analyse_history(one.iloc[0:0])[1:] == ([], [], {}, None)
    allbad = _history().assign(feasible=False)
    o, b, f, _, k = analyse_history(allbad)
    assert all(v is None for v in b) and not any(f) and k is None
    assert o == ["f_a", "f_b"]


def test_history_route_has_analysis(ws: Path) -> None:
    run = ws / "run1"
    run.mkdir()
    shutil.copyfile(FIXTURE, run / "vehicle.yaml")
    shutil.copyfile(FIXTURE, run / "best.yaml")
    _history().to_csv(run / "history.csv", index=False)
    (run / "meta.json").write_text(json.dumps({"mode": "weighted"}), encoding="utf-8")
    (run / "result.json").write_text(json.dumps({"mode": "weighted"}), encoding="utf-8")
    c = TestClient(create_app(ws, static_dir=ws / "none", study_params=PARAMS))
    ds = c.post(f"{API}/datasets", json={"path": "run1"}).json()
    h = c.get(f"{API}/runs/{ds[0]['run_id']}/history").json()
    assert h["objectives"] == ["f_a", "f_b"]
    assert h["best_so_far"] == [6.0, 4.0, 3.0, 3.0, 3.0]
    assert h["pareto_all"] == [False, True, True, False, False]
    assert h["pareto_pairs"] == {"f_a|f_b": h["pareto_all"]}
    assert h["knee_eval_id"] in (1, 2)


def test_export_snapshot_embeds_only_given_paths(ws: Path) -> None:
    out = ws / "out" / "compare.html"
    size = export_snapshot(
        [ws / "car.yaml"], out, ws, include_scene=False, template=ws / "tpl" / "index.html"
    )
    raw = out.read_bytes()
    assert size == len(raw) and b"\r\n" not in raw
    html = raw.decode("utf-8")
    m = re.search(r"<head><script>window\.__SUSPKIN_SNAPSHOT__=(.*?)</script>", html, re.S)
    assert m is not None
    data: dict[str, Any] = json.loads(m.group(1))
    assert [d["label"] for d in data["datasets"]] == ["car"]
    assert "other.yaml" not in html
    did = data["datasets"][0]["id"]
    assert set(data["tables"][did]) == set(data["datasets"][0]["studies"])
    assert data["summaries"][did] and data["scenes"] == {} and data["histories"] == {}
    assert data["figures"]["figures"] and data["health"]["workspace"] == ws.name


def test_snapshot_data_cannot_close_the_script(ws: Path) -> None:
    mgr = DatasetManager(ws, study_params=PARAMS)
    (ds,) = mgr.open("car.yaml")
    mgr.rename(ds.id, "x</script><script>alert(1)</script><!--")
    out = ws / "evil.html"
    export_dataset_snapshot(mgr, out, include_scene=False, template=ws / "tpl" / "index.html")
    html = out.read_text(encoding="utf-8")
    assert html.count("</script>") == TEMPLATE.count("</script>") + 1
    body = re.search(r"__SUSPKIN_SNAPSHOT__=(.*?)</script>", html, re.S)
    assert body is not None
    assert json.loads(body.group(1))["datasets"][0]["label"].startswith("x</script>")


def test_snapshot_scene_and_errors(ws: Path) -> None:
    pytest.importorskip("plotly")
    mgr = DatasetManager(ws, study_params=PARAMS)
    (ds,) = mgr.open("car.yaml")
    snap = build_snapshot(mgr, include_scene=True)
    assert "data" in snap["scenes"][ds.id]
    with pytest.raises(FileNotFoundError, match="build:single"):
        export_dataset_snapshot(mgr, ws / "o.html", template=ws / "missing.html")
    with pytest.raises(ValueError, match="nothing to export"):
        export_snapshot([], ws / "o.html", ws, template=ws / "tpl" / "index.html")


def test_app_export_cli_help() -> None:
    res = CliRunner().invoke(cli_app, ["app-export", "--help"])
    assert res.exit_code == 0
    for word in ("--out", "--workspace", "--no-scene"):
        assert word in res.output
    assert "--dev" in CliRunner().invoke(cli_app, ["app", "--help"]).output


def test_export_route(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snap_mod, "DEFAULT_SINGLE_DIR", ws / "tpl")
    c = TestClient(create_app(ws, static_dir=ws / "none", study_params=PARAMS))
    assert c.post(f"{API}/export", json={}).status_code == 400  # nothing open
    c.post(f"{API}/datasets", json={"path": "car.yaml"})
    ok = c.post(f"{API}/export", json={"out": "snap/a.html"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["path"] == "snap/a.html" and (ws / "snap" / "a.html").is_file()
    default = c.post(f"{API}/export").json()["path"]
    assert re.fullmatch(r"suspkin_snapshot_\d{8}_\d{6}\.html", default)
    assert c.post(f"{API}/export", json={"out": "../evil.html"}).status_code == 403
    assert c.post(f"{API}/export", json={"out": "notes.txt"}).status_code == 400
