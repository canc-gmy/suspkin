"""Experimental web backend: datasets, routes, path security, caching, CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import suspkin.app.datasets as ds_mod
from suspkin.app.datasets import DatasetManager
from suspkin.app.server import create_app
from suspkin.cli import app as cli_app
from suspkin.geometry import load_vehicle
from suspkin.plotting.spec import FIGURES
from suspkin.report import build_report, run_standard_studies, save_bundle
from tests.unit.helpers import FIXTURE

PARAMS = {"steps": 5, "travel": 0.02, "roll_deg": 1.0}
API = "/api/v1"


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    shutil.copyfile(FIXTURE, tmp_path / "car.yaml")
    return tmp_path


@pytest.fixture
def client(ws: Path) -> TestClient:
    return TestClient(create_app(ws, static_dir=ws / "nostatic", study_params=PARAMS))


def _open(c: TestClient, path: str) -> list[dict[str, Any]]:
    r = c.post(f"{API}/datasets", json={"path": path})
    assert r.status_code == 200, r.text
    out: list[dict[str, Any]] = r.json()
    return out


def _make_run(root: Path, *, pareto: bool = True) -> Path:
    run = root / "run1"
    run.mkdir()
    shutil.copyfile(FIXTURE, run / "vehicle.yaml")
    shutil.copyfile(FIXTURE, run / "best.yaml")
    (run / "history.csv").write_text(
        "eval_id,x_a,f_total,status\n0,1.0,2.0,ok\n1,1.5,1.0,ok\n2,2.0,,failed\n",
        encoding="utf-8",
    )
    (run / "meta.json").write_text(json.dumps({"mode": "single", "seed": 1}), encoding="utf-8")
    result: dict[str, Any] = {"mode": "single", "best_eval_id": 1, "knee_eval_id": None}
    if pareto:
        (run / "pareto").mkdir()
        shutil.copyfile(FIXTURE, run / "pareto" / "pareto_0.yaml")
        (run / "pareto.csv").write_text("k,eval_id,knee\n0,1,True\n", encoding="utf-8")
        result["mode"] = "pareto"
        result["knee_eval_id"] = 1
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return run


def test_health_and_figures(client: TestClient) -> None:
    h = client.get(f"{API}/health").json()
    assert h["status"] == "ok" and h["version"]
    assert Path(h["workspace"]).is_absolute()
    spec = client.get(f"{API}/figures").json()
    assert json.dumps(spec)  # serialisable
    assert len(spec["figures"]) == len(FIGURES)


def test_open_yaml_tables_summary(client: TestClient) -> None:
    (d,) = _open(client, "car.yaml")
    assert d["kind"] == "vehicle" and d["label"] == "car" and d["source"] == "car.yaml"
    assert "front_corner" in d["studies"] and "heave" in d["studies"]
    assert d["params"]["steps"] == 5
    assert client.get(f"{API}/datasets").json() == [d]
    t = client.get(f"{API}/datasets/{d['id']}/tables/front_corner").json()["columns"]
    assert len(next(iter(t.values()))) == 5
    s = client.get(f"{API}/datasets/{d['id']}/summary").json()
    assert "front" in s and "rear" in s


def test_unknown_study_and_dataset_404(client: TestClient) -> None:
    (d,) = _open(client, "car.yaml")
    assert client.get(f"{API}/datasets/{d['id']}/tables/nope").status_code == 404
    assert client.get(f"{API}/datasets/zzz/summary").status_code == 404
    assert client.delete(f"{API}/datasets/zzz").status_code == 404
    assert client.get(f"{API}/runs/zzz/history").status_code == 404


def test_nan_becomes_null(ws: Path) -> None:
    import numpy as np

    m = DatasetManager(ws, study_params=PARAMS)
    (d,) = m.open("car.yaml")
    d.tables["heave"] = d.tables["heave"].assign(bad=np.nan)
    c = TestClient(create_app(ws, static_dir=ws / "x"))
    c.app.state.manager._datasets = {d.id: d}  # type: ignore[attr-defined]
    cols = c.get(f"{API}/datasets/{d.id}/tables/heave").json()["columns"]
    assert cols["bad"] == [None] * len(cols["bad"])


def test_label_dedupe_rename_recolor_delete(client: TestClient, ws: Path) -> None:
    sub = ws / "other"
    sub.mkdir()
    shutil.copyfile(FIXTURE, sub / "car.yaml")
    a = _open(client, "car.yaml")[0]
    b = _open(client, "other/car.yaml")[0]
    assert a["label"] == "car" and b["label"] == "car (2)"
    assert a["color"] != b["color"] and a["id"] != b["id"]
    # reopening the same file is idempotent
    assert _open(client, "car.yaml")[0]["id"] == a["id"]
    assert len(client.get(f"{API}/datasets").json()) == 2

    r = client.patch(f"{API}/datasets/{a['id']}", json={"label": "front car", "color": 5})
    assert r.status_code == 200 and r.json()["label"] == "front car" and r.json()["color"] == 5
    assert client.patch(f"{API}/datasets/{b['id']}", json={"label": "front car"}).status_code == 400
    assert client.patch(f"{API}/datasets/{b['id']}", json={"color": -1}).status_code == 422

    assert client.delete(f"{API}/datasets/{a['id']}").status_code == 204
    assert [d["id"] for d in client.get(f"{API}/datasets").json()] == [b["id"]]
    # the freed colour is reused
    c = _open(client, "car.yaml")[0]
    assert c["color"] == a["color"] or c["color"] not in {b["color"]}


def test_path_outside_workspace_403(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "car.yaml"
    shutil.copyfile(FIXTURE, outside)
    assert client.post(f"{API}/datasets", json={"path": str(outside)}).status_code == 403
    assert client.post(f"{API}/datasets", json={"path": "../x.yaml"}).status_code == 403
    assert client.get(f"{API}/browse", params={"dir": ".."}).status_code == 403
    assert client.post(f"{API}/datasets", json={"path": "missing.yaml"}).status_code == 404


def test_bad_inputs_400(client: TestClient, ws: Path) -> None:
    (ws / "junk.yaml").write_text("a: 1\n", encoding="utf-8")
    (ws / "emptydir").mkdir()
    assert client.post(f"{API}/datasets", json={"path": "junk.yaml"}).status_code == 400
    assert client.post(f"{API}/datasets", json={"path": "emptydir"}).status_code == 400


def test_cache_hit(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = ds_mod.run_standard_studies

    def counting(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ds_mod, "run_standard_studies", counting)
    shutil.copyfile(FIXTURE, ws / "copy.yaml")
    m = DatasetManager(ws, study_params=PARAMS)
    (d,) = m.open("car.yaml")
    m.close(d.id)
    m.open("car.yaml")
    m.open("copy.yaml")  # same bytes, other file
    assert len(calls) == 1
    # different study params do not share the cache
    m2 = DatasetManager(ws, study_params={**PARAMS, "steps": 7})
    m2.open("car.yaml")
    assert len(calls) == 2


def test_bundle_and_report_folder(client: TestClient, ws: Path) -> None:
    res = run_standard_studies(load_vehicle(FIXTURE), **PARAMS)
    save_bundle(res, ws / "b1", label="mybundle")
    (d,) = _open(client, "b1")
    assert d["kind"] == "bundle" and d["label"] == "mybundle"
    assert client.get(f"{API}/datasets/{d['id']}/tables/heave").status_code == 200

    build_report(
        [("one", res), ("two", res)],
        ws / "rep",
        formats=("png",),
        include_3d=False,
    )
    ds = _open(client, "rep")
    assert sorted(x["label"] for x in ds) == ["one", "two"]
    assert all(x["kind"] == "bundle" for x in ds)


def test_optimizer_run_folder(client: TestClient, ws: Path) -> None:
    run = _make_run(ws)
    ds = _open(client, run.name)
    assert [d["label"] for d in ds] == ["run1 baseline", "run1 best", "run1 knee"]
    assert {d["run_id"] for d in ds} == {ds[0]["run_id"]} and ds[0]["run_id"]
    assert all(d["kind"] == "run" for d in ds)
    h = client.get(f"{API}/runs/{ds[0]['run_id']}/history").json()
    assert h["columns"]["eval_id"] == [0, 1, 2]
    assert h["columns"]["f_total"][2] is None
    assert h["meta"]["seed"] == 1 and h["result"]["mode"] == "pareto"
    # closing all datasets of the run forgets it
    for d in ds:
        client.delete(f"{API}/datasets/{d['id']}")
    assert client.get(f"{API}/runs/{ds[0]['run_id']}/history").status_code == 404


def test_optimizer_run_without_pareto(client: TestClient, ws: Path) -> None:
    run = _make_run(ws, pareto=False)
    assert [d["label"] for d in _open(client, run.name)] == [
        f"{run.name} baseline",
        f"{run.name} best",
    ]


def test_scene(client: TestClient) -> None:
    pytest.importorskip("plotly")
    (d,) = _open(client, "car.yaml")
    r = client.get(f"{API}/datasets/{d['id']}/scene")
    assert r.status_code == 200 and len(r.json()["data"]) > 0
    r2 = client.get(f"{API}/datasets/{d['id']}/scene", params={"heave": 0.01, "roll": 0.01})
    assert r2.status_code == 200 and len(r2.json()["data"]) > 0
    assert r2.json() != r.json()


def test_browse_kinds(client: TestClient, ws: Path) -> None:
    res = run_standard_studies(load_vehicle(FIXTURE), **PARAMS)
    save_bundle(res, ws / "bun", label="b")
    build_report([("one", res)], ws / "rep", formats=("png",), include_3d=False)
    _make_run(ws)
    (ws / "plain").mkdir()
    (ws / "notes.txt").write_text("x", encoding="utf-8")
    (ws / "junk.yaml").write_text("a: 1\n", encoding="utf-8")
    (ws / ".hidden").mkdir()
    kinds = {e["name"]: e["kind"] for e in client.get(f"{API}/browse").json()["entries"]}
    assert kinds == {
        "bun": "bundle",
        "plain": "dir",
        "rep": "report",
        "run1": "run",
        "car.yaml": "vehicle",
        "junk.yaml": "other",
        "notes.txt": "other",
    }
    sub = client.get(f"{API}/browse", params={"dir": "rep"}).json()
    assert sub["dir"] == "rep" and sub["parent"] == ""
    assert all(e["path"].startswith("rep/") for e in sub["entries"])


def test_no_frontend_page_and_spa(ws: Path, tmp_path: Path) -> None:
    c = TestClient(create_app(ws, static_dir=ws / "missing", study_params=PARAMS))
    r = c.get("/")
    assert r.status_code == 200 and "npm run build" in r.text

    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    (static / "assets" / "a.js").write_text("1", encoding="utf-8")
    c2 = TestClient(create_app(ws, static_dir=static, study_params=PARAMS))
    assert "spa" in c2.get("/").text
    assert "spa" in c2.get("/some/route").text
    assert c2.get("/assets/a.js").text == "1"
    assert c2.get(f"{API}/health").status_code == 200
    assert c2.get("/api/v1/nothing").status_code == 404
    assert "spa" in c2.get("/..%2f..%2fsecret").text  # traversal falls back to index


def test_cors_only_vite(client: TestClient) -> None:
    ok = client.get(f"{API}/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    bad = client.get(f"{API}/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_preload(ws: Path) -> None:
    c = TestClient(
        create_app(
            ws,
            static_dir=ws / "x",
            preload=[ws / "car.yaml", ws / "nope.yaml", Path("/outside/x.yaml")],
            study_params=PARAMS,
        )
    )
    assert [d["label"] for d in c.get(f"{API}/datasets").json()] == ["car"]


def test_cli_app_help() -> None:
    r = CliRunner().invoke(cli_app, ["app", "--help"])
    assert r.exit_code == 0
    for opt in ("--port", "--host", "--workspace", "--no-browser"):
        assert opt in r.output


def test_openapi_dump(tmp_path: Path) -> None:
    from suspkin.app.openapi import main

    out = tmp_path / "sub" / "openapi.json"
    assert main([str(out)]) == 0
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert "workspace" in schema["components"]["schemas"]["HealthOut"]["properties"]
    assert main([]) == 2
