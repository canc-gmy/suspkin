"""Study regions in the web backend: opening a study, run studies, wireframes, snapshots."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from suspkin.app.datasets import DatasetManager
from suspkin.app.server import create_app
from suspkin.app.snapshot import build_snapshot
from suspkin.optimize.design import DesignSpace
from suspkin.optimize.outputs import copy_inputs
from suspkin.optimize.spec import load_study
from suspkin.optimize.spec.criteria import InsideConstraint, OutsideConstraint
from suspkin.plotting.regions import _domain_vertices
from tests.unit.test_app_api import API, TestClient, _open

CFG = Path(__file__).parents[2] / "configs"
PARAMS = {"steps": 5, "travel": 0.02, "roll_deg": 1.0}


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "vehicles").mkdir()
    (tmp_path / "studies").mkdir()
    shutil.copyfile(CFG / "vehicles" / "fs_reference.yaml", tmp_path / "vehicles" / "fs.yaml")
    text = (CFG / "studies" / "fs_reference_constraints.yaml").read_text(encoding="utf-8")
    text = text.replace("../vehicles/fs_reference.yaml", "../vehicles/fs.yaml")
    (tmp_path / "studies" / "cons.yaml").write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(ws: Path) -> TestClient:
    return TestClient(create_app(ws, static_dir=ws / "nostatic", study_params=PARAMS))


def test_browse_marks_study_and_vehicle(client: TestClient) -> None:
    kinds = {
        e["name"]: e["kind"]
        for e in client.get(f"{API}/browse", params={"dir": "studies"}).json()["entries"]
    }
    assert kinds["cons.yaml"] == "study"
    kinds = {
        e["name"]: e["kind"]
        for e in client.get(f"{API}/browse", params={"dir": "vehicles"}).json()["entries"]
    }
    assert kinds["fs.yaml"] == "vehicle"


def test_open_study_gives_regions(client: TestClient, ws: Path) -> None:
    (d,) = _open(client, "studies/cons.yaml")
    assert d["label"] == "cons" and d["has_study"] and d["kind"] == "vehicle"
    regs = client.get(f"{API}/datasets/{d['id']}/regions").json()
    study, vehicle = load_study(ws / "studies" / "cons.yaml")
    n_sym = sum(1 for v in study.variables if not v.asymmetric)
    n_asym = len(study.variables) - n_sym
    doms = [r for r in regs if r["role"] == "domain"]
    assert len(doms) == 2 * n_sym + n_asym
    n_in = sum(isinstance(c, InsideConstraint) for c in study.constraints)
    n_out = sum(isinstance(c, OutsideConstraint) for c in study.constraints)
    assert sum(r["role"] == "inside" for r in regs) == n_in
    assert sum(r["role"] == "outside" for r in regs) == n_out
    # geometry equals the plotting helpers (hexahedron: 12 edges), right copy is the y mirror
    space = DesignSpace(study, vehicle)
    v0 = _domain_vertices(space, study.variables[0])
    first, mirror = doms[0], doms[1]
    assert first["point"] == study.variables[0].point and first["side"] == "left"
    assert mirror["mirrored"] and mirror["side"] == "right"
    a = np.array(first["lines"])
    assert a.shape == (12, 2, 3)
    assert {tuple(np.round(p, 12)) for ln in a for p in ln} == {tuple(np.round(p, 12)) for p in v0}
    assert np.allclose(np.array(mirror["lines"]), a * np.array([1, -1, 1]))
    over = [r for r in regs if r["over"]]
    assert over and all("sweep" in r["note"] for r in over)


def test_no_study_is_404(client: TestClient, ws: Path) -> None:
    (d,) = _open(client, "vehicles/fs.yaml")
    assert d["has_study"] is False
    assert client.get(f"{API}/datasets/{d['id']}/regions").status_code == 404


def test_invalid_study_is_rejected(client: TestClient, ws: Path) -> None:
    text = (ws / "studies" / "cons.yaml").read_text(encoding="utf-8")
    (ws / "studies" / "bad.yaml").write_text(
        text.replace("front.uca_front", "front.nope", 1), encoding="utf-8"
    )
    r = client.post(f"{API}/datasets", json={"path": "studies/bad.yaml"})
    assert r.status_code == 400 and "study" in r.text


def test_run_folder_uses_its_study(ws: Path) -> None:
    run = ws / "run"
    copy_inputs(ws / "studies" / "cons.yaml", ws / "vehicles" / "fs.yaml", run)
    shutil.copyfile(run / "vehicle.yaml", run / "best.yaml")
    (run / "history.csv").write_text("eval_id,x_a,f_total,status\n0,1.0,2.0,ok\n", encoding="utf-8")
    (run / "meta.json").write_text(json.dumps({"mode": "single"}), encoding="utf-8")
    (run / "result.json").write_text(
        json.dumps({"mode": "single", "best_eval_id": 0}), encoding="utf-8"
    )
    c = TestClient(create_app(ws, static_dir=ws / "nostatic", study_params=PARAMS))
    ds = _open(c, "run")
    assert [d["has_study"] for d in ds] == [True, True]
    assert len(c.get(f"{API}/datasets/{ds[0]['id']}/regions").json()) > 0


def test_snapshot_carries_regions(ws: Path) -> None:
    mgr = DatasetManager(ws, study_params=PARAMS)
    (ds,) = mgr.open("studies/cons.yaml")
    (plain,) = mgr.open("vehicles/fs.yaml")
    snap: dict[str, Any] = build_snapshot(mgr, include_scene=False)
    assert ds.id in snap["regions"] and len(snap["regions"][ds.id]) > 0
    assert plain.id not in snap["regions"]
    assert json.dumps(snap["regions"], allow_nan=False)
