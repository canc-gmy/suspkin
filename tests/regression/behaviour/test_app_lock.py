"""Behaviour lock of the web-app API: status code and JSON body of every endpoint.

One scripted session over a workspace holding the reference vehicle, a study, a report
bundle, a report folder and a fake optimizer run folder. Requests follow the order used by
``tests/unit/test_app_*.py``; each response is stored as ``NN_<label>.json`` with its status
and body. ``POST /export`` runs against a stub single-file template (the real one is a
git-ignored frontend build). ``test_openapi_unchanged`` pins the generated OpenAPI schema to
``frontend/src/api/openapi.json``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient as _TestClient

from behaviour._lock import REPO, snapshot_dir
from suspkin.app.server import create_app
from suspkin.geometry import load_vehicle
from suspkin.kinematics import _numba
from suspkin.report import build_report, run_standard_studies, save_bundle

pytestmark = pytest.mark.lock

API = "/api/v1"
CFG = REPO / "configs"
PARAMS = {"steps": 5, "travel": 0.02, "roll_deg": 1.0}


class TestClient(_TestClient):
    """TestClient talking to ``localhost`` (the app rejects unknown Host headers)."""

    __test__ = False

    def __init__(self, app: Any, **kw: Any) -> None:
        kw.setdefault("base_url", "http://localhost")
        super().__init__(app, **kw)


class Recorder:
    """Writes every response to ``out/NN_<label>.json``."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.n = 0

    def __call__(self, label: str, resp: Any) -> Any:
        self.n += 1
        try:
            body: Any = resp.json()
        except ValueError:
            body = {"text": resp.text}
        doc = {"status": resp.status_code, "body": body}
        (self.out / f"{self.n:02d}_{label}.json").write_text(
            json.dumps(doc, indent=1, allow_nan=True), encoding="utf-8"
        )
        return body


def _make_run(root: Path) -> None:
    run = root / "run1"
    run.mkdir()
    veh = CFG / "vehicles" / "fs_reference.yaml"
    shutil.copyfile(veh, run / "vehicle.yaml")
    shutil.copyfile(veh, run / "best.yaml")
    (run / "history.csv").write_text(
        "eval_id,x_a,f_total,status\n0,1.0,2.0,ok\n1,1.5,1.0,ok\n2,2.0,,failed\n",
        encoding="utf-8",
    )
    (run / "meta.json").write_text(json.dumps({"mode": "single", "seed": 1}), encoding="utf-8")
    (run / "pareto").mkdir()
    shutil.copyfile(veh, run / "pareto" / "pareto_0.yaml")
    (run / "pareto.csv").write_text("k,eval_id,knee\n0,1,True\n", encoding="utf-8")
    result = {"mode": "pareto", "best_eval_id": 1, "knee_eval_id": 1}
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_app_lock(tmp_path: Path, update_lock: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUSPKIN_NO_NUMBA", "1")
    _numba.reset()
    import suspkin.app.snapshot as snap

    stub = tmp_path / "single"
    stub.mkdir()
    (stub / "index.html").write_text(
        "<!doctype html><html><head><title>stub</title></head><body></body></html>\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(snap, "DEFAULT_SINGLE_DIR", stub)

    ws = tmp_path / "ws"
    (ws / "vehicles").mkdir(parents=True)
    (ws / "studies").mkdir()
    (ws / "other").mkdir()
    veh = CFG / "vehicles" / "fs_reference.yaml"
    shutil.copyfile(veh, ws / "car.yaml")
    shutil.copyfile(veh, ws / "vehicles" / "fs.yaml")
    shutil.copyfile(veh, ws / "other" / "car.yaml")
    text = (CFG / "studies" / "fs_reference_constraints.yaml").read_text(encoding="utf-8")
    (ws / "studies" / "cons.yaml").write_text(
        text.replace("../vehicles/fs_reference.yaml", "../vehicles/fs.yaml"), encoding="utf-8"
    )
    (ws / "junk.yaml").write_text("a: 1\n", encoding="utf-8")
    (ws / "notes.txt").write_text("x", encoding="utf-8")
    (ws / "plain").mkdir()
    res = run_standard_studies(load_vehicle(veh), **PARAMS)
    save_bundle(res, ws / "b1", label="mybundle")
    build_report([("one", res), ("two", res)], ws / "rep", formats=("png",), include_3d=False)
    _make_run(ws)

    out = tmp_path / "out"
    out.mkdir()
    rec = Recorder(out)
    c = TestClient(create_app(ws, static_dir=ws / "nostatic", study_params=PARAMS))

    rec("health", c.get(f"{API}/health"))
    rec("figures", c.get(f"{API}/figures"))
    rec("browse_root", c.get(f"{API}/browse"))
    rec("browse_rep", c.get(f"{API}/browse", params={"dir": "rep"}))
    rec("browse_outside_403", c.get(f"{API}/browse", params={"dir": ".."}))

    (car,) = rec("open_car", c.post(f"{API}/datasets", json={"path": "car.yaml"}))
    cid = car["id"]
    rec("open_car_again", c.post(f"{API}/datasets", json={"path": "car.yaml"}))
    rec("list", c.get(f"{API}/datasets"))
    for study in car["studies"]:
        rec(f"table_{study}", c.get(f"{API}/datasets/{cid}/tables/{study}"))
    rec("table_unknown_404", c.get(f"{API}/datasets/{cid}/tables/nope"))
    rec("summary", c.get(f"{API}/datasets/{cid}/summary"))
    rec("hardpoints", c.get(f"{API}/datasets/{cid}/hardpoints"))
    rec("regions_no_study_404", c.get(f"{API}/datasets/{cid}/regions"))
    rec("scene_static", c.get(f"{API}/datasets/{cid}/scene"))
    rec(
        "scene_posed",
        c.get(f"{API}/datasets/{cid}/scene", params={"heave": 0.01, "roll": 0.01}),
    )
    rec("scene_out_of_range_422", c.get(f"{API}/datasets/{cid}/scene", params={"heave": 5}))

    (other,) = rec("open_other_car", c.post(f"{API}/datasets", json={"path": "other/car.yaml"}))
    rec(
        "patch_label_color",
        c.patch(f"{API}/datasets/{cid}", json={"label": "front car", "color": 5}),
    )
    rec(
        "patch_label_taken_400",
        c.patch(f"{API}/datasets/{other['id']}", json={"label": "front car"}),
    )
    rec("patch_color_422", c.patch(f"{API}/datasets/{other['id']}", json={"color": -1}))
    rec("patch_steps", c.patch(f"{API}/datasets/{cid}", json={"steps": 7}))
    rec("patch_steps_422", c.patch(f"{API}/datasets/{cid}", json={"steps": 2}))
    rec("table_heave_after_steps", c.get(f"{API}/datasets/{cid}/tables/heave"))

    edit = {"axle": "front", "point": "uca_outer", "side": "both", "z": 0.30}
    (derived,) = [
        rec(
            "derive",
            c.post(f"{API}/datasets/{cid}/derive", json={"edits": [edit], "label": "raised"}),
        )
    ]
    did = derived["id"]
    rec("derive_hardpoints", c.get(f"{API}/datasets/{did}/hardpoints"))
    rec(
        "derive_unknown_point_400",
        c.post(
            f"{API}/datasets/{cid}/derive",
            json={"edits": [{"axle": "front", "point": "nope", "side": "both", "x": 0.0}]},
        ),
    )
    rec("derive_empty_422", c.post(f"{API}/datasets/{cid}/derive", json={"edits": []}))
    rec("derive_unknown_dataset_404", c.post(f"{API}/datasets/nope/derive", json={"edits": [edit]}))
    rec("save_no_folder_400", c.post(f"{API}/datasets/{did}/save", json={"path": "saved/new.yaml"}))
    (ws / "saved").mkdir()
    rec("save", c.post(f"{API}/datasets/{did}/save", json={"path": "saved/new.yaml"}))
    rec("save_exists_409", c.post(f"{API}/datasets/{did}/save", json={"path": "saved/new.yaml"}))
    rec(
        "save_overwrite",
        c.post(f"{API}/datasets/{did}/save", json={"path": "saved/new.yaml", "overwrite": True}),
    )
    rec("save_escape_403", c.post(f"{API}/datasets/{did}/save", json={"path": "../escape.yaml"}))
    rec("save_bad_ext_400", c.post(f"{API}/datasets/{did}/save", json={"path": "car.txt"}))
    shutil.copyfile(ws / "saved" / "new.yaml", out / "saved_new.yaml")

    (study,) = rec("open_study", c.post(f"{API}/datasets", json={"path": "studies/cons.yaml"}))
    rec("regions", c.get(f"{API}/datasets/{study['id']}/regions"))
    (bundle,) = rec("open_bundle", c.post(f"{API}/datasets", json={"path": "b1"}))
    rec("patch_bundle_steps_400", c.patch(f"{API}/datasets/{bundle['id']}", json={"steps": 9}))
    rec("open_report", c.post(f"{API}/datasets", json={"path": "rep"}))
    runs = rec("open_run", c.post(f"{API}/datasets", json={"path": "run1"}))
    rec("run_history", c.get(f"{API}/runs/{runs[0]['run_id']}/history"))
    rec("run_history_404", c.get(f"{API}/runs/zzz/history"))
    rec("list_all", c.get(f"{API}/datasets"))

    rec("open_outside_403", c.post(f"{API}/datasets", json={"path": "../x.yaml"}))
    rec("open_missing_404", c.post(f"{API}/datasets", json={"path": "missing.yaml"}))
    rec("open_junk_400", c.post(f"{API}/datasets", json={"path": "junk.yaml"}))
    rec("open_emptydir_400", c.post(f"{API}/datasets", json={"path": "plain"}))
    rec("summary_unknown_404", c.get(f"{API}/datasets/zzz/summary"))

    rec("export_bad_name_400", c.post(f"{API}/export", json={"out": "snap.txt"}))
    rec("export", c.post(f"{API}/export", json={"out": "snap.html"}))
    rec("export_exists_409", c.post(f"{API}/export", json={"out": "snap.html"}))
    rec("export_overwrite", c.post(f"{API}/export", json={"out": "snap.html", "overwrite": True}))
    shutil.copyfile(ws / "snap.html", out / "snap.html")

    for i, d in enumerate(c.get(f"{API}/datasets").json()):
        rec(f"delete_{i}", c.delete(f"{API}/datasets/{d['id']}"))
    rec("delete_unknown_404", c.delete(f"{API}/datasets/zzz"))
    rec("list_empty", c.get(f"{API}/datasets"))
    rec("run_history_after_close_404", c.get(f"{API}/runs/{runs[0]['run_id']}/history"))
    rec("api_unknown_404", c.get(f"{API}/nothing"))

    # dataset / run ids hash the absolute path of the workspace: map them to stable tokens
    ids: dict[str, str] = {}
    for f in sorted(out.glob("*.json")):
        for m in re.finditer(r'"(?:id|run_id)": ?"([0-9a-f]{10})"', f.read_text(encoding="utf-8")):
            ids.setdefault(m.group(1), f"<ID{len(ids) + 1}>")
    for f in sorted([*out.glob("*.json"), out / "snap.html"]):
        txt = f.read_text(encoding="utf-8")
        for real, token in ids.items():
            txt = txt.replace(real, token)
        f.write_text(txt, encoding="utf-8")
    snapshot_dir("app", out, "", 0, update=update_lock, tmp=tmp_path)


def test_openapi_unchanged() -> None:
    schema = create_app(Path.cwd()).openapi()
    committed = json.loads((REPO / "frontend" / "src" / "api" / "openapi.json").read_text("utf-8"))
    got = json.loads(json.dumps(schema))
    # info.version of the committed file lags the package version (stale since 1.1.0); the
    # paths and schemas are what the frontend types are generated from.
    got["info"].pop("version", None)
    committed["info"].pop("version", None)
    assert got == committed
