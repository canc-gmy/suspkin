"""Hardpoint editing endpoints: derive (mirror, right-only, errors, ids) and save."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import yaml
from fastapi.testclient import TestClient as _TestClient

import suspkin.app.datasets as ds_mod
from suspkin.app.server import create_app
from suspkin.geometry import load_vehicle
from suspkin.kinematics import KinematicsError
from tests.unit.helpers import FIXTURE

PARAMS = {"steps": 5, "travel": 0.02, "roll_deg": 1.0}
API = "/api/v1"


class TestClient(_TestClient):
    __test__ = False

    def __init__(self, app: Any, **kw: Any) -> None:
        kw.setdefault("base_url", "http://localhost")
        super().__init__(app, **kw)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    shutil.copyfile(FIXTURE, tmp_path / "car.yaml")
    return tmp_path


@pytest.fixture
def client(ws: Path) -> TestClient:
    return TestClient(create_app(ws, static_dir=ws / "nostatic", study_params=PARAMS))


def _car(c: TestClient) -> dict[str, Any]:
    r = c.post(f"{API}/datasets", json={"path": "car.yaml"})
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()[0]
    return out


def _derive(c: TestClient, src: str, edits: list[dict[str, Any]], **kw: Any) -> Any:
    return c.post(f"{API}/datasets/{src}/derive", json={"edits": edits, **kw})


def _hp(c: TestClient, did: str, axle: str, point: str, side: str) -> dict[str, Any]:
    rows = c.get(f"{API}/datasets/{did}/hardpoints").json()
    (row,) = [r for r in rows if (r["axle"], r["point"], r["side"]) == (axle, point, side)]
    return row  # type: ignore[no-any-return]


def _edit(side: str, **xyz: float) -> dict[str, Any]:
    return {"axle": "front", "point": "uca_outer", "side": side, **xyz}


def test_derive_both_mirrors(client: TestClient) -> None:
    src = _car(client)
    base_l = _hp(client, src["id"], "front", "uca_outer", "left")
    r = _derive(client, src["id"], [_edit("both", z=base_l["z"] + 0.01)])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["kind"] == "edited" and d["parent_id"] == src["id"]
    assert d["label"] == "car edited" and d["resolution_fixed"] is False
    assert d["edits"][0]["side"] == "both" and d["steps"] == PARAMS["steps"]
    left = _hp(client, d["id"], "front", "uca_outer", "left")
    right = _hp(client, d["id"], "front", "uca_outer", "right")
    assert left["z"] == pytest.approx(base_l["z"] + 0.01)
    assert left["x"] == pytest.approx(base_l["x"]) and left["y"] == pytest.approx(base_l["y"])
    assert (right["x"], right["y"], right["z"]) == pytest.approx((left["x"], -left["y"], left["z"]))
    assert right["overridden"] is False
    assert len(client.get(f"{API}/datasets/{d['id']}/tables/heave").json()["columns"]["heave"]) == 5
    assert [x["id"] for x in client.get(f"{API}/datasets").json()] == [src["id"], d["id"]]


def test_derive_both_drops_existing_override(ws: Path, client: TestClient) -> None:
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    raw["front"]["right_overrides"] = {"uca_outer": [0.01, -0.56, 0.31]}
    (ws / "asym.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    r = client.post(f"{API}/datasets", json={"path": "asym.yaml"})
    src = r.json()[0]
    d = _derive(client, src["id"], [_edit("both", z=0.30)]).json()
    right = _hp(client, d["id"], "front", "uca_outer", "right")
    left = _hp(client, d["id"], "front", "uca_outer", "left")
    assert right["overridden"] is False and right["y"] == pytest.approx(-left["y"])


def test_derive_left_keeps_override_and_right_writes_override_only(
    ws: Path, client: TestClient
) -> None:
    src = _car(client)
    base_r = _hp(client, src["id"], "front", "uca_outer", "right")
    base_l = _hp(client, src["id"], "front", "uca_outer", "left")
    # right only: the left point does not move, the override is in right-side coordinates
    d = _derive(client, src["id"], [_edit("right", z=base_r["z"] + 0.005)]).json()
    right = _hp(client, d["id"], "front", "uca_outer", "right")
    assert right["overridden"] is True and right["z"] == pytest.approx(base_r["z"] + 0.005)
    assert right["y"] == pytest.approx(base_r["y"]) and right["y"] < 0
    left = _hp(client, d["id"], "front", "uca_outer", "left")
    assert (left["x"], left["y"], left["z"]) == pytest.approx(
        (base_l["x"], base_l["y"], base_l["z"])
    )
    # then a left edit of that revision keeps the right override
    d2 = _derive(client, d["id"], [_edit("left", x=base_l["x"] + 0.004)]).json()
    right2 = _hp(client, d2["id"], "front", "uca_outer", "right")
    assert right2["overridden"] is True
    assert (right2["x"], right2["y"], right2["z"]) == pytest.approx(
        (right["x"], right["y"], right["z"])
    )
    assert _hp(client, d2["id"], "front", "uca_outer", "left")["x"] == pytest.approx(
        base_l["x"] + 0.004
    )
    assert d2["parent_id"] == d["id"]


def test_derive_label_steps_and_dedupe(client: TestClient) -> None:
    src = _car(client)
    e = [_edit("both", z=0.30)]
    d = _derive(client, src["id"], e, label="raised", steps=7).json()
    assert d["label"] == "raised" and d["steps"] == 7
    d2 = _derive(client, src["id"], [_edit("both", z=0.31)], label="raised").json()
    assert d2["label"] == "raised (2)"
    # resolution of an edited revision can change
    r = client.patch(f"{API}/datasets/{d['id']}", json={"steps": 9})
    assert r.status_code == 200 and r.json()["steps"] == 9


def test_derive_id_is_deterministic_and_cached(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = ds_mod.run_standard_studies

    def counting(*a: Any, **k: Any) -> Any:
        calls.append(k["steps"])
        return real(*a, **k)

    monkeypatch.setattr(ds_mod, "run_standard_studies", counting)
    c = TestClient(create_app(ws, static_dir=ws / "nostatic", study_params=PARAMS))
    src = _car(c)
    e = [_edit("both", z=0.30)]
    a = _derive(c, src["id"], e).json()
    n = len(calls)
    assert _derive(c, src["id"], e).json()["id"] == a["id"]  # still open: returned as is
    assert len(calls) == n
    c.delete(f"{API}/datasets/{a['id']}")
    b = _derive(c, src["id"], e).json()  # closed: same id again, solved tables from the cache
    assert b["id"] == a["id"] and len(calls) == n
    other = _derive(c, src["id"], e, steps=9).json()  # the resolution is part of the id
    assert other["id"] != a["id"]


def test_derive_errors_400(client: TestClient) -> None:
    src = _car(client)
    sid = src["id"]
    # unknown point
    r = _derive(client, sid, [{"axle": "front", "point": "nope", "side": "both", "x": 0.0}])
    assert r.status_code == 400 and "front.nope" in r.json()["detail"]
    # no coordinate
    r = _derive(client, sid, [_edit("both")])
    assert r.status_code == 400 and "uca_outer" in r.json()["detail"]
    # invalid geometry: a right point on the left side (y > 0 in right_overrides)
    r = _derive(client, sid, [_edit("right", y=0.5)])
    assert r.status_code == 400
    assert "front.uca_outer" in r.json()["detail"] and "right_overrides" in r.json()["detail"]
    # malformed body
    assert _derive(client, sid, []).status_code == 422
    assert _derive(client, sid, [_edit("middle", x=0.0)]).status_code == 422
    assert (
        client.post(f"{API}/datasets/nope/derive", json={"edits": [_edit("both", x=0)]}).status_code
        == 404
    )
    # nothing was registered
    assert [x["id"] for x in client.get(f"{API}/datasets").json()] == [sid]


def test_derive_solver_failure_422(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _car(client)

    def boom(*a: Any, **k: Any) -> Any:
        raise KinematicsError("no convergence at travel 0.020000 m")

    monkeypatch.setattr(ds_mod, "run_standard_studies", boom)
    r = _derive(client, src["id"], [_edit("both", z=0.30)])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "sweep" in detail and "no convergence" in detail
    assert len(client.get(f"{API}/datasets").json()) == 1


def test_failing_sweep_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    veh = load_vehicle(FIXTURE)
    assert ds_mod._failing_sweep(veh, PARAMS) is None

    def boom(*a: Any, **k: Any) -> Any:
        raise KinematicsError("x")

    monkeypatch.setattr(ds_mod, "sweep_vehicle", boom)
    assert ds_mod._failing_sweep(veh, PARAMS) == "steering rack"


def test_save_writes_schema_1_lf_and_reloads(ws: Path, client: TestClient) -> None:
    src = _car(client)
    d = _derive(client, src["id"], [_edit("both", z=0.30)]).json()
    r = client.post(f"{API}/datasets/{d['id']}/save", json={"path": "out/new.yaml"})
    assert r.status_code == 400  # folder must exist
    (ws / "out").mkdir()
    r = client.post(f"{API}/datasets/{d['id']}/save", json={"path": "out/new.yaml"})
    assert r.status_code == 200, r.text
    assert r.json()["path"] == "out/new.yaml"
    raw = (ws / "out" / "new.yaml").read_bytes()
    assert b"\r" not in raw and yaml.safe_load(raw)["schema"] == 1
    assert r.json()["size"] == len(raw)
    veh = load_vehicle(ws / "out" / "new.yaml")
    assert veh.front.hardpoints.uca_outer[2] == pytest.approx(0.30)


def test_save_guards_and_overwrite(ws: Path, client: TestClient) -> None:
    src = _car(client)
    sid = src["id"]
    post = lambda body: client.post(f"{API}/datasets/{sid}/save", json=body)  # noqa: E731
    assert post({"path": "../escape.yaml"}).status_code == 403
    assert post({"path": str(ws.parent / "escape.yaml")}).status_code == 403
    assert post({"path": "car.txt"}).status_code == 400
    assert not (ws.parent / "escape.yaml").exists()
    # existing file: 409 until overwrite is set
    before = (ws / "car.yaml").read_bytes()
    r = post({"path": "car.yaml"})
    assert r.status_code == 409 and r.json()["detail"] == "already exists: car.yaml"
    assert (ws / "car.yaml").read_bytes() == before
    d = _derive(client, sid, [_edit("both", z=0.30)]).json()
    r = client.post(f"{API}/datasets/{d['id']}/save", json={"path": "car.yaml", "overwrite": True})
    assert r.status_code == 200
    assert load_vehicle(ws / "car.yaml").front.hardpoints.uca_outer[2] == pytest.approx(0.30)
    # a file that is not a vehicle is never replaced
    (ws / "notes.yaml").write_text("a: 1\n", encoding="utf-8")
    r = post({"path": "notes.yaml", "overwrite": True})
    assert r.status_code == 400 and (ws / "notes.yaml").read_text(encoding="utf-8") == "a: 1\n"
    assert client.post(f"{API}/datasets/nope/save", json={"path": "x.yaml"}).status_code == 404


def test_scene_and_hardpoints_work_for_edited(client: TestClient) -> None:
    pytest.importorskip("plotly")
    src = _car(client)
    d = _derive(client, src["id"], [_edit("both", z=0.30)]).json()
    s = client.get(f"{API}/datasets/{d['id']}/scene").json()
    assert s["hardpoints"]["FL:uca_outer"][2] == pytest.approx(0.30)
