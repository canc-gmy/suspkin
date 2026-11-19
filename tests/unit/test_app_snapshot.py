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

from fastapi.testclient import TestClient as _TestClient
from typer.testing import CliRunner

import suspkin.app.snapshot as snap_mod
from suspkin.app.datasets import DatasetManager
from suspkin.app.routes import analyse_history
from suspkin.app.server import create_app
from suspkin.app.snapshot import (
    SNAPSHOT_MARKER,
    build_snapshot,
    export_dataset_snapshot,
    export_snapshot,
)
from suspkin.cli import app as cli_app
from tests.unit.helpers import FIXTURE, plain


class TestClient(_TestClient):
    """TestClient that talks to ``localhost`` (the app rejects unknown Host headers)."""

    __test__ = False

    def __init__(self, app: Any, **kw: Any) -> None:
        kw.setdefault("base_url", "http://localhost")
        super().__init__(app, **kw)


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
    m = re.search(
        r"<head>"
        + re.escape(SNAPSHOT_MARKER)
        + r"<script>window\.__SUSPKIN_SNAPSHOT__=(.*?)</script>",
        html,
        re.S,
    )
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
    assert set(snap["scenes"][ds.id]["centres"]) >= {"roll", "fv_ic", "sv_ic", "contact", "pitch"}
    with pytest.raises(FileNotFoundError, match="build:single"):
        export_dataset_snapshot(mgr, ws / "o.html", template=ws / "missing.html")
    with pytest.raises(ValueError, match="nothing to export"):
        export_snapshot([], ws / "o.html", ws, template=ws / "tpl" / "index.html")


def test_app_export_cli_help() -> None:
    res = CliRunner().invoke(cli_app, ["app-export", "--help"])
    assert res.exit_code == 0
    for word in ("--out", "--workspace", "--no-scene"):
        assert word in plain(res.output)
    assert "--dev" in plain(CliRunner().invoke(cli_app, ["app", "--help"]).output)


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


# ------------------------------------------------------------------------------ hardening
import os  # noqa: E402

import suspkin.app.datasets as ds_mod  # noqa: E402
from suspkin.app.datasets import WorkspaceError  # noqa: E402
from suspkin.geometry import load_vehicle  # noqa: E402
from suspkin.report import run_standard_studies, save_bundle  # noqa: E402


def _client(ws: Path) -> TestClient:
    return TestClient(create_app(ws, static_dir=ws / "none", study_params=PARAMS))


def _run_folder(root: Path, name: str = "run1", *, result: str | None = None) -> Path:
    run = root / name
    run.mkdir()
    shutil.copyfile(FIXTURE, run / "vehicle.yaml")
    shutil.copyfile(FIXTURE, run / "best.yaml")
    _history().to_csv(run / "history.csv", index=False)
    (run / "meta.json").write_text(json.dumps({"mode": "weighted"}), encoding="utf-8")
    (run / "result.json").write_text(result or json.dumps({"mode": "weighted"}), encoding="utf-8")
    return run


def test_unc_and_other_drive_paths_rejected(ws: Path) -> None:
    mgr = DatasetManager(ws)
    for bad in ("\\\\server\\share\\car.yaml", "//server/share/car.yaml"):
        with pytest.raises(WorkspaceError, match="network"):
            mgr.resolve(bad)
    c = _client(ws)
    assert c.post(f"{API}/datasets", json={"path": "//server/share/x"}).status_code == 403
    assert c.get(f"{API}/browse", params={"dir": "\\\\server\\share"}).status_code == 403
    if os.name == "nt":
        own = os.path.splitdrive(str(ws))[0].upper()
        other = next(f"{d}:" for d in "DEFGHZ" if f"{d}:" != own)
        with pytest.raises(WorkspaceError):
            mgr.resolve(f"{other}\\car.yaml")
        with pytest.raises(WorkspaceError):
            mgr.resolve(f"{other}car.yaml")  # drive-relative


def test_dotdot_and_case_variants(ws: Path) -> None:
    mgr = DatasetManager(ws)
    with pytest.raises(WorkspaceError):
        mgr.resolve("../outside.yaml")
    with pytest.raises(WorkspaceError):
        mgr.resolve(str(ws.parent / "outside.yaml"))
    if os.name == "nt":  # case-insensitive file system: still inside the workspace
        assert mgr.resolve(str(ws).swapcase() + "\\car.yaml").name.lower() == "car.yaml"
        assert mgr.open(str(ws / "CAR.YAML"))


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")


def test_symlinked_children_outside_workspace_denied(
    ws: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "history.csv"
    _history().to_csv(secret, index=False)
    run = _run_folder(ws, "linked")
    (run / "history.csv").unlink()
    _symlink(run / "history.csv", secret)
    c = _client(ws)
    r = c.post(f"{API}/datasets", json={"path": "linked"})
    assert r.status_code == 403 and "outside the workspace" in r.json()["detail"]
    # a symlinked vehicle file is denied too
    shutil.copyfile(FIXTURE, outside / "car.yaml")
    _symlink(ws / "link.yaml", outside / "car.yaml")
    assert c.post(f"{API}/datasets", json={"path": "link.yaml"}).status_code == 403


def test_export_outside_and_overwrite_rules(
    ws: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snap_mod, "DEFAULT_SINGLE_DIR", ws / "tpl")
    c = _client(ws)
    c.post(f"{API}/datasets", json={"path": "car.yaml"})
    outside = tmp_path_factory.mktemp("elsewhere") / "x.html"
    assert c.post(f"{API}/export", json={"out": str(outside)}).status_code == 403
    assert not outside.exists()
    # a file that is not a snapshot is never replaced, even with overwrite
    (ws / "notes.html").write_text("<html>mine</html>", encoding="utf-8")
    for flag in (False, True):
        r = c.post(f"{API}/export", json={"out": "notes.html", "overwrite": flag})
        assert r.status_code == 400 and "not a suspkin snapshot" in r.json()["detail"]
    assert (ws / "notes.html").read_text(encoding="utf-8") == "<html>mine</html>"
    # a snapshot needs overwrite=true
    assert c.post(f"{API}/export", json={"out": "s.html"}).status_code == 200
    r = c.post(f"{API}/export", json={"out": "s.html"})
    assert r.status_code == 409 and r.json()["detail"] == "already exists: s.html"
    assert c.post(f"{API}/export", json={"out": "s.html", "overwrite": True}).status_code == 200
    assert (ws / "s.html").read_text(encoding="utf-8").count(SNAPSHOT_MARKER) == 1
    assert c.post(f"{API}/export", json={"out": "sub"}).status_code == 400  # not .html


def test_file_changed_after_open_gets_new_id_and_solve(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = ds_mod.run_standard_studies

    def counting(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ds_mod, "run_standard_studies", counting)
    mgr = DatasetManager(ws, study_params=PARAMS)
    (a,) = mgr.open("car.yaml")
    assert mgr.open("car.yaml")[0].id == a.id and len(calls) == 1
    with (ws / "car.yaml").open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n# edited\n")
    (b,) = mgr.open("car.yaml")
    assert b.id != a.id and len(calls) == 2 and len(mgr.list()) == 2


def test_oversize_and_unreadable_inputs(ws: Path) -> None:
    c = _client(ws)
    (ws / "big.yaml").write_text("# " + "x" * 2_100_000 + "\n", encoding="utf-8")
    r = c.post(f"{API}/datasets", json={"path": "big.yaml"})
    assert r.status_code == 413 and "limited" in r.json()["detail"]
    run = _run_folder(ws, "badhist")
    (run / "history.csv").write_bytes(b"\xff\xfe\x00\x00 not csv")
    assert c.post(f"{API}/datasets", json={"path": "badhist"}).status_code == 400
    (ws / "broken.yaml").write_text("a: [unclosed\n", encoding="utf-8")
    assert c.post(f"{API}/datasets", json={"path": "broken.yaml"}).status_code == 400


def test_run_id_follows_history_file(ws: Path) -> None:
    run = _run_folder(ws)
    mgr = DatasetManager(ws, study_params=PARAMS)
    first = mgr.open("run1")[0].run_id
    assert mgr.open("run1")[0].run_id == first
    h = _history()
    h.loc[0, "f_total"] = 99.0
    h.to_csv(run / "history.csv", index=False)
    os.utime(run / "history.csv", ns=(10**18, 10**18))
    assert mgr.open("run1")[0].run_id != first


def test_result_with_nan_and_knee_without_pareto_csv(ws: Path) -> None:
    result = (
        '{"mode": "weighted", "knee_eval_id": 1, '
        '"baseline": {"f_total": NaN, "g": {"a": Infinity}}}'
    )
    _run_folder(ws, "nanrun", result=result)
    c = _client(ws)
    ds = c.post(f"{API}/datasets", json={"path": "nanrun"}).json()
    assert [d["label"] for d in ds] == ["nanrun baseline", "nanrun best"]  # no pareto.csv: no knee
    r = c.get(f"{API}/runs/{ds[0]['run_id']}/history")
    assert r.status_code == 200 and "NaN" not in r.text and "Infinity" not in r.text
    assert r.json()["result"]["baseline"]["f_total"] is None


def test_report_folder_opens_all_or_nothing(ws: Path) -> None:
    res = run_standard_studies(load_vehicle(FIXTURE), **PARAMS)
    (ws / "rep" / "bundles").mkdir(parents=True)
    save_bundle(res, ws / "rep" / "bundles" / "a", label="a")
    save_bundle(res, ws / "rep" / "bundles" / "b", label="b")
    (ws / "rep" / "bundles" / "b" / "summary.json").unlink()
    c = _client(ws)
    assert c.post(f"{API}/datasets", json={"path": "rep"}).status_code == 400
    assert c.get(f"{API}/datasets").json() == []


def test_bundle_id_changes_with_files(ws: Path) -> None:
    res = run_standard_studies(load_vehicle(FIXTURE), **PARAMS)
    save_bundle(res, ws / "b1", label="b1")
    mgr = DatasetManager(ws)
    a = mgr.open("b1")[0].id
    assert mgr.open("b1")[0].id == a
    tab = next((ws / "b1" / "tables").glob("*.csv"))
    tab.write_text(tab.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert mgr.open("b1")[0].id != a


def test_host_header_guard(ws: Path) -> None:
    c = _client(ws)
    assert c.get(f"{API}/health", headers={"Host": "evil.example"}).status_code == 400
    assert c.get(f"{API}/health", headers={"Host": "evil.example:8765"}).status_code == 400
    for good in ("localhost:8765", "127.0.0.1:8765", "[::1]:8765", "LOCALHOST"):
        assert c.get(f"{API}/health", headers={"Host": good}).status_code == 200, good
    custom = TestClient(
        create_app(ws, static_dir=ws / "none", allowed_hosts=["mybox"]), base_url="http://mybox"
    )
    assert custom.get(f"{API}/health").status_code == 200
    anyhost = TestClient(
        create_app(ws, static_dir=ws / "none", allowed_hosts=["*"]), base_url="http://whatever"
    )
    assert anyhost.get(f"{API}/health").status_code == 200


def test_scene_rejects_non_finite_and_out_of_range(ws: Path) -> None:
    c = _client(ws)
    (d,) = c.post(f"{API}/datasets", json={"path": "car.yaml"}).json()
    base = f"{API}/datasets/{d['id']}/scene"
    for q in ("heave=nan", "roll=inf", "pitch=-inf", "rack=nan", "heave=0.5", "roll=1.0"):
        assert c.get(f"{base}?{q}").status_code == 422, q


def test_health_has_pose_metadata_and_kinds(ws: Path) -> None:
    c = _client(ws)
    pose = {p["name"]: p for p in c.get(f"{API}/health").json()["pose"]}
    assert set(pose) == {"heave", "roll", "pitch", "rack"}
    assert pose["heave"]["display_unit"] == "mm" and pose["heave"]["factor"] == pytest.approx(1000)
    assert pose["roll"]["factor"] == pytest.approx(180 / 3.141592653589793)
    schema = c.get("/openapi.json").json()["components"]["schemas"]
    assert "report" not in schema["DatasetOut"]["properties"]["kind"]["enum"]
    assert "report" in schema["BrowseEntry"]["properties"]["kind"]["enum"]


def test_browse_kind_cache(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = ds_mod.classify

    def spy(p: Path) -> str:
        calls.append(p.name)
        return real(p)

    monkeypatch.setattr(ds_mod, "classify", spy)
    c = _client(ws)
    c.get(f"{API}/browse")
    n = len(calls)
    assert n >= 2
    c.get(f"{API}/browse")
    assert len(calls) == n  # unchanged entries are not classified again


def test_run_warns_for_non_loopback_host(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    from suspkin.app.server import run

    served: list[Any] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.append(kw))
    run(ws, host="127.0.0.1", open_browser=False)
    assert "WARNING" not in capsys.readouterr().err
    run(ws, host="0.0.0.0", open_browser=False)
    err = capsys.readouterr().err
    assert "WARNING" in err and "0.0.0.0" in err and len(served) == 2


def test_snapshot_has_hardpoints(ws: Path) -> None:
    mgr = DatasetManager(ws, study_params=PARAMS)
    (ds,) = mgr.open("car.yaml")
    snap = build_snapshot(mgr, include_scene=False)
    rows = snap["hardpoints"][ds.id]
    assert rows and {"axle", "point", "side", "x", "y", "z", "overridden"} <= set(rows[0])
