"""The dist check that the built web app is packaged in the wheel and sdist."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_dist_frontend.py"

FULL = [
    "suspkin/__init__.py",
    "suspkin/app/static/index.html",
    "suspkin/app/static/assets/index-x.js",
    "suspkin/app/static_single/index.html",
]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dist_frontend", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_dist_frontend"] = mod
    spec.loader.exec_module(mod)
    return mod


def _wheel(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for n in names:
            zf.writestr(n, "x")
    return path


def _sdist(path: Path, names: list[str]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for n in names:
            info = tarfile.TarInfo(n)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
    return path


def test_complete_wheel_ok(tmp_path: Path) -> None:
    mod = _load_script()
    assert mod.missing_frontend(_wheel(tmp_path / "s-2.0.0-py3-none-any.whl", FULL)) == []


def test_complete_sdist_ok(tmp_path: Path) -> None:
    mod = _load_script()
    names = ["suspkin-2.0.0/src/" + n for n in FULL]
    assert mod.missing_frontend(_sdist(tmp_path / "suspkin-2.0.0.tar.gz", names)) == []


def test_missing_single_reported(tmp_path: Path) -> None:
    mod = _load_script()
    names = [n for n in FULL if "static_single" not in n]
    whl = _wheel(tmp_path / "s.whl", names)
    assert mod.missing_frontend(whl) == ["suspkin/app/static_single/index.html"]


def test_missing_assets_reported(tmp_path: Path) -> None:
    mod = _load_script()
    names = [n for n in FULL if "assets" not in n]
    assert mod.missing_frontend(_wheel(tmp_path / "s.whl", names)) == ["suspkin/app/static/assets/"]


def test_empty_wheel_reports_all(tmp_path: Path) -> None:
    mod = _load_script()
    assert len(mod.missing_frontend(_wheel(tmp_path / "s.whl", FULL[:1]))) == 3


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    mod = _load_script()
    with pytest.raises(ValueError):
        mod.missing_frontend(_wheel(tmp_path / "foo.zip", FULL))


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    ok = _wheel(tmp_path / "ok.whl", FULL)
    bad = _wheel(tmp_path / "bad.whl", FULL[:1])
    assert mod.main([str(ok)]) == 0
    assert mod.main([str(ok), str(bad)]) == 1
    assert "missing" in capsys.readouterr().out
    assert mod.main([]) == 2
    assert mod.main([str(_wheel(tmp_path / "foo.zip", FULL))]) == 2


def test_node_modules_in_sdist_reported(tmp_path: Path) -> None:
    mod = _load_script()
    names = ["suspkin-2.0.0/src/" + n for n in FULL]
    names.append("suspkin-2.0.0/frontend/node_modules/vite/package.json")
    sdist = _sdist(tmp_path / "suspkin-2.0.0.tar.gz", names)
    assert mod.unexpected_members(sdist) == ["node_modules/"]
    assert mod.main([str(sdist)]) == 1


def test_clean_sdist_has_no_unexpected(tmp_path: Path) -> None:
    mod = _load_script()
    names = ["suspkin-2.0.0/src/" + n for n in FULL]
    assert mod.unexpected_members(_sdist(tmp_path / "suspkin-2.0.0.tar.gz", names)) == []
