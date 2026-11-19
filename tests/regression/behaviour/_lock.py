"""Behaviour-lock helpers: scrub volatile text, snapshot a folder of outputs, compare.

A lock case stores ``stdout.txt``, ``exit_code.txt`` and every file written under the output
folder in ``snapshots/<name>/``. Text is scrubbed of paths, timestamps, durations, version and
hashes first. CSV / JSON / YAML are compared by value (numeric tolerance, exact columns and
row counts); ``.md`` / ``.html`` / ``.txt`` exactly; binary files (``.png`` ``.gif`` ``.mp4``
...) by existence and ``size > 0`` only. ``_files.txt`` lists every output file so extra or
missing files fail the comparison.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _golden import flatten, refuse_in_ci

import suspkin

SNAP = Path(__file__).resolve().parent / "snapshots"
REPO = Path(__file__).resolve().parents[3]
RTOL = 1e-12
ATOL = 1e-12

# Wall-clock columns / keys: layout is locked, the measured values are not.
_VOLATILE = re.compile(r"(^|[._])(seconds|elapsed|wall_?time|duration|timestamp)($|[._])")

_TEXT_EXACT = {".md", ".html", ".txt"}
_TABLE = {".csv"}
_TREE = {".json", ".yaml", ".yml"}
_BINARY = {".png", ".gif", ".mp4", ".svg", ".pdf", ".npz", ".npy", ".pt", ".pth"}

_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_STAMP = re.compile(r"\b\d{8}_\d{6}\b")
_DUR = re.compile(r"\d+(?:\.\d+)? ?s\b")
_DEC_SECONDS = re.compile(r"\d+\.\d+ s\b")
_TIMING_LINE = re.compile(r"^done:|time|elapsed|took|wall|runtime|duration|\bin \d", re.IGNORECASE)
_GIT = re.compile(r"\b[0-9a-f]{40}\b")
_STUDY_HASH = re.compile(r"(study_hash[\"']?\s*[:=]\s*[\"']?)[0-9A-Za-z]+")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_B64 = re.compile(r"(data:[\w/+.-]+;base64,)[A-Za-z0-9+/=]+")
_PATH_TAIL = re.compile(r"(<TMP>|<REPO>)([\w./\\-]*)")
_SCRIPT = re.compile(r"(<script[^>]*>)(.*?)(</script>)", re.DOTALL)
_BIG = 100_000


def _big_script(m: re.Match[str]) -> str:
    if len(m.group(2)) < _BIG:
        return m.group(0)
    digest = hashlib.sha256(m.group(2).encode("utf-8")).hexdigest()[:16]
    return f"{m.group(1)}<LARGE-SCRIPT sha256={digest} len={len(m.group(2))}>{m.group(3)}"


def scrub(text: str, *, tmp: Path) -> str:
    """Replace volatile substrings (paths, timestamps, durations, hashes, version) by tokens."""
    text = text.replace("\r\n", "\n")
    text = _ANSI.sub("", text)
    roots = {tmp, tmp.resolve()}
    variants: list[str] = []
    for r in roots:
        for s in (str(r), r.as_posix(), str(r).replace("\\", "\\\\")):
            variants.append(s)
            variants.append(s.replace("/", "\\/"))
    for s in sorted(set(variants), key=len, reverse=True):
        text = text.replace(s, "<TMP>")
    repo = []
    for s in (str(REPO), REPO.as_posix(), str(REPO).replace("\\", "\\\\")):
        repo.append(s)
    for s in sorted(set(repo), key=len, reverse=True):
        text = text.replace(s, "<REPO>")
    text = _PATH_TAIL.sub(lambda m: m.group(1) + m.group(2).replace("\\", "/"), text)
    text = _B64.sub(r"\1<B64>", text)
    text = _SCRIPT.sub(_big_script, text)
    text = _DEC_SECONDS.sub("<DUR>", text)
    text = _ISO.sub("<TS>", text)
    text = _STAMP.sub("<STAMP>", text)
    text = _UUID.sub("<UUID>", text)
    text = _GIT.sub("<GITHASH>", text)
    text = _STUDY_HASH.sub(r"\1<STUDYHASH>", text)
    text = re.sub(rf"(?<![\w.]){re.escape(suspkin.__version__)}(?![\w.])", "<VERSION>", text)
    lines = [_DUR.sub("<DUR>", ln) if _TIMING_LINE.search(ln) else ln for ln in text.split("\n")]
    return "\n".join(lines)


def _cmp_scalar(new: Any, old: Any, key: str, where: str) -> None:
    if isinstance(old, bool) or isinstance(new, bool) or old is None or new is None:
        assert new == old, f"{where}:{key}: {new!r} != {old!r}"
    elif isinstance(old, (int, float)) and isinstance(new, (int, float)):
        np.testing.assert_allclose(
            new, old, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=f"{where}:{key}"
        )
    else:
        assert new == old, f"{where}:{key}: {new!r} != {old!r}"


def _cmp_tree(new_text: str, old_text: str, loader: Any, where: str) -> None:
    new, old = flatten(loader(new_text)), flatten(loader(old_text))
    added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    assert not (added or removed), f"{where}: keys added={added[:10]} removed={removed[:10]}"
    for k, v in old.items():
        if _VOLATILE.search(k):
            continue
        _cmp_scalar(new[k], v, k, where)


def _cmp_csv(new_text: str, old_text: str, where: str) -> None:
    new = pd.read_csv(io.StringIO(new_text))
    old = pd.read_csv(io.StringIO(old_text))
    assert list(new.columns) == list(old.columns), f"{where}: columns differ"
    assert len(new) == len(old), f"{where}: rows {len(new)} != {len(old)}"
    for col in old.columns:
        if _VOLATILE.search(str(col)):
            continue
        a, b = new[col], old[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            np.testing.assert_allclose(
                a.to_numpy(dtype=float),
                b.to_numpy(dtype=float),
                rtol=RTOL,
                atol=ATOL,
                equal_nan=True,
                err_msg=f"{where}:{col}",
            )
        else:
            assert a.astype(str).tolist() == b.astype(str).tolist(), f"{where}:{col}"


def _compare_text(rel: str, new_text: str, old_text: str) -> None:
    suffix = Path(rel).suffix.lower()
    if suffix in _TABLE:
        _cmp_csv(new_text, old_text, rel)
    elif suffix == ".json":
        _cmp_tree(new_text, old_text, json.loads, rel)
    elif suffix in {".yaml", ".yml"}:
        _cmp_tree(new_text, old_text, yaml.safe_load, rel)
    else:
        assert new_text == old_text, f"{rel}: text differs"


def _collect(out: Path) -> list[Path]:
    return sorted(p for p in out.rglob("*") if p.is_file())


def snapshot_dir(name: str, out: Path, stdout: str, code: int, *, update: bool, tmp: Path) -> None:
    """Write (``update``) or compare the snapshot ``snapshots/<name>`` of one lock case."""
    dest = SNAP / name
    files = {p.relative_to(out).as_posix(): p for p in _collect(out)} if out.exists() else {}
    texts: dict[str, str] = {"stdout.txt": scrub(stdout, tmp=tmp), "exit_code.txt": f"{code}\n"}
    for rel, p in files.items():
        suffix = p.suffix.lower()
        if suffix in _BINARY or suffix not in (_TEXT_EXACT | _TABLE | _TREE):
            assert p.stat().st_size > 0, f"{name}/{rel}: empty file"
            continue
        texts[f"files/{rel}"] = scrub(p.read_text(encoding="utf-8"), tmp=tmp)
    texts["_files.txt"] = "\n".join(sorted(files)) + "\n"

    if update:
        refuse_in_ci()
        if dest.exists():
            shutil.rmtree(dest)
        for rel, text in texts.items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
        return

    if not dest.exists():
        pytest.fail(f"missing lock snapshot {name}: run with --update-lock to create it")
    stored = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert stored == set(texts), (
        f"{name}: snapshot files differ: added={sorted(set(texts) - stored)} "
        f"removed={sorted(stored - set(texts))}"
    )
    for rel, new_text in texts.items():
        old_text = (dest / rel).read_text(encoding="utf-8")
        if rel.startswith("files/"):
            _compare_text(f"{name}/{rel}", new_text, old_text)
        else:
            assert new_text == old_text, f"{name}/{rel}: differs"
