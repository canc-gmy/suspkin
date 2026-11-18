"""Static snapshot export: open datasets embedded in one self-contained HTML file.

The page is the single-file frontend build (``npm run build:single``) with the data the
frontend needs inlined as ``window.__SUSPKIN_SNAPSHOT__`` (read by ``SnapshotSource``).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from suspkin.app.datasets import DatasetManager
from suspkin.app.routes import (
    _column,
    _dataset_out,
    hardpoints_out,
    health_out,
    history_out,
    regions_out,
    scene_json,
)
from suspkin.optimize.spec import StudyConfigError
from suspkin.plotting.spec import spec_json

__all__ = [
    "DEFAULT_SINGLE_DIR",
    "SNAPSHOT_MARKER",
    "WARN_BYTES",
    "build_snapshot",
    "export_dataset_snapshot",
    "export_snapshot",
    "is_snapshot_file",
]

DEFAULT_SINGLE_DIR = Path(__file__).parent / "static_single"
SNAPSHOT_MARKER = '<meta name="suspkin-snapshot" content="1">'
WARN_BYTES = 25_000_000
log = logging.getLogger("suspkin.app")


def is_snapshot_file(path: Path) -> bool:
    """True if ``path`` is a page written by :func:`export_dataset_snapshot` (has the marker)."""
    try:
        with path.open("rb") as f:
            head = f.read(8192)
    except OSError:
        return False
    return SNAPSHOT_MARKER.encode("utf-8") in head


def build_snapshot(manager: DatasetManager, *, include_scene: bool = True) -> dict[str, Any]:
    """Everything the frontend reads from a snapshot (JSON-safe)."""
    datasets = manager.list()
    tables = {
        d.id: {
            study: {"columns": {str(c): _column(df[c]) for c in df.columns}}
            for study, df in d.tables.items()
        }
        for d in datasets
    }
    histories = {
        rid: history_out(manager.run(rid)).model_dump(mode="json")
        for rid in sorted({d.run_id for d in datasets if d.run_id})
    }
    scenes: dict[str, Any] = {}
    if include_scene:
        try:
            scenes = {d.id: scene_json(d) for d in datasets}
        except ImportError:
            scenes = {}  # plotly is not installed: the 3D tab reports no scene
    regions: dict[str, Any] = {}
    for d in datasets:
        if d.study_path is not None:
            try:
                regions[d.id] = [r.model_dump(mode="json") for r in regions_out(d)]
            except StudyConfigError:
                continue  # the study file changed or vanished: that revision shows no regions
    return {
        "health": health_out(manager).model_dump(mode="json")
        | {"workspace": manager.workspace.name},
        "figures": spec_json(),
        "datasets": [_dataset_out(d).model_dump(mode="json") for d in datasets],
        "tables": tables,
        "summaries": {d.id: d.summary for d in datasets},
        "histories": histories,
        "scenes": scenes,
        "regions": regions,
        "hardpoints": {
            d.id: [h.model_dump(mode="json") for h in hardpoints_out(d)] for d in datasets
        },
    }


def _embed(snapshot: dict[str, Any]) -> str:
    """JSON for a ``<script>`` body: every ``<`` is escaped, so ``</script>`` cannot appear."""
    text = json.dumps(snapshot, allow_nan=False, separators=(",", ":"))
    # ensure_ascii already escapes U+2028/2029; only "<" needs care (</script>, <!--)
    return text.replace("<", "\\u003c")


def export_dataset_snapshot(
    manager: DatasetManager,
    out_html: Path,
    *,
    include_scene: bool = True,
    template: Path | None = None,
) -> int:
    """Write the open datasets of ``manager`` as a snapshot page; returns the size in bytes."""
    tpl = template if template is not None else DEFAULT_SINGLE_DIR / "index.html"
    if not tpl.is_file():
        raise FileNotFoundError(
            f"single-file frontend not built ({tpl}); run: cd frontend && npm run build:single"
        )
    html = tpl.read_text(encoding="utf-8")
    data = _embed(build_snapshot(manager, include_scene=include_scene))
    script = f"<script>window.__SUSPKIN_SNAPSHOT__={data}</script>"
    html, n = re.subn(
        r"<head[^>]*>", lambda m: m.group(0) + SNAPSHOT_MARKER + script, html, count=1
    )
    if n != 1:
        raise ValueError(f"template has no <head>: {tpl}")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    blob = html.encode("utf-8")
    with out_html.open("wb") as f:
        f.write(blob)
    log.info("snapshot %s: %.1f MB", out_html, len(blob) / 1e6)
    if len(blob) > WARN_BYTES:
        log.warning(
            "snapshot %s is %.0f MB; consider --no-scene or fewer revisions",
            out_html,
            len(blob) / 1e6,
        )
    return len(blob)


def export_snapshot(
    paths: Sequence[Path | str],
    out_html: Path | str,
    workspace: Path | str,
    *,
    include_scene: bool = True,
    template: Path | None = None,
) -> int:
    """Open ``paths`` (inside ``workspace``) and write a self-contained snapshot page.

    Returns the size of ``out_html`` in bytes. Only the given paths are opened.
    """
    manager = DatasetManager(Path(workspace))
    for p in paths:
        manager.open(p)
    if not manager.list():
        raise ValueError("nothing to export: no path given")
    return export_dataset_snapshot(
        manager, Path(out_html), include_scene=include_scene, template=template
    )
