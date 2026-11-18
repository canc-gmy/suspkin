"""FastAPI application factory and uvicorn helper (experimental)."""

from __future__ import annotations

import logging
import threading
import webbrowser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from suspkin import __version__
from suspkin.app.datasets import (
    DatasetError,
    DatasetManager,
    DatasetNotFoundError,
    WorkspaceError,
)
from suspkin.app.routes import make_router

__all__ = ["DEFAULT_STATIC_DIR", "create_app", "run"]

log = logging.getLogger("suspkin.app")

DEFAULT_STATIC_DIR = Path(__file__).parent / "static"
_VITE_ORIGIN = "http://localhost:5173"

_NO_FRONTEND = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>suspkin app</title></head>
<body style="font-family: sans-serif; max-width: 40em; margin: 3em auto">
<h1>suspkin app (experimental)</h1>
<p>The API is running (version {version}); the frontend has not been built.</p>
<pre>cd frontend &amp;&amp; npm ci &amp;&amp; npm run build</pre>
<p>API docs: <a href="/docs">/docs</a>,
figure spec: <a href="/api/v1/figures">/api/v1/figures</a>.</p>
</body></html>
"""


def create_app(
    workspace: Path,
    *,
    static_dir: Path | None = None,
    preload: Sequence[Path] = (),
    study_params: Mapping[str, Any] | None = None,
) -> FastAPI:
    """Create the app: ``/api/v1`` plus the built frontend (or a hint page) at ``/``.

    Parameters
    ----------
    workspace : Path
        Root folder; every path the API touches must be inside it.
    static_dir : Path, optional
        Built frontend (default ``suspkin/app/static``); may be missing.
    preload : sequence of Path
        Paths opened at start-up; failures are logged and skipped.
    study_params : mapping, optional
        Overrides for ``run_standard_studies`` (e.g. ``{"steps": 7}``).
    """
    app = FastAPI(title="suspkin app (experimental)", version=__version__)
    manager = DatasetManager(workspace, study_params=study_params)
    app.state.manager = manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[_VITE_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _err(status: int) -> Any:
        def handler(_: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"detail": str(exc)}, status_code=status)

        return handler

    app.add_exception_handler(WorkspaceError, _err(403))
    app.add_exception_handler(FileNotFoundError, _err(404))
    app.add_exception_handler(DatasetNotFoundError, _err(404))
    app.add_exception_handler(DatasetError, _err(400))

    app.include_router(make_router(manager))

    for p in preload:
        try:
            manager.open(p)
        except Exception as exc:
            log.warning("could not open %s: %s", p, exc)

    sdir = (DEFAULT_STATIC_DIR if static_dir is None else Path(static_dir)).resolve()
    if (sdir / "index.html").is_file():
        if (sdir / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=sdir / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> Response:
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "not found"}, status_code=404)
            cand = (sdir / full_path).resolve()
            if full_path and cand.is_file() and cand.is_relative_to(sdir):
                return FileResponse(cand)
            return FileResponse(sdir / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def no_frontend() -> HTMLResponse:
            return HTMLResponse(_NO_FRONTEND.format(version=__version__))

    return app


def run(
    workspace: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    preload: Sequence[Path] = (),
    open_browser: bool = True,
    static_dir: Path | None = None,
) -> None:
    """Serve the app with uvicorn (blocking); optionally open the browser shortly after start."""
    import uvicorn

    app = create_app(workspace, static_dir=static_dir, preload=preload)
    if open_browser:
        shown = "localhost" if host in ("0.0.0.0", "::") else host
        threading.Timer(1.0, webbrowser.open, args=(f"http://{shown}:{port}/",)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
