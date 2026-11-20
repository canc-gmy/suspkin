"""FastAPI application factory and uvicorn helper (experimental)."""

from __future__ import annotations

import logging
import sys
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
    AlreadyExistsError,
    DatasetError,
    DatasetManager,
    DatasetNotFoundError,
    DatasetTooLargeError,
    SolveFailedError,
    WorkspaceError,
)
from suspkin.app.routes import make_router

__all__ = ["DEFAULT_STATIC_DIR", "create_app", "run"]

log = logging.getLogger("suspkin.app")

DEFAULT_STATIC_DIR = Path(__file__).parent / "static"
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
_WILDCARD_HOSTS = ("0.0.0.0", "::", "")


def _host_name(header: str) -> str:
    """Host name of a ``Host`` header value (port and IPv6 brackets removed, lower case)."""
    h = header.strip().lower()
    if h.startswith("["):
        return h[1 : h.find("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


_VITE_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

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
    allowed_hosts: Sequence[str] = (),
) -> FastAPI:
    """Create the app: ``/api/v1`` plus the built frontend (or a hint page) at ``/``.

    Args:
        workspace (Path): Root folder; every path the API touches must be inside it.
        static_dir (Path, optional): Built frontend (default ``suspkin/app/static``); may be
            missing.
        preload (sequence of Path): Paths opened at start-up; failures are logged and skipped.
        study_params (mapping, optional): Overrides for ``run_standard_studies`` (e.g. ``{"steps":
            7}``).
        allowed_hosts (sequence of str): Extra ``Host`` header names accepted besides localhost
            (DNS-rebinding guard); ``"*"``
            disables the check.
    """
    app = FastAPI(title="suspkin app (experimental)", version=__version__)
    manager = DatasetManager(workspace, study_params=study_params)
    app.state.manager = manager

    hosts = {h.lower() for h in (*_LOCAL_HOSTS, *allowed_hosts)}

    # Reject requests whose Host header is not an allowed local name (DNS-rebinding guard).
    @app.middleware("http")
    async def host_guard(request: Request, call_next: Any) -> Response:
        if "*" not in hosts and _host_name(request.headers.get("host", "")) not in hosts:
            return JSONResponse({"detail": "invalid Host header"}, status_code=400)
        response: Response = await call_next(request)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_VITE_ORIGINS,
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
    app.add_exception_handler(DatasetTooLargeError, _err(413))
    app.add_exception_handler(DatasetError, _err(400))
    app.add_exception_handler(SolveFailedError, _err(422))
    app.add_exception_handler(AlreadyExistsError, _err(409))

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

        # Serve built frontend files; unknown paths fall back to index.html (client routing).
        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> Response:
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "not found"}, status_code=404)
            cand = (sdir / full_path).resolve()
            if full_path and cand.is_file() and cand.is_relative_to(sdir):
                return FileResponse(cand)
            return FileResponse(sdir / "index.html")

    else:
        # No built frontend found: show a placeholder page.
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
    dev: bool = False,
) -> None:
    """Serve the app with uvicorn (blocking); optionally open the browser shortly after start.

    ``dev``: serve only the API (with CORS for the Vite dev server on port 5173) and print
    the Vite URL; no static files are served and no browser is opened.

    Args:
        workspace: Root folder served to the API.
        host: Bind address.
        port: TCP port.
        preload: Paths opened at start-up.
        open_browser: Open the default browser shortly after start (ignored with ``dev``).
        static_dir: Built frontend folder; ``None`` uses the packaged one.
        dev: API only, see above.
    """
    import uvicorn

    if dev:
        static_dir = workspace / ".suspkin-no-static"  # missing on purpose: API only
        open_browser = False
        print(f"API on http://{host}:{port}/api/v1 (docs: /docs)")
        print("Frontend dev server: cd frontend && npm run dev, then open http://localhost:5173/")
    loopback = host in ("localhost", "127.0.0.1", "::1")
    if not loopback:
        print(
            f"WARNING: binding to {host!r}: anyone who can reach this machine can read every file "
            f"under {workspace}. Use the default 127.0.0.1 unless you trust the network.",
            file=sys.stderr,
        )
    allowed = ["*"] if host in _WILDCARD_HOSTS else [host]
    app = create_app(workspace, static_dir=static_dir, preload=preload, allowed_hosts=allowed)
    if open_browser:
        shown = "localhost" if host in ("0.0.0.0", "::") else host
        threading.Timer(1.0, webbrowser.open, args=(f"http://{shown}:{port}/",)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
