"""Web app commands: app, app-export."""

from pathlib import Path
from typing import Annotated

import typer

from suspkin.cli._common import fail


def app_cmd(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Vehicle YAMLs, bundles, report or optimizer run folders to open."),
    ] = None,
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8765,
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Root folder the app may read (default: cwd)."),
    ] = None,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Do not open the browser.")
    ] = False,
    dev: Annotated[
        bool,
        typer.Option("--dev", help="API only: use the Vite dev server (npm run dev) for the UI."),
    ] = False,
) -> None:
    """Start the experimental local web app (needs the 'app' extra).

    Snapshot export is a separate command (``app-export``) because a Typer group with a
    variadic PATHS argument cannot also have sub-commands.
    """
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        fail("the web app needs the 'app' extra: uv sync --extra app", 2)
    from suspkin.app.server import run

    root = (workspace or Path.cwd()).resolve()
    if not root.is_dir():
        fail(f"error: workspace is not a directory: {root}", 2)
    run(
        root,
        host=host,
        port=port,
        preload=[p.resolve() for p in (paths or [])],
        open_browser=not no_browser,
        dev=dev,
    )


def app_export_cmd(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Vehicle YAMLs, bundles, report or optimizer run folders to embed."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Output HTML file.")] = Path("compare.html"),
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Root folder the paths must be inside (default: cwd)."),
    ] = None,
    no_scene: Annotated[
        bool, typer.Option("--no-scene", help="Leave out the 3D scenes (smaller file).")
    ] = False,
) -> None:
    """Export a self-contained HTML snapshot of the given results (needs the 'app' extra)."""
    try:
        import fastapi  # noqa: F401
    except ImportError:
        fail("the web app needs the 'app' extra: uv sync --extra app", 2)
    from suspkin.app.snapshot import export_snapshot

    root = (workspace or Path.cwd()).resolve()
    try:
        size = export_snapshot([p.resolve() for p in paths], out, root, include_scene=not no_scene)
    except Exception as exc:
        fail(f"error: {exc}")
    typer.echo(f"Snapshot exported to {out} ({size / 1e6:.1f} MB)")
