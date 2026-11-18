"""Dump the OpenAPI schema of the app (input of the frontend type generation).

Usage: ``python -m suspkin.app.openapi OUT.json``
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from suspkin.app.server import create_app

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Write ``create_app(cwd).openapi()`` to the given file."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m suspkin.app.openapi OUT.json", file=sys.stderr)
        return 2
    out = Path(args[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = create_app(Path.cwd()).openapi()
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(schema, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
