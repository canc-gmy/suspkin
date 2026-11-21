"""Check that built distributions contain the built web app frontend.

The frontend build (``src/suspkin/app/static`` and ``static_single``) is git-ignored and is
added to the wheel and sdist through hatch ``artifacts``. If it is missing, ``suspkin app``
serves only the API. Run after ``uv build``::

    python scripts/check_dist_frontend.py dist/*.whl dist/*.tar.gz

Exit codes: 0 all files complete, 1 a file lacks the frontend, 2 usage error.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

REQUIRED: tuple[str, ...] = (
    "suspkin/app/static/index.html",
    "suspkin/app/static_single/index.html",
)
ASSETS_PREFIX = "suspkin/app/static/assets/"


def archive_members(path: Path) -> list[str]:
    """List the member names of a wheel or sdist.

    Args:
        path: A ``.whl`` (zip) or ``.tar.gz`` (gzip tar) file.

    Returns:
        Member names with ``/`` separators.

    Raises:
        ValueError: The file is neither a ``.whl`` nor a ``.tar.gz``.
    """
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tf:
            return tf.getnames()
    raise ValueError(f"unsupported distribution file: {path}")


def missing_frontend(path: Path) -> list[str]:
    """Return the frontend entries a distribution lacks.

    A member matches an entry when it equals it (wheel layout) or ends with ``/`` + entry
    (sdist layout, ``suspkin-X.Y.Z/src/suspkin/...``).

    Args:
        path: A ``.whl`` or ``.tar.gz`` file.

    Returns:
        The missing entries of ``REQUIRED``, plus ``ASSETS_PREFIX`` when no member lies
        under it; empty when the frontend is complete.

    Raises:
        ValueError: Unsupported file type.
    """
    names = archive_members(path)
    missing = [e for e in REQUIRED if not any(n == e or n.endswith("/" + e) for n in names)]
    has_assets = any(
        len(n) > len(ASSETS_PREFIX) and (n.startswith(ASSETS_PREFIX) or "/" + ASSETS_PREFIX in n)
        for n in names
    )
    if not has_assets:
        missing.append(ASSETS_PREFIX)
    return missing


def main(argv: list[str]) -> int:
    """Check each given distribution file and print what is missing.

    Args:
        argv: Paths of distribution files.

    Returns:
        0 when every file is complete, 1 when one lacks the frontend, 2 on usage errors.
    """
    if not argv:
        print("usage: check_dist_frontend.py DIST_FILE [DIST_FILE ...]")
        return 2
    bad = False
    for arg in argv:
        path = Path(arg)
        try:
            missing = missing_frontend(path)
        except (ValueError, OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
            print(f"error: {exc}")
            return 2
        for entry in missing:
            print(f"{path}: missing {entry}")
        bad = bad or bool(missing)
        if not missing:
            print(f"{path}: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
