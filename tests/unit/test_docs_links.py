"""Guard: documentation pages must not link to files outside ``docs/``.

The site builder cannot resolve such links, so they would break on the published site.

Limits: only inline links in the raw ``.md`` files are checked; reference-style links, ``<path>``
targets and files pulled in via snippets (CHANGELOG.md, examples/*.py) are not.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK = re.compile(r"(\]\()(?!\w+:|#|/)([^)\s]+)(\))")


def _escaping_targets(page: Path) -> list[str]:
    page_dir = posixpath.dirname(page.relative_to(DOCS).as_posix())
    bad: list[str] = []
    fence: str | None = None
    for line in page.read_text(encoding="utf-8").splitlines():
        m = _FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)[0]
        elif fence is not None and m and m.group(1)[0] == fence:
            fence = None
        elif fence is None:
            for link in _LINK.finditer(line):
                path = link.group(2).partition("#")[0]
                if not path:
                    continue
                repo_path = posixpath.normpath(posixpath.join("docs", page_dir, path))
                if not repo_path.startswith("docs/"):
                    bad.append(f"{page.relative_to(DOCS)}: {link.group(2)}")
    return bad


def test_no_relative_link_leaves_docs() -> None:
    bad: list[str] = []
    for page in sorted(DOCS.rglob("*.md")):
        if "superpowers" in page.relative_to(DOCS).parts:
            continue
        bad += _escaping_targets(page)
    assert bad == []
