"""MkDocs hook: rewrite relative links that leave docs/ to GitHub blob URLs.

Only inline links ``[text](path)`` are rewritten, and fenced code blocks (``` and ~~~)
are skipped. Known limits: reference-style links (``[text][ref]`` / ``[ref]: path``) and
angle-bracket targets (``[text](<path>)``) are left untouched.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any

BLOB_URL = "https://github.com/canc-gmy/suspkin/blob/main/"
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK = re.compile(r"(\]\()(?!\w+:|#|/)([^)\s]+)(\))")


def on_page_markdown(markdown: str, page: Any, **kwargs: Any) -> str:
    page_dir = posixpath.dirname(page.file.src_uri)

    def repl(match: re.Match[str]) -> str:
        target = match.group(2)
        path, sep, anchor = target.partition("#")
        if not path:
            return match.group(0)
        repo_path = posixpath.normpath(posixpath.join("docs", page_dir, path))
        if repo_path.startswith("docs/") or repo_path.startswith(".."):
            return match.group(0)
        return f"{match.group(1)}{BLOB_URL}{repo_path}{sep}{anchor}{match.group(3)}"

    out: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        m = _FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)[0]
        elif fence is not None and m and m.group(1)[0] == fence:
            fence = None
        elif fence is None:
            line = _LINK.sub(repl, line)
        out.append(line)
    return "".join(out)
