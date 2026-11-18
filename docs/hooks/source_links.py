"""MkDocs hook: rewrite relative links that leave docs/ to GitHub blob URLs."""

from __future__ import annotations

import posixpath
import re
from typing import Any

BLOB_URL = "https://github.com/canc-gmy/suspkin/blob/main/"
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

    return _LINK.sub(repl, markdown)
