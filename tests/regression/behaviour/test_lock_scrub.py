"""The report lock must not depend on today's date."""

from __future__ import annotations

import re
from pathlib import Path

from behaviour._lock import SNAP, scrub


def test_bare_date_is_tokenised(tmp_path: Path) -> None:
    assert scrub("<td>Date</td><td>2026-11-20</td>", tmp=tmp_path) == "<td>Date</td><td><DATE></td>"
    assert scrub("- Date: 1999-01-02\n", tmp=tmp_path) == "- Date: <DATE>\n"


def test_full_timestamp_still_ts(tmp_path: Path) -> None:
    assert scrub("at 2026-11-20T13:28:58Z", tmp=tmp_path) == "at <TS>"


def test_no_literal_date_in_snapshots() -> None:
    hits = [
        p
        for p in SNAP.rglob("*")
        if p.is_file()
        and re.search(r"\b\d{4}-\d{2}-\d{2}\b", p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert hits == []
