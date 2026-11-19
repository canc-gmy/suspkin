"""Regenerate the surrogate search report from an existing surrogate root.

Usage: ``python scripts/surrogate_hpo_report.py --root DIR [--out FILE] [--axles front rear]``
where ``DIR`` holds ``models/<axle>/optuna.db``. Read-only on the databases, so it is safe
while a search is still running. Needs no torch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from suspkin.surrogate.report import write_hpo_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="surrogate root (has models/)")
    ap.add_argument("--out", type=Path, default=None, help="default: ROOT/models/hpo_report.md")
    ap.add_argument("--axles", nargs="+", default=["front", "rear"], choices=["front", "rear"])
    args = ap.parse_args()
    path = write_hpo_report(args.root / "models", args.axles, out=args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
