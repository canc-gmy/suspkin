"""Retrain the best configuration of an existing search, without running new trials.

Usage: ``python scripts/surrogate_retrain_best.py STUDY.yaml --root DIR --axle rear
[--device auto|cpu|cuda]``. Needs ``DIR/dataset/<axle>`` and ``DIR/models/<axle>/optuna.db``;
writes ``models/<axle>/model`` and refreshes ``models/hpo_report.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from suspkin.surrogate.workflow import retrain_surrogate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", type=Path, help="study YAML with a surrogate block")
    ap.add_argument("--root", type=Path, required=True, help="surrogate root (has models/)")
    ap.add_argument("--axle", required=True, choices=["front", "rear"])
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    res = retrain_surrogate(args.study, args.root, axle=args.axle, device=args.device)
    print(
        f"{res.axle}: best trial score {res.best_value:.3f}, retrained {res.final_score:.3f}, "
        f"model {res.model_dir}"
    )


if __name__ == "__main__":
    main()
