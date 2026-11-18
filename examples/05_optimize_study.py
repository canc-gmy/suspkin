"""EXPERIMENTAL: evaluate a study baseline and run a tiny optimisation.

``suspkin.optimize`` is experimental: its Python API may change in minor releases (the
study YAML schema is stable).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import suspkin.optimize as opt
from suspkin.optimize.spec import load_study

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "configs" / "studies" / "fs_reference_optimize.yaml"


def main(out: Path) -> None:
    # 1. The `suspkin check` equivalent: evaluate the baseline against the constraints.
    study, vehicle = load_study(STUDY)
    space = opt.DesignSpace(study, vehicle)
    ev = opt.evaluate(space)
    df = ev.display_frame()
    print(df[["name", "value", "min", "max", "unit", "g", "status"]].to_string(index=False))
    print(f"baseline feasible: {ev.feasible}, f_total = {ev.f_total:.4f}")

    # 2. A tiny differential-evolution run (the YAML budget is overridden).
    result = opt.run_optimization(STUDY, out, method="de", evaluations=30, workers=1)
    print(f"{result.n_evals} evaluations in {result.seconds:.1f} s, feasible: {result.feasible}")
    print(f"baseline f_total: {result.baseline.f_total:.4f}")
    print(f"best f_total    : {result.f_total:.4f}")
    print(f"run folder: {result.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "examples" / "05_optimize_study")
    main(ap.parse_args().out)
