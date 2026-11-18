"""Resumable benchmark driver of the v1.1 surrogate (DESIGN_PLAN 13.13).

Stages (each skipped when its output exists; ``--force STAGE`` redoes it):

1. ``sample``    both axles (``suspkin surrogate sample``), into ROOT/dataset
2. ``train``     architecture search + training of both axles, into ROOT/models
3. ``validate``  test-split error per channel, into ROOT/validation
4. ``al``        active-learning optimisation (``optimize --surrogate``), into RUN/al
5. ``de_same``   direct DE with the same number of FOM evaluations as the AL run
6. ``de_total``  direct DE with the FOM-equivalent cost of AL + dataset
                 (one corner sample is counted as one evaluation: assumption)
7. ``report``    RUN/benchmark.md and RUN/benchmark.json

Run: ``python -m uv run python scripts/benchmark_surrogate.py [--dry-run]``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

STAGES = ("sample", "train", "validate", "al", "de_same", "de_total", "report")
AXLES = ("front", "rear")
DEFAULT_STUDY = Path("configs/studies/fs_reference_surrogate.yaml")
DEFAULT_BASE = Path("results/benchmark_surrogate")
DE_TIME = "48 h"  # effectively unlimited: the evaluation budget governs the direct DE runs
_CLI = "from suspkin.cli import app; app()"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--study", type=Path, default=DEFAULT_STUDY, help="Benchmark study YAML.")
    p.add_argument(
        "--root", type=Path, default=DEFAULT_BASE / "surrogate", help="Surrogate folder."
    )
    p.add_argument("--run", type=Path, default=DEFAULT_BASE / "run", help="Run folder.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--workers", default=None, help="Worker processes: an integer or auto.")
    p.add_argument(
        "--force",
        action="append",
        default=[],
        choices=STAGES,
        metavar="STAGE",
        help=f"Redo a stage ({', '.join(STAGES)}); repeatable.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return p


def _cli(*args: str) -> list[str]:
    return [sys.executable, "-c", _CLI, *args]


def stage_commands(a: argparse.Namespace, n_al: int = 0, n_total: int = 0) -> dict[str, list[str]]:
    """Command of each subprocess stage (``n_al``, ``n_total`` fill the DE budgets)."""
    study, root, run = str(a.study), str(a.root), Path(a.run)
    dev = ["--device", a.device] if a.device else []
    wrk = ["--workers", str(a.workers)] if a.workers else []
    return {
        "sample": _cli("surrogate", "sample", study, "--root", root, "--resume", *wrk),
        "train": _cli("surrogate", "train", study, "--root", root, *dev),
        "validate": _cli("surrogate", "validate", study, "--root", root, *dev),
        "al": _cli("optimize", study, "--surrogate", root, "--out", str(a.run / "al"), *dev, *wrk),
        "de_same": _cli(
            "optimize",
            study,
            "--method",
            "de",
            "--evaluations",
            str(n_al),
            "--time",
            DE_TIME,
            "--no-report",
            "--out",
            str(run / "de_same"),
            *wrk,
        ),
        "de_total": _cli(
            "optimize",
            study,
            "--method",
            "de",
            "--evaluations",
            str(n_total),
            "--time",
            DE_TIME,
            "--no-report",
            "--out",
            str(run / "de_total"),
            *wrk,
        ),
    }


def outputs(a: argparse.Namespace) -> dict[str, Path]:
    """File whose existence marks a stage as done."""
    return {
        "sample": a.root / "dataset" / "rear" / "meta.json",
        "train": a.root / "models" / "rear" / "model" / "model.pt",
        "validate": a.root / "validation" / "rear" / "validation.csv",
        "al": a.run / "al" / "result.json",
        "de_same": a.run / "de_same" / "result.json",
        "de_total": a.run / "de_total" / "result.json",
        "report": a.run / "benchmark.json",
    }


def print_plan(a: argparse.Namespace) -> None:
    cmds = stage_commands(a, n_al=0, n_total=0)
    outs = outputs(a)
    print("Benchmark plan (FOM budgets of de_same and de_total are read at run time)")
    for st in STAGES:
        state = "done" if outs[st].exists() and st not in a.force else "todo"
        print(f"  [{state}] {st:<9} -> {outs[st]}")
        if st in cmds:
            print("      " + " ".join(cmds[st][3:]))


# --------------------------------------------------------------------------- helpers


def _stages_path(a: argparse.Namespace) -> Path:
    return a.run / "stages.json"


def _load_stages(a: argparse.Namespace) -> dict[str, Any]:
    p = _stages_path(a)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _remove(a: argparse.Namespace, stage: str) -> None:
    target = {
        "sample": a.root / "dataset",
        "train": a.root / "models",
        "validate": a.root / "validation",
        "al": a.run / "al",
        "de_same": a.run / "de_same",
        "de_total": a.run / "de_total",
    }.get(stage)
    if target is not None and target.exists():
        shutil.rmtree(target)
    if stage == "report":
        for name in ("benchmark.json", "benchmark.md"):
            (a.run / name).unlink(missing_ok=True)


def _run(cmd: list[str], ok_codes: tuple[int, ...] = (0,)) -> float:
    print("$ " + " ".join(cmd[3:]), flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, check=False).returncode
    dt = time.perf_counter() - t0
    if rc not in ok_codes:
        raise SystemExit(f"stage command failed with exit code {rc}: {' '.join(cmd[3:])}")
    return dt


def _n_dataset(a: argparse.Namespace) -> dict[str, int]:
    out: dict[str, int] = {}
    for ax in AXLES:
        meta = json.loads((a.root / "dataset" / ax / "meta.json").read_text(encoding="utf-8"))
        out[ax] = int(meta["n_initial"])
    return out


def _al_info(a: argparse.Namespace) -> dict[str, Any]:
    al_dir = a.run / "al"
    al = pd.read_csv(al_dir / "al.csv")
    meta = json.loads((al_dir / "meta.json").read_text(encoding="utf-8"))
    info = meta.get("active_learning", {})
    return {
        "n_fom": int(al["n_fom"].iloc[-1]) if len(al) else 0,
        "iterations": len(al),
        "stop_reason": str(info.get("stop_reason", "unknown")),
    }


def _history_rows(run_dir: Path) -> int:
    return len(pd.read_csv(run_dir / "history.csv", usecols=["eval_id"]))


def _run_summary(run_dir: Path) -> dict[str, Any]:
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    return {
        "best_f": float(res["best"]["f_total"]),
        "feasible": bool(res["best"]["feasible"]),
        "baseline_f": float(res["baseline"]["f_total"]),
        "baseline_feasible": bool(res["baseline"]["feasible"]),
        "fom_evaluations": _history_rows(run_dir),
    }


def _validation(a: argparse.Namespace) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for ax in AXLES:
        path = a.root / "validation" / ax / "validation.csv"
        if path.is_file():
            tab = pd.read_csv(path)
            out[ax] = {str(c): bool(p) for c, p in zip(tab["channel"], tab["passed"], strict=True)}
    return out


def build_summary(a: argparse.Namespace, seconds: dict[str, float]) -> dict[str, Any]:
    n_ds = _n_dataset(a)
    al = _al_info(a)
    runs = {k: _run_summary(a.run / k) for k in ("al", "de_same", "de_total")}
    return {
        "study": str(a.study),
        "assumption": "one corner dataset sample is counted as one FOM evaluation",
        "seconds": seconds,
        "dataset_samples": n_ds,
        "al": al,
        "runs": runs,
        "baseline_f": runs["al"]["baseline_f"],
        "baseline_feasible": runs["al"]["baseline_feasible"],
        "validation": _validation(a),
        "equivalent_cost": {
            "al": al["n_fom"],
            "dataset": sum(n_ds.values()),
            "total": al["n_fom"] + sum(n_ds.values()),
        },
    }


def render_markdown(s: dict[str, Any]) -> str:
    lines = [
        "# Surrogate benchmark",
        "",
        f"Study: `{s['study']}`. Baseline f = {s['baseline_f']:.4f} "
        f"(feasible: {'yes' if s['baseline_feasible'] else 'no'}).",
        f"Cost assumption: {s['assumption']}.",
        "",
        "## Approaches",
        "",
        "| approach | FOM evaluations | best FOM-verified f | feasible |",
        "|---|---:|---:|:---:|",
    ]
    for key, label in (
        ("al", "active learning"),
        ("de_same", "direct DE, same FOM evals as AL"),
        ("de_total", "direct DE, FOM-equivalent cost of AL + dataset"),
    ):
        r = s["runs"][key]
        lines.append(
            f"| {label} | {r['fom_evaluations']} | {r['best_f']:.4f} | "
            f"{'yes' if r['feasible'] else 'no'} |"
        )
    cost = s["equivalent_cost"]
    lines += [
        "",
        f"FOM-equivalent cost of AL + dataset: {cost['al']} (AL) + {cost['dataset']} "
        f"(dataset: {s['dataset_samples']}) = {cost['total']}.",
        f"Active learning: {s['al']['iterations']} iterations, "
        f"stop reason `{s['al']['stop_reason']}`.",
        "",
        "## Wall time per stage",
        "",
        "| stage | seconds |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v:.1f} |" for k, v in s["seconds"].items()]
    lines += ["", "## Validation per channel", ""]
    for ax, chans in s["validation"].items():
        bad = [c for c, ok in chans.items() if not ok]
        lines.append(f"- {ax}: {len(chans) - len(bad)}/{len(chans)} channels within tolerance")
        lines.append(f"  - over tolerance: {', '.join(bad) if bad else 'none'}")
    lines += ["", "| axle | channel | status |", "|---|---|:---:|"]
    for ax, chans in s["validation"].items():
        lines += [f"| {ax} | {c} | {'pass' if ok else 'FAIL'} |" for c, ok in chans.items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- driver


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.dry_run:
        print_plan(a)
        return 0
    a.run.mkdir(parents=True, exist_ok=True)
    seconds: dict[str, float] = dict(_load_stages(a))
    outs = outputs(a)

    def stage(name: str, cmd: list[str] | None, ok_codes: tuple[int, ...] = (0,)) -> None:
        if name in a.force:
            _remove(a, name)
        # sample and train write their marker early: also require a recorded finish.
        finished = name in seconds or name not in ("sample", "train")
        if outs[name].exists() and finished:
            print(f"[skip] {name}: {outs[name]} exists", flush=True)
            return
        print(f"[stage] {name}", flush=True)
        if cmd is None:
            return
        seconds[name] = _run(cmd, ok_codes)
        _stages_path(a).write_text(json.dumps(seconds, indent=2), encoding="utf-8")

    cmds = stage_commands(a)
    stage("sample", cmds["sample"])
    stage("train", cmds["train"])
    # validate exits 1 when a channel is over tolerance: that is a result, not a failure.
    stage("validate", cmds["validate"], ok_codes=(0, 1))
    # AL and the DE runs exit 1 when no feasible design is found: also a result.
    stage("al", cmds["al"], ok_codes=(0, 1))
    al = _al_info(a)
    n_ds = _n_dataset(a)
    n_total = al["n_fom"] + sum(n_ds.values())
    print(
        f"FOM evaluations: AL {al['n_fom']}, dataset {n_ds}, equivalent total {n_total} "
        "(one corner sample counted as one evaluation)",
        flush=True,
    )
    cmds = stage_commands(a, n_al=al["n_fom"], n_total=n_total)
    stage("de_same", cmds["de_same"], ok_codes=(0, 1))
    stage("de_total", cmds["de_total"], ok_codes=(0, 1))

    if "report" in a.force:
        _remove(a, "report")
    summary = build_summary(a, seconds)
    (a.run / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (a.run / "benchmark.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))
    print(f"wrote {a.run / 'benchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
