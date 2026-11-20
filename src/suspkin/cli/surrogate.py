"""Surrogate commands: sample, train, validate."""

from pathlib import Path
from typing import Annotated, Any

import typer

from suspkin.cli._common import fail


def surrogate_sample(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML with a surrogate block.")],
    root: Annotated[
        Path | None, typer.Option("--root", help="Surrogate folder (dataset/ is written here).")
    ] = None,
    axle: Annotated[str, typer.Option("--axle", help="front | rear | both.")] = "both",
    n: Annotated[
        int | None, typer.Option("--n", min=1, help="LHS samples per axle (default: study).")
    ] = None,
    workers: Annotated[
        str | None, typer.Option("--workers", help="Worker processes: an integer or auto.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue an interrupted sampling in --root.")
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", hidden=True, help="Deprecated alias of --root.")
    ] = None,
) -> None:
    """Sample the initial Latin-hypercube dataset of the surrogate (experimental).

    Writes ROOT/dataset/<axle>/shard_*.npz and meta.json. Exit code 2 on a study or argument
    error.
    """
    from suspkin.optimize.spec import StudyConfigError
    from suspkin.surrogate.data import Axle, NoAxleVariablesError
    from suspkin.surrogate.dataset import DatasetError, sample_axle

    if root is None:
        root = out
    elif out is not None and out != root:
        fail("error: give --root only (--out is a deprecated alias)", 2)
    if root is None:
        fail("error: missing option --root", 2)
    axles: list[Axle]
    if axle == "both":
        axles = ["front", "rear"]
    elif axle == "front":
        axles = ["front"]
    elif axle == "rear":
        axles = ["rear"]
    else:
        fail(f"error: --axle must be front, rear or both, got {axle!r}", 2)
    try:
        n_workers: int | str | None = None
        if workers is not None:
            if workers.strip().lower() == "auto":
                n_workers = "auto"
            else:
                n_workers = int(workers)
                if n_workers < 1:
                    raise ValueError("--workers must be >= 1 or auto")
        done = 0
        for ax in axles:
            try:
                rep = sample_axle(
                    study_yaml, root, ax, n=n, workers=n_workers, resume=resume, echo=typer.echo
                )
            except NoAxleVariablesError as exc:
                if axle != "both":
                    raise
                typer.echo(f"{ax}: skipped ({exc})")
                continue
            done += 1
            typer.echo(
                f"{ax}: {rep.n_samples} samples ({rep.n_new} new), "
                f"feasible {100 * rep.feasible_fraction:.1f} %, {rep.seconds:.1f} s"
            )
            if rep.warning:
                typer.echo(f"warning: {rep.warning}", err=True)
        if done == 0:
            raise StudyConfigError("no axle to sample")
    except (StudyConfigError, DatasetError, ValueError, OSError) as exc:
        fail(f"error: {exc}", 2)


def _parse_axles(axle: str) -> list[str]:
    if axle == "both":
        return ["front", "rear"]
    if axle in ("front", "rear"):
        return [axle]
    fail(f"error: --axle must be front, rear or both, got {axle!r}", 2)


def _present_axles(root: Path, axles: list[str], axle: str) -> list[Any]:
    """Axles of ``axles`` that have a dataset (``both`` skips the missing ones)."""
    have = [a for a in axles if (root / "dataset" / a / "meta.json").is_file()]
    if axle != "both":
        missing = [a for a in axles if a not in have]
        if missing:
            fail(f"error: no dataset for {missing[0]} in {root / 'dataset'}", 2)
    elif not have:
        fail(f"error: no dataset in {root / 'dataset'}; run `surrogate sample` first", 2)
    return list(have)


def surrogate_train(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML with a surrogate block.")],
    root: Annotated[Path, typer.Option("--root", help="Surrogate folder (holds dataset/).")],
    axle: Annotated[str, typer.Option("--axle", help="front | rear | both.")] = "both",
    trials: Annotated[
        int | None, typer.Option("--trials", min=1, help="Optuna trials (default: study).")
    ] = None,
    timeout: Annotated[
        str | None,
        typer.Option("--timeout", help="Search time budget, e.g. '2 h' (default: study)."),
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="auto | cpu | cuda (default: study).")
    ] = None,
) -> None:
    """Search the architecture and train the networks (experimental).

    Writes ROOT/models/<axle>/ (optuna.db, trials.csv, model/). Exit code 2 on a study,
    argument or data error.
    """
    from suspkin.optimize.spec import StudyConfigError
    from suspkin.surrogate.dataset import DatasetError
    from suspkin.surrogate.workflow import train_surrogate
    from suspkin.units import parse_quantity

    axles = _parse_axles(axle)
    if device is not None and device not in ("auto", "cpu", "cuda"):
        fail(f"error: --device must be auto, cpu or cuda, got {device!r}", 2)
    try:
        tmo = None if timeout is None else parse_quantity(timeout, "time")
        if tmo is not None and tmo <= 0:
            raise ValueError("--timeout must be positive")
        use = _present_axles(root, axles, axle)
        train_surrogate(
            study_yaml,
            root,
            axles=use,
            n_trials=trials,
            timeout=tmo,
            device=device,
            echo=typer.echo,
        )
    except (StudyConfigError, DatasetError, ValueError, OSError, RuntimeError) as exc:
        fail(f"error: {exc}", 2)


def surrogate_validate(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML with a surrogate block.")],
    root: Annotated[Path, typer.Option("--root", help="Surrogate folder (dataset/, models/).")],
    axle: Annotated[str, typer.Option("--axle", help="front | rear | both.")] = "both",
    device: Annotated[
        str | None, typer.Option("--device", help="auto | cpu | cuda (default: study).")
    ] = None,
) -> None:
    """Test-split error per channel against its tolerance (experimental).

    Writes ROOT/validation/<axle>/ (CSV tables and figures). Exit code 1 when a channel is
    over its tolerance, 2 on a study, argument or data error.
    """
    from suspkin.optimize.spec import StudyConfigError
    from suspkin.surrogate.dataset import DatasetError
    from suspkin.surrogate.workflow import validate_surrogate, validation_table

    axles = _parse_axles(axle)
    if device is not None and device not in ("auto", "cpu", "cuda"):
        fail(f"error: --device must be auto, cpu or cuda, got {device!r}", 2)
    try:
        use = _present_axles(root, axles, axle)
        reports = validate_surrogate(study_yaml, root, axles=use, device=device, echo=typer.echo)
    except (StudyConfigError, DatasetError, ValueError, OSError, RuntimeError) as exc:
        fail(f"error: {exc}", 2)
    n_over = 0
    for ax, rep in reports.items():
        tab = validation_table(rep)
        n_over += int((~tab["passed"]).sum())
        typer.echo(
            f"{ax}: {rep.n} test samples, score {rep.score:.3f}, validity accuracy "
            f"{100 * rep.validity_accuracy:.1f} %"
        )
        typer.echo(f"  {'channel':<26}{'rmse':>11}{'tol':>11}  {'unit':<5}{'ratio':>8}  status")
        for row in tab.itertuples(index=False):
            typer.echo(
                f"  {row.channel:<26}{row.rmse:>11.4g}{row.tol:>11.4g}  {row.unit:<5}"
                f"{row.ratio:>8.2f}  {'ok' if row.passed else 'OVER'}"
            )
        typer.echo(str(root / "validation" / ax))
    if n_over:
        fail(f"error: {n_over} channel(s) over tolerance")
