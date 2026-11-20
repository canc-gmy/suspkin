"""Study commands: check, optimize."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from suspkin.cli._common import _cell, _echo_table, fail


def check(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML file.")],
    csv: Annotated[
        Path | None, typer.Option("--csv", help="Write the constraint table (SI) as CSV.")
    ] = None,
    plot: Annotated[
        Path | None, typer.Option("--plot", help="Write the 3D study view as an image.")
    ] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit with code 1 if the baseline is not feasible.")
    ] = False,
) -> None:
    """Evaluate the baseline vehicle against all constraints of a study.

    Prints one row per constraint (display units); status is PASS, FAIL, UNSOLVED,
    INFEASIBLE or OFF (disabled). Exit code 2 on a study file or output error, 1 with
    --strict if the baseline is not feasible or lies outside a variable domain.
    """
    from suspkin.optimize import DesignSpace, evaluate
    from suspkin.optimize.design import InfeasibleDesign
    from suspkin.optimize.spec import StudyConfigError, load_study

    try:
        study, vehicle = load_study(study_yaml)
        space = DesignSpace(study, vehicle)
    except (StudyConfigError, ValueError) as exc:
        fail(str(exc), 2)
    baseline_ok = True
    try:
        space.from_vehicle()
    except InfeasibleDesign as exc:
        baseline_ok = False
        typer.echo(f"warning: baseline is outside a variable domain: {exc}", err=True)
    ev = evaluate(space)
    df = ev.display_frame()
    status = [("OFF" if s == "disabled" else s.upper()) for s in df["status"]]
    records: list[Any] = list(df.itertuples(index=False))
    rows = [
        (
            str(r.name),
            str(r.type),
            _cell(float(r.value)),
            _cell(float(r.min)),
            _cell(float(r.max)),
            str(r.unit) or "-",
            _cell(float(r.g), "{:.3f}"),
            st,
            str(r.detail),
        )
        for r, st in zip(records, status, strict=True)
    ]
    head = ("name", "type", "value", "min", "max", "unit", "g", "status", "detail")
    _echo_table(head, rows, {2, 3, 4, 6})  # right-aligned columns
    enabled = [s for s in status if s != "OFF"]
    n_pass = sum(s == "PASS" for s in enabled)
    typer.echo(
        f"{n_pass}/{len(enabled)} constraints pass, feasible: {'yes' if ev.feasible else 'no'}"
    )
    if ev.objectives:
        odf = ev.objectives_display_frame()
        orecords: list[Any] = list(odf.itertuples(index=False))
        orows = [
            (
                str(r.name),
                str(r.type),
                _cell(float(r.raw)),
                str(r.unit) or "-",
                _cell(float(r.f), "{:.3f}"),
                _cell(float(r.weight), "{:g}"),
                str(r.status).upper(),
                str(r.detail),
            )
            for r in orecords
        ]
        typer.echo("")
        _echo_table(
            ("objective", "type", "raw", "unit", "f", "weight", "status", "detail"),
            orows,
            {2, 4, 5},
        )
        typer.echo(f"f_total (weighted sum): {_cell(ev.f_total, '{:.3f}')}")
        opt = study.optimizer
        if opt is not None:
            typer.echo(f"optimizer: mode {opt.mode}, method {opt.method}")
    try:
        if csv is not None:
            csv.parent.mkdir(parents=True, exist_ok=True)
            ev.frame().to_csv(csv, index=False)
            typer.echo(str(csv))
        if plot is not None:
            from suspkin.plotting import plot_study_3d

            plot.parent.mkdir(parents=True, exist_ok=True)
            plot_study_3d(space).savefig(plot, dpi=200)
            typer.echo(str(plot))
    except Exception as exc:
        fail(f"error: cannot write output: {exc}", 2)
    if strict and not (ev.feasible and baseline_ok):
        raise typer.Exit(code=1)


def _compare_table(
    title: str, head: tuple[str, ...], rows: Sequence[tuple[str, ...]], right: set[int]
) -> None:
    typer.echo("")
    typer.echo(title)
    _echo_table(head, rows, right)


def optimize(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML with objectives and optimizer.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Run folder (default results/opt/<study>_<timestamp>)."),
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue the run in --out from its history.csv.")
    ] = False,
    workers: Annotated[
        str | None,
        typer.Option("--workers", help="Worker processes: an integer or auto (default: study)."),
    ] = None,
    evaluations: Annotated[
        int | None, typer.Option("--evaluations", help="Evaluation budget override.")
    ] = None,
    time: Annotated[
        str | None, typer.Option("--time", help='Wall-time budget override, e.g. "30 min".')
    ] = None,
    method: Annotated[
        str | None,
        typer.Option("--method", help="nsga2 | de | slsqp | trust-constr | de+slsqp."),
    ] = None,
    no_report: Annotated[
        bool, typer.Option("--no-report", help="Skip the before/after report.")
    ] = False,
    surrogate: Annotated[
        Path | None,
        typer.Option(
            "--surrogate",
            help="Surrogate folder (dataset/, models/): active-learning optimizer (experimental).",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="auto | cpu | cuda for the networks (needs --surrogate)."),
    ] = None,
) -> None:
    """Optimise a study's design variables against its objectives and constraints.

    Writes a run folder (history.csv, best.yaml, result.json, figures, report). With
    --surrogate the networks of `surrogate train` search a trust region and the FOM verifies
    the infill points (`--evaluations` is then the FOM budget; al.csv logs the iterations).
    Exit code 2 on a study or argument error, 1 if no feasible design was found, 0 otherwise.
    """
    from suspkin.optimize.drivers import RunError, run_optimization
    from suspkin.optimize.history import HistoryError
    from suspkin.optimize.spec import StudyConfigError
    from suspkin.units import parse_quantity

    if surrogate is None and device is not None:
        fail("error: --device needs --surrogate", 2)
    if device is not None and device not in ("auto", "cpu", "cuda"):
        fail(f"error: --device must be auto, cpu or cuda, got {device!r}", 2)
    try:
        seconds = None if time is None else parse_quantity(time, "time")
        n_workers: int | str | None = None
        if workers is not None:
            if workers.strip().lower() == "auto":
                n_workers = "auto"
            else:
                n_workers = int(workers)
                if n_workers < 1:
                    raise ValueError("--workers must be >= 1 or auto")
        if surrogate is None:
            res = run_optimization(
                study_yaml,
                out,
                resume=resume,
                workers=n_workers,
                evaluations=evaluations,
                time=seconds,
                method=method,
                report=not no_report,
                echo=typer.echo,
            )
        else:
            from suspkin.surrogate.active import run_active_learning

            active = run_active_learning(
                study_yaml,
                surrogate,
                out,
                resume=resume,
                workers=n_workers,
                evaluations=evaluations,
                time=seconds,
                method=method,
                device=device,
                report=not no_report,
                echo=typer.echo,
            )
            res = active.run
    except (StudyConfigError, RunError, HistoryError, ValueError, OSError) as exc:
        fail(f"error: {exc}", 2)
    except (RuntimeError, ImportError) as exc:
        if surrogate is None:
            raise
        fail(f"error: {exc}", 2)

    base, best = res.baseline, res.best
    typer.echo(
        f"done: {res.n_evals} new evaluations ({res.n_total} in history), "
        f"{res.seconds:.1f} s, {res.workers} workers"
    )
    if surrogate is not None:
        typer.echo(
            f"active learning: {active.new_iterations} new iterations ({active.iterations} in "
            f"al.csv), stopped on {active.stop_reason}, device {active.device}"
        )
    if res.mode == "pareto":
        typer.echo(f"Pareto set: {res.n_pareto} designs; table shows the knee point")
    o0: list[Any] = list(base.objectives_display_frame().itertuples(index=False))
    o1: list[Any] = list(best.objectives_display_frame().itertuples(index=False))
    orows = [
        (
            str(a.name),
            _cell(float(a.raw)),
            _cell(float(b.raw)),
            str(a.unit) or "-",
            _cell(float(a.f), "{:.3f}"),
            _cell(float(b.f), "{:.3f}"),
        )
        for a, b in zip(o0, o1, strict=True)
    ]
    _compare_table(
        "objectives",
        ("objective", "baseline", "best", "unit", "f base", "f best"),
        orows,
        {1, 2, 4, 5},
    )
    typer.echo(f"f_total: baseline {base.f_total:.3f} -> best {best.f_total:.3f}")
    c0: list[Any] = list(base.display_frame().itertuples(index=False))
    c1: list[Any] = list(best.display_frame().itertuples(index=False))
    crows = [
        (
            str(a.name),
            _cell(float(a.value)),
            _cell(float(b.value)),
            str(a.unit) or "-",
            _cell(float(a.g), "{:.3f}"),
            _cell(float(b.g), "{:.3f}"),
            "PASS" if b.status == "pass" else str(b.status).upper(),
        )
        for a, b in zip(c0, c1, strict=True)
        if a.status != "disabled"
    ]
    _compare_table(
        "constraints",
        ("constraint", "baseline", "best", "unit", "g base", "g best", "status"),
        crows,
        {1, 2, 4, 5},
    )
    typer.echo(
        f"feasible: baseline {'yes' if base.feasible else 'no'}, "
        f"best {'yes' if best.feasible else 'no'}"
    )
    typer.echo(str(res.out_dir))
    if not res.feasible:
        fail("error: no feasible design found")
