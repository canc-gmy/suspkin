# suspkin

[![CI](https://github.com/canc-gmy/suspkin/actions/workflows/ci.yml/badge.svg)](https://github.com/canc-gmy/suspkin/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-zensical-blue)](https://canc-gmy.github.io/suspkin/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`suspkin` is a Python toolkit for the kinematics of Formula Student double-wishbone pushrod
suspensions. You describe the car as a YAML file of hardpoints; it solves how the wheels move
under bump, roll, pitch and steering, computes the metrics a designer reads (camber, toe, roll
centre, motion ratio, anti-dive, ...), and turns them into plots and reports. Geometry can then be
optimised against a constraint specification, a neural-network surrogate speeds up large
studies, and a local web app compares designs. The optimiser, surrogate and web app are
experimental.

Documentation: <https://canc-gmy.github.io/suspkin/>

## Features

- Corner kinematics solved in 3D for bump, droop and rack travel, with pushrod actuation on the
  upright, lower or upper wishbone, for front and rear axles.
- Whole-car body motion (heave, roll, pitch, rack) with anti-roll bar coupling.
- More than 40 per-corner metrics with gradients, plus a vehicle summary (rates, roll stiffness,
  load transfer distribution, anti-features, steering).
- Command line, Python API, matplotlib figures, interactive 3D, animations and a self-contained
  HTML report.
- Constrained single- and multi-objective optimisation of hardpoints on a manufacturing grid
  (experimental).
- Neural surrogate with active learning for studies with many variables (experimental).
- Local web app to compare designs, edit hardpoints and export snapshots (experimental).

## Install

`suspkin` is installed from a clone of the repository; it is not on PyPI. Requires Python 3.11+.
The one-command setup (`scripts/setup.ps1` on Windows, `scripts/setup.sh` elsewhere) installs
everything. On some Windows machines `uv` is not on `PATH`; use `python -m uv` instead.

```bash
git clone https://github.com/canc-gmy/suspkin.git && cd suspkin
uv sync --all-extras          # with uv: core, all extras and dev tools
pip install -e ".[viz,opt,app,fast]"   # with pip (in a venv): extras, no dev tools
```

Extras (see [Install](docs/getting-started/install.md) for details):

- `viz`: plotly, for interactive 3D (`suspkin show --html`).
- `opt`: pymoo, for the `nsga2` optimiser.
- `app`: fastapi and uvicorn, for the web app (`suspkin app`).
- `fast`: numba, a compiled corner solver (about 1.8 times faster).
- `surrogate`: torch and optuna, for the neural surrogate (`pip install -e ".[surrogate]"`).

## Quick start

Run these from the repository root on the reference car:

```bash
uv run suspkin summary configs/vehicles/fs_reference.yaml
uv run suspkin sweep configs/vehicles/fs_reference.yaml --corner FL --travel -30 30 --metrics --out fl.csv
uv run suspkin vsweep configs/vehicles/fs_reference.yaml --roll -2 2 --out roll.csv
uv run suspkin report configs/vehicles/fs_reference.yaml --out report/
uv run suspkin check configs/studies/fs_reference_constraints.yaml --plot results/study.png
```

Inputs to the CLI are in mm and degrees; files written (CSV, JSON) are SI. `uv run suspkin --help`
and `uv run suspkin <command> --help` list every option. The
[Tutorial](docs/getting-started/tutorial.md) walks through these outputs.

## Documentation

- [Documentation home](docs/index.md): how the docs are organised and the reading order.
- Getting started: [Install](docs/getting-started/install.md), [Tutorial](docs/getting-started/tutorial.md).
- Guides: [Vehicle YAML](docs/guides/vehicle-yaml.md), [Study YAML](docs/guides/study-yaml.md),
  [Command line](docs/guides/cli.md), [Examples](docs/guides/examples.md),
  [Surrogate](docs/guides/surrogate.md), [Web app](docs/guides/app.md),
  [Stability and versioning](docs/guides/stability.md),
  [Development](docs/guides/development.md).
- Concepts (the math and the code that computes it): [Conventions](docs/concepts/conventions.md),
  [Vehicle model](docs/concepts/vehicle-model.md),
  [Corner kinematics](docs/concepts/corner-kinematics.md),
  [Vehicle kinematics](docs/concepts/vehicle-kinematics.md), [Metrics](docs/concepts/metrics.md),
  [Sweeps and reports](docs/concepts/sweeps-and-reports.md),
  [Optimisation](docs/concepts/optimization.md), [Surrogate](docs/concepts/surrogate.md),
  [Web app](docs/concepts/web-app.md), [Architecture](docs/concepts/architecture.md).
- Reference: [docs/reference/](docs/reference/index.md), per-package API pages.
- Architecture decisions: [docs/adr/](docs/adr/0001-webapp-architecture.md).

The published site is <https://canc-gmy.github.io/suspkin/>.

## Project layout

- `src/suspkin/`: the Python package (core kinematics, metrics, CLI, plotting, optimisation,
  surrogate, web app backend).
- `configs/`: reference vehicle YAML and study YAML files (SI units, ISO 8855 frame).
- `docs/`: the Zensical source for the documentation site.
- `examples/`: runnable Python scripts (load and sweep, compare designs, optimise a study).
- `frontend/`: the web app browser UI (Node project, built into the package).
- `tests/`: unit, golden regression and behaviour-lock tests.
- `scripts/`: setup scripts for Windows and Linux/macOS, plus the benchmark, profiling and surrogate scripts (`bench_kinematics.py`, `profile_fom.py`, `benchmark_surrogate.py`, `surrogate_*.py`).
- `results/`: tracked result folders, including the benchmark surrogate models.

## Development

```bash
uv run pre-commit install
uv run pytest -q -m "not examples"
```

See [Development](docs/guides/development.md) for the tooling (ruff, mypy, pytest) and the
contribution workflow. Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
