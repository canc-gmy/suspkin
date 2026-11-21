# Development guide

How to set up a working copy of `suspkin`, run exactly the checks CI runs, keep the golden
and behaviour-lock snapshots honest, extend the code (metrics, CLI, API, study criteria),
benchmark, release and write docs. Every command below is meant to be copied as is, from the
repository root.

Background reading: [Architecture](../concepts/architecture.md) (package map, layering,
testing strategy), [Conventions](../concepts/conventions.md) (frame, units, signs) and
[Metrics](../concepts/metrics.md) (every metric column).

!!! note "uv on Windows"
    On some Windows machines `uv` is not on `PATH`. Everywhere below, `uv ...` can be written
    `python -m uv ...` (same program, started through the Python that has it installed).
    The examples use plain `uv`.

## 1. Setup

### 1.1 Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | 3.12 is the development default (`.python-version`); the package supports `>=3.11` and CI runs 3.11, 3.12 and 3.13 | runtime |
| [uv](https://docs.astral.sh/uv/) | any recent | environment and dependency manager |
| Node.js + npm | Node `>=22.12` (CI uses 22) | only to build or test the web app frontend in `frontend/` |
| git | | the hooks and `.gitattributes` rules assume it |

uv downloads a matching Python itself if none is installed (`uv python install 3.12`).

### 1.2 One-command setup

```bash
# Linux, macOS, Git Bash
bash scripts/setup.sh              # flags: --minimal  --test  --no-frontend
```

```powershell
# Windows PowerShell 5.1+
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1   # switches: -Minimal -Test -NoFrontend
```

Both scripts do the same seven steps: check for Python 3.11 or newer; find uv (or install it
with `pip install --user uv`); `uv sync --all-extras` (`--minimal` runs plain `uv sync`, core
dependencies only); `pre-commit install` when `.git` exists; build the frontend
(`npm ci`, `npm run build`, `npm run build:single`) when `node` and `npm` are on `PATH`
(`--no-frontend` skips it); a smoke check `suspkin version`; and the test suite with `--test`.

### 1.3 Manual setup

```bash
uv sync --all-extras               # core + dev group + every optional extra
uv sync --all-extras --group docs  # additionally the documentation toolchain (Zensical)
uv run suspkin version
uv run pre-commit install          # git hooks, see section 3
```

The dev dependency group (`pytest`, `hypothesis`, `ruff`, `mypy`, `pre-commit`, `pytest-cov`,
`httpx`, stub packages) is installed by default by `uv sync`. The optional extras declared in
`pyproject.toml` are:

| Extra | Brings | Used by |
|---|---|---|
| `viz` | plotly | interactive HTML figures |
| `opt` | pymoo | NSGA-II driver, `suspkin optimize` |
| `app` | fastapi, uvicorn | web app (`suspkin app`, `app-export`) |
| `fast` | numba | compiled corner solver (`SUSPKIN_NO_NUMBA=1` disables it) |
| `surrogate` | torch, optuna | neural-network surrogate |

Without an extra the matching code must degrade with a clear error, not at import time; see
the lazy-import rules in [Architecture](../concepts/architecture.md).

### 1.4 PyTorch index per platform

`[tool.uv.sources]` in `pyproject.toml` pins where `torch` comes from:

| Platform | Index | Wheel |
|---|---|---|
| Windows (`sys_platform == 'win32'`) | `pytorch-cu124` | CUDA 12.4 build |
| Linux (`sys_platform == 'linux'`) | `pytorch-cpu` | CPU-only build |
| macOS | default PyPI | the standard (Apple Silicon) wheel |

Both indexes are `explicit = true`, so only `torch` is ever taken from them. The `surrogate`
extra requires `torch>=2.5,<2.7`. If you do not need the surrogate, skip it:
`uv sync --extra opt --extra app --extra viz --extra fast` avoids the large download.

### 1.5 Frontend

```bash
cd frontend
npm ci                  # exact install from package-lock.json
npm run dev             # Vite dev server (talks to a running `suspkin app`)
npm run build           # -> src/suspkin/app/static (git-ignored, shipped in the wheel)
npm run build:single    # -> src/suspkin/app/static_single (single-file app for app-export)
```

`npm ci` (not `npm install`) is what CI runs, so `package-lock.json` is the source of truth.

## 2. The quality gate

CI (`.github/workflows/ci.yml`) runs four jobs on every push to `main` and every pull
request against `main`, with `MPLBACKEND=Agg` set. Run the same commands locally before you
open a pull request.

### 2.1 Python (job `test`)

CI runs this on a matrix of Ubuntu, Windows and macOS (Apple Silicon) times Python 3.11,
3.12 and 3.13, each after `uv sync --all-extras`:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src
uv run pytest -q -m "not examples" --cov=suspkin --cov-report=term --cov-report=xml
```

- `ruff` rules: `E F W I N UP B SIM NPY RUF`, line length 100, target `py311`. Locally use
  `uv run ruff check --fix` and `uv run ruff format` to fix instead of just report.
  Markdown files are excluded from ruff (docs are hand-written).
- `mypy` runs in `strict` mode over `src` only, with the pydantic plugin. Tests are not
  type-checked.
- `pytest` runs everything under `tests/` except the slow example-script tests. The coverage
  floor is **94 %** (`[tool.coverage.report] fail_under = 94` in `pyproject.toml`); the run
  fails when the total drops below it. `-ra` is always on (`addopts`), so skipped tests are
  listed with their reason.
- Markers: `examples` (runs `examples/*.py`, slow), `perf` (wall-clock guards with generous
  limits, machine dependent), `regression` (golden snapshots), `lock` (behaviour lock).

Typical fast loops:

```bash
uv run pytest tests/unit/test_alignment.py -q          # one file
uv run pytest -q -k "camber and not surrogate"         # by keyword
uv run pytest tests/regression -q                      # golden + behaviour lock only
```

### 2.2 Examples (job `examples`)

```bash
uv run pytest -q -m examples
```

Runs every script in `examples/` end to end into a temporary folder (`tests/examples`). CI
runs it once, on Ubuntu with Python 3.12.

### 2.3 Docs (job `docs`)

```bash
uv sync --all-extras --group docs
uv run zensical build --strict --clean
```

`--strict` turns every warning (broken link, unresolved `:::` cross-reference, missing nav
target) into an error. Pages that exist but are not yet in `nav` (orphans) build without any
message, so the `nav` in `zensical.toml` is the only check for them.

### 2.4 Frontend and app (job `frontend`)

```bash
cd frontend
npm ci
npm run lint            # eslint
npm run typecheck       # tsc --noEmit
npm test                # vitest run
npm run build
npm run build:single
cd ..
uv sync --all-extras
uv run pytest -q tests/unit/test_app_api.py tests/unit/test_app_snapshot.py
```

The last line is the "app job": the API and snapshot tests, run after the frontend was
built, so the snapshot export finds the compiled single-file app.

### 2.5 One-shot local gate

```bash
uv run ruff check --fix && uv run ruff format && uv run mypy src && uv run pytest -q -m "not examples"
```

Add `--cov=suspkin` to check the coverage floor. The full suite takes minutes; the torch
based tests skip themselves (`pytest.importorskip("torch")`) when the extra is absent.

## 3. Pre-commit, line endings

`.pre-commit-config.yaml` defines two groups of hooks, installed by `uv run pre-commit install`
(or by the setup scripts):

| Hook | Does | Notes |
|---|---|---|
| `ruff-check` | `uv run ruff check --fix` | local hook, Python files |
| `ruff-format` | `uv run ruff format` | local hook |
| `mypy` | `uv run mypy src` | runs on the whole of `src`, not per file |
| `trailing-whitespace`, `end-of-file-fixer` | strip whitespace, final newline | **excluded** for `tests/regression/behaviour/snapshots/` |
| `check-yaml` | YAML syntax | |
| `check-toml` | TOML syntax | |
| `mixed-line-ending` | `--fix=lf` | rewrites CRLF to LF |

The local hooks call `python -m uv run ...`, so they work on machines where `uv` is not on
`PATH`. Run all hooks on the whole tree with `uv run pre-commit run --all-files`.

**Snapshot exclusion.** The behaviour-lock snapshots store the exact text a command printed.
Trailing whitespace and the missing final newline are part of what is locked, so the
whitespace hooks must not touch them. Never reformat or hand-edit files under
`tests/regression/behaviour/snapshots/`; regenerate them (section 4).

**Line endings.** `.gitattributes` sets `* text=auto eol=lf`: text files are LF in the
repository and in your working tree on every platform. The exceptions are `*.ps1`
(`eol=crlf`, because Windows PowerShell 5.1 mis-parses LF scripts), binary types (`*.png`,
`*.gif`, `*.mp4`) and the benchmark surrogate weights (`*.pt`, Git LFS). If your editor writes
CRLF, `mixed-line-ending --fix=lf` fixes it at commit time; configure the editor to LF to
avoid noisy diffs. Golden CSVs are written with `lineterminator="\n"` for the same reason.

## 4. Golden tests and the behaviour lock

Two regression nets protect the numbers and the user-visible behaviour. They are different
tools with different update rules.

| | Golden snapshots | Behaviour lock |
|---|---|---|
| Location | `tests/regression/golden/` | `tests/regression/behaviour/snapshots/` |
| Tests | `tests/regression/test_golden.py` | `test_cli_lock.py`, `test_app_lock.py`, `test_surrogate_lock.py` under `tests/regression/behaviour/` |
| Protects | numeric results of the reference car and the two reference studies | CLI stdout, exit codes, every file a command writes, the app API responses, the OpenAPI schema, the surrogate dataset and oracle evaluation |
| Compared by | values, `rtol=1e-6`, `atol=1e-9` (`1e-8` for finite-difference gradient columns) | CSV/JSON/YAML by value at `1e-12`; `.md`/`.html`/`.txt` by exact text skeleton after scrubbing volatile text, with float literals at `rtol = atol = 1e-9` (`TEXT_RTOL` in `tests/regression/behaviour/_lock.py`); binary files by existence and size only |
| Update flag | `--update-golden` | `--update-lock` |

### 4.1 When a test fails

Never start by regenerating. First find out what changed.

```bash
uv run pytest tests/regression/test_golden.py -x -q
```

The assertion message names the snapshot and column (`corner_FL.camber`), or reports added
and removed columns or a different row count. To see the size and shape of the deviation,
write the new output to a scratch copy and diff:

```bash
uv run pytest tests/regression --update-golden     # rewrites tests/regression/golden/*
git diff --stat tests/regression/golden            # which files moved
git diff tests/regression/golden/corner_FL.csv     # how much
git checkout -- tests/regression/golden            # undo if the change is not wanted
```

Because the regeneration is an ordinary working-tree change, `git diff` is the inspection
tool, and `git checkout -- <dir>` the undo. For the lock, the same workflow applies to
`tests/regression/behaviour/snapshots`; a failing lock message reads
`<case>/<file>: differs` or lists added or removed files.

Typical causes, by increasing seriousness: an unrelated formatting change in a CLI message
(lock: `stdout.txt` differs), a metric column added or renamed (golden: columns added or
removed), a solver tolerance change (small value drift), a real regression (large drift or
`nan`s). A drift far above the tolerance without a deliberate model change is a bug.

### 4.2 When updating is legitimate

Update only for a **deliberate, reviewed** change of behaviour:

- a metric was added, removed, renamed or its definition corrected (golden);
- a CLI message, option or help text changed on purpose (lock);
- an app endpoint or model changed (lock, including `frontend/src/api/openapi.json`);
- the solver changed in a way that moves results beyond the tolerances, and the change is
  explained in `CHANGELOG.md`.

Not legitimate: making a red test green. Refactors must leave both nets untouched; that is
exactly their purpose.

```bash
uv run pytest tests/regression --update-golden   # golden: tests/regression/golden/
uv run pytest tests/regression --update-lock     # lock:   tests/regression/behaviour/snapshots/
```

Each flag turns the matching tests into "write and skip": they rewrite the files and report
`skipped` with the list of written files. Run the same suite again without the flag
afterwards; it must pass.

Both flags **refuse to run in CI**. If the environment variable `CI` or `GITHUB_ACTIONS` is
set, the test fails with "regenerate snapshots locally and commit them", so a pipeline can
never silently rewrite its own reference. Regenerate on your machine, commit the files.

Notes:

- The lock's `optimize` case compares only layout, CSV column names and exit codes when `CI`
  is set or the platform is not Windows (the DE+SLSQP run can differ slightly between
  platforms). Locally on Windows it is strict. The full snapshot set was recorded on
  Windows, so regenerate there when you can.
- `test_surrogate_lock.py` needs torch, and some lock cases need plotly, pymoo or fastapi;
  with a missing extra the case is skipped and its snapshot is **not** regenerated. Use a
  full `uv sync --all-extras` before `--update-lock`.
- `tests/regression/behaviour/BENCH_BASELINE.txt` is a machine-dependent timing note from
  before the refactor, not a test oracle.

### 4.3 Review expectations

In the pull request:

1. The snapshot diff is in the same commit as the code change that caused it.
2. The diff is small and explainable: say which columns or lines moved and why. A golden
   rewrite that touches every file for no stated reason is rejected.
3. `CHANGELOG.md` mentions the change under `Changed` or `Added`.
4. The golden README (`tests/regression/golden/README.md`) still describes the files.

## 5. Recipes

### 5.1 Add a metric end to end

Example: a new per-state corner metric `my_metric` (SI value, here a length).

A corner metric exists in **two** vectorised implementations that must agree, plus a scalar
oracle in tests. Cross-check, do not copy: write the formula once, then mirror it.

1. **Compute it in `suspkin.metrics.arrays.corner_arrays`** (`src/suspkin/metrics/arrays.py`).
   It fills a dict of columns over $N$ states with no Python loop per state; add
   `out["my_metric"] = ...` beside related columns (alignment, wheel position, centres, ...).
   Follow the sign conventions in [Conventions](../concepts/conventions.md); degenerate
   cases give `nan` or `inf`, not exceptions. The column then appears in
   `suspkin.metrics.table.metrics_frame` automatically, because `corner_metric_arrays`
   feeds it.
2. **Mirror it in `suspkin.kinematics.batch.corner_columns`** (`src/suspkin/kinematics/batch.py`),
   the design-and-state batched path the optimiser and surrogate use. The two modules share
   geometry helpers, but the column list is written out in each; a column present in one and
   missing in the other fails `tests/regression/test_vectorised_equivalence.py`.
3. **Scalar oracle in tests.** Add the per-state reference to `tests/unit/_scalar_oracles.py`
   (the simple, readable version of the formula) and compare it to the vectorised column in
   `tests/regression/test_vectorised_equivalence.py` (tolerance `1e-12`). Prefer an
   analytical or invariant check in a unit test too (a mirrored corner gives the same value,
   a rigid translation leaves it unchanged, a known simple geometry gives a closed form);
   see `tests/unit/test_alignment.py` for the pattern.
4. **Gradient (optional).** If the metric has a useful derivative, add it to
   `suspkin.kinematics.batch.GRADIENTS`: `{"travel": {"my_metric": "my_metric_rate"}}`. Both
   `metrics.table.add_gradients` (per-sweep) and the batched path read that single table.
   Remember the gradient columns use the looser golden tolerance; extend `_TABLE_GRADIENTS`
   in `tests/regression/_golden.py` only if the new name is not derived from `GRADIENTS`.
5. **Metric registry and units.** Register the dimension in `src/suspkin/optimize/registry.py`
   (`_CORNER` maps column name to a `Dim` such as `"length"`, `"angle"`, `"ratio"`), so
   constraints and objectives in study YAML accept the name and parse units
   (`"3 mm"`). For a vehicle-summary quantity the entries are in `_SUMMARY_OWN` and
   `_VEHICLE_OWN`; for per-corner columns of vehicle sweeps (`FL_cp_x` style) `_CORNER_SUFFIX_OWN`. A wrong dimension is
   caught by `suspkin check` as a unit error.
6. **Display unit.** Figures take the unit from the series *kind* in
   `suspkin.plotting.figures` (`_KINDS`: `"mm"`, `"deg"`, ...). Add the metric as a series
   (`_s("my_metric", "my metric", "mm")`) in the relevant `_PANELS` entry; the web figure
   spec in `suspkin.plotting.spec` mirrors these panels. For a vehicle-summary value add a
   row to `SUMMARY_ROWS` in `src/suspkin/plotting/units.py` (label, key path, factor from
   SI, unit, format), and for its `suspkin vehicle` line an entry in `VEHICLE_CLI_UNITS`.
7. **Surrogate channel (only if the surrogate must predict it).** Channels are listed in
   `suspkin.surrogate.channels` with a tolerance; adding one changes the dataset layout and
   invalidates trained models, so it is a separate, deliberate step.
8. **Update the golden snapshots**, because `corner_*.csv` and the vehicle CSVs gain a
   column: `uv run pytest tests/regression --update-golden`, then read the diff: exactly the
   new column should appear, all other columns byte-identical. Also run the lock
   (`uv run pytest tests/regression/behaviour -q`); outputs that print every column change
   there and need `--update-lock`.
9. **Docs.** Add the metric to [Metrics](../concepts/metrics.md): definition, physical
   meaning, sign, formula, code location, unit and display unit. Add a line to
   `CHANGELOG.md` under `Added`.
10. Gate: section 2.1 and `uv run zensical build --strict --clean`.

### 5.2 Add a CLI command

The CLI is the package `suspkin.cli`, one module per command group
(`vehicle`, `study`, `webapp`, `surrogate`); shared helpers (`fail`, `load_vehicle_or_exit`,
`_echo_table`) are in `suspkin.cli._common`.

1. Write the command as a plain function in the matching group module with typed parameters
   (`typer.Argument`, `typer.Option`) and a docstring: its first line is the `--help` summary.
   Errors go through `suspkin.cli._common.fail` (stderr, exit code 1) so every command fails
   the same way. Import heavy or optional dependencies **inside** the function, so
   `suspkin --help` stays fast and works without extras.
2. Register it in `src/suspkin/cli/__init__.py`: `app.command()(vehicle.my_command)`, or
   `app.command("name")(module.func)` for a different command name. **Registration order is
   the order of `suspkin --help`**, so place the line deliberately (the current order is
   vehicle commands, `check`, `optimize`, `app`, `app-export`, then the `surrogate`
   sub-group). Sub-group commands go on `surrogate_app`.
3. Tests: a `CliRunner` unit test in `tests/unit/test_cli*.py` (see `test_cli.py`), checking
   exit code and key output lines.
4. Behaviour lock: `tests/regression/behaviour/test_cli_lock.py` holds `COMMANDS`,
   `SURROGATE` and `CASE_ARGV`. Add the command name to `COMMANDS` (this creates a
   `cli_help_<name>` snapshot) and a case to `CASE_ARGV` for a run. The root `--help`
   snapshot (`cli_help_root`) changes too. Then `uv run pytest tests/regression/behaviour
   --update-lock` and review the new and changed snapshot folders.
5. Docs: add the command and all its flags to the CLI guide (`docs/guides/cli.md`);
   changelog line under `Added`.

### 5.3 Add an API endpoint

The web app (`suspkin.app`, optional extra `app`) builds its FastAPI app in
`suspkin.app.server.create_app`; all routes are in `suspkin.app.routes.make_router`, under
the prefix `/api/v1`. The frontend types are generated from the OpenAPI schema.

1. **Models.** Add request and response pydantic models to `src/suspkin/app/models.py`.
   Every route declares a `response_model`, which is what makes the generated TypeScript
   precise. Use `Literal` types for closed sets.
2. **Route.** Add the handler to `make_router` in `src/suspkin/app/routes.py` (`@r.get(...)`
   or `@r.post(...)`). Keep handlers thin: dataset logic belongs in `suspkin.app.datasets`,
   centres in `suspkin.app.centres`, edits in `suspkin.app.edits`. Raise `HTTPException` with
   a clear message for client errors.
3. **Tests.** `tests/unit/test_app_api.py` (uses FastAPI's test client).
4. **Regenerate the schema and the types**:

    ```bash
    cd frontend
    npm run gen:api
    ```

    `gen:api` runs `python -m suspkin.app.openapi src/api/openapi.json` (through `uv run`, or
    `python -m uv run` as fallback) and then `openapi-typescript` to write
    `frontend/src/api/schema.d.ts`. Commit both generated files.
5. **Frontend.** Use the new type in `frontend/src/api` (`httpSource.ts`, `hooks.ts`,
   `types.ts`; the snapshot source `snapshotSource.ts` must serve the same data when the
   endpoint is needed in the exported single-file app), then `npm run lint`,
   `npm run typecheck`, `npm test`.
6. **Lock.** `test_openapi_unchanged` pins the live schema to the committed
   `frontend/src/api/openapi.json`; it fails until step 4 is done. The API lock snapshots
   change if the endpoint appears in `test_app_lock.py`: `--update-lock`, then review.
7. Docs: the endpoint table in the web-app concept page (`docs/concepts/web-app.md`);
   changelog line.

### 5.4 Add a study constraint or objective type

Study YAML is parsed by pydantic models in `src/suspkin/optimize/spec/criteria.py`, and
evaluated in `src/suspkin/optimize/evaluate.py`.

1. **Model.** Add a frozen model with a `type: Literal["my_type"]` discriminator field. For a
   constraint, subclass `_Constraint` (or `_Limits` for min/max limits); for an objective,
   `_Objective`. Add it to the discriminated union `Constraint` or `Objective`.
2. **Unit parsing.** Values with units (`"3 mm"`) are converted to SI in `prep_constraint` /
   `prep_objective` before validation; add a branch for the new type, using the metric
   registry (`corner_metric_dim`, `vehicle_metric_dim`) to get the expected dimension.
   Errors must name the constraint and the field.
3. **Evaluation.** Constraints: add an `isinstance` branch in `_eval_one` in
   `src/suspkin/optimize/evaluate.py`, returning a normalised `g` value (`g <= 0` is
   feasible; dividing by a scale makes different constraints comparable). Objectives: add the
   branch in the objective evaluation of the same module, returning one or more normalised
   `f` values (lower is better). `suspkin.optimize.history.constraint_names` builds the
   `g_<name>` history columns from the study, so check that a new type is reflected there.
4. **Surrogate path.** `suspkin.surrogate.evaluate` evaluates the same criteria on predicted
   curves. If the new type needs data the surrogate does not predict, reject it clearly
   there (see how `ReachConstraint` and `SummaryConstraint` are special-cased).
5. **Tests.** Spec parsing in `tests/unit/test_study_spec.py`, evaluation in
   `tests/unit/test_evaluate.py` / `test_objectives.py` with a case whose answer is known in
   closed form. Add an example entry to `configs/studies/fs_reference_constraints.yaml` if
   the type should be demonstrated (that file lists every constraint type); that changes the
   golden `study_constraints.csv` and some lock snapshots, so review those diffs.
6. **Docs.** Add the type to the study guide (`docs/guides/study-yaml.md`) with every field,
   and to the optimisation concept page.

## 6. Benchmarking

Timing scripts live in `scripts/`; they are measurements, not tests. Numbers are machine
dependent, so compare before and after on the same machine and the same Python.

```bash
uv run python scripts/bench_kinematics.py [--repeats N] [--profile] [--no-numba]
```

Times, on `configs/vehicles/fs_reference.yaml`: one cold corner solve, corner and vehicle
sweeps at 31, 61 and 121 points (solver only), and `run_standard_studies` end to end.
`--no-numba` forces the pure-Python path; `--profile` prints a cProfile top list of
`run_standard_studies(steps=61)`.

```bash
uv run python scripts/profile_fom.py [--study PATH] [--repeats N]
```

Profiles one full-order design evaluation (the optimiser entry point) on
`configs/studies/fs_reference_optimize.yaml`: total time, a per-stage breakdown, the number
of corner states solved, and the cost of one surrogate dataset sample.

```bash
uv run python scripts/benchmark_surrogate.py [--dry-run]
```

Resumable benchmark of the surrogate workflow (sample, train, validate, active learning,
direct differential evolution at equal evaluation cost, report). It runs for a long time
and writes under `results/benchmark_surrogate/`; use `--dry-run` to see the stages first.
`scripts/surrogate_hpo_report.py` and `scripts/surrogate_retrain_best.py` are follow-ups
of the same study. The numba fast path costs 0.91 s once per process (kernel load plus first sweep, cached
on disk), so use enough repeats or discard the first run when you compare.

## 7. Releases

- **Changelog.** `CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/):
  an `## [Unreleased]` section on top with `Added`, `Changed`, `Deprecated`, `Removed`,
  `Fixed` subsections, one section `## [X.Y.Z] - YYYY-MM-DD` per release. Every user-visible
  change gets a line in the same pull request. Only the maintainer edits it at release time.
- **Versioning.** [SemVer](https://semver.org/): new features are MINOR, bug fixes PATCH,
  incompatible changes to the Python API, CLI, YAML formats or app API are MAJOR. The version
  is `version` in `pyproject.toml`; `suspkin.__version__` reports the installed one.
- **Cut a release.** On `main`, green: rename `[Unreleased]` to the new version and date
  (and open a new empty `[Unreleased]`), bump `version` in `pyproject.toml`, commit, then
  tag and push:

    ```bash
    git tag vX.Y.Z
    git push origin main vX.Y.Z
    ```

- **What the release workflow does** (`.github/workflows/release.yml`, triggered by a `v*`
  tag): checks that the tag equals `v` + the `pyproject.toml` version; checks that the tagged
  commit is contained in `origin/main`; builds the web app frontend with Node 22
  (`npm ci`, `npm run build`, `npm run build:single`); runs `uv sync --all-extras` and
  `pytest -q -m "not examples"`; builds the wheel and sdist with `uv build` and checks with
  `scripts/check_dist_frontend.py` that both contain the frontend and no `node_modules`
  (the CI `frontend` job runs the same check on every push); extracts the
  matching `## [X.Y.Z]` section of `CHANGELOG.md` as the release notes (it fails if there
  is none); and publishes a GitHub release with `dist/*.whl` and `dist/*.tar.gz` attached.
  A tag containing a hyphen (for example `v1.2.0-rc1`) is marked as a pre-release.
- Releases do not publish to PyPI. The documentation site is deployed separately
  (section 8).

## 8. Documentation workflow

```bash
uv sync --all-extras --group docs
uv run zensical serve                  # live preview at http://127.0.0.1:8000, reloads on save
uv run zensical build --strict --clean # what CI runs; output in site/
```

The site (`zensical.toml`, Zensical with the classic Material look) renders math with MathJax (`$...$` inline, `$$...$$`
display) and diagrams from ` ```mermaid ` fences. API reference pages use mkdocstrings
`:::` entries, so a good docstring is the reference documentation; keep docstrings precise
(units and shapes of arguments, what is returned, what is raised). The `Docs` workflow
(`.github/workflows/docs.yml`) rebuilds and deploys to GitHub Pages on pushes to `main` that
touch `docs/`, `zensical.toml`, `src/`, `examples/`, `CHANGELOG.md` or `README.md`.

Page conventions:

- English only; SI units with symbols (mm and degrees only when quoting display output).
- Name code by fully qualified name in backticks (`suspkin.metrics.arrays.corner_arrays`),
  never by line number. Link other pages with relative links
  (`../concepts/metrics.md#camber`).
- Concept pages follow one structure: purpose and place in the pipeline; theory with
  equations and signs; how the code does it (steps, equations, constants and tolerances taken
  from the code, inputs and outputs with units and shapes); edge cases and failure modes; a
  worked example with real numbers and the command that produced them; where it is tested;
  see also.
- Worked numbers come from running the code on `configs/vehicles/fs_reference.yaml` (or a
  study config), and the command is shown next to them.
- Every statement about the code is checked against the current code, not copied from an
  older page. New pages must be added to `nav` in `zensical.toml`.
