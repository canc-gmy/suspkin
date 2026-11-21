# Stability and versioning

This page says what you can rely on when you update `suspkin`, how versions are numbered, what
is experimental, how deprecations work and what CI checks on every change. The reasoning behind
these rules is in [ADR 0006](../adr/0006-stability-testing-and-release.md).

## Versioning

`suspkin` follows [Semantic Versioning](https://semver.org/) from 1.0.0: `MAJOR.MINOR.PATCH`.
The current release is **1.1.0** (`suspkin.__version__`, `suspkin version`).

| Change | Version bump |
|---|---|
| A breaking change to anything in "Stable surface" below | major |
| New optional YAML fields, commands, options, metrics, columns, API names; changes to experimental parts; numerics that move because of a bug fix | minor |
| Bug fixes that do not move published numbers beyond the golden tolerance | patch |

Releases are tagged `vX.Y.Z` on `main`; the tag must equal the `pyproject.toml` version. The
release notes are the matching section of `CHANGELOG.md` ([Changelog](../changelog.md)), in the
[Keep a Changelog](https://keepachangelog.com/) format.

!!! note "The 2026 restructure"
    The unreleased restructure after 1.1.0 ([ADR 0008](../adr/0008-2026-restructure.md))
    moved Python modules: for example `suspkin.geom_math` is now
    `suspkin.geometry.vectors`, and `suspkin.cli` is a package. The CLI, the YAML formats, the
    web API and `suspkin.__all__` did not change, and numerical results are locked to
    `rtol = 1e-12` by the behaviour lock. Code that imported modules by their old path must be
    updated; the version bump that goes with it (a minor, 1.2.0) is decided at release time.

## Stable surface

These are covered by SemVer: a breaking change needs a new major version.

| Surface | What is stable |
|---|---|
| Vehicle YAML | `schema: 1` ([Vehicle YAML](vehicle-yaml.md)): field names, units, meaning |
| Study YAML | `schema: 1` ([Study YAML](study-yaml.md)): blocks, entries, units |
| CLI | commands, options, defaults and exit codes ([Command line](cli.md)); CSV and JSON column names; printed table layout is for humans and may be refined in a minor release |
| Python API | the names in `suspkin.__all__` |

The Python API is exactly these 19 names plus `__version__`:

| Group | Names |
|---|---|
| Vehicle model and files | `Vehicle`, `VehicleConfigError`, `load_vehicle`, `save_vehicle` |
| Corners and solvers | `build_corners`, `solve_corner`, `sweep_corner`, `sweep_rack`, `solve_vehicle`, `KinematicsError` |
| Sweeps and tables | `sweep_vehicle`, `vehicle_frame`, `states_to_frame`, `metrics_frame`, `add_gradients` |
| Summary, report, plot | `vehicle_summary`, `build_report`, `run_standard_studies`, `plot_vehicle_3d` |

The list is snapshotted by `tests/unit/test_public_api.py` (`EXPECTED_ALL`) which also checks
that each name imports and has a docstring, so adding or removing a public name is a deliberate,
reviewed change. Names are imported lazily (`import suspkin` stays cheap, see
[Architecture](../concepts/architecture.md)). The signatures are in the
[Python API reference](../reference/index.md).

**Not stable:** everything not in `suspkin.__all__`, including the module layout under
`suspkin.geometry`, `suspkin.kinematics`, `suspkin.metrics`, `suspkin.plotting`,
`suspkin.report` and all private (`_`-prefixed) names. Use the top-level names. If you need a
function from a submodule, it may move in a minor release.

What may appear in a minor release: new optional fields in the YAML files, new CLI commands and
options, new metrics and new table columns. Code that reads tables should select columns by
name, not by position.

## Schema versions

Each file format carries an integer `schema`:

| File | Key | Current | Rejection message |
|---|---|---|---|
| Vehicle YAML | `schema` (default 1 when omitted; `save_vehicle` always writes it) | 1 | `unsupported vehicle schema N; this suspkin supports 1` |
| Study YAML | `schema` (required) | 1 | `unsupported study schema N; this suspkin supports 1` |
| Result bundle `manifest.json` (report bundles, opened by the web app) | `schema` | 1 | `unsupported bundle schema N in <folder> (this suspkin reads 1)` |

A change that an old reader cannot handle (a renamed or removed field, a changed unit or
meaning) bumps `schema`; readers reject unknown numbers with the message above instead of
guessing. Additions that old readers can ignore do not bump it: both YAML models forbid unknown
keys (`extra="forbid"`), so an old suspkin refuses a newer file that uses a new key, with a
message that names it. The `surrogate:` block, added in 1.1.0, is an example of such an
addition to study schema 1.

## Experimental

These parts are usable but their Python API may change in a minor release:

| Part | Stable pieces inside it |
|---|---|
| `suspkin.optimize` (design space, evaluator, drivers, history, run folders) | study YAML schema 1 and the commands `check` and `optimize` follow the stable rules |
| `suspkin.surrogate` and the `suspkin surrogate` commands | the `surrogate:` block is an addition to study schema 1; the Python API is not in `suspkin.__all__` |
| `suspkin.app` and the `app` / `app-export` commands | none promised; the 16 endpoints under `/api/v1` and the UI may change |

The behaviour of the experimental CLI commands is protected by the behaviour lock like
everything else; "experimental" means the maintainers may still change it deliberately.

## Deprecation policy

A stable name is not removed without warning. Before removal it raises a `DeprecationWarning`
that states the version of the deprecation and of the planned removal, and it stays available for
at least one minor release. Removal then happens in a later release. Deprecations are listed in
the [Changelog](../changelog.md).

Today there is **no deprecation helper in the code**. The `_deprecation.deprecated` decorator was
deleted in the 2026 restructure because nothing used it
([ADR 0008](../adr/0008-2026-restructure.md), [ADR 0006](../adr/0006-stability-testing-and-release.md));
the policy itself stands, and a deprecation is implemented by calling `warnings.warn(...,
DeprecationWarning, stacklevel=2)` directly where needed. The only deprecated item at present is
the hidden CLI option `suspkin surrogate sample --out`, an alias of `--root` (it still works; use
`--root`).

## What CI checks

The workflow `.github/workflows/ci.yml` runs on every push to `main` and every pull request
to `main`.

| Job | Runs on | What |
|---|---|---|
| `test` | matrix: ubuntu, windows, macOS (Apple silicon) x Python 3.11, 3.12, 3.13 (9 jobs, `fail-fast: false`) | `uv sync --all-extras`, `ruff check`, `ruff format --check`, `mypy src` (strict), then `pytest -q -m "not examples"` with coverage (floor 94 %) |
| `examples` | ubuntu, Python 3.12 | `pytest -m examples`: runs the five scripts in `examples/` as subprocesses |
| `docs` | ubuntu, Python 3.12 | `uv sync --all-extras --group docs`, `zensical build --strict --clean` |
| `frontend` | ubuntu, Node 22 | in `frontend/`: `npm ci`, lint, typecheck, `npm test`, `npm run build`, `npm run build:single`; then the Python app tests `tests/unit/test_app_api.py` and `tests/unit/test_app_snapshot.py` against the built UI |

`MPLBACKEND=Agg` is set for all jobs. Two more workflows exist:

- `docs.yml` builds the site on every push to `main` that touches `docs/`, `zensical.toml`, `pyproject.toml`,
  `uv.lock`, `src/`, `examples/`, `CHANGELOG.md` or `README.md`, and deploys it to GitHub Pages.
- `release.yml` runs on a tag `v*`: it checks that the tag equals the `pyproject.toml` version
  and that the commit is on `main`, runs the tests, builds the sdist and wheel
  (`uv build`), extracts the notes from `CHANGELOG.md` and attaches the files to a GitHub
  Release (a tag with `-` in it is marked as a prerelease). Publishing to PyPI is not part of
  it.

## Regression nets

Two kinds of stored snapshots guard the numbers.

**Golden snapshots** (`tests/regression/golden/`): CSV and JSON of the reference car (corner
sweeps of all metrics, vehicle sweeps, the vehicle summary, the reference study's evaluation
frames). Tests compare freshly computed tables with the snapshots at `rtol = 1e-6`, with a looser
absolute tolerance on finite-difference gradient columns. A snapshot may change only when the
numerics change on purpose (a bug fix or a formula change); `--update-golden` is refused when `CI` or `GITHUB_ACTIONS` is set:

```bash
uv run pytest tests/regression --update-golden
git diff tests/regression/golden      # review every changed number
```

The commit message and the changelog must state why the numbers moved.

**Behaviour lock** (`tests/regression/behaviour/`): recorded stdout, exit codes and written
files of every CLI command, one JSON snapshot per web endpoint (plus the generated OpenAPI
schema) and the surrogate file layouts, compared by value at `rtol = 1e-12`; human-readable text by exact skeleton with float literals at `rtol = atol = 1e-9` (tolerances in [Development](development.md#4-golden-tests-and-the-behaviour-lock)).
It is regenerated only with `--update-lock`, which is refused in CI, and only for a deliberate
output change with a stated reason. How to run and update both nets is in the
[Development guide](development.md).

## What this means for you

- Scripts that use only `suspkin.__all__`, the CLI and the YAML files keep working across minor
  and patch releases. Suspkin is not on PyPI: pin a git tag (for example `v1.1.0`) in a project that must not change
  numbers.
- Compare results between versions by diffing the CSV tables. A change to a published number
  must show up as a golden-snapshot diff with a stated reason in the commit message and the
  changelog, so the changelog tells you which numbers moved; differences below the golden
  tolerance (`rtol = 1e-6`) are platform noise (SVD and BLAS last digits), not changes.
- Pickle nothing from suspkin objects across versions; save vehicles with `save_vehicle` (YAML,
  schema-checked) and tables as CSV.
