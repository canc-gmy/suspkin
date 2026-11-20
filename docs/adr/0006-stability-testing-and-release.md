# ADR 0006: Stable surface, regression testing, CI and release

Status: accepted. Date: 2026-11-18 (CI matrix extended later).

## Context

From 1.0.0 other people (and the team's own studies) depend on the YAML formats, the command
line and the numbers. Numerical code is easy to break silently: a refactor can change a sign or
a tolerance and every test of "does it run" still passes. The project also needs a
repeatable way to release, and examples that cannot rot.

## Decision

* **Stable surface** (SemVer from 1.0.0): the vehicle YAML (`schema: 1`), the study YAML
  (`schema: 1`), the CLI (commands, options, exit codes, CSV/JSON column names) and the names
  in `suspkin.__all__`. `suspkin.optimize` as a Python API is experimental. Unknown schema
  numbers are rejected with a clear message. See the stability guide.
* **Deprecation policy:** a stable name is not removed without a `DeprecationWarning` for at
  least one minor release. (The helper that wrapped this was deleted in the 2026 restructure
  because nothing used it; the policy itself stands and a warning can be raised directly.)
* **Golden snapshots:** CSV and JSON of the reference car (corner bump and rack sweeps of all
  corner metrics, vehicle heave/roll/pitch/rack sweeps, the vehicle summary and the reference
  study's evaluation frame) live in `tests/regression/golden/`. They are compared with tight
  tolerances (rtol $10^{-6}$, looser absolute tolerance on finite-difference gradient columns)
  and regenerated only with `pytest --update-golden`, with the reason in the commit message.
  Hand-picked analytic tests stay as well.
* **No external validation data** for now: validation rests on analytic tests and the golden
  snapshots. A real car can be added as another regression case.
* **Documentation site:** MkDocs Material with mkdocstrings, built with `--strict` in CI and
  deployed to GitHub Pages.
* **CI:** GitHub Actions on Ubuntu, Windows and macOS with Python 3.11, 3.12 and 3.13; a
  coverage floor (94 %); a docs build; and the examples run as a separate job.
* **Release:** on a `v*` tag a workflow checks that the tag equals the `pyproject.toml`
  version and that the commit is on `main`, runs the tests, builds the sdist and wheel, takes
  the notes from `CHANGELOG.md` and attaches the files to a GitHub Release. PyPI publication
  may come later. MIT licence.
* **Examples:** plain `.py` scripts under `examples/`, executed in CI (marker `examples`) and
  shown in the docs.
* **Package name:** `suspkin`.

## Consequences

* Any change that moves a published number is visible as a golden diff and has to be
  justified. The 2026 restructure used a stricter, separate lock (ADR 0008).
* Platforms differ in the last digits of SVD and BLAS results; tolerances are therefore not
  zero, and the behaviour lock widens only where CI showed differences.
* The experimental/stable split lets the optimiser internals change without a major release.
