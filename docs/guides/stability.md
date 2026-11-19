# Stability and versioning

suspkin follows [Semantic Versioning](https://semver.org/) from 1.0.0.

## Stable surface

These are covered by SemVer: a breaking change needs a new major version.

| Surface | What is stable |
|---|---|
| Vehicle YAML | `schema: 1` ([Vehicle YAML](vehicle-yaml.md)): field names, units, meaning |
| Study YAML | `schema: 1` ([Study YAML](study-yaml.md)): blocks, entries, units |
| CLI | commands, options and exit codes ([Command line](cli.md)); CSV/JSON column names |
| Python API | the names in `suspkin.__all__` ([Python API](../reference/api.md)) |

New optional fields, new commands, new metrics and new columns may appear in minor releases.
A change that an old reader cannot handle bumps `schema`; the loader rejects unknown schema
numbers with a clear message.

## Experimental

`suspkin.optimize` (design space, evaluator, drivers, history, run folders) is
**experimental**: its Python API may change in minor releases. The study YAML schema 1
itself and the CLI commands `check` and `optimize` follow the stable rules above.

## Deprecation policy

A stable name is not removed without warning. It first emits a `DeprecationWarning`
stating the version of deprecation and of removal, and stays available for at least one minor release; removal happens in a later release.
Deprecations are listed in the [Changelog](../changelog.md).

## Golden regression snapshots

`tests/regression/golden/` holds CSV/JSON snapshots of the reference car (corner sweeps,
vehicle sweeps, vehicle summary, study results). Tests compare freshly computed tables with
the snapshots (rtol 1e-6; a looser absolute tolerance on finite-difference gradient columns).

A snapshot may change only when the numerics change on purpose (a bug fix or a formula
change). Then:

```bash
uv run pytest tests/regression --update-golden
git diff tests/regression/golden      # review every changed number
```

The commit message and the changelog must state why the numbers moved.
