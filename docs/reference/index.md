# API reference

## What is stable

The stable public API is exactly the set of names in `suspkin.__all__` (see [suspkin](suspkin.md)).
Everything else is importable but may change in minor releases; `suspkin.optimize` and
`suspkin.surrogate` are explicitly experimental. See [Stability and versioning](../guides/stability.md).

## Organisation

One page per package, each with every public module (no leading underscore):

[suspkin](suspkin.md), [units](units.md), [geometry](geometry.md), [kinematics](kinematics.md),
[metrics](metrics.md), [sweeps](sweeps.md), [report](report.md), [plotting](plotting.md),
[optimize](optimize.md), [surrogate](surrogate.md), [app](app.md), [cli](cli.md).

Pages are generated from the source docstrings (Google style). Frame, unit and sign rules are in
[Conventions](../concepts/conventions.md); explanations and derivations are in the concepts pages:
[Architecture](../concepts/architecture.md), [Vehicle model](../concepts/vehicle-model.md),
[Corner kinematics](../concepts/corner-kinematics.md), [Vehicle kinematics](../concepts/vehicle-kinematics.md),
[Metrics](../concepts/metrics.md), [Sweeps and reports](../concepts/sweeps-and-reports.md),
[Optimization](../concepts/optimization.md), [Surrogate](../concepts/surrogate.md),
[Web app](../concepts/web-app.md).
