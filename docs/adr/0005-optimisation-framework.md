# ADR 0005: Optimisation framework

Status: accepted (the Python API of `suspkin.optimize` is experimental, the study YAML and the
`check` and `optimize` commands are stable; see ADR 0006). Date: 2026-11-18, extended
2026-11-19.

## Context

Choosing hardpoints by hand is slow. The team wants to say "move these points inside these
volumes, keep these curves inside these bands, minimise this error" and let a search run. The
description must be readable and reviewable by non-programmers, reusable across cars, and the
runs must be reproducible and resumable because they take hours.

## Decision

**Study file.** A separate study YAML (`configs/studies/`) points to a vehicle YAML; one car
can have many studies. Values carry unit strings (ADR 0003).

**Design space.**

* Each region has a per-point role. A `box` or `hexahedron` is the variable domain through a
  trilinear map from $(u,v,w) \in [0,1]^3$; any region can also act as an `inside` or `outside`
  (keep-out) constraint. The design vector is the unit cube $[0,1]^n$.
* Symmetry: a variable moves the left point and the right side mirrors it. `asymmetric: true`
  moves one side only, through `right_overrides`.
* **Producible hardpoints** (2026-11-18): every coordinate an optimiser chooses is a multiple
  of the study `resolution` (default 0.5 mm, absolute vehicle-frame grid; `null` = continuous;
  per-variable override). Snapping happens at decode time (nearest node, or nearest node inside
  the domain for bounds and hexahedra). Optimisers keep a continuous population and evaluate,
  record and export the snapped design. The baseline is evaluated as given.

**Constraints and objectives.**

* Curve constraints: `band`, `slope`, `value`, `reach`. Geometric: `inside`, `outside`,
  `distance`, `relation`, `joint_angle`. Scalar: `summary` (from `vehicle_summary`).
* Objectives: `curve` (RMS error to a target curve), `target` ($|value - target|$),
  `minimize` / `maximize` (of value, slope, max, min, max_abs or mean over a range),
  `linearity` (RMS residual of a straight-line fit).
* Optimiser mode `weighted` ($\sum w_i f_i$, one objective) or `pareto` (separate $f_i$,
  NSGA-II only).
* Methods: `nsga2` (pymoo, `opt` extra), `de` (scipy differential evolution), `slsqp` and
  `trust-constr` (scipy `minimize`), and `de+slsqp` (DE, then an SLSQP polish from the best
  feasible point); `de+slsqp` is the default.

**Runs.**

* A multiprocessing pool (`workers: auto` = CPU count minus 1) evaluates designs in
  parallel. Budget = evaluation count plus wall time.
* The history CSV is **append-only**, so a run resumes: the cache skips evaluated points and
  the population is warm-started from history.
* **Failures:** an unsolved or infeasible design gets $g = G_{FAIL} = 10^3$ on every
  constraint and every objective gets the finite penalty $F_{FAIL} = 10^3$, so drivers never
  see NaN or inf (`suspkin.optimize.evaluate`).
* **Reproducibility:** `seed` in the YAML; the run folder stores `study.yaml`, `vehicle.yaml`
  and `meta.json` (package version, git hash, seed, method, times, evaluation count).
* **Outputs:** the optimised vehicle YAML (best, or every Pareto point), `history.csv`,
  plots (convergence, Pareto front, parallel coordinates, constraint margins) and a
  before/after report through the report builder.

Superseded detail: the original plan split this into a v0.7.0 (spec, constraints, evaluator,
`check`) and a v0.8.0 (objectives and optimisers); both shipped before 1.0.0.

## Consequences

* A study is data, not code; it can be reviewed, versioned and re-run.
* Finite penalties keep gradient-free and gradient-based drivers stable at the cost of
  flat regions in infeasible space; `de+slsqp` mitigates this by polishing only from a
  feasible point.
* The unit cube plus snapping means a surrogate or a driver never sees physical units; the
  grid makes results machine-shop ready but also makes the objective piecewise constant at
  sub-grid scales.
* The same driver interface (`evaluate_batch` on a `Problem`) is reused by the surrogate
  (ADR 0007).
