# Study YAML reference

!!! warning "Experimental"
    The optimizer (`suspkin.optimize`, the `optimize` command) is **experimental**: its Python
    API may change in minor releases. The study file format, schema 1, is stable.

A study file describes *what may move* (variables), *what is measured* (sweeps) and *what is
required or wanted* (constraints and objectives). One vehicle can have many studies. Studies
are loaded with `suspkin.optimize.spec.load_study`, which returns `(Study, Vehicle)` or raises
`StudyConfigError`.

Reference files: `configs/studies/fs_reference_constraints.yaml` (constraints only; every
region, sweep and constraint type) and `configs/studies/fs_reference_optimize.yaml` (adds
objectives and an optimizer block).

## Top level

| Key | Meaning |
|---|---|
| `schema` | Must be `1`. |
| `vehicle` | Path to the vehicle YAML, relative to the study file. |
| `variables` | Hardpoints that may move, each with a domain. |
| `sweeps` | Named corner or vehicle sweeps that constraints and objectives refer to. |
| `constraints` | Requirements. |
| `objectives` | Optional goals. |
| `optimizer` | Optional optimizer settings. |
| `surrogate` | Optional, experimental: output grid and dataset of the NN surrogate. |

Values are SI numbers or strings with a unit (`"30 mm"`, `"-1.5 deg"`, `"0.5 deg/deg"`). A
block-level `units: mm` applies to every bare coordinate in that block. Durations take `s`,
`min` or `h`. Coordinates are ISO 8855 vehicle-frame positions on the left side (y > 0).
Point references look like `front.uca_front` or, for a single coordinate,
`front.uca_front.z`.

## `variables`

Each entry: `point: <axle>.<hardpoint>`, `domain` (exactly one of the below), optional
`asymmetric` and `side` (to optimise the right-hand point independently).

| Domain | Keys |
|---|---|
| `hexahedron` | `vertices`: 8 points in FEM order (0-3 bottom face counter-clockwise seen from +z, 4-7 above 0-3); must be convex. |
| `box` | `min`, `max` corner points. |
| `offset` | Per-axis ranges around the baseline point, e.g. `{x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}`. |

`sphere` and `cylinder` regions are constraint-only (`inside` / `outside`); they are not
valid variable domains.

## `sweeps`

A mapping of name to definition.

| `kind` | Keys |
|---|---|
| `corner` | `corner` (FL, FR, RL, RR), `travel` `[min, max]`, `steps`. |
| `vehicle` | `input` (`heave`, `roll`, `pitch`, `rack`), `range` or `full`, `steps`, optional `base`. |

## `constraints`

A list. Common keys: `name`, `type`, optional `min`, `max` (where the type has limits),
`scale` (normalisation), `enabled` (default true), `note`.

| `type` | Checks |
|---|---|
| `band` | A sweep metric stays within `min`/`max` over a `where` range. |
| `slope` | The slope of a sweep metric over a `where` range is within limits. |
| `value` | A sweep metric at a point `at` is within limits. |
| `reach` | A corner can travel the given range with a minimum rocker toggle margin (`min_toggle_margin`) and optional coilover `stroke` check. |
| `summary` | A static vehicle-summary metric is within limits. |
| `inside`, `outside` | A hardpoint (optionally over a sweep, `over`) stays inside or outside a `region` (`box`, `hexahedron`, `sphere`, `cylinder`). |
| `distance` | Distance between points `a` and `b` is within limits. |
| `relation` | Linear relation between coordinates: `lhs op rhs` with `op` in `<=`, `>=` and an `offset`. |
| `joint_angle` | Misalignment angle of a spherical joint over a sweep stays within limits. |

Semantics of each type are given in [Design plan, section 7.4](../DESIGN_PLAN.md).

## `objectives`

| `type` | Goal |
|---|---|
| `curve` | RMS error of a metric to a target curve (`target: {x: [...], y: [...]}`). |
| `target` | `abs(value - target)`. |
| `minimize`, `maximize` | A reduction `of` the metric (`value`, `slope`, `max`, `min`, `max_abs`, `mean`) over `where`/`at`, or a summary metric without `sweep`. |
| `linearity` | RMS residual of a straight-line fit. |

Keys: `name`, `type`, `sweep`, `metric`, `where`, `at`, `of`, `target`, `scale`, `weight`
(default 1, used in weighted mode only), `note`.

## `optimizer`

| Key | Meaning |
|---|---|
| `mode` | `weighted` (single objective, weighted sum) or `pareto` (multi-objective). |
| `method` | `nsga2` (needs the `opt` extra), `de`, `slsqp`, `trust-constr`, `de+slsqp`. |
| `seed` | Random seed. |
| `workers` | Integer or `auto` (CPU count minus one). |
| `budget` | `evaluations` and/or `time` (`"15 min"`). |
| `start` | `baseline` or `best` (warm start from a previous run). |
| `options` | Pass-through method options (`pop_size`, `popsize`, `maxiter`, `ftol`, `xtol`, `gtol`, `fd_step`, ...). |

## `surrogate` (experimental)

Used by `suspkin surrogate sample`, `train` and `validate` (see [stability](stability.md)). The surrogate learns the
left corner of each axle on a fixed grid. Only symmetric variables are supported.

| Key | Meaning |
|---|---|
| `grid.travel` | `min`, `max`, `points`: wheel-travel grid (contains 0, at least 3 points). |
| `grid.rack` | Same for the rack, front axle only (omit: rack fixed at 0). |
| `dataset.n_initial` | Latin-hypercube samples per axle (default 20000). |
| `dataset.seed` | LHS and split seed (default 1). |
| `dataset.shard_size` | Rows per shard file (default 500). |
| `training.device` | `auto` (CUDA if available, else CPU), `cpu` or `cuda` (default `auto`). |
| `training.max_epochs`, `training.patience` | Epoch cap and early-stopping patience (defaults 2000, 100). |
| `training.num_threads` | CPU threads of torch (default: torch default). |
| `training.finetune_lr_factor` | Learning-rate factor when fine-tuning from a trained model (default 0.1). |
| `hpo.n_trials` | Optuna trials per axle (default 100). |
| `hpo.timeout` | Search time budget with a unit, e.g. `4 h` (default `4 h`; `null` for none). |
| `hpo.parallel` | `auto` (axles in parallel on CUDA, sequential on CPU), `true` or `false`. |
| `active_learning.max_iterations` | Iteration cap of `optimize --surrogate` (default 50). |
| `active_learning.fom_evaluations` | FOM evaluation budget of the whole run, baseline included (default 2000). |
| `active_learning.time` | Wall-time budget with a unit, e.g. `2 h` (default none). |
| `active_learning.surrogate_evaluations` | Budget of each inner optimisation on the networks (default 20000). |
| `active_learning.infill` | `top` extra candidates (default 0) and `perturb` random points (default 4) besides the predicted optimum. |
| `active_learning.finetune` | `epochs` (default 200), `weight` (default 5.0) of the new dataset rows, and `replay` (default 4096): each fine-tuning uses all active-learning rows plus a fresh random subset of `replay` earlier train rows (validation split unchanged); early stopping scores the mean of the validation score and the score on all active-learning rows. |
| `active_learning.margin` | Constraint margin on the normalised `g` during the network search (default 0.05). |
| `active_learning.trust_region` | `initial`, `min`, `max` radius in the unit cube (defaults 0.2, 0.01, 0.5). |
| `active_learning.stop` | `eps_f` (relative agreement, default 0.01) and `patience` (iterations, default 5). |

The inner optimiser is the study's `optimizer` block (`method`, `mode`, `seed`).

```yaml
surrogate:
  grid:
    travel: {min: -30 mm, max: 30 mm, points: 41}
    rack: {min: -25 mm, max: 25 mm, points: 21}
  dataset: {n_initial: 20000, seed: 1, shard_size: 500}
  training: {device: auto, max_epochs: 2000, patience: 100}
  hpo: {n_trials: 100, timeout: 4 h}
  active_learning:
    max_iterations: 50
    fom_evaluations: 2000
    surrogate_evaluations: 20000
    infill: {top: 0, perturb: 4}
    finetune: {epochs: 200, weight: 5.0, replay: 4096}
    margin: 0.05
    trust_region: {initial: 0.2, min: 0.01, max: 0.5}
    stop: {eps_f: 0.01, patience: 5}
```

## Abridged example

```yaml
schema: 1
vehicle: ../vehicles/fs_reference.yaml

variables:
  - point: front.tie_inner
    domain: {box: {min: [70, 135, 139], max: [100, 165, 169], units: mm}}
  - point: rear.pushrod_outer
    domain: {offset: {x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}}

sweeps:
  bump_f: {kind: corner, corner: FL, travel: [-30 mm, 30 mm], steps: 13}
  roll: {kind: vehicle, input: roll, range: [-2 deg, 2 deg], steps: 9}

constraints:
  - {name: camber_band_f, type: band, sweep: bump_f, metric: camber,
     where: [-25 mm, 25 mm], min: -4 deg, max: 0 deg, scale: 1 deg}
  - {name: mr_front, type: value, sweep: bump_f, metric: mr_spring, at: 0 mm, min: 0.9, max: 1.1}
  - {name: tie_len, type: distance, a: front.tie_inner, b: front.tie_outer, min: 400 mm}
```

Check a study's baseline with `suspkin check`, optimise with `suspkin optimize`
([Command line](cli.md)). Maths and file roles: [Architecture, section 14](../ARCHITECTURE.md).
