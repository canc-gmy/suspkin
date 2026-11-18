# Suspension Kinematics Toolkit — Design Plan

Status: **DRAFT v0.2** — open questions resolved (see §12)
Language policy: English only (code, comments, docs, commit messages).

---

## 1. Goal

Build a Python package that simulates the rigid-body kinematics of a
four-wheeled Formula Student car with double-wishbone, pushrod-actuated
suspension on both axles. It produces every kinematic curve and scalar of
interest for vehicle dynamics. Later it supports geometry optimization
against objectives under geometric constraints.

### In scope (this plan)

- Rigid-body kinematics of all four corners (ideal joints, no compliance).
- Front: steering rack + tie rods. Rear: fixed toe link.
- Pushrod → rocker (bellcrank) → coilover on every corner.
- Anti-roll bar per axle, selectable type: **torsional** (U-bar),
  **blade** (torsion tube + blade arms) or **beam** (bending bar).
- Corner-level sweeps and vehicle-level motions (heave, roll, pitch,
  steer, combined).
- Full metric set + plots + 3D geometry view.
- Geometry optimization driven directly by the kinematic solver.

### Out of scope (future phases, see §11)

- Quasi-static force solver (link loads).
- FEM structural analysis of members.
- Neural-network surrogate.
- Elasto-kinematics / compliance (K&C).

---

## 2. Conventions

| Item | Choice |
|---|---|
| Coordinate system | ISO 8855: x forward, y left, z up |
| Origin | Ground plane, mid-point between front contact patches at static ride height |
| Units | SI internally and in input files (m, rad, N, N/m, N·m/rad) |
| Display units | Configurable per plot; default mm and deg for readability (display only) |
| Corner naming | `FL`, `FR`, `RL`, `RR` |
| Symmetry | Hardpoints given for the left side; right side mirrored (y → −y) unless an explicit asymmetric override is given |
| Sign conventions | Camber negative = top of wheel leaning toward car centre. Toe-in positive. Wheel travel positive = bump (wheel moves up relative to body). Roll positive = right-hand rule about +x (ISO). Steer positive = left turn. |

---

## 3. Geometry model

### 3.1 Hardpoints per corner

| Group | Points |
|---|---|
| Upper wishbone (UCA) | `uca_front`, `uca_rear` (chassis), `uca_outer` (upright) |
| Lower wishbone (LCA) | `lca_front`, `lca_rear` (chassis), `lca_outer` (upright) |
| Toe control | `tie_inner` (rack or chassis), `tie_outer` (upright) |
| Wheel | `wheel_center`, spindle axis (from static camber/toe), tyre unloaded radius, rim width |
| Pushrod | `pushrod_outer` (on upright, both axles), `pushrod_inner` (rocker) |
| Rocker | `rocker_pivot`, `rocker_axis` (direction), `rocker_spring` (spring pickup), `rocker_arb` (drop-link pickup, optional) |
| Coilover | `spring_chassis` (chassis end); spring rate, free length, preload; damper stroke limits |
| ARB | Per axle: type (`torsional` / `blade` / `beam`), bar pivot or beam centre points, arm length, drop-link points, material + section → stiffness |
| Steering (front) | Rack axis, rack ratio (c-factor, m per pinion rev), rack travel limits |

### 3.2 Vehicle-level data

Wheelbase and track come from hardpoints. Also: CG height, mass, front
mass distribution, unsprung mass, tyre vertical stiffness, brake
location (inboard/outboard per axle), and drive layout. These are needed
for anti-features and ride/roll rates.

### 3.3 Input file

One YAML file per vehicle, validated with **pydantic**.

```yaml
vehicle:
  name: fs_reference
  cg_height: 0.28
  mass: 280.0
  front_mass_ratio: 0.47
  drive: rwd            # rwd | awd | fwd
front:
  brakes: outboard      # inboard | outboard
  hardpoints:           # left side, metres, ISO 8855
    uca_front: [ 0.130, 0.250, 0.290]
    ...
  pushrod_mount: upright  # upright (default, both axles) | lca (kept as option)
  coilover: {k: 35000.0, free_length: 0.200, preload: 0.0}
  arb: {type: torsional, ...}
  steering: {c_factor: 0.090, rack_travel: [-0.030, 0.030]}
rear:
  ...
```

---

## 4. Kinematic solver

### 4.1 Corner DOF reduction

Each corner is a closed-loop mechanism. It has 1 DOF (wheel travel) at
the rear and 2 DOF (wheel travel + rack displacement) at the front.

Implemented (v0.2) as a **reduced 3-unknown formulation**:

| Symbol | Meaning |
|---|---|
| θ_U | UCA rotation about its chassis axis → UBJ position |
| θ_L | LCA rotation about its chassis axis → LBJ position |
| ψ | Upright spin about the LBJ→UBJ (kingpin) axis |

The upright pose = minimal rotation aligning the static LBJ→UBJ vector to
the current one, then rotation ψ about that axis, anchored at the LBJ.

Constraint equations (3 in 3 unknowns):

1. |UBJ − LBJ| = static ball-joint distance.
2. |`tie_outer` − `tie_inner`(rack)| = tie rod length.
3. Driver: wheel-centre z − static z = travel.

`pushrod_outer` moves with the upright, the LCA or the UCA depending on
`pushrod_mount` (reference car: UCA). The rocker is then a decoupled
1-unknown problem: |`pushrod_outer` − `pushrod_inner`(φ_R)| = pushrod
length. Spring length and ARB drop-link motion follow explicitly.

Notes from the v0.1.0 review:

- YAML hardpoints are the **static ride-height pose**. The wheel spin axis
  in the upright frame = `wheel_spin_axis(side, static_camber, static_toe)`
  at that pose, carried rigidly with the upright.
- `pushrod_mount: lca` → `pushrod_outer` moves with the LCA rotation instead
  of the upright pose.
- Right corners are reflections: the right rocker angle φ has the opposite
  sign under the right-hand rule about the mirrored axis. Solver and metrics
  use a per-side sign so left/right outputs match.
- The rack is one rigid part: both `tie_inner` points translate along the
  global `Steering.rack_axis` (not mirrored).

### 4.2 Numerical method

- Own damped Newton on the 3x3 closing system (unknowns `theta_uca`, `theta_lca`,
  `psi`) with an **analytic Jacobian** and a backtracking line search on `||F||`
  (stop at `max|F| < 1e-12`, at most 30 iterations). `scipy.optimize.root`
  (`hybr`, same analytic Jacobian) is only the fallback before `KinematicsError`.
  The inner loop uses scalar float arithmetic (no numpy calls on 3-vectors).
  A guess that is far from the target is reached by staged continuation from the guess.
- Sweeps use **continuation**: each step starts from the previous
  solution (linear extrapolation of the three angles once two states exist).
  Step refinement kicks in on non-convergence, and also when a solve moves any
  angle by more than 0.05 rad from its guess (branch-jump guard; cold solves exempt).
  In rack sweeps near lock (psi changes about 0.13 rad per 10 mm) the guard triggers
  extra halvings; this costs work but the result is correct.
- Convergence tolerance: 1e-9 m on constraint residuals.
- Health checks after each step:
  - link-length invariance
  - rocker past toggle (pushrod/rocker angle near 0° or 180°)
  - joint angle limits (optional)
  - coilover stroke limits
- **Pure function core**: `solve_corner(geometry, travel, rack) -> CornerState`.
  No global state. Vectorizable over sweeps, so the optimizer can call it
  thousands of times.

### 4.3 Tyre / contact patch

The tyre is modelled as a rigid disc of radius R. The contact patch is
the lowest point of the wheel disc relative to the ground plane, so it
accounts for camber. A loaded-radius option (R − F/k_t) comes later.

### 4.4 Vehicle-level solver

Prescribe body motion (heave z, roll φ, pitch θ). For each corner, solve
the wheel travel that puts the contact patch on the ground plane. The
ground plane is fixed and the body moves. Combined cases (roll + steer,
heave + pitch + steer) are just tuples of inputs.

The ARB couples left and right corners only in forces, not in
kinematics. Its kinematic output is the bar twist β_L − β_R (torsional,
blade) or the end deflection δ (beam) from the left/right drop-link motion;
it is zero in pure heave.

---

## 5. Metrics

All metrics are reported as **curves** vs the sweep variable and as
**scalars** at static ride height (value + gradient).

### 5.1 Wheel alignment

- Camber (to body and to ground), toe, caster, kingpin inclination (KPI / SAI)
- Scrub radius, mechanical trail (caster trail), kingpin offset at wheel centre, spindle length
- Bump steer (toe vs travel), camber gain (camber vs travel), caster change vs travel

### 5.2 Wheel position

- Half-track change, contact-patch lateral and longitudinal shift, wheelbase change
- Wheel-centre recession

### 5.3 Geometry centres

- Front-view instant centre (FVIC) and front-view swing-arm length (FVSA)
- Side-view instant centre (SVIC) and side-view swing-arm length (SVSA)
- Kinematic roll centre: height and lateral migration in heave and roll
- Roll axis
- Pitch centre

### 5.4 Anti-features

- Anti-dive (front), anti-lift (rear), anti-squat (driven axle)
- Each computed for inboard/outboard brakes and drive layout, as % vs travel

### 5.5 Actuation and rates

- Pushrod angle, rocker angle, spring length, damper length
- Motion ratios:
  - spring/wheel
  - damper/wheel
  - pushrod/wheel
  - rocker ratio
- Installation ratio progression
- Wheel rate (k·MR² from virtual work), ride rate (in series with tyre stiffness)
- Coilover stroke usage

### 5.6 Anti-roll bar

- Elastic coordinate vs roll: bar twist (torsional, blade) or end deflection (beam)
- ARB motion ratio (roll mode and single wheel)
- ARB wheel rate (roll mode ½·k·MR², single wheel k·MR²) and ARB roll stiffness (v0.5)
- Bar stiffness from material + section
  - Torsional: 1/k = L/(GJ) + 2r/(3EI) (bar twist in series with round-arm bending; rigid arms optional)
  - Blade: same, with blade arm I_eff = 1/(cos²α/I_flat + sin²α/I_edge)
  - Beam: two cantilevers of half-length a, k = 6EI/a³ on the end deflection

### 5.7 Roll

- Roll stiffness per axle (springs + ARB)
- Total roll stiffness and roll gradient (deg/g, using CG and RC heights)
- Front roll stiffness distribution (TLLTD, geometric + elastic)
- Roll camber coefficient, roll steer coefficient

### 5.8 Steering

- Steer angle per wheel vs rack travel
- Ackermann % vs steer angle
- Steering ratio (handwheel → road-wheel), turning radius
- Toe change vs travel at steer angles (combined steer + bump)

---

## 6. Plots and outputs

- matplotlib for report-quality figures (PNG/PDF/SVG). Optional plotly for interactive 3D.
- Standard plot set per study: one figure per metric group, left/right or front/rear overlaid.
- Design comparison: overlay of N geometries on the same axes.
- 3D view: hardpoints, links, wheels, rockers, ARB at any solved state. Optional animation over a sweep.
- Data export: every sweep to CSV / Parquet (tidy format: one row per step, one column per metric). Scalars to JSON.
- HTML or Markdown report that bundles the figures and the scalar table.

Plot code consumes the tidy result tables only. It never calls the solver.

---

## 7. Optimization

v0.7.0 delivers the study specification, the constraint evaluator and `suspkin check`.
Objectives and optimizers were delivered in v0.8.0 (sections 7.10–7.14, milestone in section 11). The design variables and
constraints are decided in O9-O14 (section 12). Everything runs on the pure kinematic core,
so the optimizer will call the same design-space and `evaluate` API.

Code: [optimize/](../src/suspkin/optimize/), [metrics/joints.py](../src/suspkin/metrics/joints.py),
[plotting/regions.py](../src/suspkin/plotting/regions.py). Module reference and math:
[ARCHITECTURE.md section 14](ARCHITECTURE.md#14-optimization-v07v08).

### 7.1 Study file

- A separate YAML in `configs/studies/` with `schema: 1` and `vehicle:` pointing to one vehicle
  YAML (path relative to the study file). One vehicle can have many studies (O9).
- Blocks: `variables`, `sweeps`, `constraints`. Unknown keys are rejected (`extra="forbid"`).
- `load_study(path) -> (Study, Vehicle)`; errors raise `StudyConfigError`.
- Checked at load: point references `<axle>.<hardpoint>` exist; sweeps and metrics exist
  (metric dimensions come from a registry); `where` and `at` lie inside the sweep range;
  `min <= max`; `scale` is positive; unit dimensions match the metric.

### 7.2 Design variables and the domain map

Each variable moves one hardpoint inside a domain:

- `hexahedron`: 8 vertices. The trilinear map from (u, v, w) in [0, 1]^3 onto the hexahedron is
  the variable domain (O10). Vertex order (FEM convention): bottom face 0-1-2-3
  counter-clockwise seen from +z, top face 4-5-6-7 above 0-1-2-3. The Jacobian determinant must
  be positive at the 8 corners and the centre (non-degenerate, not twisted).
- `box`: `min` and `max` corners.
- `offset`: a range per axis relative to the baseline point; axes not given stay fixed and are
  not design variables.

The design vector is `x` in [0, 1]^n (bounds are all [0, 1]). `DesignSpace.to_vehicle(x)` builds
a `Vehicle` through the existing validators, and `from_vehicle` returns the baseline `x`. Invalid
geometry raises `InfeasibleDesign`.

Symmetry (O12): a symmetric variable moves the left hardpoint and the right side mirrors it.
`asymmetric: true` with `side: left | right` moves one side only; right-side variables are
written through `right_overrides`, and their regions are in right-side coordinates.

Regions are in the vehicle frame (ISO 8855), on the left side (y > 0) unless stated otherwise.

### 7.3 Units (O11)

A value is a plain number (SI) or a string with a unit (`"30 mm"`, `"-1.5 deg"`,
`"0.5 deg/deg"`). Strings are converted to SI on load by `units.parse_quantity`, and the unit's
dimension must match the metric. Accepted symbols: `m`, `cm`, `mm`, `rad`, `deg`, `N`, `kN`,
`m/s2`, `g`, `kg`, `%`, and quotients such as `deg/mm`, `N/mm`, `deg/g`, `deg/deg`. No external
units library.

### 7.4 Constraints (O13)

Common keys: `name`, `enabled`, `scale`, `note`. Limits are `min` and/or `max`.

| Type | Acts on | Semantics |
|---|---|---|
| `band` | corner or vehicle sweep metric | worst point over `where` (default: the whole sweep) inside [min, max]; ends interpolated |
| `slope` | sweep metric | least-squares slope of the metric against the sweep input over `where` |
| `value` | sweep metric | metric linearly interpolated at `at` inside [min, max] |
| `reach` | one corner | the corner solves over its own `travel`; minimum toggle margin and, with `stroke: true`, remaining bump and droop stroke |
| `summary` | scalar of `vehicle_summary` (dotted name, e.g. `front.rc_height`) | scalar inside [min, max] |
| `inside` / `outside` | one hardpoint and a region | signed distance of the point to the region; `outside` is a keep-out zone; `over` takes the worst state of a corner sweep |
| `distance` | two hardpoints | `\|a - b\|` inside [min, max]; `over` takes the worst state of a sweep |
| `relation` | two hardpoint coordinates (`axle.point.axis`) or a constant | `lhs >= rhs + offset` or `lhs <= rhs + offset` on static coordinates |
| `joint_angle` | one joint of one corner over a sweep | joint misalignment angle (section 7.7) inside [min, max]; `axis` override allowed |

Regions: `box` (min, max), `hexahedron` (must be convex for `inside` and `outside`, checked at
load), `sphere` (center, radius), `cylinder` (p0, p1, radius).

### 7.5 Normalised values

For limits `[min, max]` (either may be missing):

    g = max((v - max) / s, (min - v) / s)

`g <= 0` is feasible. The scale `s` is, in order: the constraint's `scale`; `max - min` when both
limits are given and differ; a per-dimension default (length 1 mm, angle 0.1 deg, ratio 0.01,
percent 1, force 10 N, stiffness 1 N/mm); otherwise `|limit|` (a zero limit needs an explicit
`scale`, checked at load). Region constraints use `g = sd / s` (`inside`) or `-sd / s`
(`outside`), where `sd` is the signed distance (negative inside). `reach` takes the largest of its
normalised terms.

### 7.6 Evaluator

`evaluate(space, x) -> Evaluation` runs each sweep used by an enabled constraint at most once per
call: corner sweeps through `sweep_corner`, `states_to_frame(metrics=True)` and `add_gradients`;
vehicle sweeps through `sweep_vehicle` and `vehicle_frame`. `vehicle_summary` runs only if a
`summary` constraint exists.

Each constraint result carries `value`, its limits, `g` and a status: `pass`, `fail`, `unsolved`,
`infeasible` or `disabled`. A solver failure (`KinematicsError`, a non-finite value or a missing
point) gives `unsolved` with `g = G_FAIL = 1e3`. An invalid design gives `infeasible` for every
enabled constraint. `Evaluation` provides `feasible`, `max_g`, `g_vector`, `frame()` (SI) and
`display_frame()` (display units).

### 7.7 Joint angles

Each body's rotation relative to the static pose is a Kabsch fit of its hardpoints; the chassis
is the identity. Two-force links (`tie`, `pushrod`, `coilover`) compare the current link direction
with the static direction carried by the housing. Ball joints (`uca_outer`, `lca_outer`) compare
the kingpin axis carried by the upright with the one carried by the wishbone. The neutral pose
gives 0; `joint_angle` is the angle `atan2(|a x b|, a . b)`.

### 7.8 Check command and region plot

`suspkin check STUDY.yaml [--csv FILE] [--plot FILE] [--strict]` evaluates the baseline vehicle
against every constraint and prints one row per constraint in display units. `--csv` writes the SI
table, `--plot` writes the 3D view. Exit code 2 on a study file error; exit code 1 with `--strict`
when the baseline is not feasible.

The region plot draws the vehicle, the variable domains (green) and the keep-outs (red). It reads
only the study and the design space, and it runs no solve.

### 7.9 Deferred to v0.8.0

Implemented in v0.8.0 (see 7.10–7.14), except the surrogate, which stays deferred to v3.x:

- Objectives: target curves or scalars (bump steer, camber gain, roll centre height and migration,
  motion ratio, anti-dive), combined by weighted sum or as a Pareto front.
- Optimizers: `pymoo` NSGA-II (multi-objective); `scipy.optimize` SLSQP and differential evolution
  (single objective).
- Outputs: optimized YAML, Pareto front plot, before/after comparison report.
- The neural-network surrogate (v3.x) plugs in behind the same evaluator interface.

### 7.10 Objectives (v0.8.0, implemented)

Objectives are evaluated on the same sweep cache as the constraints ([evaluate.py](../src/suspkin/optimize/evaluate.py)).

| Type | Raw value (SI) |
|---|---|
| `curve` | RMS of `metric − target(x)` over the window (default: overlap of the target x range with the sweep); linear interpolation of the target |
| `target` | `\|value − target\|` at `at` of a sweep, or of a summary metric when no sweep is given |
| `minimize` / `maximize` | `of`: `value` (at `at`), `slope`, `max`, `min`, `max_abs` or `mean` over `where`; or a summary metric |
| `linearity` | RMS residual of a least-squares line over `where` |

- Normalised value: `f = raw / scale`. The scale is the user `scale`, else `default_scale(dim)`; a `target` objective without either uses `|target|`. A missing scale on a dimension without a default is rejected at load.
- `maximize` is stored as `−raw`.
- `weight` (default 1) is used only in weighted mode: `f_total = Σ wᵢ fᵢ`.
- A failed objective gets `f = F_FAIL = 1e3` (section 12, O19).

### 7.11 Optimizer block (v0.8.0, implemented)

`optimizer:` keys: `mode` (`weighted` | `pareto`), `method` (`nsga2` | `de` | `slsqp` | `trust-constr` | `de+slsqp`), `seed`, `workers` (int or `auto` = CPU count − 1), `budget` (`evaluations`, `time`), `start` (`baseline` | `best`, read by the local methods), `options` (passthrough: `pop_size`, `popsize`, `maxiter`, `ftol`, `xtol`, `gtol`, `fd_step`, DE strategy keys). `pareto` with any method other than `nsga2` is rejected when the run starts.

### 7.12 Drivers (v0.8.0, implemented)

All methods work in the unit cube `[0, 1]^n` through the `Problem` (cache, history, budget) ([drivers.py](../src/suspkin/optimize/drivers.py)).

- **nsga2** (pymoo, `opt` extra): constraints as `n_ieq_constr` with `G ≤ 0` feasible; weighted mode is one objective (`f_total`), pareto mode has one per objective. Population `max(20, 4n)` unless `pop_size` is given; generations `max_evals // pop_size` (or 10 000 without an evaluation budget). The budget stops the run through `BudgetExhausted`.
- **de** (scipy `differential_evolution`): vectorised, deferred updating, no polish. Constraints are one `NonlinearConstraint(g, −∞, 0)`. The population is the Latin hypercube (or the warm start on resume).
- **slsqp** and **trust-constr** (scipy `minimize`): start from `start` (baseline or best history point). SLSQP gets `ineq: −g` with its Jacobian. trust-constr gets `NonlinearConstraint(g, −∞, 0)` with its Jacobian and a BFGS Hessian. Both use the forward-difference Jacobian of section 7.13.
- **de+slsqp**: DE gets 80 % of the remaining evaluations and wall time, then SLSQP polishes from the best history point (feasible with minimal `f_total`, else least `max_g`). This method reads the best history point, not the `start` key.

### 7.13 Parallel runs, cache, budget and resume (v0.8.0, implemented)

- **Pool**: a `ProcessPoolExecutor` with the `spawn` context; each worker loads the study once in its initializer. With one worker, evaluation runs in-process.
- **Cache**: keyed on `x` rounded to 12 decimals. Cached points are free, and duplicates inside one batch are evaluated once.
- **Budget**: `evaluations` counts new evaluations and `time` is wall seconds. A batch that starts within budget completes even if it crosses the limit; `BudgetExhausted` is raised for a batch that starts after the limit.
- **Jacobian**: forward differences with step `1e-3` (backward where `x + h > 1`); the base point and the `n` perturbed points form one parallel batch.
- **History**: `history.csv` is appended row by row and flushed (section 12, O18).
- **Resume**: `--resume` with `--out` on a folder that contains `history.csv`. The cache is rebuilt from the history, `eval_id` continues, the budget counts only new evaluations, and the population is warm-started from the best history rows (feasible first, by `f_total`) and filled with Latin hypercube points. A history whose columns differ from the study is rejected.

### 7.14 Run outputs and report (v0.8.0, implemented)

Run folder (default `results/opt/<study>_<YYYYmmdd-HHMMSS>`, or `--out`):

- `study.yaml`, `vehicle.yaml` (copies; the study then points at `vehicle.yaml`), `meta.json` (package version, git hash, seed, method, mode, start and end times, evaluation counts, workers, resumed) and `history.csv`.
- `best.yaml`: the best feasible point, or the least infeasible one if none is feasible (weighted mode).
- `pareto/pareto_<k>.yaml` and `pareto.csv`: the feasible non-dominated designs (pareto mode).
- `result.json`: best `x`, `f`, `g` and a summary per objective and constraint, baseline vs best.
- `figures/`: convergence, Pareto front, parallel coordinates and constraint margins (pure readers of `history.csv`).
- `report/`: `build_report` of the baseline and the optimized vehicle. In pareto mode the table shows the knee point (minimum normalised distance to the ideal point).

The CLI is `suspkin optimize STUDY.yaml [--out DIR] [--resume] [--workers N] [--evaluations N] [--time "30 min"] [--method M] [--no-report]` (exit 0 ok, 1 no feasible design, 2 study or argument error).

---

## 8. Package architecture

```
veichle-dynamics/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── .gitignore
├── .pre-commit-config.yaml
├── docs/
│   └── DESIGN_PLAN.md
├── configs/
│   ├── vehicles/fs_reference.yaml
│   └── studies/{heave,roll,steer,optimize}.yaml
├── src/suspkin/
│   ├── __init__.py
│   ├── units.py           # display-unit conversion only
│   ├── geometry/          # pydantic models, YAML I/O, mirroring
│   ├── math/              # rotations, frames, line/plane intersections
│   ├── kinematics/        # corner solver, rocker, ARB, vehicle solver
│   ├── metrics/           # alignment, centres, anti, rates, roll, steering
│   ├── sweeps/            # sweep definitions → tidy result tables
│   ├── plotting/          # 2D figure sets, 3D viewer, report
│   ├── optimize/          # variables, constraints, objectives, drivers
│   └── cli.py             # `suspkin run|plot|report|optimize`
└── tests/
    ├── unit/
    ├── regression/        # golden results for reference geometry
    └── property/          # hypothesis-based invariants
```

The package name `suspkin` is a placeholder.

**Tooling**:

- Python ≥ 3.11
- Managed with `uv`, built with `hatchling`
- Libraries: numpy, scipy, pandas, pydantic, pyyaml, matplotlib, plotly (optional extra), typer, pymoo
- Dev tools: pytest, hypothesis, ruff, mypy, pre-commit

---

## 9. Verification strategy

- **Invariants** (every solved step): link lengths constant within 1e-9 m; upright rigid; rocker radius constant.
- **Symmetry**: mirrored geometry gives mirrored results; roll centre stays on the centreline in pure heave.
- **Analytical cases**:
  - parallel equal-length wishbones → zero camber gain, RC on the ground
  - known swing-arm geometry → known FVSA and RC
  - single-lever rocker → analytic motion ratio
- **Numerical cross-check**: compare motion ratio and wheel rate from finite differences against the virtual-work result.
- **Regression**: golden CSVs for the reference FS geometry, compared within tolerance.
- **External reference**: if real hardpoints or OptimumK / Lotus results become available, add them as regression cases (decision O1).

---

## 10. Git and versioning workflow

| Item | Rule |
|---|---|
| Branches | `main` always green. Feature branches `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, merged with a merge request or fast-forward |
| Commit messages | [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`) |
| Versioning | [SemVer](https://semver.org/). Annotated tag `vX.Y.Z` at the end of each milestone |
| Changelog | `CHANGELOG.md` in Keep-a-Changelog format, updated per release |
| Quality gate | pre-commit runs ruff + mypy + fast tests before each commit |
| Remote | Optional GitHub remote + CI workflow (lint + test matrix) |

---

## 11. Milestones

| Ver. | Milestone | Done when |
|---|---|---|
| v0.0.1 | Scaffold | Package installs, CLI stub, CI/pre-commit green |
| v0.1.0 | Geometry model | YAML schema + validation + mirroring + static 3D plot of reference car |
| v0.2.0 | Corner solver | Wheel-travel and rack sweeps solve; invariants pass |
| v0.3.0 | Alignment and centres | §5.1–5.3 metrics + analytical tests |
| v0.4.0 | Actuation and ARB | Rocker, coilover, motion ratios, both ARB types (§5.5–5.6) |
| v0.5.0 | Vehicle level | Heave / roll / pitch / combined; anti-features, roll, steering (§5.4, 5.7, 5.8) |
| v0.6.0 | Plots and report | Full figure set, 3D viewer, report, data export |
| v0.7.0 | Optimization study spec: variables, constraints, evaluator, `suspkin check` | Reference study loads and passes; `check` table and region plot work |
| v0.8.0 | Objectives + optimizers (NSGA-II, SLSQP/DE), comparison report | Optimizer runs on the v0.7 study spec; Pareto front and before/after report |
| **v1.0.0** (current, done) | Stable kinematics | Docs complete, regression suite frozen |

**Future phases (not planned in detail yet)**:

- **v1.x Forces**: quasi-static link forces from tyre load cases.
- **v2.x FEM**: 3D beam/truss FEM of members (stress, pushrod buckling); load cases from the force solver.
- **v3.x Surrogate**: dataset generation (Sobol/LHS over design variables), MLP surrogate of metrics + stresses, plugged into the optimizer behind the same objective interface.

---

## 12. Decisions log

| # | Topic | Decision |
|---|---|---|
| D1 | Vehicle class | Formula Student |
| D2 | Architecture | Double wishbone + pushrod + rocker + coilover on both axles; front rack, rear toe link |
| D3 | Physics level | Rigid-body kinematics only (no forces, no compliance) for v0.x |
| D4 | Solver | Own constraint solver on scipy, with continuation |
| D5 | Frame and units | ISO 8855, SI |
| D6 | Motions | All: corner bump/rebound, heave, roll, pitch, steer, combined |
| D7 | Metrics | Every metric in §5 |
| D8 | FEM / NN | Deferred to future phases |
| O1 | Validation data | User **may** have real hardpoints. v0.x uses a reference FS geometry + analytical tests. The YAML schema must accept user hardpoints directly so their data drops in as a regression case later. |
| O2 | Pushrod mount | Solver supports `upright`, `lca` and `uca`. **Reference car (from v0.2): pushrod on the UCA, both axles**, rod in compression going up to a rocker mounted above the UCA (changed 2026-11-17, was upright). |
| O3 | ARB | **Both axles**. Type chosen per axle in YAML (decided 2026-11-17 for v0.4): `torsional` (round arms, bar twist GJ/L **in series with arm bending** 3EI/L³), `blade` (torsion tube + flat blade arms with an orientation angle: flat = soft, edge-on = stiff), `beam` (transverse bending bar: heave-free central slider, clamped in rotation; roll bends the two halves as cantilevers). Replaces the earlier single `flexural` type. |
| O4 | Brakes / drive | **Outboard** brakes on both axles, **RWD**. Configurable. |
| O5 | Tyre | **Rigid disc** for now. Loaded radius later. |
| O6 | Optimization | Decided in O9-O14 (2026-11-18). v0.7.0: study spec, constraints and evaluator. v0.8.0: objectives and optimizers. |
| O9 | Study file | Separate study YAML in `configs/studies/`. It points to a vehicle YAML; one car can have many studies. |
| O10 | Regions and domains | Each region has a per-point role. A hexahedron or box is the variable domain via a trilinear map from (u, v, w) in [0, 1]^3. Any region can also be an `inside` or `outside` (keep-out) constraint. |
| O11 | Units | Values carry explicit unit strings (`"30 mm"`, `"-1.5 deg"`, `"0.5 deg/deg"`). A plain number means SI. Everything is converted to SI on load and checked against the metric dimension. |
| O12 | Symmetry | A variable moves the left point and the right side mirrors it. `asymmetric: true` moves one side only, through `right_overrides`. |
| O13 | Constraint types | Curve: `band`, `slope`, `value`, `reach`. Geometric: `inside`, `outside`, `distance`, `relation`, `joint_angle`. Scalar: `summary` (from `vehicle_summary`). |
| O14 | Scope | v0.7.0: spec, evaluator, `suspkin check` and a region 3D plot; no optimizer. v0.8.0: objectives and optimizers (pymoo NSGA-II; scipy SLSQP/DE). |
| O15 | Objective types | `curve` (RMS error to a target curve), `target` (\|value − target\|), `minimize` / `maximize` (of value, slope, max, min, max_abs or mean over a range), `linearity` (RMS residual of a straight-line fit over a range). |
| O16 | Optimizer mode | `weighted` (Σ wᵢ·fᵢ, one objective) or `pareto` (separate fᵢ; NSGA-II only). |
| O17 | Methods | `nsga2` (pymoo, `opt` extra), `de` (scipy differential evolution), `slsqp`, `trust-constr` (scipy minimize), `de+slsqp` (DE, then SLSQP polish from the best feasible point). |
| O18 | Parallel, budget, resume | Multiprocessing pool (`workers: auto` = CPU − 1). Budget = evaluation count plus wall time. The history CSV is append-only, so a run can resume: the cache skips evaluated points and the population is warm-started from history. |
| O19 | Failures | An unsolved or infeasible design gets g = G_FAIL on every constraint, and every objective gets the finite penalty F_FAIL = 1e3. |
| O20 | Reproducibility | `seed` in the YAML. The run folder stores study.yaml, vehicle.yaml and meta.json (package version, git hash, seed, method, times, evaluation count). |
| O21 | Outputs | Optimized vehicle YAML (best, or every Pareto point), history.csv, plots (convergence, Pareto front, parallel coordinates, constraint margins) and a before/after report through `build_report`. |
| V1 | Regression freeze | Golden CSV snapshots in `tests/regression/golden/` (corner bump and rack sweeps of all corner metrics, vehicle heave/roll/pitch/rack sweeps, the vehicle summary as JSON, and the reference study's evaluation frame). They are compared with tight tolerances and regenerated only with `pytest --update-golden`. The existing hand-picked values stay. |
| V2 | Validation data | No external validation data for now. Validation rests on the analytic tests plus the golden snapshots. |
| V3 | Docs site | A MkDocs Material site with mkdocstrings, deployed to GitHub Pages: tutorial, how-to pages, YAML reference, API reference, and the existing Markdown docs. |
| V4 | Stable surface | Stable surface: vehicle YAML, study YAML, CLI and `suspkin.__all__`. `suspkin.optimize` is experimental. The vehicle YAML gets `schema: 1`. Deprecation policy: one minor release of `DeprecationWarning` before removal. |
| V5 | CI | Ubuntu and windows × Python 3.11/3.12/3.13, a coverage floor, a docs build (`mkdocs build --strict`) and the examples run. |
| V6 | Release | On a `v*` tag, a GitHub Action builds the sdist and wheel and attaches them to a GitHub Release. PyPI comes later. MIT licence (already present). |
| V7 | Examples | Plain `.py` scripts, run in CI and shown in the docs. |
| O7 | Package name | `suspkin` |
| D9 | Body rotation pivot | Default pivot for body pitch (and roll) = vehicle **CG** (x = −a, y = 0, z = h_cg), decided 2026-11-17. Overridable per call. |
| O8 | GitHub + CI | **Yes**. User creates an empty GitHub repo and gives the URL. Then: add remote, push `main` + tags, add a GitHub Actions workflow (ruff + mypy + pytest). |
| D10 | Corner solver | Own damped Newton with analytic Jacobian replaces `scipy.optimize.root` as the primary corner solver (decided 2026-11-17); scipy `hybr` with the same Jacobian stays as fallback, and far guesses go through staged continuation from the guess. Why: finite-difference `hybr` needed about 15 residual evaluations per solve with numpy overhead on 3-vectors; Newton needs about 3 on the reference car and a scalar evaluation is much cheaper, with results identical to 1e-9. Continuation gained a linear predictor and a 0.05 rad branch-jump guard that triggers step halving (extra work in rack sweeps near lock). |
