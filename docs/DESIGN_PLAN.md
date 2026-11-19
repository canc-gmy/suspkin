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
- Neural-network surrogate of forces and stresses (the kinematic surrogate is planned for v1.1, §13).
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

**Hardpoint grid (S16).** Designs must be producible: the study has a `resolution` (default
0.5 mm; `null` = continuous; optional per-variable override). `DesignSpace.points`,
`to_vehicle` and `quantise_x` round every free coordinate to an integer multiple of the
resolution in absolute vehicle-frame coordinates: per axis to the nearest node inside the bounds
for box and offset domains, and to the nearest node inside the region (searching neighbouring
nodes in growing cubes) for a hexahedron. A domain that contains no node is a study error at load
time. `Problem.evaluate_batch` evaluates, caches (raw `x` as alias) and logs the snapped design;
the finite-difference step is at least one grid step. The baseline vehicle is evaluated as it is.

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
| v1.1.0 (implemented, benchmark pending, §13) | NN surrogate (front + rear MLPs, Optuna), analytic vehicle layer, active-learning optimizer | See §13.13 |

**Future phases (not planned in detail yet)**:

- **v1.x Forces**: quasi-static link forces from tyre load cases.
- **v2.x FEM**: 3D beam/truss FEM of members (stress, pushrod buckling); load cases from the force solver.
- **v3.x Surrogate of stresses**: extend the v1.1 kinematic surrogate (§13) with force and stress outputs.

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
| D8 | FEM / NN | FEM deferred to future phases. Kinematic NN surrogate planned for v1.1.0 (§13, S1–S14). |
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
| S1 | Surrogate scope | v1.1.0 (was planned as v0.9.0 before v1.0.0 shipped) brings the NN surrogate forward from v3.x, for kinematics only (no stresses). Every reported design is FOM-verified (§13). Symmetric variables only; asymmetric variables deferred (decided 2026-11-18). |
| S2 | Axle split (decided 2026-11-18) | Two independent models, front and rear. Inputs = the axle's design variables in the unit cube; vehicle data are not inputs. |
| S3 | Outputs (decided 2026-11-18) | Left-corner curves on a fixed grid (front: travel × rack, rear: travel), channel registry of §13.4, with a validity mask. Swing angles instead of instant-centre coordinates, each encoded as a (cos, sin) pair to avoid the ±π wrap (replaced by the wrapped line angle, S17). As built: the mask is "solved and finite" only; `toggle_margin` and the solver coordinates `theta_uca`, `theta_lca`, `psi` are channels; there is no ARB or wheel-centre-z channel (§13.4). |
| S4 | Derived quantities (decided 2026-11-18) | Rates (gains, motion ratios, wheel and ride rates) are finite differences of predicted curves, not network outputs. |
| S5 | Vehicle metrics (decided 2026-11-18) | Computed by an analytic, vectorised layer from the curves (§13.5), never predicted. Layer error measured separately from network error. As built: layer error is far below the §13.5 tolerances (§13.16). |
| S6 | Sampling (decided 2026-11-18) | Latin hypercube (`random-cd`), seeded, baseline as row 0; default 20 000 samples per axle; infeasible samples kept. As built: `dataset.lhs_optimization: auto` (`random-cd` up to 5000 samples, none above); the LHS matrix is computed once and cached in `lhs.npy`. |
| S7 | Dataset store (decided 2026-11-18) | Append-only `.npz` shards + `meta.json` with vehicle, design-space and grid hashes. Fixed 80/10/10 split, frozen test split. As built: shards are `shard_XXXXX.npz` under `dataset/<axle>/`; the meta also holds the channel list. |
| S8 | Network (decided 2026-11-18) | PyTorch MLP (`surrogate` extra), CUDA available on the user's PC (CPU fallback), regression + feasibility heads, masked MSE + BCE, optional POD output. As built: CUDA (cu124) wheels on Windows, CPU wheels on Linux; `device: auto` picks CUDA when present; fine-tuning never returns a worse model (§13.7). |
| S9 | Architecture search (decided 2026-11-18) | Optuna TPE + Hyperband, objective = validation RMSE normalised by per-channel engineering tolerances; SQLite storage; front and rear in parallel. As built: front and rear in parallel on CUDA, sequential on CPU (`hpo.parallel: auto`). |
| S10 | Active learning (decided 2026-11-18) | Optimize on the networks → FOM run at the predicted optimum (+ optional infill) → train on the residual by **variant A**: append the FOM points to the dataset with weight w and fine-tune from the current weights. Variant B (frozen base + correction model) is not implemented. |
| S11 | Safeguards (decided 2026-11-18) | Trust region around the best FOM-verified design with ratio test; constraint margin `g ≤ −δ` on the surrogate. As built: a box trust region around the best verified design; radius rules in §13.9. |
| S12 | Stopping (decided 2026-11-18) | Model–FOM agreement for p iterations, or stagnation of the best FOM-verified f, or budgets. As built: also `radius_min` (radius at its minimum without improvement for p iterations) and `no_candidates`; see §13.9. |
| S13 | Interface (decided 2026-11-18) | `SurrogateProblem` with the v0.8 `evaluate_batch` interface, so existing drivers run unchanged; `suspkin surrogate sample|train|validate`, `suspkin optimize --surrogate DIR`. As built: `SurrogateProblem` subclasses the v0.8 `Problem`; the sample, train and validate commands take `--root` (`--out` is a deprecated alias of `--root` for `sample`). |
| S14 | Go/no-go (decided 2026-11-18) | Slice S0 profiles the FOM first; the plan goes on only if a surrogate evaluation is at least 100× cheaper. Result: go (§13.15). |
| S15 | Search trial budget (decided 2026-11-18) | Search trials are screening runs: capped at `hpo.max_epochs` epochs (default 300) on a seeded `hpo.subsample` of the train rows, scored on the full validation split; the best configuration is retrained with the full budget. The budget is stored in the study and a resume with a different one is refused (§13.8). |
| S16 | Producible hardpoints (decided 2026-11-18) | Every hardpoint coordinate an optimiser chooses is a multiple of the study `resolution` (default 0.5 mm, absolute vehicle-frame grid, `null` = continuous, per-variable override). Quantised at decode time (round to the nearest node; nearest node inside the domain for bounds and hexahedra); the optimisers keep a continuous population and evaluate, record and export the snapped design. The surrogate input is the snapped design re-encoded; the training dataset and its hashes stay continuous. The baseline is evaluated as it is. Active learning verifies snapped candidates, dedupes infill by design and clamps the trust-region radius to one grid step (§13.9). |
| S17 | Swing angles as line angles (decided 2026-11-19) | The three swing-angle channels hold the angle of the line (reference point → instant centre) wrapped to (−π/2, π/2], tolerance 0.05 deg, instead of the (cos, sin) of a direction: the direction flips by π when the IC changes side (parallel arms), a step the network cannot fit (rear search score stuck near 15). Roll centre, pitch centre and anti-features need only the line. The search objective stays the mean over channels; per-channel ratios (median, max, worst 5) are stored in the model meta and shown in the HPO report so one bad channel cannot hide. `toggle_margin = min(a, π − a)` has a kink at a = π/2 (maximum margin); the smooth alternative is to predict `a` (angle between pushrod and rocker lever) and derive the margin, not done. |
| S18 | Fine-tuning cost (decided 2026-11-19) | In the smoke run the per-iteration fine-tune took 309 to 361 s (about 55 % of an iteration), about 90 % of it in training epochs and the rest in data load and normalisation; the front never improved on its starting weights (kept_init) from iteration 1 on. Replay rows and validation split are now normalised and moved to the device once and extended with the new rows (replay drawn once per run); `epochs` 60 (cosine schedule compressed to it), `replay` 2048, `patience` 30 counted from the best epoch of the run; an axle is fine-tuned only if the residual of the new verified point exceeds `trigger` (1.0 tolerance), and after `skip_after` 2 kept-init fine-tunes it is skipped for `skip_for` 2 iterations unless the residual exceeds 3x the trigger. Skips are logged (`skipped_<axle>` and `skip_reason_<axle>` in al.csv). Addendum: when the networks run on the CPU (resolved device `cpu`) fine-tuning is skipped by default (`finetune.on_cpu: skip`, reason `cpu`; `run` forces it), because a CPU fine-tune of the front network takes many minutes; CUDA is unchanged; the never-worse guarantee and the trust-region rules are unchanged. Measured on the benchmark networks: both axles about 50 s worst case (front 39 s, rear 9 s with warm cache) instead of about 315 s. |

---

## 13. Surrogate model and active learning (v1.1.0, implemented)

Status: **implemented (v1.1.0, unreleased)**. Benchmark (§13.13) pending. Decided 2026-11-18: S1–S14 (§12). The text below is the design as drafted; each **As built** note records what changed.

### 13.1 Goal and principle

Optimize **all** hardpoints of both axles together (about 40–45 free coordinates per
axle, about 90 for the car with symmetric variables). Direct optimization costs about
1 s per design (v0.8 demo history), too slow for the evaluation counts this needs.
Two MLPs (front, rear) learn the corner kinematic curves, the optimizer runs on the
networks, and an active-learning loop corrects the networks with full-order model
(FOM) runs at the predicted optima.

Rules:

- The FOM is the v0.8 path: corner solver + metric tables + evaluator.
- **Every reported design is FOM-verified.** The surrogate only proposes designs.
- Vehicle-level metrics (anti-dive, anti-squat, roll centre, roll stiffness, TLLTD,
  roll gradient, Ackermann …) are **not** predicted. An analytic layer computes them
  from the corner curves of the networks (§13.5).

### 13.2 Scope of v1.1.0

Stability (`docs/guides/stability.md`): the `surrogate` study block is a new optional
block of study schema 1, allowed in a minor release. The Python package
`suspkin.surrogate` is **experimental**, like `suspkin.optimize`: its API may change in
minor releases, and it is not exported in `suspkin.__all__`.

In: symmetric variables on both axles, LHS dataset, two MLPs with Optuna architecture
search, analytic vehicle layer, active-learning optimizer, benchmark against direct DE.

Out (later):

- asymmetric variables (the model is per corner, so they need two model calls per axle);
- spring rates and ARB stiffness as variables (they enter only the analytic layer, so
  they can be added without retraining);
- stresses (v3.x).

A trained model is bound to one vehicle baseline, one design space and one output grid
(hashes stored with the model). Changing any of them needs a new dataset.

### 13.3 Axle split

Front corner kinematics depend only on front hardpoints, rear only on rear hardpoints:

- front model: x_F ∈ [0, 1]^d_F → front corner curves;
- rear model: x_R ∈ [0, 1]^d_R → rear corner curves;
- the study design vector is split by axle with the existing `DesignSpace` slots;
- vehicle data (mass, CG, wheelbase, brake bias, drive split, spring rates, ARB, tyre)
  are not network inputs: they enter the analytic layer.

The two datasets and the two trainings are independent and run in parallel. A dataset
sample solves only the left corner of one axle (about half the cost of a full design).

### 13.4 Network outputs: corner curves on a fixed grid

Grid (from the study `surrogate.grid`):

- front: wheel travel × rack, e.g. 41 × 21 points;
- rear: wheel travel only, e.g. 41 points.

Channels per grid point, left corner, body frame, SI (registry in code; starting set):

| Group | Channels (as built) |
|---|---|
| Alignment | camber, toe, steer angle (front only), caster, KPI, scrub radius, mechanical trail |
| Wheel position | half-track change, contact-patch dx and dy, wheel-centre recession |
| Centres | front-view swing angle, side-view swing angle, side-view wheel-centre angle, each as the line angle wrapped to (−π/2, π/2] (S17) |
| Actuation | coilover length, rocker angle, pushrod angle, toggle margin |
| Joints | misalignment angle per rod end and ball joint (the 8 entries of `JOINTS`) |
| Solver coordinates | `theta_uca`, `theta_lca`, `psi`: rebuild the exact corner states in the evaluator |
| Validity | per-point mask: kinematically solved and finite only |

**As built.** The registry has 29 channels on the front (11 alignment and wheel position, 3 swing-line angles, 4 actuation, 8 joints, 3 solver coordinates) and 28 on the rear (the steer angle is front only). There is no ARB channel: the drop-link point follows exactly from the rocker angle, so the vehicle layer rebuilds it. There is no wheel-centre z: it equals the static z plus the travel. Coilover range and toggle margin are constraints and channels, not part of the mask.

- **Swing-line angles (S17).** The three swing angles are stored as the angle of the line
  reference point → instant centre, taken modulo π and wrapped to (−π/2, π/2] (0 = horizontal
  line; the sign is that of the metric tables: positive = anti-geometry for the side view, IC above
  the patch for the front view). The metric tables give a direction, which jumps by π when
  the IC moves from one side of the reference point to the other (nearly parallel arms, about 48 %
  of the rear designs of the surrogate study): a step function of the design that capped the
  search score near 15. Every consumer needs only the line: the roll centre and pitch centre are
  intersections of lines, and anti-dive / anti-squat / anti-lift are tan(angle), which has period π.
  The wrapped angle is smooth and small (a few degrees). Tolerance 0.05 deg. The channel
  list changed, so datasets and models of the old encoding are refused by the channel check.
  Caveat: a vertical line (angle ±π/2) would still wrap; it does not occur for real suspensions.
- **Swing angles, not instant-centre coordinates.** Instant centres go to infinity for
  parallel arms; the angles stay bounded and smooth. Anti-features use tan(angle)
  (`metrics.vehicle`); the symmetric roll centre follows from the front-view angle and
  the contact patch.
- **Rates are derived, not predicted.** Camber gain, bump steer, motion ratios (spring,
  ARB), wheel and ride rate are finite differences of the predicted curves on the
  grid, with the existing formulas. They stay consistent with the curves.

### 13.5 Analytic vehicle layer (curves → vehicle metrics)

Pure numpy, vectorised over a batch of designs. Inputs: front and rear curve tensors and
the vehicle data. Right corners mirror the left (the right wheel at rack r is the
mirrored left wheel at −r).

- **Heave:** corner travel = body heave.
- **Roll φ:** each wheel's travel solves "tyre contact height = static contact height" on the tilted ground, as in `solve_vehicle`. The small-angle mapping of the draft had the wrong sign and is not used.
  Roll centre from `roll_centre_pair`, roll camber = camber(s) + φ, roll steer from toe(s).
- **Anti-features:** `anti_dive_front`, `anti_lift_rear`, `anti_squat_rear` from the
  swing angles (`anti_angle_for` picks contact-patch or wheel-centre angle).
- **Roll stiffness, TLLTD, roll gradient:** existing functions in `metrics.vehicle`,
  with wheel and ARB rates from the derived motion ratios.
- **Steering:** Ackermann %, steering ratio, turning radius from steer angle vs rack of
  the inner and outer wheel.
- **Pitch:** travel per axle from pitch angle and axle distance to the pivot.

Layer error is measured **separately from network error**: FOM curves go into the layer
and the result is compared with `vehicle_summary` / `vehicle_frame` on the baseline and
on random designs. Proposed tolerances, within roll ±2 deg:

- roll-centre height: 1 mm;
- anti-features: 1 %-point;
- TLLTD: 0.5 %-point;
- roll gradient: 0.01 deg/g.

**As built.** Travel 0 and rack 0 must be grid nodes. Curves are interpolated with local cubic Hermite splines (C1, slopes from `np.gradient`, four nodes). A query outside the grid, or touching a masked node, gives NaN for the affected outputs, never an extrapolation. Measured with FOM curves on broad designs, all ARB types and brake/drive layouts: roll-centre height about 1e-13 mm, TLLTD up to 2.6e-4 %-pt, roll gradient up to 4e-6 deg/g, wheel rate about 5e-5 relative. All far inside the tolerances above. Cost about 0.46 ms per design at batch 256.

### 13.6 Dataset

- **Sampler:** Latin hypercube in the unit cube of each axle (`scipy.stats.qmc`,
  `optimization="random-cd"`), seeded, baseline as row 0.
- **Size:** `n_initial` per axle, default 20 000. Measured in S0 with the v0.8 tables:
  front sample (41 × 21 grid) 2.65 s, rear 0.16 s, so 20 000 front samples take about
  2.1 h on 7 workers. The S1 extractor computes only the registry channels, vectorised
  over all states of a sample (target ≥ 5× faster than `metrics_frame` +
  `joint_angles_frame`, equal to them within 1e-9).
- **FOM per sample:** corner sweeps on the §13.4 grid for the left corner of that axle,
  through the existing parallel pool, with the budget and resume rules of O18.
- **Infeasible samples** (the corner does not assemble, or the solver fails) are kept
  with `feasible = 0` and an all-false mask: the feasibility head needs them.
  `sample` reports the feasible fraction and warns below 30 % (domains too large).
- **Storage:** `DIR/dataset/{front,rear}/shard_XXXXX.npz` with:
  - `X (N, d)`, `Y (N, C, G)`, `mask (N, G)`, `feasible (N,)`;
  - `source` (`lhs` or `al_<k>`);
  - a `meta.json` (vehicle and design-space hashes, grid, channels, version, seed).

  Shards are append-only, so sampling resumes.
- **Split:** train / validation / test 80 / 10 / 10 by seed. The test split is frozen
  and never used for training or model selection. Active-learning points go to
  training only.

**As built.** The LHS matrix is computed once and cached in `dataset/<axle>/lhs.npy`. `lhs_optimization: auto` uses `random-cd` up to 5000 samples and none above. Measured for the front 41 × 21 grid: 14.8 samples/s, about 23 min for 20 000 samples. About 100 kB per sample in float32. `--root` is the surrogate folder for `sample`, `train` and `validate`.

### 13.7 Networks

Per axle, in PyTorch (new optional extra `surrogate`):

- **Input:** x in [0, 1]^d (already normalised by the design space).
- **Body:** L fully connected blocks `Linear → activation → (LayerNorm) → (Dropout)`,
  optional residual connections between equal-width blocks.
- **Heads:** regression (C·G values, or k POD coefficients) and feasibility (G logits).
- **Scaling:** per-channel standardisation over the training set.
- **POD (optional):** PCA of the standardised curves. The network predicts k
  coefficients and the curves are rebuilt.
- **Loss:** masked MSE on standardised channels + λ · BCE on the validity mask.
- **Training:**
  - AdamW, cosine schedule with warm-up, early stopping on validation loss;
  - seeded (deterministic algorithms where available);
  - runs on CPU or CUDA.

Front and rear train in two separate processes, each with its own Optuna study.

**As built.** `resolve_device`: `auto` picks CUDA when available, else the CPU; `cuda` raises when no CUDA device exists. Windows installs the CUDA (cu124) torch wheels, Linux the CPU wheels. Models load on any device. Fine-tuning starts from the current weights with `training.finetune_lr_factor` (default 0.1) and no warm-up, and never returns a model worse than its starting weights. GPU smoke test: 20 000 samples, 29 channels, 21 × 41 grid fit in 1.33 GB and take 4.4 s per epoch on a GTX 1650.

### 13.8 Architecture search with Optuna

| Hyperparameter | Range |
|---|---|
| layers L | 2–8 |
| width | 64–1024 (log) |
| activation | SiLU, GELU, ReLU, tanh |
| residual, LayerNorm | on / off |
| dropout | 0–0.2 |
| learning rate | 1e-4–3e-3 (log) |
| weight decay | 1e-6–1e-3 (log) |
| batch size | 128–2048 |
| output | raw curves, or POD with k in 8–128 |
| λ (feasibility weight) | 0.01–1 (log) |

- **Objective:** validation error in engineering units, the mean over channels of
  RMSE_c / tol_c. Each channel has a tolerance in the registry (for example camber
  0.02 deg, toe 0.01 deg, coilover length 0.1 mm, swing angles 0.05 deg).
- **Sampler and pruner:** TPE (seeded), with a Hyperband pruner on the per-epoch
  validation error.
- **Storage and budget:** SQLite in the model folder, so a search resumes. Budget is
  `n_trials` plus `timeout`.
- **Trial budget vs final retrain (S15):** one full-length trial on 20 000 samples took
  about 85 min, so a 90 min search ran 1 or 2 trials. Each trial now trains at most
  `hpo.max_epochs` epochs (effective cap = min with `training.max_epochs`; patience
  `min(training.patience, max(10, cap // 10))`; Hyperband `max_resource` = cap) on
  `hpo.subsample` training rows (seeded by the dataset seed, same rows for all trials).
  The validation split is always complete, so trial scores are comparable with the
  retrained model. The best configuration is then retrained on the full train split with
  `training.max_epochs` and `training.patience`. The budget is recorded in the study; a
  resume under another budget is an error, and trials left RUNNING by a killed process
  are marked FAIL.
- **Report:** `models/hpo_report.md` (written after the search, or on demand by
  `scripts/surrogate_hpo_report.py --root DIR`, read-only on `optuna.db`) summarises
  budget, trial states, best and top trials, pruning and the final model.
- **When it runs:**
  - once on the initial dataset;
  - in the active-learning loop the best architecture is kept and retrained;
  - a short new search runs when the dataset has doubled or the validation error
    stops falling.

### 13.9 Active-learning optimization

Iteration k:

1. **Optimize on the surrogate.** `SurrogateProblem` exposes the v0.8 `evaluate_batch`
   interface, so the existing drivers run unchanged with large budgets (order 1e5
   evaluations).
   - Constraints use a margin `g ≤ −δ`, so network error does not push the optimum
     out of the feasible set.
   - The search stays inside a trust region around the best FOM-verified design.
2. **Network optimum.** The infill set is:
   - weighted mode: the predicted optimum x*_k;
   - pareto mode: a spread subset of the predicted front;
   - optional: top-m distinct candidates and p perturbations of x*_k in the trust region.
3. **FOM run** of the infill set (whole design, both axles). It gives the true curves,
   f and g. These rows go to `history.csv` with `source = al_<k>`.
4. **Residual.** r = y_FOM − y_NN on the curves at the infill points, and the
   prediction errors of f and g. Logged per iteration.
5. **Training on the residual** (S10, variant A): add the infill points to the dataset
   with weight w and fine-tune from the current weights for at most `epochs` epochs
   (early stopping with `patience` on validation and the new points; S18). On the CPU
   fine-tuning is skipped unless `finetune.on_cpu: run` (S18); the reason is logged per axle
   in `al.csv` (`skip_reason_<axle>`). The loss on those points is the residual
   y_FOM − y_NN. Variant B (frozen base + correction model δ(x)) was considered and not
   chosen: the residual points are few and clustered along the optimizer path.
6. **Trust region.** Compute ρ = FOM improvement / predicted improvement. Expand the
   region if ρ > 0.75, shrink it if ρ < 0.25.

The loop stops at the first of:

- agreement: |f_NN − f_FOM| / scale < ε_f with matching feasibility, p iterations in a row;
- stagnation: no gain in the best FOM-verified f for p iterations;
- budgets: iterations, FOM evaluations or wall time.

Result: the best FOM-verified design, with the v0.8 run outputs (§7.14).

**As built.**

- Fine-tuning replays all active-learning rows plus 4096 random earlier training rows (`finetune.replay`). Early stopping scores the mean of the validation score and the score on the active-learning rows (`extra_val`).
- The dataset update is atomic per iteration; a resumed run repairs a partly written iteration.
- The trust region is a box `[x_b − r, x_b + r]` in the unit cube around the best FOM-verified design; the inner driver runs on this box.
- Trust ratio. With a feasible verified best, ρ = (f_b − f_FOM) / (f_b − f_NN); ρ is undefined (NaN) when no gain was predicted, and the radius then halves. A FOM-infeasible optimum gives ρ = 0. Without a feasible verified design, a feasible FOM result gives ρ = 1, a smaller worst g gives 0.5, anything else 0.
- Radius update: halve when ρ < 0.25 or undefined; double (up to `max`) when ρ > 0.75 **and the step reached the box boundary**; otherwise keep. The draft had no boundary condition.
- Pareto mode: the trust region is centred on the knee of the FOM-verified front; ρ = 1 when the new optimum extends the verified front, else 0.
- Stop reasons: `agreement`, `stagnation`, `radius_min`, `max_iterations`, `fom_budget`, `time_budget`, `no_candidates` (the network search returns nothing).
- The FOM budget counts every row of the run folder's history, baseline and resumed rows included. A resume refuses a study whose constraints or objectives changed.
- The inner network search budget is `active_learning.surrogate_evaluations` (default 20 000).
- Hardpoint grid (S16). The network input is the snapped design re-encoded to the unit cube; the
  FOM verifies the snapped candidate. Infill points (top-m and perturbations) are snapped, then
  deduplicated against the FOM history; if no new grid design remains the loop stops with
  `no_candidates`. The effective minimum radius is `max(trust_region.min, one grid step)` (largest
  step over the variables, capped by `max`), and the boundary test of the doubling rule allows
  half a grid step of slack. The `baseline` history row is the snapped baseline. The training
  dataset is not snapped and the hashes ignore the resolution.

Shared models: with only `ROOT/models/` (no dataset) `optimize --surrogate` checks the model
metadata alone and disables fine-tuning per axle (`skip_reason` `no_dataset`); `validate` and
`train` still need the dataset.

### 13.10 Study YAML: `surrogate` block

```yaml
surrogate:
  grid:
    travel: {min: -30 mm, max: 30 mm, points: 41}
    rack: {min: -25 mm, max: 25 mm, points: 21}     # front only
  dataset: {n_initial: 20000, seed: 1}
  training: {device: auto, max_epochs: 2000, patience: 100}
  hpo: {n_trials: 100, timeout: 4 h}
  active_learning:
    max_iterations: 50
    fom_evaluations: 2000
    infill: {top: 0, perturb: 4}
    finetune: {epochs: 60, weight: 5.0, replay: 2048, patience: 30, trigger: 1.0}   # S10, S18
    margin: 0.05                        # delta on normalised g
    trust_region: {initial: 0.2, min: 0.01, max: 0.5}
    stop: {eps_f: 0.01, patience: 5}
```

**As built.** The block also takes `dataset.shard_size` (default 500), `dataset.lhs_optimization` (default `auto`), `training.num_threads`, `training.finetune_lr_factor`, `hpo.parallel`, `active_learning.time`, `active_learning.surrogate_evaluations` and `finetune.replay`. The full key table is in the [study YAML guide](guides/study-yaml.md).

### 13.11 CLI

- `suspkin surrogate sample STUDY.yaml --root DIR [--axle front|rear|both] [--n N] [--workers N|auto] [--resume]` (`--out` is a deprecated alias of `--root`)
- `suspkin surrogate train STUDY.yaml --root DIR [--axle …] [--trials N] [--timeout T] [--device D]`
- `suspkin surrogate validate STUDY.yaml --root DIR [--axle …] [--device D]`: test error per channel against its tolerance, parity and worst-case curve plots. The layer error is measured separately (`layer_error`, §13.16).
- `suspkin optimize STUDY.yaml --surrogate DIR [--device D] [--evaluations N] [--time T] [--resume] [--out RUN]`: the active-learning optimizer of §13.9.

### 13.12 Slices

| Slice | Content | Depends on | Agent |
|---|---|---|---|
| S0 | Profile one FOM evaluation (solve vs metric tables vs evaluator). Go/no-go: a surrogate evaluation must be ≥ 100× cheaper. **Done** (`scripts/profile_fom.py`), see §13.15 | – | coder |
| S1 | Grid and channel registry; FOM curve extractor for one corner (design → `Y`, mask) | S0 | coder |
| S2 | Analytic vehicle layer + layer-error validation against the FOM | S1 | coder + reviewer |
| S3 | LHS sampler, sharded dataset, `surrogate sample`, resume | S1 | coder |
| S4 | MLP, training, Optuna search, front/rear in parallel, `surrogate train` / `validate` | S1 | coder + reviewer |
| S5 | `SurrogateProblem`: x → networks → curves → vehicle layer → evaluator → f, g | S2, S4 | coder |
| S6 | Active-learning driver, trust region, run outputs, AL plots (f_NN vs f_FOM, residual per iteration) | S5 | coder + reviewer |
| S7 | Benchmark study with all hardpoints free, docs, CHANGELOG, tag `v1.1.0` | S6 | chore + reviewer |

S2, S3 and S4 run in parallel after S1.

### 13.13 Done when

- [ ] Test RMSE per channel is below its tolerance on both axles (held-out LHS split): **pending**, no `surrogate validate` result is recorded in the repository yet.
- [x] Vehicle-layer error on FOM curves is within the §13.5 tolerances: **met** (§13.16).
- [ ] On the benchmark study (all hardpoints free), the active-learning optimizer reaches a FOM-verified best f no worse than direct `de` with the same number of FOM evaluations: **pending**, the benchmark is not yet run (§13.15).
- [x] Runs are reproducible with a fixed seed: **met per device**. CPU and CUDA runs are not bit-identical.
- [ ] `main` stays green without the `surrogate` extra (its tests are skipped when torch is missing): **to confirm in CI**.

### 13.14 Risks

- **Low feasible fraction** of LHS in ~40 dimensions, so the dataset is mostly failures.
  Mitigation: feasibility head, fraction reported by `sample`, domain check before sampling.
- **Surrogate exploitation:** the optimizer finds regions where the network is wrong.
  Mitigation: trust region, constraint margin, FOM check every iteration.
- **Evaluator overhead:** the v0.8 evaluator works on pandas tables. If S0 shows that
  they dominate, S5 needs a vectorised evaluator path.
- **Small-angle vehicle layer:** the roll and pitch mapping is approximate. Its error is
  measured on its own (§13.5), and the final design is FOM-verified anyway.
- **torch with spawn workers on Windows:** torch is imported only in training and
  surrogate processes, never in the FOM workers.

### 13.15 S0 result (2026-11-18)

Measured with `scripts/profile_fom.py` on the baseline of `fs_reference_optimize.yaml`,
one process:

| Item | Value |
|---|---|
| One FOM evaluation | 0.72–0.82 s (696 corner solves) |
| Corner solving | 47 % |
| Metric tables (pandas, per-state dicts) | 38 % |
| Vehicle-sweep overhead | 10 % |
| Constraint and objective logic | 0.4 % (3 ms) |
| `vehicle_summary` (roll and heave sweeps inside) | about 85 % of the cumulative time |
| Per state: solve / `metrics_frame` / joint angles | 0.46 ms / 1.9 ms / 0.95 ms |

Hotspots by own time are small-array numpy calls (`cross`, `norm`, `moveaxis`).

Decision: **go**. A batched MLP forward pass costs microseconds per design on the GPU.
The constraint and objective logic costs 3 ms. A vectorised vehicle layer in numpy
should stay under 1 ms per design in a batch. That leaves about 5 ms per design,
which is more than 100× cheaper than 0.75 s.

Conditions:

- S5 must not build pandas tables per design: the surrogate path feeds the evaluator
  arrays, or a vectorised evaluator is added.
- S1 vectorises channel extraction (§13.6), because the tables cost about 6× the solve
  per state.

Side finding, outside v1.1: the same vectorisation applied to the FOM tables and to
`vehicle_summary` would make every v0.8 evaluation about 2× faster.

**Benchmark (v1.1, not yet run).** Study `configs/studies/fs_reference_surrogate.yaml`: 90 variables (all 15 hardpoints of each axle as ±10 mm offsets, wheel centres ±5 mm, symmetric). Runner `scripts/benchmark_surrogate.py` with resumable stages `sample`, `train`, `validate`, `al`, `de_same`, `de_total` and `report`. Command: `python -m uv run python scripts/benchmark_surrogate.py --device cuda --workers auto`. Estimated wall time about 4 to 6 h: sampling about 30 min, architecture search about 1.5 to 2 h, active learning about 1 to 3 h, direct DE at the total cost about 30 min. The report with both numbers of §13.13 is written to `results/benchmark_surrogate/run/benchmark.md`.

**Inner-loop timing note (2026-11-19).** On the benchmark study (90 variables, DE `weighted`, GTX 1650, GPU shared with other work) one design cost about 14 ms in the active-learning inner loop (20 000 evaluations in 275 to 300 s), against 2.5 ms for the network alone. The rest was per-design Python in the design decode (`DesignSpace.points` 4.5 ms per call, called three times per design: vehicle, history row, history header), plus cache keys and history flushes. After vectorising the decode over the batch, caching the baseline encoding and flushing the history once per batch it is about 2.6 to 3.3 ms per design (about 50 to 62 s for 20 000 evaluations, depending on machine load). The stored history is bit-identical.

### 13.16 As-built performance and accuracy (v1.1.0)

- **FOM speed-ups** (made in this milestone; they benefit the FOM and the surrogate's training data): `metrics.batch` vectorisation (about 3× faster evaluation, 0.75 s to 0.24 s on the reference study); evaluator logic with a closed-form line fit and window fast paths; a convex-hull cache in `optimize/regions.py`.
- **`SurrogateProblem`** (S5): subclasses the v0.8 `Problem`. Oracle check against the FOM (exact curves through the surrogate evaluator): g within 8e-6, f within 2e-6. About 2.5 to 2.9 ms per design at batch 1024, against about 240 ms for the FOM.
- **Vehicle layer**: about 0.46 ms per design at batch 256; layer errors in §13.5.
- **Dataset**: 14.8 samples/s on the front 41 × 21 grid (§13.6).
