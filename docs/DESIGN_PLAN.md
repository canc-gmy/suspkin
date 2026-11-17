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
- Anti-roll bar per axle, selectable type: **torsional** (U-bar) or
  **flexural** (blade / bending bar).
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
| ARB | Per axle: type (`torsional` / `flexural`), bar pivot points, arm length, drop-link points, material + section → stiffness |
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

- `scipy.optimize.root` (hybrid Powell / Levenberg–Marquardt) with an
  analytic Jacobian where practical.
- Sweeps use **continuation**: each step starts from the previous
  solution. Step refinement kicks in on non-convergence.
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
kinematics. Its kinematic output is the bar twist (torsional) or bar
deflection (flexural) from the left/right drop-link motion.

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

- Bar twist angle (torsional) or blade deflection (flexural) vs roll
- ARB motion ratio
- ARB wheel rate and ARB roll stiffness
- Bar stiffness from material + section
  - Torsional: GJ/L plus arm bending
  - Flexural: EI cantilever

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

> **Deferred.** The kinematics side (v0.1–v0.6) is built first.
> Variables, constraints and objectives get decided in a later grilling
> round, before v0.7. The list below is a draft only. The kinematic core
> is still designed as pure functions so the optimizer can plug in later.

1. **Design variables**: a user-chosen subset of hardpoint coordinates with bounds (YAML).
2. **Geometric constraints**:
   - packaging boxes (e.g. inboard points inside the chassis envelope, outboard inside the rim)
   - minimum link lengths
   - joint angle limits
   - rocker toggle margin
   - coilover stroke within limits
3. **Objectives**: target curves or scalars, e.g.
   - bump steer ≈ 0
   - camber gain in a given range
   - roll centre height and migration limits
   - motion ratio target and linearity
   - anti-dive %
4. **Formulation**: weighted sum or multi-objective (Pareto).
5. **Optimizer**:
   - `pymoo` (NSGA-II) for multi-objective
   - `scipy.optimize` (SLSQP / differential evolution) for single-objective
6. **Outputs**:
   - optimized YAML
   - Pareto front plot
   - before/after comparison report

The objective function is a pure call into the kinematic core. Later, the
neural network surrogate plugs in behind the same interface (§11).

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
| v0.7.0 | Optimization | Variables, constraints, objectives, NSGA-II + SLSQP, comparison report |
| v1.0.0 | Stable kinematics | Docs complete, regression suite frozen |

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
| O6 | Optimization | **Deferred**. Kinematics first; optimization design gets its own grilling round before v0.7. |
| O7 | Package name | `suspkin` |
| O8 | GitHub + CI | **Yes**. User creates an empty GitHub repo and gives the URL. Then: add remote, push `main` + tags, add a GitHub Actions workflow (ruff + mypy + pytest). |
