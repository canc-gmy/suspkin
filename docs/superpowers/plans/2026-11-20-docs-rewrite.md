# Docs Rewrite Implementation Plan

> **For agentic workers:** executed with superpowers:subagent-driven-development. Writers own disjoint files; the orchestrator owns `mkdocs.yml`, `README.md`, `CLAUDE.md`, deletions.

**Goal:** Replace the docs with a precise Diátaxis-structured set that explains theory + code for every part of `suspkin`.

**Spec:** `docs/superpowers/specs/2026-11-20-docs-rewrite-design.md`

## Global Constraints

- English only. SI units with symbols; mm/deg only when quoting display output.
- Every statement about code is checked against the current code (HEAD of this branch), never copied from old docs unverified.
- No line-number links. Refer to code by fully-qualified name in backticks (e.g. `suspkin.kinematics.corner.solve_corner`) or by mkdocstrings cross-reference `[name][suspkin.module.name]`.
- Equations in MathJax (`$...$`, `$$...$$`, arithmatex generic). Diagrams in Mermaid fences.
- Worked numbers come from actually running the code on `configs/vehicles/fs_reference.yaml` (or the study configs); state the command used.
- Only docstrings/comments may change under `src/`. No behaviour change: the behaviour lock and full suite must still pass if `src/` is touched.
- `python -m uv run mkdocs build --strict` must pass (pages not yet in nav are INFO only).
- Writers do not edit `mkdocs.yml`, `README.md`, `CLAUDE.md`, or files owned by another task. Writers do not commit.

## Concepts page contract

Each `docs/concepts/*.md` page has: (1) purpose + position in the pipeline (diagram if useful); (2) theory with equations and sign conventions; (3) "How the code does it": algorithm step by step with exact equations, constants and tolerances from code, function names, inputs/outputs with units and shapes; (4) edge cases and failure modes; (5) worked example with real numbers; (6) where it is tested; see also.

---

### Task D1: conventions + ADRs
Files: create `docs/concepts/conventions.md`; create `docs/adr/0002-*.md` … one ADR per still-relevant decision (or tight group) from `docs/DESIGN_PLAN.md` §12 decisions log and the refactor decisions in `docs/superpowers/specs/2026-11-19-refactor-design.md` (one ADR "0NNN-2026-restructure" summarising that refactor and its rulings). ADR format: Status, Context, Decision, Consequences (short). Source: DESIGN_PLAN §2 and §12, verified against `units.py`, `geometry/`, `kinematics/`, `metrics/`.

### Task D2: vehicle model + kinematics (3 writers in parallel)
- D2a `docs/concepts/vehicle-model.md`: hardpoints, corner construction, mirroring, axles, steering/ARB geometry, YAML loading and validation (`geometry/`).
- D2b `docs/concepts/corner-kinematics.md`: corner DOF reduction, unknowns, constraint equations, Newton solver, tolerances, rocker, steering rack, contact patch, failure modes, numba fast path (`kinematics/_numba.py`) and batched path (`kinematics/batch.py`, `kinematics/ic_arrays.py`), performance numbers (from `scripts/bench_kinematics.py`).
- D2c `docs/concepts/vehicle-kinematics.md`: vehicle poses (heave/roll/pitch/rack), body rotations, pivot, anti-roll bar (`kinematics/vehicle.py`, `kinematics/arb.py`, `metrics/arb.py`).

### Task D3: metrics + sweeps (2 writers in parallel)
- D3a `docs/concepts/metrics.md`: every metric column produced by `metrics/arrays.py`, `metrics/corner.py`, `metrics/table.py`, `metrics/vehicle.py` (alignment, wheel position, centres, anti-features, actuation & rates, ARB, roll, steering, joint angles, gradients, vehicle summary). For each: definition, physical meaning, sign, formula, code, unit, display unit.
- D3b `docs/concepts/sweeps-and-reports.md`: sweeps (`sweeps.py`), tables, static summary, report bundle (`report/`), figures and figure spec (`plotting/`), display units (`plotting/units.py`).

### Task D4: optimisation, surrogate, web app (3 writers in parallel)
- D4a `docs/concepts/optimization.md` (`optimize/`): study spec package, variables/regions/quantisation, constraints and objectives (`spec/criteria.py`), normalised values, evaluator, problem, drivers (NSGA-II, DE, local, DE+SLSQP), pool, cache/history/resume, budget, outputs/report, check command.
- D4b `docs/concepts/surrogate.md` (`surrogate/`, `optimize/spec/surrogate.py`): principle, axle split, channels/grid, dataset sampling, normalisation/POD, network, training, HPO, evaluator and vehicle layer, active learning loop, outputs; with DESIGN_PLAN §13 as source material.
- D4c `docs/concepts/web-app.md` (`app/`, `plotting/spec.py`, frontend contract): server, workspace, datasets, endpoints (all 16), figure spec, scene, snapshot export, openapi/type generation; with ADR 0001 and `docs/webapp/PLAN.md` as source material.

### Task D5: architecture + development (2 writers in parallel)
- D5a `docs/concepts/architecture.md`: package map (every module, one line each), layering rules, data flow diagram, optional extras and lazy imports, performance notes (from `docs/performance.md`), testing strategy (unit, property, golden, behaviour lock).
- D5b `docs/guides/development.md`: setup, uv usage, gate commands, CI jobs, golden/lock update rules (`--update-golden`, `--update-lock`, refuse in CI), adding a metric end to end, release workflow, pre-commit and CRLF note.

### Task D6: guides + getting started (2 writers)
- D6a rewrite `docs/guides/{vehicle-yaml,study-yaml,cli}.md` (every field / every command and flag, verified against code and `--help`).
- D6b rewrite `docs/guides/{app,surrogate,examples,stability}.md`, `docs/getting-started/{install,tutorial}.md`, `docs/index.md`, `examples/README.md`.

### Task D7: reference + docstrings (parallel per package)
- Rewrite `docs/reference/*.md` so every public module has a `:::` entry (split into one page per package: geometry, kinematics, metrics, sweeps, report, plotting, optimize, surrogate, app, units).
- Docstring audit per package: every public function/class has a precise Google-style docstring (args with units/shapes, returns, raises); fix stale names (e.g. `metrics/corner.py` mention of `roll_centre_symmetric`).

### Task D8 (orchestrator): nav, deletions, README, CLAUDE.md, final review
- `mkdocs.yml` nav for the new tree; delete `docs/DESIGN_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/webapp/PLAN.md`, `docs/performance.md`; README rewrite (short, links to docs); `CLAUDE.md` pointers (conventions now `docs/concepts/conventions.md`); CHANGELOG `Changed: documentation rewritten`.
- Gate: mkdocs --strict, full pytest, examples. Final whole-docs review (opus): accuracy spot-checks per page, broken links, duplication.
