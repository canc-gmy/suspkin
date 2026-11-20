# Docs rewrite design (project 2)

Date: 2026-11-20. Branch: `claude/superpowers-refactor-docs-d114ea`, on top of the finished code refactor (project 1).

## 1. Intent

The owner wants to understand every detail of how `suspkin` works. The docs must be precise, long where the subject is deep, and explain both the vehicle-dynamics theory and the exact code that implements it. The owner delegated the docs design ("use the best docs practice and be as precise") and decided two things:

- Guides teach **theory + code**: what a quantity is physically, why it matters, its sign convention, the exact math, and the function that computes it.
- The old planning docs (`DESIGN_PLAN.md`, `ARCHITECTURE.md`, `webapp/PLAN.md`) are **replaced**: still-true content moves into the new pages; the decision log becomes short ADRs in `docs/adr/`.

## 2. Goals, non-goals, anti-goals

Goals
- One source of truth per topic; every page reachable from the MkDocs nav.
- Every claim about the code is true for the current code (verified by a reviewer reading the code, not the old docs).
- Every metric, solver step, optimiser step and surrogate step is explained: theory → math (MathJax) → code path (module + function, linked via mkdocstrings cross-references) → worked example or test that pins it.
- Complete API reference (mkdocstrings) for every public module; docstrings precise (Google style).
- `mkdocs build --strict` passes; examples still run.

Non-goals
- No code behaviour change. Docstring and comment edits only in `src/`.
- No frontend docs beyond how it talks to the backend (frontend/DESIGN.md stays as the UI design note).

Anti-goals
- Docs that paraphrase old docs without checking code.
- Line-number links (`file.py#L123`) — they rot. Link to modules/functions by name instead.
- Duplicated explanations across pages (link instead).

## 3. Approach (decided by orchestrator under delegation)

Diátaxis structure (tutorials / how-to guides / explanation / reference), which is the standard for technical docs and maps cleanly onto "learn it", "do it", "understand it", "look it up".

```
docs/
  index.md                      what suspkin is, who it is for, map of the docs
  getting-started/
    install.md                  install, extras, verify
    tutorial.md                 first end-to-end run, explained step by step
  concepts/                     EXPLANATION — theory + code, long pages
    conventions.md              ISO 8855 frame, origin, units, every sign convention
    vehicle-model.md            hardpoints, corners, mirroring, YAML -> pydantic model
    corner-kinematics.md        the corner solver: unknowns, constraints, Newton, rocker, steering, numba + batch paths
    vehicle-kinematics.md       heave/roll/pitch/steer poses, body rotations, anti-roll bar
    metrics.md                  every metric: definition, physics, sign, formula, code, gradients
    sweeps-and-reports.md       sweeps, tables, summary, report bundle, figures
    optimization.md             study spec, design space, regions, quantisation, constraints, objectives, drivers, history/resume, outputs
    surrogate.md                dataset, channels, normalisation, network, training, HPO, evaluator, vehicle layer, active learning
    web-app.md                  server, API, datasets, figure spec, snapshot export, frontend contract
    architecture.md             package map, layering, data flow, optional extras, performance, testing strategy (golden + behaviour lock)
  guides/                       HOW-TO — task oriented
    cli.md, vehicle-yaml.md, study-yaml.md, app.md, surrogate.md (how to train/use), examples.md, stability.md, development.md (tooling, tests, golden/lock update rules, release)
  reference/                    mkdocstrings pages covering every public module
  adr/                          0001 existing + ADRs distilled from DESIGN_PLAN §12
  reports/surrogate-training.md kept (dated data report)
  changelog.md
```

Deleted: `docs/DESIGN_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/webapp/PLAN.md`, `docs/performance.md` (content merged into concepts). `docs/superpowers/` stays out of the nav (process record).

Alternatives considered: (a) keep the current flat layout and lengthen each page — rejected, theory and how-to stay mixed and ARCHITECTURE.md stays a 2.3k-line monolith; (b) one page per module — rejected, mirrors code layout instead of how a reader learns.

## 4. Page contract (every concepts page)

1. Purpose and where it sits in the pipeline (one diagram, Mermaid if helpful).
2. Theory section(s) with figures/equations and sign conventions.
3. "How the code does it": step-by-step algorithm with the exact equations used, naming each function (`::: `-style cross-refs or backticked fully-qualified names), inputs/outputs with units and shapes.
4. Edge cases and failure modes (what raises, tolerances, degenerate geometry).
5. Worked example on `configs/vehicles/fs_reference.yaml` with real numbers produced by running the code (writer runs it; numbers quoted with units).
6. "Where it is tested" (test files) and "See also".

Precision rules: SI units with symbols; every sign convention stated explicitly; every tolerance/constant quoted from code; no claim without a code location.

## 5. Process

Writers (sonnet) each own disjoint files; a reviewer (sonnet) checks every page against the code (claims, equations, signs, numbers); orchestrator owns `mkdocs.yml`, `README.md`, `CLAUDE.md` link updates and the deletions. Order: conventions + ADRs first (they consume DESIGN_PLAN before deletion), then concepts pages in parallel batches, then guides, then reference/docstrings, then index/README/nav/deletions, then a final whole-docs review on the most capable model.

## 6. Left to writers

Exact headings, diagram choice, example selection, page length (long is fine; padding is not).
