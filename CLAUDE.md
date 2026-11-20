# suspkin — project rules

Python package for Formula Student double-wishbone + pushrod suspension kinematics.
How it works: `docs/concepts/` (start at `architecture.md`). Decisions: `docs/adr/`.
Behaviour is locked by `tests/regression/behaviour/` and `tests/regression/golden/`; never regenerate them to make a change pass.

## Language
English only: code, comments, docs, commit messages, configs.

## Conventions
- ISO 8855 frame: x forward, y left, z up. Origin on ground, mid front axle.
- SI units internally and in YAML (m, rad, N, N/m). mm/deg only for display.
- Sign conventions: `docs/concepts/conventions.md`.
- Docstrings: Google style (`Args:`, `Returns:`, `Raises:`), SI units and array shapes stated.
- Core math = pure functions on numpy arrays. Plotting reads result tables only, never calls the solver.
- Tooling: `uv`, ruff, mypy, pytest (+ hypothesis).
- On this PC `uv` may not be on PATH: call it as `python -m uv` (e.g. `python -m uv run pytest -q`).

## Orchestration (token budget)
The main session (Opus) only plans, writes specs, decides, and commits.
It does not write bulk code or read large files itself. Delegate:

| Agent | Model | Use for |
|---|---|---|
| `coder` | sonnet | Any real code slice + its tests (solver, metrics, plots, CLI) |
| `chore` | haiku | Scaffolding, configs, YAML, README, CHANGELOG, renames |
| `verifier` | haiku | ruff + mypy + pytest gate; read-only, terse |
| `reviewer` | sonnet | Math/spec review of math-heavy slices and each milestone |
| `scout` | haiku | Read-only lookups instead of reading files in the main session |

Slice protocol:
1. Orchestrator writes a spec: goal, files allowed, public signatures, acceptance tests, plan section.
2. `coder` (or `chore`) implements.
3. `verifier` runs the gate. On failure, send the failure lines back to the same agent (SendMessage), not a new one.
4. `reviewer` for math-heavy slices and at milestone end.
5. Orchestrator commits.

Independent slices (no shared files) may run in parallel.
Agent reports stay short; agents never paste code back.

## Git
- `main` stays green. Conventional Commits. SemVer tag `vX.Y.Z` per milestone (plan §11).
- `CHANGELOG.md` in Keep a Changelog format.
- Only the orchestrator commits.
