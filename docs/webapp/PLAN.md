# Experimental web app: interactive results viewer + comparison (branch `feat/webapp`)

## Context
Reporting today is static. `build_report` writes PNG/SVG, CSV and HTML. Plotly is only used for one 3D HTML view. Nothing can read results back in: report folders store no vehicle, and there is no loader.

The user wants a local web app that:
- shows every vehicle-dynamics graph interactively;
- opens several results at once and overlays them to compare configurations.

It is experimental, so it lives on its own branch. The design must make large frontend changes after the first iteration cheap.

Decisions (2026-11-18):
- **Stack**: FastAPI backend + React/TypeScript/Vite frontend with Plotly.js.
- **Data**: both live solving of vehicle YAMLs and loading saved results.
- **Scope**: view and compare only; editing comes later and the API leaves room for it.
- **Deployment**: local server, plus a shareable static HTML snapshot.

## Design principle: three contracts, so either side can change alone
1. **Result bundle**: a reloadable on-disk format for one design's results.
2. **Figure spec**: a declarative list of curves, defined in Python. The frontend renders it generically.
3. **Versioned JSON API** (`/api/v1`): its OpenAPI schema generates TypeScript types. The frontend only talks to this API, through a swappable data source: HTTP or an embedded snapshot.

A frontend redesign touches only `frontend/`. A new graph means adding one spec entry in Python, with no frontend code. A record of these decisions goes in `docs/adr/0001-webapp-architecture.md`.

## Python side (`src/suspkin/`)
| File | Content |
|---|---|
| `report/bundle.py` (new) | `save_bundle(results: StudyResults, dir, label)` and `load_bundle(dir) -> Bundle`. Layout: `manifest.json` (schema 1, label, suspkin version, source, study params) + `vehicle.yaml` + `tables/<study>.csv` (SI) + `summary.json`. `build_report` also writes one bundle per design under `bundles/`, so report folders become loadable. Reuses `StudyResults.tables()` (`report/studies.py:39`). |
| `plotting/spec.py` (new) | Frozen dataclasses: `FigureSpec(id, group, title, study, x: Axis, panels)`, `Panel(title, y_dim, series)`, `Series(column, label, side)`. The curve lists are lifted from the private `_Series`/`_Panel` definitions in `plotting/figures.py:60-582` (alignment, position, centres, actuation, steering, roll, heave/pitch). The summary rows come from `_SUMMARY_ROWS` (`figures.py:472`). Display factor and unit come from `units.display_factor`/`display_unit` (`units.py:207-212`). `to_json()`. Matplotlib figures stay as they are for now; a test checks the spec and `FIGURE_SET` cover the same columns. The `mr_progression` unit is set to 1/mm everywhere (cli vs figures mismatch). |
| `plotting/interactive.py` | Split out `interactive_figure(vehicle, state=None) -> go.Figure`. `interactive_view` keeps calling it and writing HTML, so its behaviour is unchanged. |
| `app/` (new, experimental) | **`server.py`**: FastAPI app factory `create_app(workspace)`; it serves the built frontend from `app/static/`. **`datasets.py`**: `DatasetManager`. Opening a path auto-detects its kind: vehicle YAML → `run_standard_studies`, cached by file hash + study params; bundle or report folder → `load_bundle`; optimizer run folder → baseline + best (plus the Pareto knee) as datasets, plus `history.csv`/`result.json` through `read_history`. **`routes.py`**: `/api/v1` endpoints. **`snapshot.py`**: static export. |
| `cli.py` | `suspkin app [PATHS...] [--port 8765] [--host 127.0.0.1] [--no-browser] [--dev]` (`--dev` proxies to the Vite dev server). `suspkin app export PATHS... --out compare.html`. |
| `pyproject.toml` | Extra `app = ["fastapi", "uvicorn[standard]"]` (plotly comes via `viz`). Built frontend files under `src/suspkin/app/static/` are included in the wheel. |

**API `/api/v1`**:
- `GET /figures` returns the spec.
- Datasets: `GET /datasets`, `POST /datasets {path}` (open), `PATCH /datasets/{id}` (label, colour), `DELETE /datasets/{id}`.
- `GET /datasets/{id}/tables/{study}` returns columnar SI JSON.
- `GET /datasets/{id}/summary`.
- `GET /datasets/{id}/scene?heave&roll&pitch&rack` returns plotly 3D figure JSON at a solved pose.
- `GET /datasets/{id}/history` (optimizer runs).
- `GET /browse?dir=` is a local file picker restricted to the workspace root. The host defaults to 127.0.0.1.

## Frontend (`frontend/`, Vite + React + TS)
- **`src/api/`**: `DataSource` interface with two implementations, `HttpSource` (fetch) and `SnapshotSource` (reads `window.__SUSPKIN_SNAPSHOT__`). The types are generated from OpenAPI (`openapi-typescript`).
- **`src/state/`**: Zustand store holding open datasets, visibility, colours, the reference dataset and units mode. TanStack Query handles caching.
- **`src/components/`**:
  - `SpecFigure` renders any FigureSpec with Plotly.js: overlay of N datasets; left/front solid and right/rear dashed; linked hover; optional "Δ vs reference" mode interpolated onto the reference x.
  - `DatasetSidebar` handles open, toggle, rename, colour and reference choice.
  - `SummaryTable` shows values side by side with Δ columns.
  - `Scene3D` shows the 3D plotly figure with pose sliders that re-request the scene.
  - `OptimizationPanel` shows convergence, Pareto front and parallel coordinates from history.
- **`src/views/`**: one tab per spec group, plus Summary, 3D and Optimization tabs. Styling uses CSS variables (theme tokens) with light/dark mode. There is no domain logic in the frontend: units and factors come from the spec.
- **Build**:
  - `npm run build` outputs to `src/suspkin/app/static/`; that folder is git-ignored and built in CI and in the setup scripts when Node is present.
  - `npm run build:single` uses `vite-plugin-singlefile` to produce one inlined HTML file, used by the snapshot export.
  - If the static files are missing, `suspkin app` exits with a hint to run `npm run build`.

## Reference docs and UI consistency (user request)
- **`docs/webapp/PLAN.md`**: this plan, committed as the first commit on `feat/webapp`. Every slice updates its status table ("Status" column: todo / in progress / done + commit hash), so the plan file stays the reference.
- **`frontend/DESIGN.md`** is the frontend design system, written before any UI code. It covers:
  - colour tokens (light/dark), with dataset colours taken from the existing colour-blind-safe palette in `plotting/style.py`;
  - typography scale, spacing grid and layout regions (sidebar, tab bar, figure grid);
  - component rules for buttons, lists, tables and sliders;
  - Plotly chart template conventions: axes, grid, line styles (left solid / right dashed), hover format, legend, which match the matplotlib style.

  The tokens live in one file, `frontend/src/theme/tokens.css`, together with one Plotly template, `frontend/src/theme/plotlyTemplate.ts`. All components use only these, so a redesign means editing the tokens and the template.
- **frontend-design skill**: the slices that write UI (W3, W4) load the `frontend-design` skill first. The skill is enabled on the user's account but is not loaded in this session yet, so it needs a session reload before W3. The orchestrator invokes it and passes its guidance plus `frontend/DESIGN.md` to the coder agents in their specs. W1/W2 (Python) do not need it.

## Slices (coder = sonnet, chore = haiku; verifier gate after each)
1. **W1** (coder): `report/bundle.py`, `plotting/spec.py`, the `interactive_figure` split and the units fix. Tests:
   - bundle round-trip;
   - every spec column exists in the standard study tables;
   - spec and FIGURE_SET parity.

   **Reviewer** pass (it is the data contract).
2. **W2** (coder, after W1): `app/` backend, the `suspkin app` CLI and the extra. Tests with FastAPI `TestClient`:
   - open a YAML, a bundle and a run folder;
   - tables, summary, scene and history;
   - path restriction;
   - cache hit.
3. **W3** (coder, parallel with W2 once the W1 spec JSON is fixed): frontend scaffold, DataSource, store, `SpecFigure`, sidebar, figure tabs, summary table. Vitest for the Δ interpolation and the series mapping; ESLint + `tsc --noEmit`.
4. **W4** (coder, after W2+W3): 3D tab, Optimization tab, snapshot export (Python + single-file build), type generation from OpenAPI.
5. **W5** (chore):
   - `docs/guides/app.md` + the ADR;
   - README "Web app (experimental)";
   - CHANGELOG Unreleased;
   - a CI `frontend` job (npm ci, lint, typecheck, test, build) + the backend tests;
   - an optional Node step in the setup scripts.

   **Reviewer**: milestone review.
6. The orchestrator commits each slice on `feat/webapp` and pushes the branch. No merge to main and no tag until the user decides.

**Step 0, before any implementation (user request):**
1. `git fetch` + `git pull --ff-only` on `main` to match origin (local main is 4 commits behind).
2. Check `git status`: it must be clean and in sync with `origin/main`.
3. Review what the 4 new commits changed (`git log`/`--stat`). If they touch report/plotting/cli, adjust the slice specs.
4. Only then create `feat/webapp` from the updated main.

## Verification
- `python -m uv run pytest -q`, ruff, mypy src, `mkdocs build --strict`: all green on the branch. The golden tests stay unchanged: the core solver is untouched.
- `cd frontend && npm ci && npm run lint && npm run typecheck && npm test && npm run build`.
- `python -m uv run suspkin app configs/vehicles/fs_reference.yaml <modified copy>.yaml results/opt/demo` opens the browser. Check in the built-in browser pane:
  - every figure tab overlays the 3 datasets;
  - Δ-vs-reference works;
  - the summary diff is correct;
  - the 3D pose sliders update;
  - the optimization tab shows convergence.
  - Take screenshots.
- `suspkin app export ... --out compare.html`: open the file directly (no server). The same graphs render offline, except that 3D shows the static pose only.

## Status
| Slice | Content | Status |
|---|---|---|
| 0 | Sync main, branch `feat/webapp`, this plan | done |
| W1 | Result bundle, figure spec, `interactive_figure` split, units fix | done |
| W2 | FastAPI backend, `suspkin app` CLI, `app` extra | todo |
| W3 | Frontend scaffold, design system (`frontend/DESIGN.md`), figure tabs, summary | todo |
| W4 | 3D tab, optimization tab, snapshot export, OpenAPI types | todo |
| W5 | Docs, ADR, CI frontend job, review | todo |
