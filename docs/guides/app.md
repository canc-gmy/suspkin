# Web app (experimental)

`suspkin app` starts a local web app that shows every study figure interactively and overlays
several results on the same curves. It lives on the branch `feat/webapp` and is not part of the
stable surface ([Stability and versioning](stability.md)); the API and the UI may change.

## Install

```bash
uv sync --extra app --extra viz
```

Node 20+ is needed only to build the frontend (the Python side runs without it, but then the
server has no UI to serve):

```bash
cd frontend && npm ci && npm run build && npm run build:single
```

`npm run build` writes the app served by `suspkin app`; `npm run build:single` writes the
single-file build used by the snapshot export. Both outputs are git-ignored.

## Start

```bash
uv run suspkin app PATHS... [--port 8765] [--host 127.0.0.1] [--workspace DIR] [--no-browser] [--dev]
```

| Option | Meaning |
|---|---|
| `PATHS...` | Results to open at start (optional; you can also open them in the UI) |
| `--port` | Port to listen on (default 8765) |
| `--host` | Interface to bind (default `127.0.0.1`) |
| `--workspace` | Root folder the app may read (default: current directory) |
| `--no-browser` | Do not open the browser |
| `--dev` | API only; the UI comes from the Vite dev server (see below) |

## What can be opened

The kind is detected from the path:

- a **vehicle YAML**: the standard studies are solved live and cached by file hash;
- a **result bundle** (a folder with `manifest.json`), or a **report folder** (its `bundles/`);
- an **optimizer run folder**: the baseline and the best design (and the Pareto knee) become
  datasets, and the run history feeds the optimization tab.

Each opened result is a **revision** with a letter (A, B, C ...) in open order. The letter is the
short name in legends, hover labels and the summary header. Revisions can be renamed, hidden and
closed. One revision is the **reference** (A by default; choose it in the title block). With
**Δ vs reference** on, every curve is shown as the difference to the reference, interpolated onto
the reference x values, and the summary table shows differences.

## Tabs

- **Figure tabs**: one per figure group (alignment, position, centres, actuation, steering, roll,
  heave and pitch). All revisions are overlaid; left/front is solid, right/rear dashed.
- **Summary**: the vehicle summary of every revision side by side, with differences.
- **3D**: the suspension at a solved pose. Sliders set heave, roll, pitch and rack (mm and deg);
  the server solves the pose and returns the scene.
- **Optimization**: convergence and parameters of an opened optimizer run.

## Snapshot export

A snapshot is one self-contained HTML file with the data embedded; it opens by double click, with
no server. The 3D tab shows the static pose only.

```bash
uv run suspkin app-export PATHS... --out compare.html [--workspace DIR] [--no-scene]
```

`--no-scene` leaves out the 3D scenes (smaller file). In the app, the **Export snapshot** button
writes a file into the workspace for the revisions that are open. Export needs the single-file
build (`npm run build:single`).

## Security

The app reads only inside the workspace (`--workspace`, default the current directory); paths
outside it are refused. The server binds to `127.0.0.1` by default and has no authentication, so
only use `--host` with another interface on a network you trust.

## Frontend development

```bash
uv run suspkin app --dev        # API only
cd frontend && npm run dev      # Vite dev server with hot reload
npm run gen:api                 # regenerate the TypeScript API types after an API change
```

Checks: `npm run lint`, `npm run typecheck`, `npm test`.

## Change the look

The design system is described in
[frontend/DESIGN.md](../../frontend/DESIGN.md) (single source of truth). Edit it first, then
`frontend/src/theme/tokens.css` (colours, fonts, spacing) and
`frontend/src/theme/plotlyTemplate.ts` (plot styling). Components never hard-code a colour, font
or size.

## Add a figure

A new graph needs no frontend code. Add the panel to the definitions in
`src/suspkin/plotting/figures.py`; the spec in `src/suspkin/plotting/spec.py` is built from
those same panels, and the app renders it generically. Only add a column to the study table
if it does not exist yet.
