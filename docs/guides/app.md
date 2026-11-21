# Web app (experimental)

`suspkin app` starts a local web app in your browser. It shows every study figure of a design
interactively, overlays several designs on the same curves, shows differences against a
reference design, lets you edit hardpoints and re-solve in seconds, shows the 3D suspension at
any body pose, follows an optimisation run, and exports a snapshot you can mail to a teammate.

It is **experimental**: the HTTP API and the UI may change in minor releases
([Stability](stability.md)). For how it works inside (server, endpoints, caching, figure spec)
see [Web app: backend, API and frontend contract](../concepts/web-app.md).

## 1. Install and build

```bash
uv sync --extra app --extra viz        # fastapi, uvicorn; plotly for the 3D tab
```

The `viz` extra is needed for the 3D tab (without it the scene endpoint answers 501).
A release wheel or sdist (from the GitHub release page) already contains the built browser
UI: install it with the `app` extra and `suspkin app` works with no Node installed.

From a source checkout, the browser UI is a separate Node project (Node 22.12 or newer) that
you build once:

```bash
cd frontend
npm ci
npm run build           # the app served by `suspkin app`
npm run build:single    # the single-file build used by snapshot export
```

Both outputs are git-ignored (`src/suspkin/app/static` and `src/suspkin/app/static_single`).
Without the first, the server runs but the root page only shows a hint and the API works;
without the second, snapshot export fails with `single-file frontend not built`.
`scripts/setup.sh` and `scripts/setup.ps1` do all this for you
([Install](../getting-started/install.md)).

## 2. Start

```bash
uv run suspkin app configs/vehicles/fs_reference.yaml
```

```text
INFO:     Started server process [37272]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8799 (Press CTRL+C to quit)
```

(port 8799 here was chosen with `--port 8799`; the default is 8765.) A browser tab opens. Stop
the server with Ctrl+C.

| Option | Default | Meaning |
|---|---|---|
| `PATHS...` | none | Things to open at start. Optional: you can also open them in the UI |
| `--port` | 8765 | Port to listen on |
| `--host` | `127.0.0.1` | Interface to bind; anything else prints a warning (see Security) |
| `--workspace` | current directory | Root folder the app may read and write; must exist (exit code 2 otherwise) |
| `--no-browser` | off | Do not open the browser |
| `--dev` | off | API only; the UI comes from the Vite dev server (section 8) |

If the `app` extra is missing, the command exits with code 2 and an install hint.

## 3. The workspace

The **workspace** is the folder you started the app in, or `--workspace DIR`. The app reads and
writes only inside it. Every path the UI sends is checked: `..` that escapes, absolute paths on
another drive, symlinks that point out and network (UNC) paths are refused with an error.
Paths shown in the UI are workspace-relative. The header shows `workspace: <path>`.

Start the app from the repository root to find `configs/` and your result folders; use
`--workspace` to point at a different project folder.

## 4. Open files

Click **Open files** (top bar, or the button on the empty screen). A dialog lists the
workspace folders; tick one or more entries and press **Open N files**. The kind of each entry
is detected from the path:

| You open | What happens |
|---|---|
| a **vehicle YAML** | the standard studies (front and rear corner sweeps, rack, heave, roll, pitch) are solved live. Solving takes about a second; results are cached by file content, so re-opening is instant |
| a **study YAML** | the vehicle it points to is opened, and the study is attached so the 3D tab can draw its design domains and constraint regions |
| a **result bundle** (a folder with `manifest.json`) | its stored tables are loaded, nothing is solved |
| a **report folder** (it has `bundles/`) | every design of the report opens as one revision |
| an **optimiser run folder** (`history.csv` plus `meta.json`/`result.json`) | up to three revisions: `<run> baseline`, `<run> best` and, in `pareto` mode, `<run> knee`; the run history feeds the Optimization tab |

You can pass the same paths on the command line. Opening the same thing twice returns the
existing revision.

Each opened design is a **revision** with a letter (A, B, C ...) in the order opened. The letter
is the short name in legends, hover text and table headers.

## 5. The window

```text
+--------------------------------------------------------------------------------+
| suspkin  workspace: ...   [Revisions] [Open files] Resolution 31|61|121 [Export]  |
+-------------------+------------------------------------------------------------+
| Title block       |  tabs: Alignment | Position | Centres | Actuation | Steering |
|  Rev  Design  Ref |        Roll | Heave & pitch | Summary | Hardpoints | 3D |      |
|  A  fs_reference (o)|       Optimization                                          |
|  B  tie_down5   ( )|  [Front|Rear|Both]            [Overlay | Delta to rev A]      |
|  [Open files]     |  figure panels ...                                          |
+-------------------+------------------------------------------------------------+
```

### Title block (revisions)

The ruled table at the left lists the open revisions: letter plus a line sample in the
revision's colour, the design name and a **Ref** radio button.

- Click a row (or its letter) to **hide or show** that revision in all plots.
- **Rename** edits the label (Enter to accept, Escape to cancel); **Close** removes the
  revision.
- The **Ref** radio chooses the **reference** revision (A by default).
- Line style repeats the colour code: **solid is left/front, dashed is right/rear**.

On a narrow window the title block is behind the **Revisions** button.

### Resolution

The **Resolution** control (31, 61 or 121) sets the number of points per sweep for revisions
that come from vehicle files and are solved live. More points give smoother curves and better
finite-difference gradients, but each change re-solves every live revision. Bundles and run
revisions keep their stored points and ignore the control.

### Light and dark theme

The round button at the right of the top bar switches between a light drafting-film theme and a
dark blueprint theme.

## 6. Figure tabs, Summary, Hardpoints, 3D, Optimization

### Figure tabs

Alignment, Position, Centres, Actuation, Steering, Roll and Heave & pitch. These are the
standard study figures of the [report](../concepts/sweeps-and-reports.md), drawn by the
browser from a declarative spec. Controls above the panels:

- **Axle**: Front, Rear or Both (for groups that have both).
- **Comparison**: **Overlay** draws all visible revisions on the same axes. **Delta to rev X**
  draws every visible revision as the **difference to the reference**, interpolated onto the
  reference's x values, so you can see exactly where two designs differ.
- Hover for values; every panel has **Copy data** and **Download CSV** (the solved points of
  the panel, in display units; the CSV starts with a byte-order mark so spreadsheets read the
  Δ sign).

### Summary

The vehicle summary of every visible revision side by side (the same numbers as
`suspkin vehicle`, in display units), plus a **Δ** column against the reference for each other
revision.

### Hardpoints (edit and compare)

A table of every hardpoint, in mm, front axle then rear axle, left and right rows, with one
x/y/z column group per visible revision and a **|Δ|** column (distance to the reference point,
with the per-axis deltas under it). Controls:

- **Rows**: All or Differing only. **Filter points**: type part of a name (`uca_outer`).
- To edit: choose a revision in **Edit revision** and press **Edit as new revision**. Its
  column becomes editable number fields (0.1 mm steps). Changed cells are highlighted. A
  **left** point also moves its right twin (the mirror image); tick **Right side only** on the
  right row to give the right side its own value (a right-side override); **Reset point**
  undoes one point.
- Press **Solve** (or tick **Auto-solve** to re-solve after every edit). The result is a **new
  revision** (kind `edited`) that opens next to the original, so you can compare immediately in
  every tab. A manual **Solve** ends the edit session; with **Auto-solve** the session stays open and
  each re-solve replaces the previous automatic revision. **Discard edits** abandons the edit session.
- If the new geometry cannot be solved, a message names the sweep that failed
  (for example the roll sweep): move the point back a little.
- **Save as YAML** writes the edited revision as a vehicle file in the workspace (enter a file
  name ending in `.yaml`). The folder must exist, and an existing file is only replaced if it
  is already a loadable vehicle YAML and you confirm.

Edits are meant for quick what-if questions. For a permanent design change, edit the YAML (or
save the revision) and keep the file under version control.

### 3D

The suspension at a solved pose. Pick the **Revision**, then move the sliders: **heave** (mm),
**roll** (deg), **pitch** (deg) and **rack** (mm). The server solves the pose and returns the
scene. The slider ranges follow the study parameters of the revision (by default heave
+-25 mm, roll +-2 deg, pitch +-1.5 deg, rack +-30 mm); the server accepts up to +-200 mm and
+-0.35 rad (about 20 deg) if you call the API directly. If a pose cannot be solved you get a
message and **Reset pose** brings you back to zero. Other controls:

- **Overlays**: roll centres and roll axis, front-view and side-view instant centres, pitch
  centre; and, when a study is attached, design domains, requirement regions (inside) and
  keep-out regions (outside).
- **Plane**: none, XY (horizontal), XZ (side) or YZ (front), with an offset slider.
- **Show reference ghost** draws the reference revision's links in a faint colour; **Highlight
  changes** marks the points that differ from the reference. **Legend**, **Reset view**.

The 3D scene is a rigid-body view: the car body stays fixed and the ground tilts
([Vehicle kinematics](../concepts/vehicle-kinematics.md)).

### Optimization

For an opened optimiser run: pick the **Run**, then see the convergence, the Pareto front (in
`pareto` mode), the design variables (their spread over the evaluations) and the constraint
margins. The data come from the run's `history.csv`; see
[Optimisation](../concepts/optimization.md).

## 7. Compare designs: a walk-through

1. Start: `uv run suspkin app configs/vehicles/fs_reference.yaml`. Revision A opens.
2. Open the **Hardpoints** tab, choose A and press **Edit as new revision**.
3. Change `tie_inner` z of the front left row from 154.0 to 149.0 mm. The right twin follows.
   Press **Solve**. Revision B appears, named `fs_reference edited` (the parent's name plus `edited`; with auto-solve, each re-solve replaces the previous automatic revision and keeps its letter).
4. Open **Alignment**, set **Comparison** to **Delta to rev A**: the toe curve of B shows
   the bump steer you added (about 0.25 deg over 30 mm bump, as in the
   [tutorial](../getting-started/tutorial.md)).
5. Open **Summary**: the Δ column shows every vehicle number that moved.
6. **Save as YAML** to keep B, or **Export snapshot** (next section).

## 8. Export a snapshot

A snapshot is **one self-contained HTML file** with all data embedded. It opens by double
click, with no server and no install, and is ideal for sharing a comparison. Its limits: it is
read-only (no opening, editing, resolution change or export), and the 3D tab shows the static
pose only.

From the UI, press **Export snapshot**: it writes `suspkin_snapshot_YYYYmmdd_HHMMSS.html` into
the workspace with the revisions that are open, and the status line shows the path. If a file
of that name already exists you are asked before replacing it.

From the command line:

```bash
uv run suspkin app-export PATHS... --out compare.html [--workspace DIR] [--no-scene]
```

```text
Snapshot exported to .../compare.html (7.8 MB)
```

That is the output of `suspkin app-export ref.yaml tie_down5.yaml opt_demo --out compare.html
--workspace .` for two vehicle files and an optimiser run folder. `--no-scene` leaves out the
3D scenes: 6.6 MB for two vehicles without scenes. Most of the size is the single-file build of
the frontend (about 5.6 MB); the data is small. The paths must be inside the workspace
(otherwise: `error: path is outside the workspace`, exit code 1). Export needs
`npm run build:single`.

## 9. Security

The app reads and writes only inside the workspace. The server binds to `127.0.0.1` by
default and has **no authentication**: with `--host 0.0.0.0` (or any other interface) everyone
who can reach the port can read the workspace and write snapshots and vehicle YAMLs into it. Do
that only on a network you trust.

## 10. Frontend development

```bash
uv run suspkin app --dev           # API only, port 8765
cd frontend && npm run dev         # Vite dev server with hot reload (port 5173)
npm run gen:api                    # regenerate the TypeScript API types after an API change
npm run lint
npm run typecheck
npm test
```

The design system is described in [frontend/DESIGN.md](https://github.com/canc-gmy/suspkin/blob/main/frontend/DESIGN.md)
(single source of truth). Change it first, then `frontend/src/theme/tokens.css` (colours, fonts,
spacing) and `frontend/src/theme/plotlyTemplate.ts` (plot styling). Components never hard-code a
colour, font or size.

### Add a figure

A new graph needs no frontend code. Add the panel to the definitions in
`suspkin.plotting.figures`; the spec in `suspkin.plotting.spec` is built from those same panels
and the app renders it generically. Only add a column to the study table if it does not exist
yet.

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Root page shows a hint instead of the app | Frontend not built: `cd frontend && npm ci && npm run build` |
| `Export snapshot` fails / `single-file frontend not built` | `npm run build:single` |
| `suspkin app` exits with an install hint | `uv sync --extra app` |
| 3D tab shows an error, scene returns 501 | `uv sync --extra viz` |
| `cannot solve <file>` when opening | The vehicle's linkage cannot assemble over a standard sweep; run `suspkin summary FILE` to see the error |
| `... is not a vehicle: ...` | The YAML is not a valid vehicle or study; see [Vehicle YAML](vehicle-yaml.md) |
| Vehicle file larger than 2 MB | Refused (413) |
| `path is outside the workspace` | Start the app from a parent folder or pass `--workspace` |
| Port already in use | Pick another with `--port` |
| Two browser tabs interfere | They share one server state: closing a revision in one closes it in both |
