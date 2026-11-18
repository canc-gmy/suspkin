# suspkin web app: design system

This file is the single source of truth for how the app looks. Code follows it. To change the look, edit this file first, then `src/theme/tokens.css` and `src/theme/plotlyTemplate.ts`. Components never hard-code a colour, font, size or spacing.

## Subject, audience, job

- **Subject:** suspension kinematics of a Formula Student car. Every result is a *design revision*: the reference car, a variant with a moved hardpoint, an optimizer's best point.
- **Audience:** team engineers at a desk during design reviews. They read curves closely, compare small differences, and screenshot plots into reports.
- **Primary job:** overlay several revisions on the same curves and show what changed and by how much.

## Direction: the drawing office

The vernacular comes from the place suspension geometry is born: the engineering drawing.
- Light theme: **drafting film**. A cool, slightly blue-grey sheet, graphite ink, and plot grids like engineering graph paper.
- Dark theme: **blueprint**. Deep Prussian-blue sheet with pale ink. This is a real artefact of the trade, not a generic dark mode.
- **The memorable element is the title block.** The dataset sidebar is a drawing title block, a ruled table of revisions. Each opened dataset gets a **revision letter** (A, B, C ...) in open order. The letter is the compact name used everywhere: legends, hover labels, Δ columns ("B − A"), the summary table header. Rev A is the reference by default.

Everything else stays quiet and disciplined: plots carry the information, and chrome stays out of the way.

## Colour

Light (drafting film):

| Token | Hex | Use |
|---|---|---|
| `--sheet` | `#EDF1F3` | App background |
| `--sheet-raised` | `#F8FAFB` | Plot area, table body, inputs |
| `--ink` | `#1E2830` | Text, axes, rules |
| `--ink-muted` | `#5B6873` | Secondary text, tick labels |
| `--grid` | `#C9DCE4` | Plot major grid (graph-paper blue) |
| `--grid-minor` | `#E0EBF0` | Plot minor grid |
| `--rule` | `#B7C3CB` | Title-block and table rules |
| `--layout-blue` | `#23468F` | The one UI accent: active tab, focus ring, selected revision, primary button. Named after machinists' layout fluid. |
| `--danger` | `#A3341F` | Errors only |

Dark (blueprint):

| Token | Hex |
|---|---|
| `--sheet` | `#0F2A44` |
| `--sheet-raised` | `#133352` |
| `--ink` | `#E6EEF4` |
| `--ink-muted` | `#9DB3C6` |
| `--grid` | `#2A4D6E` |
| `--grid-minor` | `#1D3C5A` |
| `--rule` | `#3A5F82` |
| `--layout-blue` | `#9CC3FF` |
| `--danger` | `#FF9C85` |

**Revision colours** use the Okabe-Ito order from `spec.palette`, so they match the matplotlib report. Per theme, `tokens.css` overrides the two entries that fail contrast:
- Light: yellow `#F0E442` → `#B8A800`.
- Dark: black `#000000` → `#FFFFFF`, and blue `#0072B2` → `#4DA3E0`.

The index → hue mapping never changes between themes.

## Line semantics (from `spec.linestyles`)

- Solid = left / front, dashed = right / rear, dotted and dash-dot = third and fourth series of a panel.
- Colour = revision.
- The sidebar shows each revision's colour as a short drawn line sample, not a dot.
- A legend appears only when a panel has more than one series.

## Type

- **Barlow** for UI and body text, **Barlow Semi Condensed** for panel titles, tab labels and the title block: one family in two widths. Barlow's slightly condensed, technical-signage forms suit dense engineering labels without looking like a default sans.
- Numbers use `font-variant-numeric: tabular-nums` everywhere (tables, hover, ticks), so columns of values align. No monospace for data labels.
- Fonts are bundled through `@fontsource/barlow` and `@fontsource/barlow-semi-condensed`, so the offline snapshot renders identically.
- Scale on a 1.2 ratio from a 14 px base:
  - 12: tick labels, captions
  - 14: body, table
  - 17: panel titles
  - 20: view title
  - 24: empty-state heading only
- Weights: 400 body, 500 labels, 600 titles.
- Sentence case everywhere. No all-caps labels, no letter-spaced eyebrows above headings.

## Layout

```
+---------------------------------------------------------------------------------+
| suspkin   workspace: C:\...\veichle-dynamics             [Open files] [Export] ◐ |
+--------------------------+------------------------------------------------------+
| Title block              | Alignment  Position  Centres  Actuation  Steering    |
|--------------------------| Roll  Heave & pitch  |  Summary  3D  Optimization    |
| Rev  Design         Ref  |------------------------------------------------------|
| A  ── fs_reference   ●   | Front | Rear       Overlay | Δ to rev A              |
| B  ── raised_lca         |------------------------------------------------------|
| C  ── opt best           |  Camber [deg]          |  Toe [deg]                  |
|                          |  (plot)                |  (plot)                     |
| [+ Open files]           |------------------------+-----------------------------|
|                          |  Caster [deg]          |  KPI [deg]                  |
| Solid left/front,        |  (plot)                |  (plot)                     |
| dashed right/rear        |                                                      |
+--------------------------+------------------------------------------------------+
```

- Left-aligned throughout. Title block fixed at 288 px; it collapses to a top drawer below 900 px.
- Panels sit on a responsive grid: 2 columns at ≥1200 px, 1 below. Spacing comes from an 8 px base: 8, 16, 24, 32.
- **No cards.** A panel is its title plus its plot area (`--sheet-raised`), separated by gutters. No shadows, no gradients, and a single 2 px radius used only on inputs and buttons.
- Hover is linked within a figure: moving along x on one panel shows the same x on its siblings.
- The summary table is a ruled table:
  - one column per revision;
  - Δ columns against the reference;
  - unit in its own column;
  - null shown as "-".

  Δ cells are plain signed numbers, with no red/green colouring: a difference is not good or bad by itself.

## Components (rules)

- **Buttons:**
  - Primary uses a `--layout-blue` fill and appears once per view at most.
  - Secondary has a 1 px `--ink` outline.
  - Labels say exactly what happens: "Open files", "Export snapshot", "Close".
- **Tabs:** text only. The active tab has a 2 px `--layout-blue` underline and `--ink` text; inactive tabs use `--ink-muted`.
- **Toggles** (Front/Rear, Overlay/Δ): a segmented control with a 1 px rule and the active segment filled `--layout-blue`.
- **Focus:** a 2 px `--layout-blue` outline, offset 2 px, on every interactive element.
- **Sliders** (3D pose): a native range input styled with tokens, the value shown in tabular numbers with its unit.
- **Errors:** inline under the control that caused them, in `--danger`, saying what happened and what to do.
  - Example: "Could not open raised_lca.yaml: the rack travel is outside the steering limits. Check `steering.rack_travel` in the file."
- **Empty state:** the main area says "Open a vehicle YAML, a saved bundle or an optimizer run to start comparing", with one "Open files" button.

### Readout strip (iteration 2)

It replaces floating hover labels, which covered the curves. It lives in the figure view's sticky header, directly under the toolbar, so it stays visible while the panels scroll.

```
| Wheel travel  [ 12.4 ] mm   Pinned, Esc releases           Show solved points [ ] |
|      Camber [deg]    Toe [deg]         Caster [deg]   KPI [deg]                   |
| A ── −1.312          L −0.061 R −0.058  4.938         5.164                       |
| B ── −1.471          L  0.218 R  0.214  4.938         5.246                       |
```

- One row per visible revision, using the title-block letter and line sample. One column per panel, in panel order; the columns scroll horizontally on narrow screens while the revision column stays fixed.
- A panel with left/right (or front/rear) series shows both values in one cell, prefixed L/R (or F/R) in `--ink-muted`.
- In Δ mode, the cells show signed Δ to the reference, and the reference row reads "ref".
- Numbers are tabular, right-aligned, with the panel's decimals plus one, since the cursor is finer than the solver. Values outside a revision's x range show "-".
- **Idle state:** the strip keeps its height and says "Move over a plot to read values", so the layout never jumps.
- **Pin:**
  - Clicking a plot pins the cursor: the spike line becomes solid `--layout-blue` 1 px, and the strip shows "Pinned" with the release hint.
  - ← / → (or the x input) move a pinned cursor by the fine step: 0.1 mm for lengths, 0.01 deg for angles. Shift moves it 10×.
  - The x input accepts typed values in display units, and Enter applies them.
- Plot hover labels are off. The only cursor mark on a plot is the spike line: dashed `--ink-muted` while following the mouse, solid `--layout-blue` when pinned.

### Solved points and resolution

- Curves are drawn from a dense resample of the same natural cubic spline the readout uses, so the line and the numbers always agree.
- "Show solved points" draws 3 px dots in each revision's colour, only at real solver points.
- The **Resolution** segmented control (31 / 61 / 121 points) sits in the top strip next to Export.
  - It re-solves revisions opened from vehicle YAML files.
  - Revisions from bundles or runs show their stored resolution as text ("31 points, stored") and are not re-solved.

### 3D view (iteration 2)

- The canvas fills all the space below the 3D toolbar: full width of the main area, height to the bottom of the viewport. No inner card or padding. The Plotly paper, the scene background, the axis walls and the container below the toolbar are all `--sheet-raised` (the plot surface), so the canvas is one surface edge to edge with no visible box or seam, in both themes. The legend sits inside the canvas at the top left, so it is never clipped at any width.
- The toolbar holds the revision picker, pose sliders, "Reset pose", "Reset view" and "Show reference ghost".
- The camera belongs to the user. Pose changes never move the camera, zoom or pan, and the axis ranges and aspect ratio stay fixed from the static pose. Only "Reset view" restores the default camera.
- The reference ghost is drawn in 1 px `--ink-muted` lines at 40% opacity. Hardpoints that differ from the reference by more than 0.1 mm get a 6 px ring in `--layout-blue`.
- The rings have their own checkbox, "Highlight changes", next to "Show reference ghost". It is on by default and disabled (with a "Same as the reference" title) while the selected revision is the reference. The ring sits on the posed position of the point, so it follows the pose sliders. It uses the same 0.1 mm threshold as the Hardpoints tab.

### Hardpoints tab (iteration 2)

A tab after Summary. It answers "what did I move, and by how much" for every hardpoint of every visible revision.

```
| Hardpoints                                                                       |
| Show [ All | Differing only ]   Filter points [ uca_       ]   12 of 34 rows      |
|-------------------------------------------------------------------------------------|
|                          |  A  raised_lca         |  B  opt best          | Δ B − A  |
| Point         Side       |   x      y      z      |   x      y      z     |  |Δ| mm  |
|--------------------------+------------------------+-----------------------+----------|
| Front axle                                                                          |
| lca_outer     left       |  0.0   580.0  130.0    |  0.0   580.0  142.0   |  12.00   |
|                          |                        |                       | x +0.00 y +0.00 z +12.00 |
| lca_outer     right      | ...                                                    |
| tie_inner     left       |  80.0  150.0  140.0    | ...                   |   0.00   |
| Rear axle                                                                           |
```

- One ruled table in the `--sheet-raised` body, same rules as the summary table: a 2 px `--ink` rule under the header, 1 px `--rule` between rows, left-aligned text, right-aligned tabular numbers.
- **Rows:** one per (axle, point, side). Axles are group rows ("Front axle", "Rear axle") in `--font-title`. Inside an axle, rows that differ from the reference by more than 0.1 mm in any revision come first, then the rest, each part in alphabetical order (point name, left before right). A right point that comes from `right_overrides` shows "override" in `--ink-muted` 12 px under "right".
- **Columns:** point, side, then one group of x, y, z per visible revision with the revision letter and label in the group heading (the same stacked heading as the summary table). Values are in mm with 1 decimal. After the revisions come the Δ columns, one per visible revision other than the reference, headed "Δ B − A".
- **Δ cell:** the vector magnitude |Δ| in mm (2 decimals) on the first line, and the per-axis Δ ("x +0.00 y −2.00 z +0.00", signed, 2 decimals) on a second line in `--ink-muted` 12 px. No colouring: a difference is not good or bad by itself. A reference-only column reads "ref".
- **Filter bar:** a segmented control "All / Differing only", a text input "Filter points" (case-insensitive substring of the point name), and a muted count "12 of 34 rows". A filter that hides everything says "No hardpoints match. Clear the filter or choose All."
- A revision whose hardpoints cannot be read shows "-" in all its cells. The tab scrolls horizontally on narrow screens; the point and side columns stay fixed.

### Front / Rear / Both and panel data actions (iteration 2)

**Front / Rear / Both**

```
>= 1400 px, Both:                              < 1400 px, Both:
| Front                | Rear                 |  | Front                              |
| readout strip        | readout strip        |  | readout strip, panels (2 columns)  |
| panel                | panel                |  | Rear                               |
| panel                | panel                |  | readout strip, panels (2 columns)  |
```

- The axle control is a three-segment toggle: Front, Rear, Both. Both is only offered for groups that have separate front and rear figures.
- Both shows the two figures side by side from 1400 px, each with a single column of panels, and stacked below that, each with its usual panel grid. Each figure has its title ("Front", "Rear") above its strip.
- **One readout strip per figure, not one with F/R cells.** The strip belongs to its figure: the cursor, the pin and the x input act on that figure's panels, and the cells sit directly above the plots they describe. A shared strip would need F and R cells in every column, which doubles its width, and would hide which axle a pinned x belongs to. Each strip is sticky only inside its own figure, so stacked strips replace each other while scrolling.

**Panel data actions**

```
| Camber [deg]                                          Copy data  Download CSV  Copied |
```

- Right-aligned in the panel header, on the title's baseline: two text buttons "Copy data" and "Download CSV", `--ink-muted` 12 px, no border, underlined on hover, with the usual focus ring. A muted "Copied" appears to their left for 1.5 s in a polite live region, so the buttons never move.
- The data is the solved points (never the dense spline) of every visible revision, in display units. The first column is the study x, "Wheel travel [mm]". Then one column per revision and series: "A camber [deg]". A panel with its own x axis adds an "A <x name> [unit]" column per revision. A revision that has no point at a row's x leaves the cell empty.
- In Δ mode the rows are the reference's solved x, and the columns are "B-A camber [deg]" for every visible revision except the reference.
- Copy puts tab-separated text on the clipboard so it pastes into a spreadsheet; the download is a CSV named `<figure>_<panel>.csv` (UTF-8 with a byte-order mark so Excel reads the Δ and unit text). Both work in snapshots because they use only the loaded tables.
- If the clipboard is not available, a hidden-textarea fallback is tried; if that fails too, the status reads "Could not copy, use Download CSV" in `--danger`.

## Plot template (`plotlyTemplate.ts`)

- Background `--sheet-raised` and paper `--sheet`.
- Major grid `--grid` 1 px, minor grid `--grid-minor` 1 px dotted. Axis lines `--ink` 1 px; no zero line except where x = 0 is the static pose (drawn in `--ink-muted`, 1 px).
- Tick font Barlow 12 `--ink-muted` in tabular figures. Axis titles Barlow 14 `--ink` in the form "Camber [deg]".
- Line width 1.75 px. The static-pose marker is a single small ring at x = 0 on the reference revision only.
- No hover labels on curve plots (iteration 2): values are read in the readout strip. The spike line follows the cursor; it is linked across all panels of a figure.
- The modebar shows only download-PNG and reset-axes. Downloaded PNGs use the light theme at 2× scale, for reports.

## Motion

- No ambient or entrance animation.
- Motion only answers an action: the title-block drawer opening (150 ms), and a newly opened revision row highlighting once (`--layout-blue` at 12% opacity fading out over 600 ms).
- `prefers-reduced-motion` disables both.

## Copy

- Plain verbs, sentence case, no filler.
- Name things the engineer's way: "revision", "reference", "front corner", "bump travel".
- Never use internal names like "bundle schema" or "dataset id".
- An action keeps its name through the flow: "Export snapshot" leads to "Snapshot exported to compare.html".

## Review against the brief (generic-default check)

- **Rejected:** a dark dashboard with a single neon accent, SaaS cards with shadows, Inter/system sans with mono data labels, all-caps eyebrows. These read the same for any analytics tool.
- **Chosen:** drafting film and blueprint themes, the title-block sidebar with revision letters, graph-paper plot grids, and one condensed technical family. These come from engineering drawing practice and make "compare revisions" the visible organising idea.
