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
| `--region-domain` | `#2E7D32` | 3D view: variable domains |
| `--region-required` | `#00838F` | 3D view: `inside` regions |
| `--region-keepout` | `#C2185B` | 3D view: `outside` regions |

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
| `--region-domain` | `#8BD17C` |
| `--region-required` | `#5CC8D6` |
| `--region-keepout` | `#FF7FA8` |

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
- **Names and labels:** every input, checkbox, select and icon button has an accessible name (a `label` element or `aria-label`), never just its value or placeholder. A visible label never wraps inside its own control (`white-space: nowrap`); a toolbar wraps between controls. On a phone the top strip drops the "Resolution" caption (the control keeps its name) so it takes two rows, not four.
- **Prose** (notes, definitions, empty states) is at most 80 characters wide, whatever the screen.
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

- The canvas fills all the space below the 3D toolbar: full width of the main area, height to the bottom of the viewport. No inner card or padding. The Plotly paper, the scene background and the container below the toolbar are all `--sheet-raised` (the plot surface), so the canvas is one surface edge to edge with no visible box or seam, in both themes. The legend is a compact one-row-per-line strip at the top of the canvas, above the scene (it reserves its own 48 px, so it never covers the car); it is never clipped at any width.
- The toolbar has two rows. Row 1: view title, revision picker, "Reset pose" and "Reset view". Row 2: "Overlays", "Legend", "Plane" (with its offset slider once a plane is chosen), "Notes (n)" (only when there are notes), "Show reference ghost" and "Highlight changes". The pose sliders follow, two per row on a phone.
- **Legend:** the "Legend" toggle button (`aria-pressed`) shows or hides the strip. It starts open from 900 px width and closed below, so a phone keeps its canvas. Legend type is 12 px, `--ink-muted`.
- **Default framing:** the first camera is computed from the scene box and the measured canvas (`fitCamera`): same viewing direction as before, with the distance chosen so the car fills about 80 % of the limiting canvas side, perspective included. "Reset view" returns to this framing for the current canvas size.
- The camera belongs to the user. Pose, plane and overlay changes never move the camera, zoom or pan, and the axis ranges and aspect ratio stay fixed from the static pose. Only "Reset view" restores the default camera.
- The reference ghost is drawn in 1 px `--ink-muted` lines at 40% opacity. Hardpoints that differ from the reference by more than 0.1 mm get a 6 px ring in `--layout-blue`.
- The rings have their own checkbox, "Highlight changes", next to "Show reference ghost". It is on by default and disabled (with a "Same as the reference" title) while the selected revision is the reference. The ring sits on the posed position of the point, so it follows the pose sliders. It uses the same 0.1 mm threshold as the Hardpoints tab.

#### Ground grid and reference plane

- **Grid:** the Plotly axes are hidden (no walls, box, ticks or titles). The ground reference is a grid drawn at z = 0 over a world box 3 times the car's longest horizontal side, centred on the car, so it reads as an unbounded plane. Minor lines (`--grid-minor`) every major step / 5, major lines (`--grid`) every 0.1, 0.2, 0.5, 1 ... m (about 10 across the box), 1 px. Lines fade towards the edges in four opacity steps (100, 70, 40, 15 %), starting at the car's edge.
- **Labels:** major values in metres with 1 decimal (`--ink-muted`, Barlow 12 px, no unit on each tick), on one row below the car (the x values) and one column at its right (the y values), both 20 % of the car's size outside its box, so they never sit on the points. Only ticks within 40 % of the car's size beyond its box are labelled (what the default framing shows); the grid lines themselves still run the whole world box. Each row has a quiet `--ink-muted` 12 px title ("x [m]", "y [m]") placed 9 % further out, at the middle of the car, so every visible row names its axis and nothing falls off canvas.
- **Framing:** the default camera frames the car box only (about 80 %, as above), not the grid. The grid runs off the canvas on purpose.
- **Plane:** the "Plane" select in toolbar row 2 offers None (default), XY (horizontal), XZ (side) and YZ (front). A plane shows an offset slider (mm, tabular value with unit like the pose sliders, 1 mm step, range = car box along the normal plus 10 % per side). Defaults: XY at z = 0 (ground), XZ at y = 0 (centreline), YZ at x = 0 (front axle); each orientation remembers its offset.
- The plane is a translucent surface (12 % opacity) with a 2 px outline, both `--layout-blue` (no new token), covering the car box plus 15 % per side, and a 14 px `--ink` label ("z = 120 mm", a scene annotation, not a text trace) at the plane corner nearest the default camera (high x, low y, high z), pulled in to half the plane margin outside the car box so it stays on canvas and clear of the tick labels (the XY plane's label sits at the middle of its near edge, since its near corner is the lowest point on screen), readable in both themes. It lives in the body frame, like the hardpoints, so it follows the chassis. Choice and offsets are kept for the session in localStorage; it works in snapshots. The select and the slider are native controls with `aria-label`s ("Plane", "Plane offset [mm]"), so arrows and Home/End work.

#### Overlays menu and rotation centres

An "Overlays" button in the 3D toolbar (after "Reset view") opens a small panel under it with two groups: "Rotation centres" (four checkboxes, below) and "Study regions" (three checkboxes, see Constraint regions). All are off by default. The button reads "Overlays (3)" with the number of ticked boxes. The button is a secondary button with `aria-expanded`; the panel is a 1 px `--rule` box on `--sheet-raised`, opens on Enter or Space, closes on Esc (focus returns to the button) and on a click outside. The ticked boxes are kept for the session in localStorage (centres and regions under separate keys).

| Checkbox | Draws | Marker, colour |
|---|---|---|
| Roll centres and roll axis | front and rear roll centre at the pose, 2 px solid line through them | diamond, `--layout-blue` |
| Front-view instant centres | per corner, the IC in the plane through the contact patch, with a 1 px dashed line from the patch | x, `--ink` |
| Side-view instant centres | the same in the side-view plane | cross, `--ink-muted` |
| Pitch centre | the vehicle pitch centre on the centreline | open diamond, `--ink` |

- Positions are in the plotted (body) frame, so the contact patches coincide with the scene's contact markers. The roll centre is the intersection of the two patch-to-IC lines; the pitch centre is the intersection of the front and rear side-view lines.
- A centre at infinity (parallel wishbones) is not drawn and not listed. Nothing here changes the axis ranges, the aspect or the camera: those stay fixed from the static pose.
- Lines are clipped to the scene box. An IC outside the box is shown by its clipped line ending in a small open square at the box edge, and by a note in the "Notes (n)" panel (a secondary button that opens a small floating panel like Overlays, so the notes never push the canvas down), `--ink-muted`, one line per IC, sentence case, no middle dots: "FL front-view IC 4.21 m outboard", "RL side-view IC 3.10 m rearward". The distance is the swing-arm length (patch to IC). A roll or pitch centre outside the box is not drawn and gets a note too ("Front roll centre outside the view").
- Each ticked kind has one legend entry named like the checkbox ("Roll centres", "Front-view ICs", "Side-view ICs", "Pitch centre"). Colours are the existing tokens only; the revision colours keep their meaning.
- The wishbone-plane construction lines are not drawn: the 3D wishbones already show them.

#### Constraint regions

The second group of the Overlays menu, "Study regions", draws the variable domains and constraint regions of an optimization study. The study whose regions are drawn is chosen independently of the selected revision, so regions can be shown on any revision. Default: the selected revision's own study, else the first open revision that has one (a revision of an optimizer run, or a study YAML opened from the file picker). A "Study" select inside the group lists every open revision with a study (letter and label); the muted line under it reads "Shown as the study defines them, relative to its baseline car." (offset-type design domains are relative to the study's baseline point; constraint regions are absolute). The regions are drawn in the selected revision's scene, which is already in the body frame at the current pose. When no open revision has a study the three checkboxes are disabled (`aria-describedby` the hint) and a muted one-line hint reads "Open a study YAML or an optimizer run to show its regions." followed by an inline "Open files" text button that opens the Open dialog.

| Checkbox | Draws | Colour token, line |
|---|---|---|
| Design domains | the domain of each variable (symmetric variables also as a right-side copy at 50% opacity) | `--region-domain`, solid |
| Requirements (inside) | `inside` constraint regions | `--region-required`, dotted |
| Keep-outs (outside) | `outside` constraint regions | `--region-keepout`, dashed |

- Wireframes only, 1 px: box and hexahedron edges, three great circles for a sphere, two end circles and four rails for a cylinder. Geometry comes from the same helpers as the static study figure.
- Tokens: light `#2E7D32`, `#00838F`, `#C2185B`; dark `#8BD17C`, `#5CC8D6`, `#FF7FA8`. They sit away from the Okabe-Ito revision hues and from `--danger` (errors only), and the line style repeats the role, so colour is never the only cue.
- Regions are fixed in the body frame, like the chassis points: they stay put while the wheels move under the pose sliders. A constraint checked `over` a sweep (a moving point) is still drawn in the body frame; the API row carries the note "Checked over sweep ...".
- Same rule as the centres: lines are clipped to the scene box, so the axis ranges and the camera never change. Legend: one entry per ticked role, named like the checkbox.
- Constraint regions are in vehicle-frame coordinates and are not mirrored; only symmetric variable domains get a right copy.

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

### Hardpoint editing (iteration 2)

Editing lives in the Hardpoints tab, because the table already shows every point in mm. An edit never changes a revision: it makes a new one.

```
| Edit revision [ B  opt best v ]  [ Edit as new revision ]        Save as YAML           |
|-------------------------------------------------------------------------------------------|
| Editing B opt best. A left point also moves its right twin unless "Right side only" is set. |
| [ Solve ]  [ Discard edits ]   [x] Auto-solve    2 points changed    Solving…             |
|-------------------------------------------------------------------------------------------|
| Point        Side            | B  opt best                  | Δ B − A |                   |
|                              |  x       y        z          |         |                   |
| lca_outer    left            | [ 0.0 ] [ 580.0 ] |[ 142.0 ] |  12.00  |  Reset point      |
| lca_outer    right           | [ 0.0 ] [-580.0 ] |[ 142.0 ] |  12.00  |  Reset point      |
|              [ ] Right side only                              |         |                   |
| Could not solve: invalid geometry after editing front.lca_outer: ... Change the marked      |
| values and solve again.                                                                     |
```

- **Enter and leave:** "Edit as new revision" sits in the table toolbar next to a revision picker. It works on any revision that has a vehicle: files, bundles, runs and earlier edits. While editing, the button is replaced by "Discard edits". In a snapshot all editing buttons are disabled with the title "Read-only snapshot".
- **Cells:** in the edited revision's x, y, z columns the numbers become inputs, in mm, step 0.1, one decimal, tabular, right-aligned. The other revisions stay plain numbers, so the edit can be read against them.
- **Changed cells** carry a 2 px `--layout-blue` rule on their left edge, and nothing else (no fill, no bold). A value typed back to the original is no longer an edit.
- **Mirroring:** a right row follows its left row, shown as plain muted numbers, until "Right side only" is ticked on it. With it ticked the right row's inputs are live (right-side coordinates, y is negative) and the left row stops moving it. A point that already has a right override starts ticked. Unticking it re-mirrors the right point from the left one.
- **Reset point** is a text button at the end of a row that has edits. It clears both rows of that point.
- **Solve** is the one primary button of the view. It is disabled with nothing to solve and while a solve runs. On success the edits become a new revision in the title block ("B edited", next free letter), the edit mode closes and the status says "Created B edited".
- **Auto-solve** is a checkbox. With it on, the view solves 600 ms after the last edit and shows "Solving…" in place of the changed count. An answer to an older edit is dropped. Auto-solve keeps one revision: before a new solve is registered the previous auto-solved one is closed, so the revision keeps its label and letter ("B edited", never "B edited (2)"). An answer to an older edit is closed, not kept. Switching Auto-solve off keeps the last result as an ordinary revision.
- **Errors** are inline under the table, in `--danger`, and name the point and the reason: "Could not solve: invalid geometry after editing front.lca_outer: ... Change the marked values and solve again." A solver failure names the sweep. The inputs and edits stay as they were.
- **Save as YAML** is a text button next to the revision picker; it is available for edited revisions. It opens one line: a file name input (default: the revision label as a file name, ending in .yaml) with "Save" and "Cancel". If the file exists the app asks "Replace <name>?" and only then overwrites. Success reads "Saved to <path>" in the status line. Files are written inside the workspace only.

### Optimization statistics (iteration 2)

The Optimization tab has a segmented control "History / Statistics" under its title. History is the existing charts; Statistics answers "how did all the tested samples score, and where do good designs sit". It reads the history table only, so it works in snapshots.

- **Definitions** (stated once under the controls, in `--ink-muted` 12 px): a failed evaluation is one whose status is not "ok" or whose f_total is the 1e3 failure penalty; it is counted but never enters a mean, a quantile or an axis range. Feasible means every enabled g is 0 or below. "Top 10 %" is the best 10 % of the distinct feasible designs by f_total (see below), at least 5, and is "-" when fewer than 5 are distinct. Standard deviations are sample deviations (n - 1) and are "-" for fewer than 2 samples. f is the normalised objective as stored. In pareto mode f_total is still the stored sum of the objectives; the optimizer ranks by the objective vector, and the note says so.
- **Samples filter:** when the run has more than one phase (the `source` column, for example `de` then `slsqp`), a "Samples" segmented control sits above the definitions: "All phases" and one segment per phase ("DE only", "SLSQP only"), plus a muted "88 of 128 evaluations". It filters the whole view, and the best-so-far line is recomputed for the chosen rows. **Default rule:** the first phase in evaluation order, the global search; a gradient phase adds many near-identical finite-difference probes that would dominate every mean and the top 10 %. A single-phase run has no control and shows everything.
- **Distinct designs in the top 10 %:** designs closer than 1e-6 in the unit cube (Euclidean over the `x_` columns) count once, keeping the best of them, and "Top 10 %" is a tenth of those distinct designs (at least 5; "-" when fewer than 5 are distinct). The other statistics use every sample.
- **Layout:** panels on the usual grid (2 columns from 1200 px). Tables are ruled like the Summary table (2 px `--ink` rule under the header, 1 px `--rule` between rows, left-aligned text, right-aligned tabular numbers), with no card around them.
- **Panels, in order:** Summary (counts by status and by phase, feasible share, f_total best, median, mean and standard deviation for feasible and for all ok samples, time per evaluation); Score distribution; Objectives (box plot of the feasible samples plus a mean ± std, min, median and best-design table); Design parameters; Constraints; Evolution.
- **Score distribution:** histogram of f_total on shared bins, feasible in `--layout-blue`, infeasible in `--ink-muted`, overlaid. Best, median and mean are vertical lines named in the legend (no labels over the data). A muted line states how many failed evaluations are not shown ("12 failed evaluations not shown").
- **Design parameters:** physical hardpoint coordinates in mm (`p_` columns); unit-cube variables when those are absent. The table gives mean ± std for all feasible and for the top 10 %, the best design and the baseline. The chart shows, per parameter, mean ± std for all feasible and for the top 10 % as dots with whiskers and the best design as a marker, each normalised to the explored range of that parameter (0 to 1) so all fit one axis; the table carries the real values. The baseline is found as the history row whose f_total equals the stored baseline f_total, and is "-" when none matches.
- **Constraints:** pass rate (share of ok samples with g at or below 0), mean g and max g, for all ok samples and for the top 10 %. A horizontal bar chart in `--layout-blue` shows the pass rate.
- **Evolution:** mean ± std band (a `--layout-blue` line with a 15 % fill) of f_total per window of max(10, n/20) evaluations, feasible samples only, with the best-so-far step line in `--ink` on top.
- **No feasible sample:** the feasible-only panels say "No feasible evaluation yet, so there are no statistics to show." The counts and the failed and infeasible histogram still appear.

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
