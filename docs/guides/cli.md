# Command line

Entry point: `suspkin` (a Typer application, `suspkin.cli:app`). The commands live in
`suspkin.cli.vehicle`, `suspkin.cli.study`, `suspkin.cli.webapp` and `suspkin.cli.surrogate`;
`suspkin.cli._common` holds the shared error exit and table printer. Run it as
`suspkin ...`, or without installing, `uv run suspkin ...` (on a machine where `uv` is not on
the PATH: `python -m uv run suspkin ...`). `suspkin` without arguments, `suspkin --help` and
`suspkin <command> --help` print the usage; the root also has `--install-completion` and
`--show-completion` (shell completion).

All examples below were run on `configs/vehicles/fs_reference.yaml` (suspkin 1.1.0); long
output is trimmed with `...`.

## Units and conventions

- **Inputs are millimetres and degrees** (`--travel -30 30` is -30 mm to +30 mm, `--roll 2` is
  2 deg). Durations take a unit string (`--time "30 min"`, `--timeout "2 h"`).
- **Files written are SI** (m, rad, N, N/m, kg). CSV and JSON tables are never in mm or deg.
- **Text printed to the terminal is in display units** (mm, deg, N/mm, ...), with the unit in
  its own column.
- Frame: ISO 8855, wheel travel **bump positive**; see [Conventions](../concepts/conventions.md).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | Load or solve error (message `error: ...` or `<file>: ...` on stderr). Also: `check --strict` with an infeasible baseline; `optimize` finished without a feasible design; `surrogate validate` with a channel over tolerance; `app-export` failure. |
| 2 | Usage error: bad option combination, unknown corner, empty `--formats`, malformed value (Typer's own parse errors also exit with 2). For `check`, `optimize` and the `surrogate` commands also a study file, argument, data or output-file error; `app` and `app-export` when the `app` extra is missing. |

## Commands

| Command | Purpose | Writes |
|---|---|---|
| [`version`](#version) | Print the version. | - |
| [`show`](#show) | 3D geometry, static or at a solved body pose. | PNG/PDF/... or HTML |
| [`animate`](#animate) | Animate a body-pose sweep. | GIF or MP4 |
| [`sweep`](#sweep) | Single-corner sweep over travel or rack. | CSV |
| [`summary`](#summary) | Static alignment, centres and gradients for FL and RL. | optional JSON |
| [`vehicle`](#vehicle) | Vehicle-level summary (rates, roll, anti-geometry, steering). | optional JSON |
| [`vsweep`](#vsweep) | Body-pose sweep (heave, roll, pitch or rack). | CSV |
| [`report`](#report) | Standard studies into a report folder. | folder with `report.html` |
| [`check`](#check) | Evaluate a study's constraints on the baseline. | optional CSV, image |
| [`optimize`](#optimize) | Optimise a study (experimental). | run folder |
| [`app`](#app) | Start the local web app (experimental). | - |
| [`app-export`](#app-export) | Self-contained HTML snapshot of results (experimental). | HTML |
| [`surrogate sample`](#surrogate-sample) | Sample the surrogate dataset (experimental). | dataset folder |
| [`surrogate train`](#surrogate-train) | Architecture search and training (experimental). | model folder |
| [`surrogate validate`](#surrogate-validate) | Test-split error per channel (experimental). | validation folder |

## version

`suspkin version` prints `suspkin <version>`.

```console
$ suspkin version
suspkin 1.1.0
```

## show

`suspkin show VEHICLE_YAML [options]`. Draws the 3D suspension geometry, static or at a solved
body pose (`suspkin.kinematics.solve_vehicle`).

| Option | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `VEHICLE_YAML` | path | required | - | Vehicle file. |
| `--out` | path | none | - | Save the figure here (extension decides the format; 200 dpi). Without it a Matplotlib window opens. With `--html`: the HTML path (default `<vehicle stem>.html` in the current directory). |
| `--labels` | flag | off | - | Label the hardpoints (Matplotlib figure only). |
| `--heave` | float | none | mm | Body heave. |
| `--roll` | float | none | deg | Body roll. |
| `--pitch` | float | none | deg | Body pitch. |
| `--rack` | float | none | mm | Rack displacement. |
| `--html` | flag | off | - | Write an interactive Plotly HTML instead (needs the `viz` extra). |

Any subset of the four pose options may be given; none means the static geometry. Prints the
output path. Exit 1 on a load error, a solver failure (`error: ...`) or a missing extra.

```console
$ suspkin show configs/vehicles/fs_reference.yaml --roll 2 --heave 5 --out pose.png
pose.png
$ suspkin show configs/vehicles/fs_reference.yaml --html --out pose.html
pose.html
```

## animate

`suspkin animate VEHICLE_YAML --out FILE (--heave MIN MAX | --roll MIN MAX | --pitch MIN MAX | --rack MIN MAX) [--steps N] [--fps N]`

| Option | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `--out` | path | **required** | - | `.gif`, or `.mp4` (needs ffmpeg). |
| `--heave MIN MAX` | 2 floats | none | mm | Sweep the heave. |
| `--roll MIN MAX` | 2 floats | none | deg | Sweep the roll. |
| `--pitch MIN MAX` | 2 floats | none | deg | Sweep the pitch. |
| `--rack MIN MAX` | 2 floats | none | mm | Sweep the rack. |
| `--steps` | int >= 2 | `15` | - | Number of frames. |
| `--fps` | int >= 1 | `15` | - | Frames per second. |

Exactly one sweep variable, else `error: give exactly one of --heave, --roll, --pitch, --rack`
and exit 2. Solver or writer errors (for example no ffmpeg for `.mp4`) exit 1.

```console
$ suspkin animate configs/vehicles/fs_reference.yaml --roll -2 2 --steps 5 --fps 5 --out roll.gif
roll.gif
```

## sweep

`suspkin sweep VEHICLE_YAML --out CSV (--travel MIN MAX | --rack-sweep MIN MAX) [options]`.
Solves one corner over wheel travel or rack and writes a table (see
[Sweeps and reports](../concepts/sweeps-and-reports.md) for the columns).

| Option | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `--out` | path | **required** | - | CSV path. |
| `--corner` | `FL` \| `FR` \| `RL` \| `RR` | `FL` | - | Corner. Anything else: `error: unknown corner 'XX'`, exit 2. |
| `--travel MIN MAX` | 2 floats | none | mm | Wheel travel range, bump positive. |
| `--rack-sweep MIN MAX` | 2 floats | none | mm | Rack range at zero travel. |
| `--steps` | int >= 2 | `21` | - | Number of points (linearly spaced, both ends included). |
| `--rack` | float | `0.0` | mm | Fixed rack offset for a **travel** sweep. |
| `--metrics` | flag | off | - | Add the metric and gradient columns. Needs at least 3 steps (`error: at least 3 rows are required`, exit 1). |

Exactly one of `--travel`, `--rack-sweep` (else exit 2). The CSV is SI: travel in m, angles in rad.

```console
$ suspkin sweep configs/vehicles/fs_reference.yaml --corner FL --travel -30 30 --steps 7 --metrics --out fl.csv
fl.csv
```

The file has 7 rows and 107 columns (62 without `--metrics`); a few of them:

```text
corner  travel   camber      toe  mr_spring  camber_gain
    FL   -0.03 -0.00996 -0.00338    0.93351     -0.21302
    FL   -0.02 -0.01209 -0.00151    0.94341     -0.23134
    FL    0.00 -0.01745  0.00000    0.97989     -0.30493
    FL    0.03 -0.02830 -0.00325    0.99334     -0.40008
```

Camber at zero travel is -0.01745 rad = -1.0 deg, the `static_camber` of the file; camber gain
is rad/m (-0.305 rad/m = -0.0175 deg/mm, which is what `summary` prints).

## summary

`suspkin summary VEHICLE_YAML [--json FILE]`. Solves FL and RL over a small travel sweep
(-10 to +10 mm, 21 points) and prints every static metric and gradient in display units, one
row per metric. `--json FILE` writes the same numbers in **SI** as
`{"FL": {...}, "RL": {...}}` and prints the path.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--json` | path | none | Write the SI table as JSON. |

```console
$ suspkin summary configs/vehicles/fs_reference.yaml --json summary.json
metric                                   FL            RL  unit
camber                              -1.0000       -1.5000  deg
toe                                  0.0000        0.1000  deg
caster                               4.9360        1.0416  deg
kpi                                  4.9360        4.6774  deg
scrub_radius                        17.7912       11.3202  mm
mechanical_trail                    20.1879        2.1336  mm
...
rc_height                           33.7206       39.9991  mm
pushrod_angle                       64.7399       67.6404  deg
spring_force                       574.0000      696.0000  N
camber_gain                         -0.0175       -0.0231  deg/mm
bump_steer                           0.0001        0.0003  deg/mm
mr_spring                            0.9807        0.9929  -
wheel_rate                          34.5673       40.9483  N/mm
ride_rate                           25.6877       29.0520  N/mm
summary.json
```

A metric that is undefined for the car shows `nan` or `inf`: here the side-view instant
centre (`svic_x`, `svic_z` are `nan`, `svsa` is `inf`, `sv_swing_angle` is 0), because the
reference car's wishbone axes are parallel in side view.

## vehicle

`suspkin vehicle VEHICLE_YAML [--json FILE]`. Prints the vehicle-level summary
(`suspkin.metrics.vehicle_summary`): per-axle values as `front.<name>` and `rear.<name>`,
then the car-level values. `--json FILE` writes the nested SI dictionary. A value that is
`None` for the car prints as `n/a`.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--json` | path | none | Write the SI summary as JSON. |

```console
$ suspkin vehicle configs/vehicles/fs_reference.yaml --json veh.json
front.track                      1220.0000  mm
front.wheel_rate                   34.5673  N/mm
front.arb_wheel_rate_roll          32.2912  N/mm
front.roll_stiffness              520.4454  N*m/deg
front.rc_height                    33.7206  mm
...
rear.rc_height                     39.9991  mm
wheelbase                        1550.0000  mm
sprung_mass                       250.0000  kg
roll_axis_height_at_cg             37.0784  mm
roll_stiffness_total              989.6740  N*m/deg
roll_gradient                       0.6292  deg/g
tlltd_front                        50.3025  %
max_steer_inner                    22.7415  deg
steering_ratio                      5.4164  deg/deg (handwheel/road wheel)
turning_radius                      3.6957  m
veh.json
```

Note that `tlltd_front` prints in percent here, but a study constraint on `tlltd_front` is
written as a ratio (`min: 0.45, max: 0.60`); see the metric dimensions in
[Study YAML](study-yaml.md#metric-names).

## vsweep

`suspkin vsweep VEHICLE_YAML --out CSV (--heave | --roll | --pitch | --rack) MIN MAX [options]`.
Sweeps the body pose and writes the vehicle table (all four corners, roll centres, ARB).

| Option | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `--out` | path | **required** | - | CSV path. |
| `--heave MIN MAX` | 2 floats | none | mm | Sweep heave. |
| `--roll MIN MAX` | 2 floats | none | deg | Sweep roll. |
| `--pitch MIN MAX` | 2 floats | none | deg | Sweep pitch. |
| `--rack MIN MAX` | 2 floats | none | mm | Sweep rack. |
| `--steps` | int >= 2 | `21` | - | Number of points. |
| `--base-heave` | float | `0.0` | mm | Fixed heave while sweeping another input. |
| `--base-roll` | float | `0.0` | deg | Fixed roll. |
| `--base-pitch` | float | `0.0` | deg | Fixed pitch. |
| `--base-rack` | float | `0.0` | mm | Fixed rack. |
| `--pivot X Y Z` | 3 floats | the CG | mm | Body rotation centre, vehicle frame. |

Exactly one sweep variable (else exit 2). The `--base-*` option of the swept input is ignored.
CSV in SI (m, rad).

```console
$ suspkin vsweep configs/vehicles/fs_reference.yaml --roll -2 2 --steps 5 --out roll.csv
roll.csv
```

5 rows, 170 columns, for example

```text
 heave     roll  FL_camber  FR_camber  rc_front_z
   0.0 -0.03491   -0.02478   -0.01174     0.03254
   0.0  0.00000   -0.01745   -0.01745     0.03372
   0.0  0.03491   -0.01174   -0.02478     0.03254
```

Rolling by +-2 deg (roll angle in rad: -0.03491) gives opposite camber changes on the two
front wheels (at -2 deg: FL -1.42 deg, FR -0.67 deg; static -1.0 deg) and a front roll centre height that
changes by about 1 mm (`rc_front_z` 32.5 to 33.7 mm). A pose that does not solve exits 1.

## report

`suspkin report VEHICLE_YAML --out DIR [options]`. Runs the standard studies
(`suspkin.report.run_standard_studies`: corner sweeps front and rear, heave, pitch, roll,
rack) and builds a report folder (`suspkin.report.build_report`).

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--out` | path | **required** | Output directory. |
| `--compare` | path, repeatable | none | Further vehicle YAML to overlay in every figure. Designs are labelled by file stem; duplicate stems get `_2`, `_3`, ... |
| `--no-3d` | flag | off | Skip the 3D view and the roll-sweep GIF. |
| `--formats` | text | `png,svg` | Figure formats, comma separated. Empty: `error: --formats is empty`, exit 2. |
| `--steps` | int >= 3 | `31` | Points per sweep. |

Prints the path of `report.html`. Tables are SI; figures and the HTML summary are in mm and deg.

```console
$ suspkin report configs/vehicles/fs_reference.yaml --out rep --no-3d --steps 7 --formats png
rep/report.html
```

Folder layout:

```text
rep/
  report.html          self-contained (images embedded)
  report.md            same content, links to the figure files
  summary.json
  data/                <label>_front_corner.csv, _rear_corner, _heave, _pitch, _roll, _rack
  figures/             corner_alignment_*, corner_centres_*, corner_position_*, actuation_*,
                       heave_pitch, roll, steering, summary_table_<label>  (png, svg)
                       + geometry_3d.png, roll_sweep.gif unless --no-3d
  bundles/<label>/     manifest.json, summary.json, vehicle.yaml, tables/*.csv
```

The `bundles/<label>` folder is the input format of the web app ([Web app](app.md)).

## check

`suspkin check STUDY_YAML [options]`. Evaluates the **baseline** vehicle against every
constraint and objective of a study (the same evaluation the optimiser performs) and prints a
table. See [Study YAML](study-yaml.md).

| Option | Type | Default | Meaning |
|---|---|---|---|
| `STUDY_YAML` | path | required | Study file. |
| `--csv` | path | none | Write the SI result table as CSV (parent folders are created). |
| `--plot` | path | none | Write the 3D study view (`suspkin.plotting.plot_study_3d`) as an image (200 dpi). |
| `--strict` | flag | off | Exit 1 if the baseline is not feasible **or** lies outside a variable domain. |

Output: one row per constraint with columns `name type value min max unit g status detail`
(display units; `-` = no limit); status `PASS`, `FAIL`, `UNSOLVED`, `INFEASIBLE` or `OFF`
(disabled); then `n/N constraints pass, feasible: yes|no`. If the study has objectives a second
table follows (`objective type raw unit f weight status detail`), the weighted `f_total` and the
`optimizer:` line. A baseline outside a variable domain prints
`warning: baseline is outside a variable domain: ...` on stderr. A study or output-file error
exits 2; with `--strict` an infeasible baseline exits 1.

```console
$ suspkin check configs/studies/fs_reference_optimize.yaml --csv check.csv
name            type           value      min     max  unit           g  status  detail
camber_band_f   band          -0.629   -4.000   0.000  deg       -0.629  PASS
toe_band_f      band          -0.135   -0.500   0.500  deg       -3.654  PASS
bump_steer_r    slope          0.000   -0.020   0.020  deg/mm    -0.493  PASS
mr_front        value          0.980    0.900   1.100  -         -0.402  PASS
mr_rear         value          0.993    0.900   1.100  -         -0.464  PASS
travel_f        reach         10.224    0.000       -  mm       -10.224  PASS    bump stroke remaining
ubj_angle       joint_angle    5.931        -  25.000  deg     -190.687  PASS
rc_height_f     summary       33.721   10.000  80.000  mm        -0.339  PASS
tlltd_front     summary        0.503    0.450   0.600  -         -0.353  PASS
uca_front_zone  inside       -40.000        -   0.000  mm       -40.000  PASS
tie_len         distance     435.018  400.000       -  mm       -35.018  PASS
uca_above_lca   relation     325.000  225.000       -  mm      -100.000  PASS
12/12 constraints pass, feasible: yes

objective     type         raw  unit        f  weight  status  detail
camber_curve  curve      0.870  deg     8.702       1  OK
rc_f          target     6.279  mm      1.256       1  OK
bump_steer_f  minimize   0.011  deg/mm  1.076       1  OK      max_abs
mr_linearity  linearity  0.006  -       0.633     0.5  OK
f_total (weighted sum): 11.350
optimizer: mode weighted, method de+slsqp
check.csv
```

Read `g`: negative = margin, positive = violation, in scale units (here `camber_band_f` has `scale: 1 deg`, so g = -0.629 is 0.629 deg of margin; `toe_band_f` has `scale: 0.1 deg`, so a 0.365 deg margin reads -3.654).
For an `inside` constraint the `value` is the signed distance to the region boundary
(-40 mm = 40 mm inside); for `relation`, `value` is the left-hand coordinate and `min` shows
`rhs + offset`; for `reach` it is the term with the largest `g` (named in `detail`).

```console
$ suspkin check configs/studies/fs_reference_constraints.yaml --strict --plot study.png
...
20/20 constraints pass, feasible: yes
study.png
```

## optimize

`suspkin optimize STUDY_YAML [options]` (**experimental**). Runs the study's optimizer
(see [Optimization](../concepts/optimization.md)) and writes a run folder. Command-line values
override the study's `optimizer` block for this invocation.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `STUDY_YAML` | path | required | Study with objectives (and optionally an `optimizer` block). |
| `--out` | path | `results/opt/<study stem>_<YYYYMMDD-HHMMSS>` | Run folder. |
| `--resume` | flag | off | Continue the run in `--out` from its `history.csv`. See the caveat below. |
| `--workers` | int >= 1 or `auto` | study value | Worker processes. `auto` = CPU count minus one. |
| `--evaluations` | int | study value | Evaluation budget. |
| `--time` | text | study value | Wall-time budget with unit, e.g. `"30 min"`. |
| `--method` | text | study value | `nsga2`, `de`, `slsqp`, `trust-constr` or `de+slsqp`. |
| `--no-report` | flag | off | Skip the before/after report. |
| `--surrogate` | path | none | Surrogate folder (`dataset/`, `models/`): active-learning optimizer (needs the `surrogate` extra). |
| `--device` | `auto` \| `cpu` \| `cuda` | study value | Device of the networks; only with `--surrogate` (`error: --device needs --surrogate`, exit 2). |

Written to the run folder: `study.yaml` and `vehicle.yaml` (copies; the study then points at
`vehicle.yaml`), `history.csv` (one row per evaluation: `eval_id`, the normalised variables `x_*`, the decoded
points `p_*`, objective and constraint values, source), `meta.json` (version, git hash, seed, method, mode, timings,
`study_hash`), `result.json` (baseline and best design, all `f` and `g`), `best.yaml` (the
best design as a vehicle file), `figures/` (`convergence`, `constraint_margins`, `parallel`, ...)
and, unless `--no-report`, `report/` (before/after report). In Pareto mode also a `pareto/`
folder and the printed table shows the **knee point**.

```console
$ suspkin optimize configs/studies/fs_reference_optimize.yaml --out opt --evaluations 60 --workers 1 --no-report
optimize: method de+slsqp, mode weighted, 11 variables, 12 constraints, 4 objectives, 1 workers, budget 60 evaluations / 900 s
done: 69 new evaluations (69 in history), 3.8 s, 1 workers

objectives
objective     baseline   best  unit    f base  f best
camber_curve     0.870  0.939  deg      8.702   9.395
rc_f             6.279  0.420  mm       1.256   0.084
bump_steer_f     0.011  0.012  deg/mm   1.076   1.205
mr_linearity     0.006  0.006  -        0.633   0.603
f_total: baseline 11.350 -> best 10.985

constraints
constraint      baseline     best  unit      g base    g best  status
camber_band_f     -0.629   -0.516  deg       -0.629    -0.516  PASS
...
uca_above_lca    325.000  324.000  mm      -100.000   -99.000  PASS
feasible: baseline yes, best yes
opt
```

(60 evaluations is a smoke test; use thousands for a real run. The count printed by `done:`
can exceed the budget slightly, here 69 for 60.)

Exit codes: 0 done; **1 if no feasible design was found** (the folder is still written); 2 on
a study, argument or output error (including `--resume` with an incompatible folder).

!!! warning "`--resume` only detects changed constraints and objectives"
    `meta.json` stores `study_hash`, a hash of the `constraints` and `objectives` blocks
    **only**. Edits to `variables`, `sweeps`, `resolution`, `optimizer`, `surrogate` or the
    vehicle file are not detected; resuming then mixes designs evaluated under different
    problems. For a changed constraint or objective a plain `optimize --resume` warns and drops
    the cached results; `optimize --surrogate ... --resume` refuses. Start a new `--out` after
    changing anything else. Details in [Study YAML](study-yaml.md#resuming).

```console
$ suspkin optimize configs/studies/fs_reference_optimize.yaml --out opt --resume --evaluations 100 --workers 1 --no-report
optimize: method de+slsqp, mode weighted, 11 variables, 12 constraints, 4 objectives, 1 workers, budget 100 evaluations / 900 s
done: 34 new evaluations (103 in history), 2.3 s, 1 workers
```

### `optimize --surrogate DIR`

With `--surrogate DIR` the **active-learning optimizer** runs instead of the plain one: the
networks in `DIR/models/` search a trust region around the best FOM-verified design, the FOM
(the real kinematic model) verifies the infill points (history sources `baseline`, `al_<k>`),
their curves are appended to `DIR/dataset/` and the networks are fine-tuned
(`DIR/models/<axle>/al_<k>/`; the model before the loop is kept in `pre_al/`). Settings:
`surrogate.active_learning` of the study. Differences to the plain run:

- `--evaluations` is the FOM budget counted over the whole run folder (all history rows, not
  only the new ones); `--time` is the wall time of this invocation; `--device` is available.
- `--resume` continues from `al.csv`, the FOM history and the current models, and refuses a
  study whose constraints or objectives changed.
- Extra output: `al.csv` (one row per iteration, with `kept_init_<axle>` when a fine-tune kept
  the previous weights), `al/iter_<k>/surrogate_history.csv` and the figures `al_progress`,
  `al_residual`, `al_radius`. `best.yaml` is the best FOM-verified design; in Pareto mode
  `pareto/` holds FOM-verified points only. A line `active learning: N new iterations (M in
  al.csv), stopped on <reason>, device <d>` is printed; reasons include `no_candidates` (the
  network search returned nothing, or all infill points were already verified designs).
- Trust ratio: when the FOM rejects the predicted optimum the ratio is 0; with no feasible
  verified design yet, a feasible FOM point counts as 1. In Pareto mode the region is centred on
  the knee of the FOM-verified front and the ratio is 1 when the verified optimum extends the
  front, else 0.
- Errors, including a missing torch, exit 2.

## app

`suspkin app [PATHS...] [options]` (**experimental**, needs the `app` extra: `uv sync --extra
app`). Starts the local web app (FastAPI served by uvicorn); see [Web app](app.md).

| Option | Type | Default | Meaning |
|---|---|---|---|
| `PATHS...` | paths | none | Vehicle YAMLs, bundles, report folders or optimizer run folders to open at start. |
| `--port` | int | `8765` | Port. |
| `--host` | text | `127.0.0.1` | Interface to bind. |
| `--workspace` | path | current directory | Root folder the app may read. |
| `--no-browser` | flag | off | Do not open the browser. |
| `--dev` | flag | off | API only: the UI comes from the Vite dev server (`npm run dev`). |

Exit 2 if the extra is missing (`the web app needs the 'app' extra: uv sync --extra app`) or
`--workspace` is not a directory.

```console
$ suspkin app configs/vehicles/fs_reference.yaml --no-browser --port 8799
INFO:     Started server process [33308]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8799 (Press CTRL+C to quit)
```

`app` takes a variable list of paths, so the snapshot export is a separate command
(`app-export`) rather than a sub-command.

## app-export

`suspkin app-export PATHS... [--out FILE] [--workspace DIR] [--no-scene]` (**experimental**,
needs the `app` extra). Writes one self-contained HTML file with the given results embedded
(a snapshot you can mail around; it needs no server).

| Option | Type | Default | Meaning |
|---|---|---|---|
| `PATHS...` | paths | **required** | Vehicle YAMLs, bundles, report or run folders to embed. |
| `--out` | path | `compare.html` | Output file. |
| `--workspace` | path | current directory | Root folder the paths must be inside. |
| `--no-scene` | flag | off | Leave out the 3D scenes (smaller file). |

Prints `Snapshot exported to <file> (<size> MB)`. It embeds the single-file front end
(`suspkin/app/static_single/index.html`), which is built from the `frontend/` sources. In a
source checkout where that has not been built it fails with exit 1:

```console
$ suspkin app-export configs/vehicles/fs_reference.yaml --out compare.html --no-scene
error: single-file frontend not built (.../src/suspkin/app/static_single/index.html); run: cd frontend && npm run build:single
```

## surrogate sample

`suspkin surrogate sample STUDY_YAML --root DIR [options]` (**experimental**). Samples the
initial Latin-hypercube dataset of the neural-network surrogate from the study's `surrogate`
block ([Study YAML](study-yaml.md#surrogate-experimental),
[Surrogate guide](surrogate.md)) by running the real model (FOM) on each sampled design.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `STUDY_YAML` | path | required | Study with a `surrogate` block. |
| `--root` | path | required (see `--out`) | Surrogate folder; the dataset goes to `DIR/dataset/<axle>/`. |
| `--axle` | `front` \| `rear` \| `both` | `both` | Axle(s). With `both`, an axle without design variables is skipped (`rear: skipped (...)`); with one named axle it is an error. |
| `--n` | int >= 1 | `dataset.n_initial` | Latin-hypercube samples per axle. |
| `--workers` | int >= 1 or `auto` | none (in-process, 1 worker) | Worker processes; `auto` = CPU count minus one. Not taken from the study's `optimizer` block. |
| `--resume` | flag | off | Continue an interrupted sampling in `--root`; refused if the study changed. |
| `--out` | path | none | **Hidden, deprecated alias of `--root`** (not listed in `--help`). Giving both with different values: `error: give --root only (--out is a deprecated alias)`; giving neither: `error: missing option --root`. |

Writes `DIR/dataset/<axle>/shard_*.npz` (compressed shards), `meta.json` and `lhs.npy`. Row 0 is
the baseline; the train / validation / test split (80 / 10 / 10) is fixed per row. Without
`--resume`, a non-empty dataset folder is refused (exit 2). It prints the expected size first,
then progress, then per axle the sample count, feasible fraction and time, and warns on stderr
if fewer than 30 % of the samples are feasible. Exit 2 on a study or argument error.

```console
$ suspkin surrogate sample configs/studies/fs_reference_surrogate.yaml --root sur --axle front --n 40 --workers 1
front: 40 samples, about 4045080 bytes uncompressed (101127 per sample)
front: rows 40/40
front: 40 samples (40 new), feasible 100.0 %, 3.0 s
$ suspkin surrogate sample configs/studies/fs_reference_surrogate.yaml --root sur --axle front --n 40
error: .../sur/dataset/front is not empty; use --resume or a new --out directory
```

(The refusal message still says `--out`, the old name of `--root`.)

## surrogate train

`suspkin surrogate train STUDY_YAML --root DIR [options]` (**experimental**, needs the
`surrogate` extra). Runs the Optuna architecture search per axle on `DIR/dataset/<axle>/` and
retrains the best configuration. Slow (up to hours; the example run in these docs does not
train).

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--root` | path | **required** | Folder that holds `dataset/` (the `--root` of `surrogate sample`). |
| `--axle` | `front` \| `rear` \| `both` | `both` | With `both`, axles without a dataset are skipped; a named axle without dataset is an error. |
| `--trials` | int >= 1 | `hpo.n_trials` | Optuna trials. |
| `--timeout` | text | `hpo.timeout` | Search time budget with unit, e.g. `"2 h"`; must be positive. |
| `--device` | `auto` \| `cpu` \| `cuda` | `training.device` | Training device. |

Writes `DIR/models/<axle>/` with `optuna.db`, `trials.csv` and `model/`. A dataset sampled from
another study (changed grid, design space or vehicle) is refused. Prints the device, the trials
done, the best validation score and the model path. Exit 2 on a study, argument or data error.

```console
$ suspkin surrogate train configs/studies/fs_reference_surrogate.yaml --root sur --trials 30 --timeout "2 h"
```

## surrogate validate

`suspkin surrogate validate STUDY_YAML --root DIR [--axle ...] [--device ...]`
(**experimental**). Scores the trained network of each axle on the **test split**: the RMSE of
every output channel in mm / deg against its tolerance, the ratio, and the accuracy of the
validity flag.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--root` | path | **required** | Folder with `dataset/` and `models/`. |
| `--axle` | `front` \| `rear` \| `both` | `both` | |
| `--device` | `auto` \| `cpu` \| `cuda` | `training.device` | `cuda` without CUDA is an error. |

Writes `DIR/validation/<axle>/`: `validation.csv`, `worst.csv`, `parity.csv`,
`worst_curves.csv`, `summary.json` and `figures/` (RMSE/tolerance bars, parity scatter,
worst-case curves). Exit codes: 0 every channel within tolerance; **1 some channel over
tolerance** (`error: N channel(s) over tolerance`); 2 on a study, argument or data error.

Output format (one block per axle):

```text
front: <n> test samples, score <s>, validity accuracy <p> %
  channel                         rmse        tol  unit    ratio  status
  <channel>                      <...>      <...>  <mm|deg> <r>   ok | OVER
<root>/validation/front
```

Without a trained model:

```console
$ suspkin surrogate validate configs/studies/fs_reference_surrogate.yaml --root sur --axle front
device: cpu (requested auto)
error: front: no trained model in .../sur/models/front/model; run `surrogate train` first
```

## Typical sessions

```console
# Look at a car
suspkin summary configs/vehicles/fs_reference.yaml
suspkin vehicle configs/vehicles/fs_reference.yaml
suspkin report  configs/vehicles/fs_reference.yaml --out rep --compare my_car.yaml

# Curves for your own plotting
suspkin sweep  my_car.yaml --travel -30 30 --metrics --out fl.csv
suspkin vsweep my_car.yaml --roll -2 2 --out roll.csv

# Study, optimise, inspect
suspkin check    configs/studies/fs_reference_optimize.yaml --strict
suspkin optimize configs/studies/fs_reference_optimize.yaml --out results/opt/demo
suspkin app results/opt/demo

# Surrogate-assisted
suspkin surrogate sample   study.yaml --root runs/sur
suspkin surrogate train    study.yaml --root runs/sur
suspkin surrogate validate study.yaml --root runs/sur
suspkin optimize study.yaml --surrogate runs/sur --out runs/al
```
