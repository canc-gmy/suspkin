# Command line

Entry point: `suspkin` (typer). `--help` on any command lists its options. Run with
`uv run suspkin ...` (or `python -m uv run suspkin ...`).

**Units.** Command-line inputs are mm and degrees; written CSV/JSON files are SI.

**Exit codes.** 0 ok; 1 load or solve error (message on stderr); 2 usage error (bad option
combination, unknown corner, empty `--formats`). `check` and `optimize` use 2 for a
study-file, argument or output error and 1 for an infeasible result.

| Command | Purpose |
|---|---|
| `version` | Print the version. |
| `show` | 3D geometry, static or at a solved body pose. |
| `animate` | Animate a body-pose sweep (GIF or MP4). |
| `sweep` | Single-corner sweep over travel or rack to CSV. |
| `summary` | Static alignment, centres and gradients for FL and RL. |
| `vehicle` | Vehicle-level summary (rates, roll, anti-geometry, steering). |
| `vsweep` | Body-pose sweep (heave, roll, pitch or rack) to CSV. |
| `report` | Standard studies to a report folder with `report.html`. |
| `check` | Evaluate a study's constraints on the baseline. |
| `optimize` | Optimise a study (experimental) into a run folder. |

## `show VEHICLE_YAML`

`--out PATH` saves the figure (otherwise a window opens); `--labels` annotates hardpoints;
pose `--heave MM --roll DEG --pitch DEG --rack MM` (any subset; none = static); `--html`
writes an interactive plotly file (needs the `viz` extra).

```bash
suspkin show configs/vehicles/fs_reference.yaml --roll 2 --heave 5 --out pose.png
```

## `animate VEHICLE_YAML --out FILE`

Exactly one of `--heave MIN MAX` [mm], `--roll MIN MAX` [deg], `--pitch MIN MAX` [deg],
`--rack MIN MAX` [mm]; `--steps N` (default 15); `--fps N` (default 15). `.gif` or `.mp4`
(needs ffmpeg).

## `sweep VEHICLE_YAML --out CSV`

`--corner FL|FR|RL|RR` (default FL); exactly one of `--travel MIN MAX` [mm, bump positive] or
`--rack-sweep MIN MAX` [mm]; `--steps N` (default 21); `--rack MM` fixed rack for a travel
sweep; `--metrics` adds metric and gradient columns (needs at least 3 steps).

```bash
suspkin sweep fs.yaml --corner FL --travel -30 30 --metrics --out fl.csv
```

## `summary VEHICLE_YAML [--json FILE]`

Prints every static metric and gradient of FL and RL in display units; `--json` writes SI.

## `vehicle VEHICLE_YAML [--json FILE]`

Prints the vehicle summary (nested axle values as `front.<name>`); `--json` writes SI.

## `vsweep VEHICLE_YAML --out CSV`

Exactly one of `--heave/--roll/--pitch/--rack MIN MAX`; `--steps N` (21); fixed values
`--base-heave MM --base-roll DEG --base-pitch DEG --base-rack MM`; `--pivot X Y Z` [mm]
rotation centre (default the CG).

```bash
suspkin vsweep fs.yaml --roll -2 2 --out roll.csv
```

## `report VEHICLE_YAML --out DIR`

`--compare OTHER.yaml` (repeatable) overlays more designs; `--no-3d`; `--formats png,svg`;
`--steps N` (default 31). Prints the path of `report.html`.

## `check STUDY_YAML`

Evaluates the baseline against every constraint (and objective) and prints one row per
constraint with status `PASS`, `FAIL`, `UNSOLVED`, `INFEASIBLE` or `OFF`. Options:
`--csv PATH` (SI table), `--plot PATH` (3D region view), `--strict` (exit 1 if the baseline is
infeasible or outside a variable domain).

## `optimize STUDY_YAML` (experimental)

`--out DIR` (default `results/opt/<study>_<timestamp>`), `--resume` (continue from
`history.csv`), `--workers N|auto`, `--evaluations N`, `--time "30 min"`,
`--method nsga2|de|slsqp|trust-constr|de+slsqp`, `--no-report`. Command-line values override the
study's `optimizer` block. Exit 1 if no feasible design was found (the folder is still
written).

The full reference with source links is in [Architecture & math, section 10](../ARCHITECTURE.md).
