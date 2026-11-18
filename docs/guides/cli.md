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

With `--surrogate DIR` (experimental, needs the `surrogate` extra) the active-learning
optimizer runs instead: the networks in `DIR/models/` search a trust region around the best
FOM-verified design, the FOM verifies the infill points (history sources `baseline`, `al_<k>`),
their curves are appended to `DIR/dataset/` and the networks are fine-tuned
(`DIR/models/<axle>/al_<k>/`; the model before the loop is kept in `pre_al/`). Settings:
`surrogate.active_learning` of the study. Extra options: `--device auto|cpu|cuda`;
`--evaluations` is the FOM budget (counted over the whole run folder), `--time` the wall
time of this invocation, `--resume` continues from `al.csv`, the FOM history and the current
models. The run folder gets `al.csv` (one row per iteration), `al/iter_<k>/surrogate_history.csv`
and the figures `al_progress`, `al_residual`, `al_radius`. `best.yaml` is the best
FOM-verified design; in Pareto mode `pareto/` holds FOM-verified points only. Exit codes as
above. Unlike plain `optimize`, the FOM budget of a resumed run counts all history rows (not
only the new ones), and `--resume` refuses a study whose constraints or objectives changed.
When the FOM rejects the predicted optimum, the trust ratio is 0; with no feasible verified
design yet, a feasible FOM point counts as ratio 1. In Pareto mode the trust region is centred
on the knee of the FOM-verified front and the ratio is 1 when the verified optimum extends
the front, else 0. `al.csv` also logs `kept_init_<axle>` (the fine-tuning kept the previous
weights; a warning is printed). The loop stops with reason `no_candidates` when the network
search returns nothing.

## `surrogate sample STUDY_YAML --root DIR` (experimental)

Samples the Latin-hypercube dataset of the NN surrogate from the study's `surrogate` block
into `DIR/dataset/<axle>/` (`--out` is a deprecated alias of `--root`; compressed shards plus `meta.json`). Row 0 is the baseline; the
train / val / test split (80 / 10 / 10) is fixed per row. Options: `--axle front|rear|both`
(default `both`; with `both` an axle without design variables is skipped), `--n N` (default
`dataset.n_initial`), `--workers N|auto`, `--resume` (continue an interrupted run; refused if
the study changed). Without `--resume` a non-empty dataset directory is refused. Prints the
expected size first, then per axle the sample count, feasible fraction and time, and warns
below 30 % feasible. Exit 2 on a study or argument error.

## `surrogate train STUDY_YAML --root DIR` and `surrogate validate STUDY_YAML --root DIR` (experimental)

`train` runs the Optuna architecture search per axle on `DIR/dataset/<axle>/` (the `--root`
folder of `surrogate sample`) and writes `DIR/models/<axle>/` (`optuna.db`, `trials.csv`,
`model/`). Settings come from `surrogate.training` and `surrogate.hpo`; options
`--axle front|rear|both`, `--trials N`, `--timeout T` (with unit, e.g. `2 h`) and
`--device auto|cpu|cuda` override them. A dataset sampled from another study (grid, design
space or vehicle changed) is refused. Prints the device, trials done, best validation score
and model path.

`validate` scores the network on the test split: per channel RMSE in mm / deg against its
tolerance, the ratio, validity accuracy. It writes `DIR/validation/<axle>/` with
`validation.csv`, `worst.csv`, `parity.csv`, `worst_curves.csv`, `summary.json` and
`figures/` (RMSE / tolerance bars, parity scatter, worst-case curves). Options `--axle`,
`--device`. Exit 0 all channels within tolerance, 1 some channel over tolerance, 2 study,
argument or data error (also `--device cuda` without CUDA).

The full reference with source links is in [Architecture & math, section 10](../ARCHITECTURE.md).
