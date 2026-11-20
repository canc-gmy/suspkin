# Examples

Five runnable scripts in `examples/` show the Python API end to end. They are plain scripts, not
a framework: read them top to bottom, copy the parts you need. CI runs every one of them
(`pytest -m examples`), so they cannot rot.

## How to run

From the repository root:

```bash
uv run python examples/01_load_and_sweep.py
uv run python examples/01_load_and_sweep.py --out my_results/01     # choose the output folder
```

(`python -m uv run ...` where `uv` is not on `PATH`.) Every script takes `--out DIR`; the
default is `results/examples/<script name>`, created if missing. They use the Agg matplotlib
backend for files, so they work without a display. Script 05 is experimental (it uses the
`suspkin.optimize` Python API, which may change in minor releases) and needs no extra: it uses
differential evolution. Running all five takes about 30 seconds on a laptop. The scripts print absolute paths; the expected outputs below show them relative to the repository.

| Script | Shows | Run time | Writes |
|---|---|---|---|
| `01_load_and_sweep.py` | load a vehicle, bump sweep of one corner, metrics table, static values, CSV | 3 s | `fl_bump_sweep.csv` |
| `02_vehicle_roll.py` | whole-car roll sweep, vehicle summary, camber versus roll plot | 3 s | `camber_vs_roll.png` |
| `03_compare_designs.py` | edit a hardpoint, compare two designs, write a report | 8 s | `report.html`, `report.md`, `summary.json`, `figures/`, `data/`, `bundles/` |
| `04_custom_plot.py` | 3D view at a solved pose and your own matplotlib curve | 4 s | `vehicle_3d.png`, `camber_curve.png` |
| `05_optimize_study.py` | constraint table of a study baseline and a tiny optimisation | 14 s | a run folder (`best.yaml`, `history.csv`, ...) |

Run times are from one run on a Windows laptop and include the start-up of `uv run`.

## 01 Load and sweep

Loads `configs/vehicles/fs_reference.yaml`, builds the four corners, sweeps the front-left
corner $\pm 30$ mm in 13 steps, computes the metrics and gradients table
(`suspkin.metrics_frame`, `suspkin.add_gradients`), reads the row at zero travel, and converts
SI to display units by hand.

```python
--8<-- "examples/01_load_and_sweep.py"
```

Expected output:

```text
Vehicle: fs_reference
static camber :   -1.000 deg
static toe    :    0.000 deg
RC height     :     33.7 mm
camber gain   :  -0.0175 deg/mm
saved results/examples/01_load_and_sweep/fl_bump_sweep.csv (13 rows, 47 columns)
```

These numbers match the `suspkin summary` table of the same car
([Tutorial](../getting-started/tutorial.md)). The CSV is in SI (m, rad). Note the unit
conversion for the gradient: `camber_gain` is rad/m, so degrees per mm is
`np.degrees(x) / 1e3`.

## 02 Vehicle roll

Rolls the whole car $\pm 2$ deg about the CG in 9 steps (`suspkin.sweep_vehicle`), builds the
vehicle table (`suspkin.vehicle_frame`), prints the vehicle summary numbers
(`suspkin.vehicle_summary`) and plots the camber relative to the **ground** of all four wheels
against body roll.

```python
--8<-- "examples/02_vehicle_roll.py"
```

Expected output:

```text
roll gradient    : 0.629 deg/g
RC height front  : 33.7 mm
RC height rear   : 40.0 mm
roll axis at CG  : 37.1 mm
TLLTD (front)    : 50.3 %
saved results/examples/02_vehicle_roll/camber_vs_roll.png
```

Roll positive is left side up. In the plot the outer wheel (right wheel for positive roll)
gains positive ground camber with roll, the inner one negative; the slope is about 0.8 deg per
degree of roll, which is the `roll_camber` of the summary
([Metrics](../concepts/metrics.md#10-roll-and-arb-details)).

## 03 Compare designs

Makes a variant of the reference car by raising both front LCA chassis pivots by 10 mm (editing
a dict from `Vehicle.model_dump(by_alias=True)` and re-validating with `Vehicle.model_validate`),
runs `suspkin.run_standard_studies` for both, prints camber gain and roll-centre height, and
writes an HTML/Markdown report with both designs overlaid (`suspkin.build_report`).

```python
--8<-- "examples/03_compare_designs.py"
```

Expected output:

```text
design        camber gain [deg/mm]  RC height [mm]
base                       -0.0175            33.7
lca +10 mm                 -0.0259            59.0
report: results/examples/03_compare_designs/report.html
```

Raising the lower arm's inner pivots by 10 mm raises the front roll centre from 33.7 mm to
59.0 mm and strengthens the camber gain from -0.0175 to -0.0259 deg/mm: a small change of one
point has a large effect on the roll centre. Open `report.html` to see all figures overlaid.
The report folder layout is explained in
[Sweeps and reports](../concepts/sweeps-and-reports.md).

## 04 Custom plot

Draws the car in 3D at a solved pose (1.5 deg roll and 10 mm heave,
`suspkin.solve_vehicle` and `suspkin.plot_vehicle_3d`) and then a custom matplotlib curve from a
result table. This one shows the main design rule: **plotting only reads result tables, it
never calls the solver**, so any figure you want is a few lines of matplotlib on a table.

```python
--8<-- "examples/04_custom_plot.py"
```

Expected output:

```text
saved results/examples/04_custom_plot/vehicle_3d.png
saved results/examples/04_custom_plot/camber_curve.png
```

## 05 Optimize a study (experimental)

Loads `configs/studies/fs_reference_optimize.yaml`, builds the design space, evaluates the
baseline against all constraints (the `suspkin check` equivalent) and runs a tiny
differential-evolution optimisation with a budget of 30 evaluations
(`suspkin.optimize.run_optimization`).

```python
--8<-- "examples/05_optimize_study.py"
```

Expected output:

```text
          name      value    min   max   unit           g status
 camber_band_f  -0.629207  -4.00  0.00    deg   -0.629207   pass
    toe_band_f  -0.134561  -0.50  0.50    deg   -3.654386   pass
  bump_steer_r   0.000292  -0.02  0.02 deg/mm   -0.492690   pass
      mr_front   0.980479   0.90  1.10          -0.402395   pass
       mr_rear   0.992712   0.90  1.10          -0.463561   pass
      travel_f  10.224439   0.00   NaN     mm  -10.224439   pass
     ubj_angle   5.931307    NaN 25.00    deg -190.686929   pass
   rc_height_f  33.720636  10.00 80.00     mm   -0.338866   pass
   tlltd_front   0.503025   0.45  0.60          -0.353499   pass
uca_front_zone -40.000000    NaN  0.00     mm  -40.000000   pass
       tie_len 435.018390 400.00   NaN     mm  -35.018390   pass
 uca_above_lca 325.000000 225.00   NaN     mm -100.000000   pass
baseline feasible: True, f_total = 11.3504
44 evaluations in 1.8 s, feasible: True
baseline f_total: 11.3504
best f_total    : 11.0953
run folder: results/examples/05_optimize_study
```

The run needs 44 evaluations
rather than 30 because differential evolution evaluates whole generations. The timing varies
from run to run; the numbers are reproducible because the study has a fixed seed. `g` is the
normalised constraint margin (negative = satisfied,
[Optimisation](../concepts/optimization.md)). A 30-evaluation run is a smoke test, not a
design; for a real run use the CLI with a larger budget and more workers
(`suspkin optimize STUDY --evaluations 2000 --workers auto`).

## Writing your own script

The pattern in all five scripts:

1. `vehicle = suspkin.load_vehicle(path)`; to change it, dump, edit and re-validate:
   `Vehicle.model_validate(data)` on the edited nested dict `data = vehicle.model_dump(by_alias=True)` (see `raise_front_lca` in script 03).
2. A solve: `suspkin.sweep_corner`, `suspkin.sweep_vehicle` or `suspkin.solve_vehicle` return
   solved states; they raise `suspkin.KinematicsError` if the linkage cannot assemble.
3. Tables: `suspkin.metrics_frame` / `suspkin.add_gradients` (corner) or
   `suspkin.vehicle_frame` (vehicle). Columns are SI; convert to mm and deg yourself or use the
   helpers of `suspkin.plotting.units`.
4. Output: `suspkin.build_report`, or your own matplotlib.

On Windows, a script that starts worker processes (an optimiser with `workers > 1`, or the
surrogate functions) needs the `if __name__ == "__main__":` guard that every example already
has.
