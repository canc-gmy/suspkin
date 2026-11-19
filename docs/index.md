# suspkin

Suspension kinematics toolkit for **Formula Student double-wishbone + pushrod** cars.
Give it hardpoints in a YAML file; get camber, toe, caster, kingpin, anti-dive/anti-squat,
roll centres, motion ratios, anti-roll-bar rates and steering geometry as tables, plots and
an HTML report.

![Reference car](images/fs_reference.png)

## Features

- Corner kinematics solved in 3D (bump, droop, rack travel), including rocker and pushrod
  (mounted on upright, lower or upper wishbone).
- Whole-car body motion: heave, roll, pitch, rack, with ARB coupling.
- Metrics and gradients: alignment, instant centres, roll centre, motion ratio, wheel and
  spring rates, steering, anti-geometry, load transfer.
- Plots, 3D views (matplotlib, optional plotly), animations and a standard report.
- Geometry optimisation studies (experimental): constraints, objectives, several optimisers.
- Pure-function numpy core, validated YAML models, golden regression tests.

## Install

```bash
git clone https://github.com/canc-gmy/suspkin.git && cd suspkin
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1   # Windows
```

```bash
bash scripts/setup.sh                                         # Linux / macOS / Git Bash
```

Details: [Install](getting-started/install.md).

## 30 seconds

```bash
uv run suspkin summary configs/vehicles/fs_reference.yaml
uv run suspkin report  configs/vehicles/fs_reference.yaml --out report/
```

or in Python:

```python
import suspkin as sk

vehicle = sk.load_vehicle("configs/vehicles/fs_reference.yaml")
print(sk.vehicle_summary(vehicle))
```

## Where next

- [Tutorial](getting-started/tutorial.md): the CLI and the Python API on the reference car.
- [Vehicle YAML](guides/vehicle-yaml.md) and [Study YAML](guides/study-yaml.md): file formats.
- [Python API](reference/api.md) and [Command line](guides/cli.md).
- [Surrogate optimisation](guides/surrogate.md) (experimental): neural-network surrogate for many-variable studies.
- [Architecture & math](ARCHITECTURE.md): conventions, formulas and module reference.
- [Stability and versioning](guides/stability.md): what is covered by SemVer.
- Source: [github.com/canc-gmy/suspkin](https://github.com/canc-gmy/suspkin).

Conventions: ISO 8855 frame (x forward, y left, z up, origin on the ground at mid front
axle), SI units in files and in code; mm and degrees only on the command line and in displays.
