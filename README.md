# suspkin

Suspension kinematics toolkit for Formula Student double-wishbone pushrod cars.
It computes kinematic curves (camber, toe, caster, anti-dive/anti-squat, roll centre,
motion ratios, steering geometry) from hardpoint geometry and turns them into
vehicle-dynamics plots and reports. Planned later: geometry optimization, FEM
checks of key parts, and a neural-network surrogate for fast design sweeps.

## Conventions

- ISO 8855 frame: x forward, y left, z up. Origin on the ground, mid front axle.
- SI units internally and in YAML (m, rad, N, N/m). mm and deg are display only.

## Setup

```bash
uv sync
uv run suspkin version
uv run pytest
```

## Development

```bash
uv run pre-commit install
```

Design plan: [docs/DESIGN_PLAN.md](docs/DESIGN_PLAN.md)
