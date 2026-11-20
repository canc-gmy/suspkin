# Install

`suspkin` is installed from a clone of the repository. It is not on PyPI.

## Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.11 or newer (CI runs 3.11, 3.12, 3.13; the development default is 3.12) | everything |
| git | any recent | cloning |
| [uv](https://docs.astral.sh/uv/) | any recent | recommended environment manager (optional, `pip` works) |
| Node.js | 22.12 or newer | only to build the web app frontend |
| Git LFS | any recent | only to download the committed surrogate network weights |

```bash
git clone https://github.com/canc-gmy/suspkin.git && cd suspkin
```

On some Windows machines `uv` is not on `PATH`. Everywhere in these docs you can then write
`python -m uv` instead of `uv` (same program, started through the Python that has it).

## One command

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

```bash
# Linux / macOS / Git Bash
bash scripts/setup.sh
```

The script checks Python, finds or installs uv, runs `uv sync --all-extras`, installs the
pre-commit hooks, builds the frontend when Node is available and finishes with
`suspkin version`. Flags: `--minimal` / `-Minimal` runs a plain `uv sync` (core dependencies
only); `--test` / `-Test` also runs the test suite; `--no-frontend` / `-NoFrontend` skips the
frontend build.

**macOS.** Apple silicon Macs get everything, including the neural surrogate (CI tests them on
every pull request). On Intel Macs the script installs every extra except `surrogate`, because
torch has no Intel Mac build in the supported range (`torch>=2.5,<2.7`).

## Manual install with uv

```bash
uv sync                       # core + dev tools (pytest, ruff, mypy ...), no extras
uv sync --all-extras          # everything below
uv sync --extra viz --extra opt   # pick extras one by one
uv run suspkin version
```

## Manual install with pip

```bash
python -m venv .venv
# activate it: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/macOS)
pip install -e .                       # core only
pip install -e ".[viz,opt,app,fast]"   # with extras
pip install -e ".[surrogate]"          # adds torch and optuna
suspkin version
```

The dev tools (pytest, ruff, mypy) live in a uv *dependency group*, not in an extra, so with
`pip` use `uv sync` if you want to run the test suite.

## Extras

The core install (numpy, scipy, pandas, pydantic, pyyaml, matplotlib, typer) runs every stable
command: `show`, `animate`, `sweep`, `summary`, `vehicle`, `vsweep`, `report`, `check`, and
`optimize` with the `de`, `slsqp`, `trust-constr` and `de+slsqp` methods. A missing extra only
breaks the feature that needs it, and the message says which extra to add.

| Extra | Packages | What it enables | Without it |
|---|---|---|---|
| `viz` | plotly | `suspkin show --html` (interactive 3D); the 3D tab of the web app | those features fail with an `ImportError` / exit message; matplotlib views still work |
| `opt` | pymoo | the `nsga2` optimiser (and so `pareto` mode, which only runs with it) | the other optimisers still work |
| `app` | fastapi, uvicorn | `suspkin app` and `suspkin app-export` (the web app backend) | the two commands exit with an install hint |
| `fast` | numba | a compiled corner solver, about 1.8 times faster for `run_standard_studies` (94 ms against 166 ms at 61 points, see [Corner kinematics, section 6](../concepts/corner-kinematics.md#6-performance)) | pure-Python solver, same results (within the solver tolerance) |
| `surrogate` | torch, optuna | `surrogate train`, `surrogate validate`, `optimize --surrogate` (experimental) | `surrogate sample` and the exact-model oracle still work |

Set `SUSPKIN_NO_NUMBA=1` to force the pure-Python solver even when numba is installed.

### Torch index notes

The `surrogate` extra pins `torch>=2.5,<2.7`. With uv, `pyproject.toml` routes torch to a
specific wheel index per platform (`[tool.uv.sources]`):

| Platform | Torch build | Size |
|---|---|---|
| Windows | CUDA 12.4 (`pytorch-cu124`) | about 2.5 GB download |
| Linux | CPU only (`pytorch-cpu`) | small |
| macOS | default PyPI wheel (Apple silicon only) | moderate |

Plain `pip` ignores `[tool.uv.sources]` and takes the default PyPI wheel (CUDA-enabled on
Linux). Install a CPU build yourself first
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) if you want a small
install. Training on CPU works but is slow, and active learning skips fine-tuning on CPU by
default (see the [surrogate guide](../guides/surrogate.md)). Check that CUDA is visible:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Web app frontend

The browser UI is a separate Node project in `frontend/`; its build output is git-ignored but
shipped in wheels. The Python side and the API work without it (the root page then shows a
hint), but the UI and snapshot export need it:

```bash
cd frontend
npm ci
npm run build          # app served by `suspkin app`   -> src/suspkin/app/static
npm run build:single   # single-file build for export  -> src/suspkin/app/static_single
```

Both builds take a few seconds. `scripts/setup.*` run them for you when `node` and `npm` are
on `PATH`. See the [web app guide](../guides/app.md).

## Git LFS for the committed surrogate models

The trained benchmark networks in
`results/benchmark_surrogate/surrogate/models/{front,rear}/model/model.pt` (about 76 MB and
22 MB) are stored with Git LFS (`.gitattributes`). Without Git LFS a clone contains small text
pointer files instead of weights, and loading them fails. Install it once and fetch the real
files:

```bash
git lfs install
git lfs pull
git lfs ls-files        # a "*" after the object id means the file is downloaded, "-" a pointer
```

Skip this unless you want to run `optimize --surrogate` against the committed models or
inspect them; nothing else needs the weights.

## Verify the install

```bash
uv run suspkin version                                        # suspkin 1.1.0
uv run suspkin summary configs/vehicles/fs_reference.yaml    # a table of FL / RL metrics
uv run pytest tests/unit/test_public_api.py -q                # 27 passed, a few seconds
uv run pytest -q -m "not examples"                            # the full suite (what CI runs)
```

If `summary` prints a table starting with `camber  -1.0000  -1.5000  deg`, the solver and the
YAML loader work. Which extras are importable:

```bash
uv run python -c "import numba, plotly, pymoo, fastapi, torch, optuna; print('all extras present')"
```

## Build these docs

```bash
uv sync --group docs --all-extras
uv run mkdocs serve          # live preview
uv run mkdocs build --strict
```

Next: the [tutorial](tutorial.md).
