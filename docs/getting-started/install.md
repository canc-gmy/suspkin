# Install

Prerequisites: git and Python 3.11 or newer. [uv](https://docs.astral.sh/uv/) is optional;
the setup scripts install it when missing.

```bash
git clone https://github.com/canc-gmy/suspkin.git && cd suspkin
```

## One command

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

```bash
# Linux / macOS / Git Bash
bash scripts/setup.sh
```

Flags: `-Minimal` / `--minimal` skips the optional extras; `-Test` / `--test` also runs the
test suite.

**macOS:** Apple silicon Macs get everything, including the NN surrogate; CI tests them on
every pull request. On Intel Macs the script installs every extra except `surrogate`, because
torch has no Intel Mac build in the supported range. The web app frontend needs Node 22.12+
(`brew install node`).

## Manual

```bash
uv sync --all-extras        # package, dev tools and the viz + opt extras
uv run suspkin version
```

Where `uv` is not on `PATH`, use `python -m uv` instead.

| Extra | Adds | Needed for |
|---|---|---|
| `viz` | plotly | `suspkin show --html`, interactive 3D |
| `opt` | pymoo | the `nsga2` optimiser |

## Build these docs

```bash
uv sync --group docs --all-extras
uv run mkdocs serve          # live preview
uv run mkdocs build --strict
```
