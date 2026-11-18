#!/usr/bin/env bash
# One-command setup for suspkin (Linux, macOS, Git Bash).
# Usage: bash scripts/setup.sh [--minimal] [--test] [--no-frontend]
set -euo pipefail

MINIMAL=0
RUN_TEST=0
NO_FRONTEND=0
for arg in "$@"; do
  case "$arg" in
    --minimal) MINIMAL=1 ;;
    --test) RUN_TEST=1 ;;
    --no-frontend) NO_FRONTEND=1 ;;
    -h|--help) echo "Usage: bash scripts/setup.sh [--minimal] [--test] [--no-frontend]"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 1. Python >= 3.11
step "Checking Python >= 3.11"
PY=()
for cand in "python3" "python" "py -3"; do
  # shellcheck disable=SC2086
  if $cand -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
    read -r -a PY <<< "$cand"
    break
  fi
done
if [ ${#PY[@]} -gt 0 ]; then
  echo "Using: ${PY[*]} ($("${PY[@]}" --version 2>&1))"
fi

# 2. uv
step "Checking uv"
UV=()
if command -v uv >/dev/null 2>&1; then
  UV=(uv)
elif [ ${#PY[@]} -gt 0 ] && "${PY[@]}" -m uv --version >/dev/null 2>&1; then
  UV=("${PY[@]}" -m uv)
elif [ ${#PY[@]} -gt 0 ]; then
  echo "uv not found, installing with pip (--user)"
  "${PY[@]}" -m pip install --user uv || fail "pip install --user uv failed"
  "${PY[@]}" -m uv --version >/dev/null 2>&1 || fail "uv installed but not importable"
  UV=("${PY[@]}" -m uv)
else
  fail "Python 3.11+ not found and uv not available. Install Python 3.11+ (https://www.python.org/downloads/, 'brew install python@3.12' or 'sudo apt install python3 python3-pip') or uv (https://docs.astral.sh/uv/), then re-run."
fi
echo "Using: ${UV[*]}"

if [ ${#PY[@]} -eq 0 ]; then
  step "No suitable Python found, fetching one with uv"
  "${UV[@]}" python install 3.12 || fail "uv python install 3.12 failed"
fi

# 3. Dependencies
if [ "$MINIMAL" -eq 1 ]; then
  step "uv sync (minimal)"
  "${UV[@]}" sync || fail "uv sync failed"
else
  step "uv sync --all-extras"
  "${UV[@]}" sync --all-extras || fail "uv sync --all-extras failed"
fi

# 4. pre-commit hooks
if [ -d .git ] && [ -f .pre-commit-config.yaml ]; then
  step "Installing pre-commit hooks"
  "${UV[@]}" run pre-commit install || fail "pre-commit install failed"
fi

# 5. Frontend (optional, needs Node 20+)
if [ "$NO_FRONTEND" -eq 1 ]; then
  step "Skipping frontend build (--no-frontend)"
elif [ ! -f frontend/package.json ]; then
  echo "No frontend/package.json, skipping frontend build"
elif ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "node/npm not found, skipping frontend build (the web app UI needs Node 20+)"
else
  step "Building the frontend"
  (cd frontend && npm ci && npm run build && npm run build:single) || fail "frontend build failed"
fi

# 6. Smoke check
step "Smoke check"
"${UV[@]}" run suspkin version || fail "suspkin version failed"

# 7. Tests
if [ "$RUN_TEST" -eq 1 ]; then
  step "Running tests"
  "${UV[@]}" run pytest -q || fail "pytest failed"
fi

UVTXT="${UV[*]}"
cat <<NEXT

Setup complete. Next steps:
  $UVTXT run suspkin --help
  $UVTXT run suspkin check <config.yaml>
  $UVTXT run suspkin optimize configs/studies/fs_reference_optimize.yaml
  $UVTXT run pytest -q
NEXT
