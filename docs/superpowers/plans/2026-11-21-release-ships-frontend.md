# Release Ships the Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The wheel and sdist attached to a GitHub release contain the built web app, so `pip install suspkin-X.whl[app]` followed by `suspkin app` works on a machine with no Node. Then release this as v2.0.0.

**Architecture:** Two independent defects, both confirmed on 2026-11-21:
1. `release.yml` never builds the frontend, so the git-ignored `src/suspkin/app/static*/` folders do not exist when `uv build` runs.
2. Even with the frontend built, `uv build` produces a wheel and sdist with **no** frontend files (0 entries). `uv build` builds the sdist first and then the wheel from that sdist. The sdist target respects `.gitignore` and has no `artifacts` entry, so the files are lost. `uv build --wheel` keeps them (41 entries).

The fix is to build the frontend in the release job, add `artifacts` to the sdist target, and add a small dist-check script. The script runs in CI (on every push or PR) and in the release job, so neither defect can come back silently.

**Tech Stack:** GitHub Actions, hatchling (`[tool.hatch.build.targets.*]`), uv, Node 22 / Vite, pytest.

**Spec:** No spec doc. The request from chat: "add it to the release workflow so you can use the app even if you dont have node installed and push it as version 1.2".

## Global Constraints

- Node 22 with `actions/setup-node@v4`, `cache: npm`, `cache-dependency-path: frontend/package-lock.json`, the same as the `frontend` job in `ci.yml`.
- Frontend build commands, run in `frontend/`: `npm ci`, `npm run build` (→ `src/suspkin/app/static`), `npm run build:single` (→ `src/suspkin/app/static_single`).
- Release version string (MAJOR: the unreleased changes rename Python import paths, see development.md §7; chosen by the user over the requested 1.2): `2.0.0` in `pyproject.toml`, tag `v2.0.0`, CHANGELOG heading `## [2.0.0] - <release date>`.
- All work happens on branch `release-frontend`, cut from `main` before Task 1. This plan file goes in the Task 1 commit.
- `main` stays green. Conventional Commits. Only the orchestrator commits. Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never regenerate `tests/regression/behaviour/` or `tests/regression/golden/`.
- `uv` may not be on PATH locally: use `python -m uv`.

## Review Focus

1. **Installing from the sdist** (`pip install suspkin-2.0.0.tar.gz[app]`) must give a working app too. Task 1 tests the sdist layout, and Task 2 checks the real sdist.
2. **`index.html` present but `assets/` empty or missing** gives a blank page, not an error. The script requires at least one `suspkin/app/static/assets/` member (Task 1 test `test_missing_assets_reported`).
3. **The defect coming back on a PR long before the next release.** The check runs in the CI `frontend` job (Task 3), not only on tag push.
4. **Tag pushed before the release commit is on `main`.** The release job's ancestor check fails. Task 4 orders the steps: merge, pull, then tag.
5. **"Works without Node" verified only inside the repo**, where `src/` is on the path and the frontend is served from the checkout. Task 4 installs the downloaded release wheel into a fresh venv outside the repo and checks that the page is served from `site-packages`.

---

### Task 1: Dist frontend check script

**Files:**
- Create: `scripts/check_dist_frontend.py`
- Test: `tests/unit/test_check_dist_frontend.py`

**Interfaces:**
- Produces (used by Tasks 2–4 as a CLI only): `python scripts/check_dist_frontend.py <dist files...>` exits 0 when every file contains the frontend, 1 if any file lacks it, and 2 for no arguments or an unsupported file type. Prints one line per missing entry: `<file>: missing <entry>`.
- Module API:
  - `REQUIRED: tuple[str, ...] = ("suspkin/app/static/index.html", "suspkin/app/static_single/index.html")`
  - `ASSETS_PREFIX = "suspkin/app/static/assets/"`
  - `archive_members(path: Path) -> list[str]`: member names of a `.whl` (zipfile) or `.tar.gz` (tarfile). Raises `ValueError` for other suffixes.
  - `missing_frontend(path: Path) -> list[str]`: the `REQUIRED` entries not found, plus `ASSETS_PREFIX` if no member lies under it. A member matches entry `e` when `name == e` (wheel layout) or `name.endswith("/" + e)` (sdist layout, `suspkin-2.0.0/src/suspkin/...`). For `ASSETS_PREFIX`, use the same rule with "starts with" / "contains `/` + prefix".
  - `main(argv: list[str]) -> int`

- [ ] **Step 1: Write the failing tests.** Load the script with `importlib.util.spec_from_file_location`, following the `_load_script()` pattern in `tests/unit/test_benchmark_study.py`. Build tiny archives in `tmp_path`.

```python
FULL = ["suspkin/__init__.py", "suspkin/app/static/index.html",
        "suspkin/app/static/assets/index-x.js", "suspkin/app/static_single/index.html"]

def test_complete_wheel_ok(tmp_path):          # zip with FULL -> missing_frontend == []
def test_complete_sdist_ok(tmp_path):          # tar.gz with "suspkin-2.0.0/src/" + each of FULL -> []
def test_missing_single_reported(tmp_path):    # wheel without static_single -> == ["suspkin/app/static_single/index.html"]
def test_missing_assets_reported(tmp_path):    # wheel without the assets file -> == ["suspkin/app/static/assets/"]
def test_empty_wheel_reports_all(tmp_path):    # only suspkin/__init__.py -> len == 3
def test_unsupported_suffix_raises(tmp_path):  # foo.zip -> pytest.raises(ValueError)
def test_main_exit_codes(tmp_path, capsys):    # ok -> 0; one bad of two -> 1 and "missing" in stdout; [] -> 2; foo.zip -> 2
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m uv run pytest tests/unit/test_check_dist_frontend.py -q`. Expected: FAIL (the script file does not exist).
- [ ] **Step 3: Implement `scripts/check_dist_frontend.py`** with the interfaces above. Stdlib only (`zipfile`, `tarfile`, `sys`, `pathlib`), a Google-style module docstring, and `if __name__ == "__main__": sys.exit(main(sys.argv[1:]))`.
- [ ] **Step 4: Run the tests and the gate.** Run `python -m uv run pytest tests/unit/test_check_dist_frontend.py -q`, then `python -m uv run ruff check`, then `python -m uv run ruff format --check`. Expected: all pass.
- [ ] **Step 5: Commit** `test: add dist check that the built web app is packaged`

### Task 2: Ship the frontend in the sdist (and the wheel built from it)

**Files:**
- Modify: `pyproject.toml:65-68` (the hatch build targets)

**Interfaces:**
- Consumes: the Task 1 CLI.

- [ ] **Step 1: Reproduce the defect.** Build the frontend (`cd frontend && npm ci && npm run build && npm run build:single`). Then run `python -m uv build -o <scratch>/dist` and `python -m uv run python scripts/check_dist_frontend.py <scratch>/dist/*`. Expected: exit 1, with both files reporting all three entries missing.
- [ ] **Step 2: Add the sdist artifacts.** Add the table below next to the wheel one, with the same comment style.

```toml
[tool.hatch.build.targets.sdist]
artifacts = ["src/suspkin/app/static/**", "src/suspkin/app/static_single/**"]
```

- [ ] **Step 3: Rebuild and check.** Run the same two commands as Step 1 into a fresh folder. Expected: exit 0 for both the `.whl` and the `.tar.gz`. If the wheel built from the sdist still misses files, stop and report: do not switch to `--wheel` only, because the sdist must also install a working app (Review Focus 1).
- [ ] **Step 4: Run the gate.** `verifier`: ruff, mypy, pytest. Expected: green.
- [ ] **Step 5: Commit** `fix(build): include the built web app in the sdist`

### Task 3: Build the frontend in the release job, guard it in CI, document it

**Files:**
- Modify: `.github/workflows/release.yml` (the steps between `Verify the tagged commit is on main` and `uv build`)
- Modify: `.github/workflows/ci.yml` (the `frontend` job, after `uv run pytest ... test_app_snapshot.py`)
- Modify: `docs/guides/development.md` §7 "What the release workflow does"
- Modify: `docs/guides/app.md` around line 19 (the "build once" paragraph)
- Modify: `CHANGELOG.md` `[Unreleased]`

**Interfaces:**
- Consumes: the Task 1 CLI and the Task 2 packaging.

- [ ] **Step 1: Update `release.yml`.** After the "on main" check and before `uv sync --all-extras`, add the setup-node step and `npm ci` / `npm run build` / `npm run build:single`, each with `working-directory: frontend`, exactly as in `ci.yml`'s `frontend` job. Building before pytest means the app tests run against the built UI. After `uv build`, add a step named `Verify the web app is packaged` that runs `uv run python scripts/check_dist_frontend.py dist/*.whl dist/*.tar.gz`.
- [ ] **Step 2: Update `ci.yml`.** Append two steps to the `frontend` job: `uv build -o dist`, then `uv run python scripts/check_dist_frontend.py dist/*.whl dist/*.tar.gz`.
- [ ] **Step 3: Update the docs.** In `development.md` §7, say that the workflow builds the frontend (Node 22) before testing and checks the wheel and sdist contain it. In `app.md`, say that a release wheel already contains the UI and needs no Node, and that the Node build is only needed when running from a source checkout. Add this line under `### Fixed` in `CHANGELOG.md` `[Unreleased]` (create the subsection): "Release wheels and sdists now contain the built web app, so `suspkin app` and `suspkin app-export` work from an installed release without Node (they were missing in 1.1.0)."
- [ ] **Step 4: Validate.** Run `python -m uv run zensical build --strict --clean`. Expected: "No issues found". Confirm both workflow files parse: `python -c "import yaml,sys;[yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/workflows/release.yml .github/workflows/ci.yml`.
- [ ] **Step 5: Commit** `ci(release): build the web app before packaging and check the dists`

### Task 4: Release v2.0.0

**Files:**
- Modify: `pyproject.toml:7` (`version = "2.0.0"`), `uv.lock` (via `python -m uv lock`), `CHANGELOG.md` (`[Unreleased]` → `[2.0.0] - <date>`, new empty `[Unreleased]` above it)

- [ ] **Step 1: Prepare the release.** Bump the version, run `python -m uv lock`, rename the CHANGELOG section, and commit `chore(release): 2.0.0`.
- [ ] **Step 2: Open the PR.** Push the branch `release-frontend` (Tasks 1–4 commits) and open a PR to `main`. Expected: the CI `frontend` job passes, including the new dist check.
- [ ] **Step 3: Merge and tag.** Merge the PR, then run `git checkout main && git pull`, `git tag v2.0.0`, and `git push origin v2.0.0`. Expected: the Release workflow succeeds, and `gh release view v2.0.0` lists `suspkin-2.0.0-py3-none-any.whl` and `suspkin-2.0.0.tar.gz`.
- [ ] **Step 4: Check the published files.** Run `gh release download v2.0.0` into a new empty scratch folder, then `python scripts/check_dist_frontend.py <folder>/*`. Expected: exit 0.
- [ ] **Step 5: Test a real install.** Outside the repo, create a fresh venv and install `"<wheel>[app]"`. Run `suspkin app --no-browser --port 8799` in the background, then `curl -s http://127.0.0.1:8799/`. Expected: the response contains `<div id="root">` and `/assets/index-`. Also confirm that `python -c "import suspkin.app.server as s;print(s.DEFAULT_STATIC_DIR)"` points into the venv's `site-packages`. Then stop the server.
