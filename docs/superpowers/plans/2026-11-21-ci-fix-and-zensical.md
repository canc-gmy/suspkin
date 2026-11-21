# CI Date Fix and Zensical Docs Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get `main` green again, and replace MkDocs with Zensical so the site built locally with `uv run zensical build` is the one GitHub Pages serves at https://canc-gmy.github.io/suspkin/.

**Architecture:** The CI failure comes from the test harness, not the package: the behaviour lock does not tokenize a bare `YYYY-MM-DD`, and `suspkin report` writes today's date, so the two report lock cases only pass on the day they were recorded (2026-11-20). The fix adds one scrub rule. For the docs, Zensical replaces MkDocs with a native `zensical.toml` (MkDocs `hooks:` are not supported). CI and the Pages workflow run the same `uv run zensical build` command against the same `uv.lock`, so the CI build matches the local one.

**Tech Stack:** Zensical 0.1.x (`zensical`, `mkdocstrings-python`), uv, GitHub Actions (`configure-pages`, `upload-pages-artifact`, `deploy-pages`), pytest.

**Spec:** No separate spec document. The user's request (2026-11-21): fix the CI that keeps failing; migrate the docs from MkDocs to Zensical; publish the Zensical build on GitHub Pages. The facts below were established on 2026-11-21 and are part of the spec.

### Established facts

- CI run 44909063946 (`main`, commit 179ff39): all 9 `test` jobs fail on `test_cli_lock[report]` and `test_cli_lock[report_compare]` with `report.html: text differs`. It reproduces locally. The only difference is `<td>Date</td><td>2026-11-20</td>` against today's date. The source is `src/suspkin/report/builder.py:210` (`datetime.date.today().isoformat()`).
- The earlier red PR runs (macOS paths, help text) were fixed by 44fdffb and are not part of this plan.
- Four stored snapshots contain the literal date, one line each: `snapshots/cli_report{,_compare}/files/rep/report.{html,md}` (html line 19, md line 4, `- Date: 2026-11-20`).
- GitHub Pages is already set to `build_type: workflow`. The `pages build and deployment` runs are the `deploy-pages` action, so no repository setting changes.
- Zensical 0.1.3 (PyPI, 2026-11-20, Python >= 3.11) reads `zensical.toml` (top-level `[project]`). It supports mkdocstrings natively (`mkdocstrings-python`), `pymdownx.snippets`, `pymdownx.arithmatex` (`generic = true`), superfences mermaid, `extra_javascript` (local and URL), `exclude_docs` (a gitignore-style multiline string), `[project.validation] invalid_links` / `invalid_link_anchors` (on by default) and `zensical build --strict` (exit 1). It does **not** support MkDocs `hooks:`.
- `docs/hooks/source_links.py` rewrites relative links that leave `docs/`. A scan on 2026-11-21 found zero such links, so dropping the hook changes no rendered page.

## Global Constraints

- English only; Conventional Commits; only the orchestrator commits (CLAUDE.md).
- Never regenerate `tests/regression/behaviour/` snapshots with `--update-lock`. The only snapshot edit allowed is the date-to-`<DATE>` token swap in Task 1, on the four listed lines.
- Call uv as `python -m uv` locally.
- Pin `zensical>=0.1.3,<0.2`. It is pre-1.0, so minor bumps may break.
- Keep the URL `https://canc-gmy.github.io/suspkin/` and the existing page paths (the nav is carried over 1:1).
- Theme: `variant = "classic"`, so the site keeps the current Material look. Switching to Zensical's `modern` look is a later one-line change.
- Never commit `site/`. It stays in `.gitignore`. Pages is fed by the workflow artifact.

## Review Focus

1. **A bare date in another output.** Any locked text file with a `YYYY-MM-DD` that is not a full timestamp has to tokenize to `<DATE>`. Task 1's guard test fails if any stored snapshot still contains a literal date.
2. **mkdocstrings options in TOML silently ignored.** The `[project.plugins.mkdocstrings.handlers.python]` table name is inferred, not documented. If it is wrong, the API pages render empty and the build still succeeds. Task 2 checks that `solve_corner` appears in the built `reference/kinematics` page.
3. **Snippet includes.** `docs/changelog.md` includes `CHANGELOG.md`, and `guides/examples.md` includes three `examples/*.py` files. A wrong `base_path` gives empty pages, not an error. Task 2 checks for text from each included file in the output.
4. **A future relative link out of `docs/`.** Without the hook, a link like `../../src/x.py` breaks on the site. Task 2 adds a pytest guard that fails on such a link.
5. **Math and mermaid.** `document$` in `docs/javascripts/mathjax.js` is a Material global. Task 2 checks that the built HTML still loads `mathjax.js` and that pages with `$$` keep their `arithmatex` spans.

---

### Task 1: Make the report lock independent of today's date

**Files:**
- Modify: `tests/regression/behaviour/_lock.py` (add `_DATE` beside `_ISO`, near line 52; apply it in `scrub()` right after the `_ISO` line, near line 121)
- Modify (token swap only): the four snapshot files listed in Established facts
- Create: `tests/regression/behaviour/test_lock_scrub.py`

**Interfaces:**
- Consumes: `scrub(text: str, *, tmp: Path) -> str`, `SNAP: Path` from `_lock.py`.
- Produces: the token `<DATE>` for a bare ISO date. Nothing outside the test harness changes.

- [ ] **Step 1: Write the failing tests** in `test_lock_scrub.py`

```python
def test_bare_date_is_tokenised(tmp_path):
    assert scrub("<td>Date</td><td>2026-11-20</td>", tmp=tmp_path) == "<td>Date</td><td><DATE></td>"
    assert scrub("- Date: 1999-01-02\n", tmp=tmp_path) == "- Date: <DATE>\n"

def test_full_timestamp_still_ts(tmp_path):
    assert scrub("at 2026-11-20T13:28:58Z", tmp=tmp_path) == "at <TS>"

def test_no_literal_date_in_snapshots():
    hits = [p for p in SNAP.rglob("*") if p.is_file()
            and re.search(r"\b\d{4}-\d{2}-\d{2}\b", p.read_text(encoding="utf-8", errors="ignore"))]
    assert hits == []
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m uv run pytest -q tests/regression/behaviour/test_lock_scrub.py`
Expected: `test_bare_date_is_tokenised` and `test_no_literal_date_in_snapshots` FAIL (the second lists the 4 report files). `test_full_timestamp_still_ts` PASSES.

- [ ] **Step 3: Add `_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")` and apply `text = _DATE.sub("<DATE>", text)` right after the `_ISO` substitution in `scrub()`**

The order matters: `_ISO` has to consume full timestamps first.

- [ ] **Step 4: Swap the literal date for the token in the four stored snapshots**

Run: `sed -i '' 's/2026-11-20/<DATE>/' tests/regression/behaviour/snapshots/cli_report{,_compare}/files/rep/report.{html,md}`
Then: `git diff --numstat tests/regression/behaviour/snapshots`
Expected: exactly 4 files, `1 1` each. Nothing else changes.

- [ ] **Step 5: Run the new tests and the CLI lock**

Run: `python -m uv run pytest -q tests/regression/behaviour/test_lock_scrub.py tests/regression/behaviour/test_cli_lock.py`
Expected: all pass, including `test_cli_lock[report]` and `test_cli_lock[report_compare]`.

- [ ] **Step 6: Full gate (verifier)**

Run: `python -m uv run ruff check && python -m uv run ruff format --check && python -m uv run mypy src && python -m uv run pytest -q -m "not examples"`
Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add tests/regression/behaviour/_lock.py tests/regression/behaviour/test_lock_scrub.py tests/regression/behaviour/snapshots/cli_report tests/regression/behaviour/snapshots/cli_report_compare
git commit -m "test(regression): tokenise bare dates so the report lock does not expire daily"
```

---

### Task 2: Replace MkDocs with Zensical

**Files:**
- Create: `zensical.toml`
- Delete: `mkdocs.yml`, `docs/hooks/source_links.py` (and `docs/hooks/`)
- Modify: `pyproject.toml:41-44` (`[dependency-groups] docs`), `uv.lock` (via `uv lock`)
- Modify: `.pre-commit-config.yaml:28` (drop `args: [--unsafe]` and its comment, since no YAML uses `!!python/name` any more)
- Modify: `docs/guides/development.md` lines 52, 145-149, 193, 356 (commands become `uv run zensical build --strict`, add `uv run zensical serve` for local preview, drop the `--unsafe` row's reason), `README.md:4` badge (`docs-zensical`)
- Create: `docs/adr/0009-zensical-docs.md` (short: context = MkDocs maintenance status and the Material team moving to Zensical; decision; consequences = no hooks, pre-1.0 pin). Add it to the nav.
- Create: `tests/unit/test_docs_links.py`

**Interfaces:**
- Consumes: the nav and settings of the current `mkdocs.yml` (carry them over 1:1).
- Produces: the build command `uv run zensical build --strict` with output in `site/`, and `uv sync --group docs` installing it. Task 3 uses both.

- [ ] **Step 1: Write the link guard test** `tests/unit/test_docs_links.py::test_no_relative_link_leaves_docs`

It walks `docs/**/*.md` (skipping `docs/superpowers/`) and fenced blocks, matches inline links `](target)` that are not absolute, `#` or `/`, normalises `docs/<page dir>/<path>`, and asserts that the list of targets not under `docs/` is `[]`. This is the same matching as the deleted hook, now as a check instead of a rewrite. It passes today and pins Review Focus 4.

- [ ] **Step 2: Swap the dependencies**

Set the `docs` group to `["zensical>=0.1.3,<0.2", "mkdocstrings-python>=1.0"]`, then run `python -m uv lock`.
Expected: `uv.lock` no longer contains `name = "mkdocs"` or `mkdocs-material`. Check with `grep -c 'name = "mkdocs' uv.lock` → `0`, unless `mkdocstrings-python` pulls `mkdocs-autorefs` in transitively, which is acceptable.

- [ ] **Step 3: Write `zensical.toml` and delete `mkdocs.yml` and the hook**

Map each `mkdocs.yml` key:
- `[project]`: `site_name`, `site_description`, `site_url`, `repo_url`, `repo_name`, `docs_dir = "docs"`, `exclude_docs = "superpowers/\n"`, `extra_javascript = ["javascripts/mathjax.js", "https://unpkg.com/mathjax@3.2.2/es5/tex-mml-chtml.js"]`, `nav` (same order and titles as today)
- `[project.theme]`: `variant = "classic"`, `features = [...]` (the same five), the two-entry `palette` with the same media, scheme, icon and name values
- `[project.validation]`: `invalid_links = true`, `invalid_link_anchors = true`
- `[project.markdown_extensions...]`: `admonition`, `tables`, `toc.permalink = true`, `pymdownx.arithmatex.generic = true`, `pymdownx.superfences.custom_fences = [{ name = "mermaid", class = "mermaid", format = "pymdownx.superfences.fence_code_format" }]`, `pymdownx.snippets = { base_path = ["."], check_paths = true }`
- `[project.plugins.mkdocstrings.handlers.python]`: `paths = ["src"]`, and `options` with the same five keys as today

- [ ] **Step 4: Build strictly**

Run: `python -m uv sync --all-extras --group docs && python -m uv run zensical build --strict --clean`
Expected: exit 0 and `site/index.html` exists. If an unknown key is reported, fix that key; do not drop `--strict`.

- [ ] **Step 5: Check the content for Review Focus 2, 3 and 5**

Run each and expect a non-zero count:
- `grep -c solve_corner site/reference/kinematics/index.html`
- `grep -c "Keep a Changelog" site/changelog/index.html`
- `grep -c "01_load_and_sweep\|def " site/guides/examples/index.html`
- `grep -l "javascripts/mathjax.js" site/concepts/corner-kinematics/index.html`
- `grep -c arithmatex site/concepts/corner-kinematics/index.html` (this page has `$$` blocks)

Then open `uv run zensical serve` in the browser pane: check that one formula renders as math and that the light/dark toggle works.

- [ ] **Step 6: Update the docs, README, ADR and CHANGELOG** (`### Changed` under Unreleased: "Documentation is built with Zensical instead of MkDocs"), then rebuild with `--strict`.

Expected: exit 0, and the new ADR page is in the nav.

- [ ] **Step 7: Full gate (verifier)** using the Task 1 Step 6 command, plus `python -m uv run pytest -q tests/unit/test_docs_links.py` and `pre-commit run check-yaml --all-files`.
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add -A zensical.toml mkdocs.yml docs pyproject.toml uv.lock .pre-commit-config.yaml README.md CHANGELOG.md tests/unit/test_docs_links.py
git commit -m "docs: build the site with Zensical instead of MkDocs"
```

---

### Task 3: Build and publish the Zensical site from CI

**Files:**
- Modify: `.github/workflows/ci.yml` job `docs` (last step becomes `uv run zensical build --strict --clean`)
- Modify: `.github/workflows/docs.yml`

**Interfaces:**
- Consumes: the Task 2 build command and the `site/` output.
- Produces: a deploy to `https://canc-gmy.github.io/suspkin/` on every docs-relevant push to `main`.

- [ ] **Step 1: Rewrite `docs.yml`**

Changes:
- In the `paths` trigger, replace `mkdocs.yml` with `zensical.toml` and add `pyproject.toml` and `uv.lock`, because a version bump must redeploy.
- Build job steps: `actions/checkout@v4`, `actions/configure-pages@v5`, `astral-sh/setup-uv@v5` (python 3.12, **no** `enable-cache`, as Zensical's publishing guide advises against CI caches), `uv sync --group docs --all-extras`, `uv run zensical build --strict --clean`, `actions/upload-pages-artifact@v3` with `path: site`.
- Keep the deploy job, permissions and concurrency as they are.

Use the action majors already used in this repo, unless the GitHub Marketplace shows the newer majors that Zensical's guide quotes (`upload-pages-artifact@v5`, `deploy-pages@v5`, `configure-pages@v6`) as released. Then use those, in a matched set.

- [ ] **Step 2: Validate the workflow syntax**

Run: `pre-commit run check-yaml --files .github/workflows/docs.yml .github/workflows/ci.yml`
Expected: Passed.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/docs.yml
git commit -m "ci(docs): build and deploy the Zensical site to GitHub Pages"
```

- [ ] **Step 4: After the user approves the push, verify on GitHub**

Run: `gh run list --limit 5`
Expected: `CI` and `Docs` both `success` for the pushed commit.
Then run `curl -s https://canc-gmy.github.io/suspkin/reference/kinematics/ | grep -c solve_corner`. Expected: greater than 0. Also check for a Zensical marker in the HTML, such as the generator meta tag, so you know the new build is live and not a cached MkDocs page.
