# ADR 0009: Zensical as the documentation builder

Status: accepted. Date: 2026-11-21.

## Context

The site was built with MkDocs and the Material theme. MkDocs is in a maintenance-only state
and the Material team now develops Zensical, a successor that reads Material-style projects,
ships the same look and reimplements mkdocstrings natively. Staying on MkDocs would leave the
docs toolchain without a future.

## Decision

Build the site with Zensical (`zensical>=0.1.3,<0.2`, plus `mkdocstrings-python`). The
configuration moved 1:1 from `mkdocs.yml` to `zensical.toml`: nav, classic theme variant,
palette, features, MathJax, mermaid fences, snippets and the mkdocstrings Python options.
The build command is `uv run zensical build --strict --clean`, with link and anchor
validation (`invalid_links`, `invalid_link_anchors`) turned on so warnings fail the build.

## Consequences

* MkDocs hooks are unsupported, so `docs/hooks/source_links.py` (which rewrote links leaving
  `docs/` to GitHub URLs) is deleted. `tests/unit/test_docs_links.py` instead fails if any
  page links outside `docs/`; use absolute GitHub URLs for such targets.
* Zensical is pre-1.0, so the version is pinned below 0.2 and upgrades are deliberate.
* `check-yaml` no longer needs `--unsafe`: no YAML file uses `!!python/name` tags.
