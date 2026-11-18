# ADR 0001: web app architecture

Status: accepted (experimental, branch `feat/webapp`, not on `main`).

## Context

Reporting was static (PNG, SVG, CSV, HTML). We want a local app that shows every graph
interactively and overlays several results to compare configurations. It is experimental, so
large changes to the UI must stay cheap after the first iteration.

## Decision

- **Backend**: FastAPI (`suspkin.app`, extra `app`). **Frontend**: React + TypeScript + Vite with
  Plotly.js, in `frontend/`.
- **Three contracts** so either side can change alone:
    1. the **result bundle**, a reloadable on-disk format for one design's results;
    2. the **figure spec**, a declarative list of curves defined in Python
       (`plotting/spec.py`), built from the same panels as the matplotlib figures; the frontend
       renders it generically, so a new graph needs no frontend code;
    3. a versioned **JSON API** `/api/v1` whose OpenAPI schema generates the TypeScript types.
- **DataSource**: the frontend talks to the API only through a swappable data source, `Http`
  (live server) or `Snapshot` (data embedded in a single HTML file for sharing).
- **Design system** in `frontend/DESIGN.md` is the single source; `tokens.css` and
  `plotlyTemplate.ts` implement it.

## Alternatives considered

- **Dash, Panel, Streamlit**: Python only and quick, but the look and interaction (linked hover,
  per-curve control, drawing-office styling) are hard to control, and a static shareable snapshot
  is awkward.
- **No-build JavaScript** (plain ES modules): no Node needed, but no typed API client, weaker
  tooling and tests, and harder to grow.

## Consequences

- Node 22.12+ is needed to build the frontend; Python-only users are not affected.
- The static build is git-ignored and built in CI and for releases; the Python tests do not need
  it.
- The app is experimental: no stability promise, and it stays on the branch until it is mature.
- API changes need `npm run gen:api` so the TypeScript types follow.
