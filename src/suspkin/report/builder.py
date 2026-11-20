"""Build a report folder (CSV, JSON, figures, HTML, Markdown) from study results."""

from __future__ import annotations

import base64
import datetime
import html
import io
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from suspkin import __version__
from suspkin.geometry import Vehicle
from suspkin.plotting.figures import FIGURE_SET, _lookup, save_figure
from suspkin.plotting.units import SUMMARY_ROWS, format_value
from suspkin.report.bundle import json_safe, save_bundle
from suspkin.report.studies import StudyResults

Designs = Sequence[tuple[str, StudyResults]]
_Section = tuple[str, str, str, str]  # (title, mime, base64 data, relative link)

# FIGURE_SET key -> section title
FIGURE_TITLES: dict[str, str] = {
    "corner_alignment": "Wheel alignment",
    "corner_position": "Wheel position",
    "corner_centres": "Geometry centres",
    "actuation": "Actuation and rates",
    "steering": "Steering",
    "roll": "Roll",
    "heave_pitch": "Heave and pitch",
    "summary_table": "Summary table",
}
_AXLE_FIGURES = ("corner_alignment", "corner_position", "corner_centres", "actuation")

_CSS = """
body{font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:0 auto;
padding:1.5rem;color:#222}
h1{font-size:1.7rem;margin-bottom:.2rem}
h2{margin-top:2.2rem;border-bottom:1px solid #ccc;padding-bottom:.2rem;font-size:1.25rem}
table{border-collapse:collapse;margin:.8rem 0}
th,td{padding:.25rem .8rem;border-bottom:1px solid #e2e2e2;text-align:right}
th:first-child,td:first-child,td.unit{text-align:left}
th{background:#f4f4f4}
.meta td{text-align:left}
img{max-width:100%;height:auto}
@media print{body{max-width:none;padding:0}h2{break-after:avoid}img{break-inside:avoid}}
"""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "design"


def _make_figures(designs: Designs) -> list[tuple[str, str, Figure]]:
    """(file stem, section title, figure) for every FIGURE_SET entry."""
    out: list[tuple[str, str, Figure]] = []
    for key in _AXLE_FIGURES:
        for axle, attr in (("front", "front_corner"), ("rear", "rear_corner")):
            ds = [(lab, getattr(r, attr)) for lab, r in designs]
            title = f"{FIGURE_TITLES[key]} ({axle} corner)"
            out.append((f"{key}_{axle}", title, FIGURE_SET[key](ds, axle=axle)))
    racked = [(lab, r) for lab, r in designs if len(r.rack) > 0]
    if racked:
        fig = FIGURE_SET["steering"]([(lab, r.rack) for lab, r in racked])
        out.append(("steering", FIGURE_TITLES["steering"], fig))
    roll = FIGURE_SET["roll"]([(lab, r.roll) for lab, r in designs])
    out.append(("roll", FIGURE_TITLES["roll"], roll))
    hp = FIGURE_SET["heave_pitch"](
        [(lab, r.heave) for lab, r in designs], [(lab, r.pitch) for lab, r in designs]
    )
    out.append(("heave_pitch", FIGURE_TITLES["heave_pitch"], hp))
    for lab, r in designs:
        fig = FIGURE_SET["summary_table"](r.summary, title=f"Vehicle summary: {lab}")
        out.append((f"summary_table_{_slug(lab)}", f"{FIGURE_TITLES['summary_table']}: {lab}", fig))
    return out


def _png_bytes(fig: Figure, saved: Sequence[Path]) -> bytes:
    for p in saved:
        if p.suffix == ".png":
            return p.read_bytes()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _summary_rows(designs: Designs) -> list[tuple[str, str, list[str]]]:
    rows: list[tuple[str, str, list[str]]] = []
    for label, path, factor, unit, fmt in SUMMARY_ROWS:
        vals = [_lookup(r.summary, path) for _, r in designs]
        if all(v is None for v in vals):
            continue
        cells = [
            "-" if v is None or not math.isfinite(v) else format_value(fmt, v * factor)
            for v in vals
        ]
        rows.append((label, unit, cells))
    return rows


def _html_summary(designs: Designs) -> str:
    head = "".join(f"<th>{html.escape(lab)}</th>" for lab, _ in designs)
    body = "".join(
        f"<tr><td>{html.escape(n)}</td>"
        + "".join(f"<td>{c}</td>" for c in cells)
        + f'<td class="unit">{html.escape(u)}</td></tr>'
        for n, u, cells in _summary_rows(designs)
    )
    return f"<table><tr><th>Quantity</th>{head}<th>Unit</th></tr>{body}</table>"


def _md_summary(designs: Designs) -> str:
    lines = [
        "| Quantity | " + " | ".join(lab for lab, _ in designs) + " | Unit |",
        "|---|" + "---:|" * len(designs) + "---|",
    ]
    lines += [f"| {n} | " + " | ".join(c) + f" | {u} |" for n, u, c in _summary_rows(designs)]
    return "\n".join(lines)


def _three_d(vehicle: Vehicle, fig_dir: Path) -> list[_Section]:
    from suspkin.plotting.animation import animate_sweep
    from suspkin.plotting.geometry3d import plot_vehicle_3d
    from suspkin.sweeps import sweep_vehicle

    fig = plot_vehicle_3d(vehicle)
    png = fig_dir / "geometry_3d.png"
    fig.savefig(png, dpi=150)
    states = sweep_vehicle(vehicle, roll=np.radians(np.linspace(-2.0, 2.0, 9)))
    gif = animate_sweep(vehicle, states, fig_dir / "roll_sweep.gif", fps=8)
    return [
        ("3D geometry (static)", "image/png", _b64(png.read_bytes()), "figures/geometry_3d.png"),
        ("3D roll sweep (+-2 deg)", "image/gif", _b64(gif.read_bytes()), "figures/roll_sweep.gif"),
    ]


def build_report(
    results: Designs,
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("png", "svg"),
    include_3d: bool = True,
    vehicle_for_3d: Vehicle | None = None,
) -> Path:
    """Write the report folder and return the path of ``report.html``.

    Layout: ``data/<design>_<study>.csv`` (SI), ``summary.json`` (SI, per design),
    ``bundles/<design>/`` (one result bundle per design, see ``report.bundle``),
    ``figures/*`` (all ``FIGURE_SET`` figures, designs overlaid, in ``formats``; with
    ``include_3d`` also the static 3D view and a GIF of the roll sweep of
    ``vehicle_for_3d``, default the first design), ``report.html`` (self-contained,
    images embedded) and ``report.md`` (links to the figure files).
    """
    if not results:
        raise ValueError("results must contain at least one (label, StudyResults) pair")
    designs: Designs = list(results)
    labels = [lab for lab, _ in designs]
    by_slug: dict[str, list[str]] = {}
    for lab in labels:
        by_slug.setdefault(_slug(lab), []).append(lab)
    clashes = {k: v for k, v in by_slug.items() if len(v) > 1}
    if clashes:
        detail = "; ".join(f"{', '.join(map(repr, v))} -> {k!r}" for k, v in clashes.items())
        raise ValueError(
            f"design labels must be unique as file names (data/ and bundles/): {detail}"
        )
    out = Path(out_dir)
    data_dir, fig_dir = out / "data", out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for lab, res in designs:
        for name, df in res.tables().items():
            df.to_csv(data_dir / f"{_slug(lab)}_{name}.csv", index=False)
    summary = {lab: json_safe(res.summary.to_dict()) for lab, res in designs}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for lab, res in designs:
        save_bundle(res, out / "bundles" / _slug(lab), label=lab, overwrite=True)

    sections: list[_Section] = []
    for stem, title, fig in _make_figures(designs):
        saved = save_figure(fig, fig_dir / stem, formats=formats)
        link = f"figures/{saved[0].name}" if saved else ""
        sections.append((title, "image/png", _b64(_png_bytes(fig, saved)), link))
    if include_3d:
        sections = _three_d(vehicle_for_3d or designs[0][1].vehicle, fig_dir) + sections

    meta = [
        ("suspkin version", __version__),
        ("Date", datetime.date.today().isoformat()),
        *[(f"Design {lab}", res.vehicle.vehicle.name) for lab, res in designs],
        ("Units", "figures and tables in mm, deg, N, N/mm; CSV and JSON files in SI"),
    ]
    (out / "report.html").write_text(_html_page(designs, meta, sections), encoding="utf-8")
    (out / "report.md").write_text(_markdown(designs, meta, sections), encoding="utf-8")
    return out / "report.html"


def _html_page(designs: Designs, meta: list[tuple[str, str]], sections: list[_Section]) -> str:
    esc = html.escape
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>suspkin report</title><style>{_CSS}</style></head><body>",
        "<h1>Suspension kinematics report</h1>",
        "<table class='meta'>",
        *[f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in meta],
        "</table>",
        "<h2>Summary comparison</h2>",
        _html_summary(designs),
    ]
    for title, mime, data, _ in sections:
        parts.append(f"<h2>{esc(title)}</h2>")
        parts.append(f"<img alt='{esc(title)}' src='data:{mime};base64,{data}'>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _markdown(designs: Designs, meta: list[tuple[str, str]], sections: list[_Section]) -> str:
    lines = ["# Suspension kinematics report", ""]
    lines += [f"- {k}: {v}" for k, v in meta]
    lines += ["", "## Summary comparison", "", _md_summary(designs), ""]
    for title, _, _, link in sections:
        lines += [f"## {title}", "", f"![{title}]({link})", ""]
    return "\n".join(lines)
