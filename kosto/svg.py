"""Tiny dependency-free SVG charts for the HTML report (fonts come from the browser → CJK safe)."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

PALETTE = ["#2563eb", "#6b7280", "#d97706", "#7c3aed", "#0d9488", "#db2777", "#65a30d"]
STAGE_COLOR = {"A1": "#16a34a", "A2": "#86efac", "A3": "#f59e0b", "B1": "#f97316", "B2": "#dc2626",
               "B3": "#3b82f6", "UNKNOWN": "#d1d5db"}


def _nice_ticks(lo, hi, n=5):
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda m: abs(m * mag - raw)) * mag
    start = np.ceil(lo / step) * step
    return list(np.arange(start, hi + step * 0.5, step))


def line_chart(series: dict, title: str = "", w: int = 920, h: int = 300, log: bool = False, pct: bool = False,
               colors=None) -> str:
    series = {k: v.dropna() for k, v in series.items() if v is not None and len(v.dropna()) > 1}
    if not series:
        return ""
    colors = colors or PALETTE
    L, R, T, B = 56, 12, 28, 26
    idx = sorted(set().union(*[s.index for s in series.values()]))
    x0, x1 = pd.Timestamp(idx[0]).value, pd.Timestamp(idx[-1]).value
    vals = np.concatenate([s.to_numpy(float) for s in series.values()])
    tr = (lambda v: np.log(np.maximum(v, 1e-9))) if log else (lambda v: v)
    lo, hi = np.nanmin(tr(vals)), np.nanmax(tr(vals))
    pad = (hi - lo) * 0.05 or 0.01
    lo, hi = lo - pad, hi + pad
    X = lambda t: L + (pd.Timestamp(t).value - x0) / max(x1 - x0, 1) * (w - L - R)  # noqa: E731
    Y = lambda v: T + (hi - tr(v)) / (hi - lo) * (h - T - B)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{html.escape(title)}">',
           f'<text x="{L}" y="16" class="ct">{html.escape(title)}</text>']
    ticks = _nice_ticks(np.exp(lo) if log else lo, np.exp(hi) if log else hi)
    for tv in ticks:
        if tv <= 0 and log:
            continue
        y = Y(tv)
        if T <= y <= h - B:
            lab = f"{tv:.0%}" if pct else (f"{tv:.2f}" if abs(tv) < 10 else f"{tv:,.0f}")
            out.append(f'<line x1="{L}" x2="{w-R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                       f'<text x="{L-6}" y="{y+4:.1f}" class="ax" text-anchor="end">{lab}</text>')
    years = sorted({pd.Timestamp(d).year for d in idx})
    for yv in years:
        t = pd.Timestamp(yv, 1, 1)
        if x0 <= t.value <= x1:
            out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{h-B}" class="grid"/>'
                       f'<text x="{X(t)+3:.1f}" y="{h-8}" class="ax">{yv}</text>')
    lx = L + 8
    for k, (name, s) in enumerate(series.items()):
        c = colors[k % len(colors)]
        step = max(1, len(s) // 600)
        s2 = s.iloc[::step]
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in s2.items() if np.isfinite(v))
        width = 2.2 if k == 0 else 1.4
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="{width}"/>')
        out.append(f'<rect x="{lx}" y="{T+4}" width="10" height="10" fill="{c}"/>'
                   f'<text x="{lx+14}" y="{T+13}" class="lg">{html.escape(str(name))}</text>')
        lx += 22 + 8 * len(str(name)) + 6 * sum(1 for ch in str(name) if ord(ch) > 255)
    out.append("</svg>")
    return "".join(out)


def stage_strip(stages: pd.Series, w: int = 920, h: int = 26) -> str:
    s = stages.dropna()
    if s.empty:
        return ""
    L, R = 56, 12
    x0, x1 = s.index[0].value, s.index[-1].value
    X = lambda t: L + (t.value - x0) / max(x1 - x0, 1) * (w - L - R)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    run_start, cur = s.index[0], s.iloc[0]
    for t, v in list(s.items())[1:] + [(s.index[-1], None)]:
        if v != cur:
            out.append(f'<rect x="{X(run_start):.1f}" y="2" width="{max(X(t)-X(run_start), 0.8):.1f}" height="{h-4}" '
                       f'fill="{STAGE_COLOR.get(cur, "#ddd")}"><title>{cur}</title></rect>')
            run_start, cur = t, v
    out.append("</svg>")
    legend = " ".join(f'<span class="sw" style="background:{c}"></span>{k}' for k, c in STAGE_COLOR.items())
    return "".join(out) + f'<div class="legend">{legend}</div>'


def bar_chart(cats: list, series: dict, title="", w=920, h=260, pct=True) -> str:
    if not cats or not series:
        return ""
    L, R, T, B = 56, 12, 28, 40
    vals = np.array([v for s in series.values() for v in s], float)
    lo, hi = min(0, np.nanmin(vals)), max(0, np.nanmax(vals))
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    Y = lambda v: T + (hi - v) / (hi - lo) * (h - T - B)  # noqa: E731
    gw = (w - L - R) / len(cats)
    bw = gw * 0.8 / len(series)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart"><text x="{L}" y="16" class="ct">{html.escape(title)}</text>']
    for tv in _nice_ticks(lo, hi):
        y = Y(tv)
        out.append(f'<line x1="{L}" x2="{w-R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{L-6}" y="{y+4:.1f}" class="ax" text-anchor="end">{tv:.0%}</text>' if pct else
                   f'<line x1="{L}" x2="{w-R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{L-6}" y="{y+4:.1f}" class="ax" text-anchor="end">{tv:.2f}</text>')
    for k, (name, s) in enumerate(series.items()):
        c = PALETTE[k % len(PALETTE)]
        for i, v in enumerate(s):
            if not np.isfinite(v):
                continue
            x = L + i * gw + gw * 0.1 + k * bw
            y0, y1 = sorted([Y(0), Y(v)])
            out.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bw-1:.1f}" height="{max(y1-y0, 0.5):.1f}" fill="{c}">'
                       f'<title>{html.escape(str(cats[i]))} {name}: {v:.1%}</title></rect>')
        out.append(f'<rect x="{L+8+k*120}" y="{h-14}" width="10" height="10" fill="{c}"/>'
                   f'<text x="{L+22+k*120}" y="{h-5}" class="lg">{html.escape(name)}</text>')
    for i, cname in enumerate(cats):
        out.append(f'<text x="{L+i*gw+gw/2:.1f}" y="{h-B+14}" class="ax" text-anchor="middle">{html.escape(str(cname))}</text>')
    out.append(f'<line x1="{L}" x2="{w-R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="#888"/></svg>')
    return "".join(out)


def heat_table(df: pd.DataFrame, fmt="{:.1%}", reverse=False) -> str:
    v = df.to_numpy(float)
    if v.size == 0 or not np.isfinite(v).any():
        return "<p class='mut'>（無資料）</p>"
    lo, hi = np.nanmin(v), np.nanmax(v)
    rows = ["<table class='heat'><tr><th></th>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"]
    for r, (ix, row) in enumerate(df.iterrows()):
        cells = []
        for x in row:
            if not np.isfinite(x):
                cells.append("<td>–</td>")
                continue
            q = (x - lo) / (hi - lo) if hi > lo else 0.5
            q = 1 - q if reverse else q
            col = f"hsl({int(q*120)},65%,{88-int(q*20)}%)"
            cells.append(f"<td style='background:{col}'>{fmt.format(x)}</td>")
        rows.append(f"<tr><th>{html.escape(str(ix))}</th>{''.join(cells)}</tr>")
    return "".join(rows) + "</table>"
