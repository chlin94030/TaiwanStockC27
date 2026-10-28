"""介面樣式與 HTML 小元件（看盤軟體風格：深色、等寬數字、漲紅跌綠）。"""
from __future__ import annotations

import html

import numpy as np

UP, DOWN, FLAT = "#f0464b", "#16a35f", "#94a3b8"

PALETTE = {
    True: {"bg": "#0b1220", "panel": "#111a2b", "panel2": "#0f1726", "line": "#1e2a3f", "line2": "#162034", "hover": "#15213a",
           "text": "#d6deeb", "strong": "#f1f5f9", "muted": "#8a97ad", "accent": "#f5b942", "tagbg": "#17233a", "tagline": "#273857",
           "tagtx": "#b6c3d9", "gbg": "#132a20", "gtx": "#7dd3a8", "gline": "#1f4734", "ybg": "#2a2110", "ytx": "#f5c55a", "yline": "#4a3a18"},
    False: {"bg": "#f5f7fb", "panel": "#ffffff", "panel2": "#ffffff", "line": "#dde3ec", "line2": "#edf1f6", "hover": "#f3f6fb",
            "text": "#1e293b", "strong": "#0f172a", "muted": "#64748b", "accent": "#b7791f", "tagbg": "#eef2f7", "tagline": "#d5dde8",
            "tagtx": "#334155", "gbg": "#e8f7ef", "gtx": "#13704a", "gline": "#bfe6d0", "ybg": "#fdf3df", "ytx": "#8a5a00", "yline": "#f0d9a8"},
}


def css(dark: bool = True) -> str:
    v = PALETTE[bool(dark)]
    root = ":root{" + "".join(f"--{k}:{c};" for k, c in v.items()) + "}"
    return CSS_BODY.replace("<style>", "<style>" + root, 1)


CSS_BODY = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Roboto+Mono:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"], .stMarkdown, button, input, textarea, select {
  font-family: 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif; }
[data-testid="stAppViewContainer"] { background: var(--bg); }
.block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1480px; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: var(--bg); border-bottom: 1px solid var(--line); }
h1, h2, h3 { letter-spacing: .5px; }
h3 { font-size: 1.05rem !important; color: var(--strong); margin-top: .4rem !important; }
.kq-bar { display:flex; flex-wrap:wrap; gap:6px 22px; align-items:center; background:var(--panel2); border:1px solid var(--line);
  border-radius:10px; padding:9px 16px; margin-bottom:12px; }
.kq-brand { font-weight:700; font-size:17px; color:var(--accent); letter-spacing:2px; margin-right:6px; }
.kq-brand small { color:var(--muted); font-weight:400; letter-spacing:0; margin-left:6px; font-size:12px; }
.kq-item { color:var(--muted); font-size:12.5px; }
.kq-item b { color:var(--strong); font-family:'Roboto Mono', monospace; font-weight:600; font-size:14px; margin-left:4px; }
.kq-pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:500; }
.kq-cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:10px; margin: 4px 0 12px; }
.kq-card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:11px 14px; }
.kq-card .l { color:var(--muted); font-size:12px; }
.kq-card .v { font-family:'Roboto Mono', monospace; font-size:21px; font-weight:600; color:var(--strong); margin-top:2px; }
.kq-card .s { color:var(--muted); font-size:11.5px; margin-top:2px; }
.kq-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--panel2); }
table.kq { width:100%; border-collapse:collapse; font-size:13.5px; white-space:nowrap; }
table.kq th { color:var(--muted); font-weight:500; font-size:12px; text-align:right; padding:8px 10px; border-bottom:1px solid var(--line);
  background:var(--panel2); position:sticky; top:0; }
table.kq td { padding:8px 10px; border-bottom:1px solid var(--line2); text-align:right; color:var(--text);
  font-family:'Roboto Mono', 'Noto Sans TC', monospace; }
table.kq th.l, table.kq td.l { text-align:left; }
table.kq tr:hover td { background:var(--hover); }
table.kq td .nm { font-family:'Noto Sans TC', sans-serif; font-weight:500; color:var(--strong); }
table.kq td .cd { color:var(--accent); font-weight:600; margin-right:6px; }
table.kq td .sub { color:var(--muted); font-size:11.5px; font-family:'Noto Sans TC', sans-serif; }
.kq-tag { display:inline-block; padding:1px 8px; border-radius:999px; font-size:11.5px; background:var(--tagbg); color:var(--tagtx);
  border:1px solid var(--tagline); margin:1px 3px 1px 0; font-family:'Noto Sans TC', sans-serif; }
.kq-tag.g { background:var(--gbg); color:var(--gtx); border-color:var(--gline); }
.kq-tag.gold { background:var(--ybg); color:var(--ytx); border-color:var(--yline); }
.kq-badge { display:inline-block; padding:2px 9px; border-radius:6px; font-size:12px; color:#fff; font-family:'Noto Sans TC', sans-serif; }
.kq-panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:10px; }
.kq-panel h4 { margin:0 0 8px; font-size:14px; color:var(--accent); font-weight:600; }
.kq-panel ul { margin:0; padding-left:18px; } .kq-panel li { margin:3px 0; color:var(--text); font-size:13.5px; }
.kq-verdict { border-left:4px solid var(--accent); background:var(--panel); padding:12px 16px; border-radius:8px; margin:6px 0 12px; font-size:14.5px; }
.kq-mono { font-family:'Roboto Mono', monospace; }
.kq-muted { color:var(--muted); font-size:12.5px; }
.kq-step { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; height:100%; }
.kq-step .n { color:var(--bg); background:var(--accent); border-radius:999px; width:24px; height:24px; display:inline-flex;
  align-items:center; justify-content:center; font-weight:700; font-size:13px; margin-right:8px; }
.kq-step .t { font-weight:600; font-size:15px; color:var(--strong); }
.kq-step p, .kq-step li { color:var(--text); font-size:13.5px; line-height:1.65; }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:10px !important; background:var(--panel2); }
div[data-testid="stMetricValue"] { font-family:'Roboto Mono', monospace; }
@media (max-width: 640px) { .block-container { padding-left: .6rem; padding-right: .6rem; } .kq-card .v { font-size:18px; } }
</style>
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def ok(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def price(x) -> str:
    if not ok(x):
        return "–"
    x = float(x)
    d = 2 if x < 50 else (1 if x < 500 else 0)          # 依台股升降單位顯示小數位
    return f"{x:,.{d}f}"


def chg(x, d=2) -> str:
    """漲跌幅：紅漲綠跌。"""
    if not ok(x):
        return "<span>–</span>"
    x = float(x)
    c = UP if x > 0 else (DOWN if x < 0 else FLAT)
    arrow = "▲" if x > 0 else ("▼" if x < 0 else "")
    return f"<span style='color:{c}'>{arrow}{abs(x)*100:.{d}f}%</span>"


def pct(x, d=1, sign=True, color=False) -> str:
    if not ok(x):
        return "–"
    x = float(x)
    s = f"{x*100:+.{d}f}%" if sign else f"{x*100:.{d}f}%"
    if color:
        c = UP if x > 0 else (DOWN if x < 0 else FLAT)
        return f"<span style='color:{c}'>{s}</span>"
    return s


def lots_fmt(x) -> str:
    if not ok(x):
        return "–"
    x = float(x)
    c = UP if x > 0 else (DOWN if x < 0 else FLAT)
    return f"<span style='color:{c}'>{x:+,.0f}</span>"


def tag(t, kind="") -> str:
    return f"<span class='kq-tag {kind}'>{esc(t)}</span>"


def badge(text, color) -> str:
    return f"<span class='kq-badge' style='background:{color}'>{esc(text)}</span>"


def tone_badge(tone: str) -> str:
    col = {"偏多": "#b4232a", "偏空": "#167a4b", "中性": "#475569"}.get(tone, "#475569")
    return badge(tone, col)


def cards(items) -> str:
    """items: [(label, value_html, sub_html), ...]"""
    out = ["<div class='kq-cards'>"]
    for lab, val, sub in items:
        out.append(f"<div class='kq-card'><div class='l'>{esc(lab)}</div><div class='v'>{val}</div>"
                   f"<div class='s'>{sub or ''}</div></div>")
    out.append("</div>")
    return "".join(out)


def table(headers, rows, left=()) -> str:
    """headers: list[str]; rows: list[list[html]]; left: indices of left-aligned columns."""
    th = "".join(f"<th class='{'l' if i in left else ''}'>{esc(h)}</th>" for i, h in enumerate(headers))
    body = []
    for r in rows:
        body.append("<tr>" + "".join(f"<td class='{'l' if i in left else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    return f"<div class='kq-wrap'><table class='kq'><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def panel(title, body_html) -> str:
    return f"<div class='kq-panel'><h4>{esc(title)}</h4>{body_html}</div>"


def bullets(lines) -> str:
    return "<ul>" + "".join(f"<li>{x}</li>" for x in lines) + "</ul>"
