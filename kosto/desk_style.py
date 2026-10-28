"""Compact brokerage-style layout. No external font/network dependency."""
from __future__ import annotations
import html


def css(dark=False):
    colors=('#0c1421','#121e30','#22324a','#e6edf7','#95a5bd','#1ab6a0') if dark else ('#f2f5f9','#ffffff','#dce3ec','#14253d','#64758b','#087e76')
    bg,panel,line,text,muted,accent=colors
    return f'''<style>
    :root{{--desk-bg:{bg};--desk-panel:{panel};--desk-line:{line};--desk-text:{text};--desk-muted:{muted};--desk-accent:{accent}}}
    .stApp,[data-testid="stAppViewContainer"]{{background:var(--desk-bg);color:var(--desk-text)}}
    .block-container{{max-width:1510px;padding-top:2.2rem;padding-bottom:3rem}}
    html,body,button,input,select,.stMarkdown{{font-family:Inter,"PingFang TC","Noto Sans TC","Microsoft JhengHei","Heiti TC",sans-serif}}
    [data-testid="stHeader"]{{height:2rem;background:var(--desk-bg)}}
    h1,h2,h3{{color:var(--desk-text);letter-spacing:-.025em}} h3{{font-size:1.12rem!important;margin-top:.35rem}}
    .desk-top{{display:flex;justify-content:space-between;align-items:center;padding:14px 2px 18px;border-bottom:2px solid #182f4d;gap:18px}}
    .desk-wordmark{{font-size:20px;font-weight:800;letter-spacing:1.3px;color:var(--desk-text)}}
    .desk-wordmark span{{color:var(--desk-accent)}} .desk-product{{font-size:12px;color:var(--desk-muted);margin-top:3px;letter-spacing:.08em}}
    .desk-meta{{font-size:12px;text-align:right;line-height:1.8;color:var(--desk-muted)}}
    .desk-strip{{display:flex;gap:14px;flex-wrap:wrap;background:var(--desk-panel);border:1px solid var(--desk-line);padding:11px 15px;margin:12px 0;border-radius:8px;font-size:12px;color:var(--desk-muted);box-shadow:0 5px 18px rgba(15,35,58,.035)}}
    .desk-strip b{{color:var(--desk-text);font-variant-numeric:tabular-nums}}
    .desk-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:10px 0 18px}}
    .desk-stat{{background:var(--desk-panel);border:1px solid var(--desk-line);border-top:3px solid var(--desk-accent);padding:15px 18px;border-radius:8px;box-shadow:0 7px 22px rgba(15,35,58,.045)}}
    .desk-stat .label{{font-size:12px;color:var(--desk-muted)}}.desk-stat .value{{font-size:26px;font-weight:750;color:var(--desk-text);margin:4px 0;font-variant-numeric:tabular-nums}}
    .desk-stat .hint{{font-size:11px;color:var(--desk-muted)}}
    .desk-note{{font-size:12px;line-height:1.75;color:var(--desk-muted);margin:6px 0 12px}}
    .desk-title{{font-size:23px;font-weight:750;letter-spacing:-.04em;color:var(--desk-text);margin:15px 0 4px}}
    .desk-sub{{font-size:13px;color:var(--desk-muted);margin-bottom:15px;line-height:1.6}}
    .desk-stock{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;border-top:1px solid var(--desk-line);padding:20px 0 12px;margin-top:10px}}
    .desk-stock .code{{font-size:25px;font-weight:800;color:var(--desk-accent);font-variant-numeric:tabular-nums}}
    .desk-stock .name{{font-size:24px;font-weight:700;color:var(--desk-text)}}
    .desk-tag{{font-size:11px;padding:4px 8px;border:1px solid var(--desk-line);color:var(--desk-muted);border-radius:4px;background:var(--desk-panel)}}
    .desk-verdict{{border-left:4px solid var(--desk-accent);padding:12px 16px;background:var(--desk-panel);margin-bottom:12px;font-size:14px;color:var(--desk-text);line-height:1.7}}
    .desk-global{{display:flex;justify-content:space-between;gap:14px;align-items:center;margin:10px 0 16px;padding:12px 15px;border:1px solid var(--desk-line);border-left:4px solid #718399;border-radius:7px;background:var(--desk-panel);font-size:12px;box-shadow:0 6px 20px rgba(15,35,58,.035)}}
    .desk-global b{{font-size:13px;color:var(--desk-text);white-space:nowrap}}.desk-global span{{color:var(--desk-muted);text-align:right}}
    .desk-global.positive{{border-left-color:#c83a40}}.desk-global.negative{{border-left-color:#148759}}.desk-global.neutral{{border-left-color:var(--desk-accent)}}
    [data-testid="stDataFrame"]{{border:1px solid var(--desk-line);border-radius:5px;overflow:hidden}}
    [data-testid="stExpander"]{{background:var(--desk-panel);border:1px solid var(--desk-line);border-radius:5px}}
    [data-testid="stMetric"]{{background:var(--desk-panel);border:1px solid var(--desk-line);border-radius:5px;padding:12px}}
    [data-testid="stMetricValue"]{{font-size:24px;font-variant-numeric:tabular-nums}}
    .stButton button{{border-radius:5px;font-weight:600}} [data-testid="stSidebar"]{{background:var(--desk-panel)}}
    div[role="radiogroup"]{{gap:0!important;border-bottom:1px solid var(--desk-line);padding:7px 0 10px;background:var(--desk-bg)}}
    div[role="radiogroup"] label{{padding:4px 12px!important;font-size:13px!important}}
    .desk-demo{{background:#fff7df;color:#78531b;border:1px solid #edcf80;padding:8px 13px;font-size:12px;margin:8px 0;border-radius:4px}}
    .desk-footer{{border-top:1px solid var(--desk-line);margin-top:22px;padding-top:12px;color:var(--desk-muted);font-size:11px;line-height:1.8}}
    @media(max-width:720px){{
      .block-container{{padding:3rem .65rem 5rem}}
      .desk-top{{align-items:flex-start;padding-top:7px}}.desk-product{{font-size:10px}}.desk-meta{{max-width:132px;font-size:9px;line-height:1.6}}.desk-wordmark{{font-size:16px}}
      .desk-strip{{gap:7px;padding:9px 10px;overflow-x:auto;flex-wrap:nowrap}}.desk-strip span{{white-space:nowrap}}
      .desk-grid{{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}.desk-stat{{padding:10px 11px}}.desk-stat .value{{font-size:20px}}.desk-title{{font-size:21px}}
      .desk-global{{display:block;padding:10px 12px}}.desk-global span{{display:block;text-align:left;margin-top:4px;line-height:1.55}}
      div[role="radiogroup"]{{position:sticky;top:1.7rem;z-index:90;overflow-x:auto;flex-wrap:nowrap!important;box-shadow:0 5px 12px rgba(15,35,58,.08)}}
      div[role="radiogroup"] label{{padding:7px 9px!important;white-space:nowrap;font-size:12px!important}}
      [data-testid="stDataFrame"]{{font-size:11px}}
    }}
    </style>'''


def esc(x):return html.escape(str(x) if x is not None else '')


def stats(items):
    return '<div class="desk-grid">'+''.join('<div class="desk-stat"><div class="label">'+esc(a)+'</div><div class="value">'+esc(b)+'</div><div class="hint">'+esc(c)+'</div></div>' for a,b,c in items)+'</div>'


def title(main,sub=''):
    return '<div class="desk-title">'+esc(main)+'</div><div class="desk-sub">'+esc(sub)+'</div>'
