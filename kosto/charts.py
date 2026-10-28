"""看盤圖（Plotly）：K 線＋均線＋布林通道＋成交量＋KD＋MACD＋三大法人＋融資券＋大戶持股。台股配色：漲紅跌綠。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .ta import MA_LINES

UP, DOWN = "#f0464b", "#16a35f"
THEME = {True: ("#0f1726", "#1e2a3f", "#d6deeb", "#8a97ad"), False: ("#ffffff", "#e5eaf1", "#1e293b", "#64748b")}
MA_COLOR = {5: "#f5b942", 10: "#e879f9", 20: "#38bdf8", 60: "#a3e635", 120: "#fb923c", 240: "#c084fc"}
PANELS = {"volume": "成交量", "kd": "KD", "macd": "MACD", "inst": "三大法人", "margin": "融資券", "holders": "大戶持股",'broker':'主力分點'}
PANEL_H = {"volume": 0.13, "kd": 0.12, "macd": 0.12, "inst": 0.14, "margin": 0.12, "holders": 0.11,'broker':.12}


def kline(ind: pd.DataFrame, mas=(5, 20, 60), boll: bool = True, panels=("volume", "kd", "macd"),
          plan=None, bars: int = 120, height: int | None = None, title: str = "", dark: bool = True,ma_unit='D') -> go.Figure:
    BG, GRID, TXT, MUTED = THEME[bool(dark)]
    df = ind.tail(bars).copy()
    if df.empty:
        return go.Figure()
    panels = [p for p in panels if p in PANELS]
    required={'inst':['foreign','trust','dealer'],'margin':['mbal','sbal'],'broker':['broker_net']}
    panels=[p for p in panels if p not in required or any(c in df and df[c].notna().any() for c in required[p])]
    if "holders" in panels and ("big400" not in df or df["big400"].notna().sum() == 0):
        panels.remove("holders")
    x = [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(df.index)]
    rows = 1 + len(panels)
    hs = [max(0.30, 1 - sum(PANEL_H[p] for p in panels))] + [PANEL_H[p] for p in panels]
    s = sum(hs)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.018, row_heights=[h / s for h in hs],
                        specs=[[{"secondary_y": p == "margin"}] for p in ["price"] + panels])
    fig.add_trace(go.Candlestick(x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K線", showlegend=False,
                                 increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
                                 hoverlabel=dict(namelength=0)), row=1, col=1)
    for n in mas:
        c = f"ma{n}"
        if c in df and df[c].notna().any():
            ma_name=MA_LINES.get(n,f'MA{n}') if ma_unit=='D' else f'{n}'+{'W':'週','M':'月'}.get(ma_unit,'期')+'均線'
            fig.add_trace(go.Scatter(x=x, y=df[c], name=ma_name, mode="lines",
                                     line=dict(color=MA_COLOR.get(n, "#ccc"), width=1.2)), row=1, col=1)
    if boll and "bb_up" in df:
        fig.add_trace(go.Scatter(x=x, y=df["bb_up"], name="布林通道", mode="lines",
                                 line=dict(color="rgba(148,163,184,0.7)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["bb_dn"], showlegend=False, name="布林下軌", mode="lines", fill="tonexty",
                                 fillcolor="rgba(148,163,184,0.06)",
                                 line=dict(color="rgba(148,163,184,0.7)", width=1, dash="dot")), row=1, col=1)
    if plan is not None:
        for y, nm, col, dash in ((plan.buy_high, "買上", "#3b82f6", "solid"), (plan.buy_low, "拉回", "#06b6d4", "dot"),
                                 (plan.chase, "追價", "#f59e0b", "dash"), (plan.alert, "警戒", "#ef4444", "dash")):
            if y and np.isfinite(y):
                fig.add_hline(y=y, line=dict(color=col, width=1, dash=dash), row=1, col=1,
                              annotation_text=f"{nm} {y:g}", annotation_position="top left",
                              annotation_font=dict(color=col, size=10), annotation_bgcolor=BG)
    r = 2
    up = (df["close"] >= df["open"]).to_numpy()
    for p in panels:
        if p == "volume":
            fig.add_trace(go.Bar(x=x, y=df["volume"], showlegend=False, name="成交量(張)", marker_color=np.where(up, UP, DOWN),
                                 marker_line_width=0, opacity=0.85), row=r, col=1)
            if "vma20" in df:
                fig.add_trace(go.Scatter(x=x, y=df["vma20"], showlegend=False, name="20日均量", mode="lines",
                                         line=dict(color="#f5b942", width=1)), row=r, col=1)
        elif p == "kd":
            fig.add_trace(go.Scatter(x=x, y=df["k"], showlegend=False, name="K(9)", mode="lines", line=dict(color="#f59e0b", width=1.2)), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=df["d"], showlegend=False, name="D(9)", mode="lines", line=dict(color="#38bdf8", width=1.2)), row=r, col=1)
            for lv in (20, 80):
                fig.add_hline(y=lv, line=dict(color=GRID, width=1, dash="dot"), row=r, col=1)
        elif p == "macd":
            fig.add_trace(go.Bar(x=x, y=df["osc"], showlegend=False, name="OSC", marker_color=np.where(df["osc"] >= 0, UP, DOWN),
                                 marker_line_width=0, opacity=0.8), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=df["dif"], showlegend=False, name="DIF", mode="lines", line=dict(color="#f5b942", width=1.2)), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=df["dem"], showlegend=False, name="MACD", mode="lines", line=dict(color="#38bdf8", width=1.2)), row=r, col=1)
        elif p == "inst":
            for col, nm, cc in (("foreign", "外資", "#60a5fa"), ("trust", "投信", "#f472b6"), ("dealer", "自營商", "#facc15")):
                if col in df:
                    fig.add_trace(go.Bar(x=x, y=df[col], name=f"{nm}(張)", marker_color=cc, marker_line_width=0), row=r, col=1)
        elif p == "margin":
            if "mbal" in df:
                fig.add_trace(go.Scatter(x=x, y=df["mbal"], showlegend=False, name="融資餘額(張)", mode="lines",
                                         line=dict(color="#fb7185", width=1.4)), row=r, col=1, secondary_y=False)
            if "sbal" in df:
                fig.add_trace(go.Scatter(x=x, y=df["sbal"], showlegend=False, name="融券餘額(張)", mode="lines",
                                         line=dict(color="#34d399", width=1.4)), row=r, col=1, secondary_y=True)
        elif p == "holders":
            fig.add_trace(go.Scatter(x=x, y=df["big400"], showlegend=False, name="大戶≥400張(%)", mode="lines", connectgaps=True,
                                     line=dict(color="#f59e0b", width=1.6, shape="hv")), row=r, col=1)
            if "retail50" in df:
                fig.add_trace(go.Scatter(x=x, y=df["retail50"], showlegend=False, name="散戶<50張(%)", mode="lines", connectgaps=True,
                                         line=dict(color="#94a3b8", width=1.2, shape="hv")), row=r, col=1)
        elif p=='broker' and 'broker_net' in df:
            fig.add_trace(go.Bar(x=x,y=df.broker_net,name='主力分點淨買超(張)',marker_color=np.where(df.broker_net>=0,UP,DOWN),showlegend=False),row=r,col=1)
        fig.update_yaxes(title_text=PANELS[p], title_font=dict(size=10, color=MUTED), row=r, col=1)
        r += 1
    if "inst" in panels:
        fig.update_layout(barmode="relative")
    fig.update_xaxes(type="category", showgrid=False, rangeslider_visible=False, nticks=8, tickfont=dict(color=MUTED, size=10),
                     linecolor=GRID, showspikes=True, spikemode="across", spikethickness=1, spikecolor="#64748b")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED, size=10), side="right")
    fig.update_layout(height=height or int(380 + 120 * len(panels)), paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(color=TXT, family="Noto Sans TC, PingFang TC, Microsoft JhengHei, sans-serif"),
                      margin=dict(l=8, r=64, t=36 if title else 10, b=8), hovermode="x unified",
                      title=dict(text=title, font=dict(size=14)) if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                                  traceorder="normal"),
                      xaxis_rangeslider_visible=False, dragmode="pan")
    return fig


def nav_figure(navs: dict, height: int = 300, log: bool = True, dark: bool = True) -> go.Figure:
    BG, GRID, TXT, MUTED = THEME[bool(dark)]
    colors = ["#f5b942" if dark else "#b7791f", "#38bdf8", "#a3e635", "#f472b6", "#94a3b8"]
    fig = go.Figure()
    for k, (nm, sr) in enumerate(navs.items()):
        sr = sr.dropna()
        if len(sr):
            fig.add_trace(go.Scatter(x=sr.index, y=sr / sr.iloc[0], name=nm, mode="lines",
                                     line=dict(width=2 if k == 0 else 1.4, color=colors[k % len(colors)])))
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
                      margin=dict(l=8, r=48, t=10, b=8), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, tickfont=dict(color=MUTED), side="right", type="log" if log else "linear")
    return fig
