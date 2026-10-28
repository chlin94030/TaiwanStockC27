"""交易計畫：建議買進區、追價上限、警戒價、壓力參考、盤中狀態、籌碼短評、資金配置。

重要：模型回測的進出場是「訊號日收盤後決定 → 隔日開盤價成交 → 下次檢查日依排名續抱或換股」。
本模組的價位（買進區、追價上限、警戒價）是執行層面的風險控管建議，未納入回測；買進價位以「不追高、分批」為原則。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ta import round_tick

HZ_ZH = {"short": "短線", "mid": "中線", "long": "長線"}
CHECK_EVERY = {"short": "每 5 個交易日（建議每週固定一天）", "mid": "每週調整 1/4 資金（每份抱約 4 週）",
               "long": "每 4 週調整 1/3 資金（每份抱約 12 週）"}
ALERT = {"short": (2.0, 0.07), "mid": (3.0, 0.10), "long": (4.0, 0.15)}   # (ATR 倍數, 最大跌幅)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


@dataclass
class Plan:
    close: float
    buy_low: float          # 建議買進區下緣（拉回承接）
    buy_high: float         # 建議買進區上緣（開盤或盤中可掛的價位）
    chase: float            # 追價上限（開盤或盤中高於此價不追）
    alert: float            # 警戒價（跌破請重新檢視；模型回測不設停損）
    support: float          # 支撐參考
    resist: float           # 壓力參考
    atr_pct: float
    rsi: float
    bias20: float
    hot: bool               # 短線過熱（RSI>80 或 月線乖離>15%）

    def as_dict(self):
        return self.__dict__.copy()


def trade_plan(ind: pd.DataFrame, horizon: str) -> Plan | None:
    """ind = ta.indicators(日K)。"""
    if ind is None or ind.empty:
        return None
    r = ind.iloc[-1]
    c = _f(r["close"])
    if not np.isfinite(c):
        return None
    a = _f(r.get("atr"))
    a = a if np.isfinite(a) and a > 0 else c * 0.02
    ap = a / c
    ma5, ma20, ma60 = _f(r.get("ma5")), _f(r.get("ma20")), _f(r.get("ma60"))
    lo20 = _f(ind["low"].tail(20).min())
    hi60 = _f(ind["high"].tail(60).max())
    bb_up = _f(r.get("bb_up"))
    buy_high = c * (1 + min(0.015, 0.3 * ap))
    pull = c - 1.0 * a
    if np.isfinite(ma5) and ma5 < c:
        pull = max(pull, ma5)
    buy_low = max(pull, c * 0.95)
    chase = c * (1 + min(0.035, 0.6 * ap))
    k_atr, cap = ALERT.get(horizon, (3.0, 0.10))
    alert = max(c - k_atr * a, c * (1 - cap))
    sup = [x for x in (ma20, lo20, ma60) if np.isfinite(x) and x < c]
    support = max(sup) if sup else c - 2 * a
    res = [x for x in (bb_up, hi60) if np.isfinite(x) and x > c * 1.005]
    resist = min(res) if res else c + 2 * a
    rs = _f(r.get("rsi"))
    bias = c / ma20 - 1 if np.isfinite(ma20) and ma20 > 0 else np.nan
    hot = bool((np.isfinite(rs) and rs > 80) or (np.isfinite(bias) and bias > 0.15))
    return Plan(c, round_tick(buy_low, "down"), round_tick(buy_high, "down"), round_tick(chase, "down"),
                round_tick(alert, "down"), round_tick(support, "down"), round_tick(resist, "up"), ap, rs, bias, hot)


# ------------------------------------------------------------------ 盤中狀態
STATUS_STYLE = {"buy": "#2563eb", "dip": "#0891b2", "wait": "#d97706", "warn": "#dc2626", "lock": "#6b7280",
                "hold": "#7c3aed", "none": "#6b7280"}


STATUS_HELP = {"buy": "現價在建議買進區（或略高但未超過追價上限）→ 可分批進場",
               "dip": "現價回到拉回買點 → 適合承接／加碼", "wait": "現價高於追價上限 → 不追，等拉回",
               "warn": "跌破警戒價 → 空手暫緩；持有者重新檢視籌碼與基本面", "lock": "漲停鎖住 → 買不到不追",
               "hold": "排名仍在前 25 名 → 持有者續抱，空手者不新買", "none": "不在模型名單／非交易時段"}


def live_status(price: float, plan: Plan | None, role: str = "pick", limit_up: float = np.nan) -> tuple[str, str]:
    """role: pick(推薦名單) / keep(前 25 名，持有者續抱) / none(不在名單)。回傳 (代碼, 簡短狀態)。"""
    p = _f(price)
    if plan is None or not np.isfinite(p):
        return "none", "無報價"
    if np.isfinite(limit_up) and p >= limit_up - 1e-9:
        return "lock", "漲停不追"
    if p <= plan.alert:
        return "warn", "跌破警戒"
    if role == "none":
        return "none", "非模型名單"
    if role == "keep":
        return "hold", "續抱（不新買）"
    if p > plan.chase:
        return "wait", "等拉回"
    if p <= plan.buy_low:
        return "dip", "拉回買點"
    if p <= plan.buy_high:
        return "buy", "可進場"
    return "buy", "可進場（小量）"


# ------------------------------------------------------------------ 籌碼短評
def _streak(x: pd.Series) -> int:
    """連續買超(+)/賣超(−)天數（最近一日起算）。"""
    v = np.sign(x.fillna(0).to_numpy())
    if len(v) == 0 or v[-1] == 0:
        return 0
    s = v[-1]
    n = 0
    for z in v[::-1]:
        if z != s:
            break
        n += 1
    return int(n * s)


def chip_summary(df: pd.DataFrame) -> dict:
    """三大法人、融資券、大戶（集保）短評。df = 日資料（未重組）。"""
    out = {"lines": [], "score": 0, "table": None, "tone": "中性"}
    if df is None or df.empty or "foreign" not in df:
        return out
    d = df.tail(260)
    rows = []
    for col, nm in (("foreign", "外資"), ("trust", "投信"), ("dealer", "自營商")):
        s = d[col].fillna(0)
        rows.append({"法人": nm, "今日(張)": s.iloc[-1], "近5日(張)": s.tail(5).sum(), "近20日(張)": s.tail(20).sum(),
                     "連續": _streak(s)})
    tot = d[["foreign", "trust", "dealer"]].fillna(0).sum(axis=1)
    rows.append({"法人": "三大法人合計", "今日(張)": tot.iloc[-1], "近5日(張)": tot.tail(5).sum(),
                 "近20日(張)": tot.tail(20).sum(), "連續": _streak(tot)})
    out["table"] = pd.DataFrame(rows)
    f5, t5, d5 = (rows[i]["近5日(張)"] for i in range(3))
    lines, sc = [], 0
    if f5 > 0 and t5 > 0 and d5 > 0:
        lines.append("三大法人近 5 日同步買超 → 籌碼偏多")
        sc += 2
    elif f5 < 0 and t5 < 0 and d5 < 0:
        lines.append("三大法人近 5 日同步賣超 → 籌碼偏空")
        sc -= 2
    elif f5 > 0 and t5 > 0:
        lines.append("外資、投信同步買超（自營商反向）")
        sc += 1
    elif f5 < 0 and t5 < 0:
        lines.append("外資、投信同步賣超")
        sc -= 1
    elif f5 < 0 < t5:
        lines.append("土洋對作：投信買、外資賣")
    elif t5 < 0 < f5:
        lines.append("土洋對作：外資買、投信賣")
    fs = rows[0]["連續"]
    if abs(fs) >= 3:
        lines.append(f"外資連續{'買' if fs > 0 else '賣'}超 {abs(fs)} 日")
        sc += 1 if fs > 0 else -1
    ts = rows[1]["連續"]
    if abs(ts) >= 3:
        lines.append(f"投信連續{'買' if ts > 0 else '賣'}超 {abs(ts)} 日")
        sc += 1 if ts > 0 else -1
    v20 = d["volume"].tail(20).sum()
    if v20 > 0:
        conc = tot.tail(20).sum() / v20
        out["conc20"] = conc
        if abs(conc) >= 0.05:
            lines.append(f"法人 20 日{'買' if conc > 0 else '賣'}超占成交量 {abs(conc):.0%}（{'主力進駐' if conc > 0 else '主力調節'}）")
    # 融資券（散戶＝猶豫者）
    if "mbal" in d and d["mbal"].notna().sum() > 25:
        mb = d["mbal"].ffill()
        c = d["close"]
        m20 = mb.iloc[-1] / mb.iloc[-21] - 1 if mb.iloc[-21] > 0 else np.nan
        p20 = c.iloc[-1] / c.iloc[-21] - 1
        out["m20"], out["p20"] = m20, p20
        out["mbal"] = float(mb.iloc[-1])
        if "sbal" in d:
            out["sbal"] = float(d["sbal"].ffill().iloc[-1])
        if np.isfinite(m20):
            if m20 > 0.05 and p20 < 0:
                lines.append(f"融資 20 日增 {m20:.0%} 但股價跌 → 散戶攤平、籌碼凌亂（股票流向猶豫者）")
                sc -= 1
            elif m20 < -0.05 and p20 > 0:
                lines.append(f"融資 20 日減 {-m20:.0%}、股價漲 → 籌碼沉澱（股票流向固執者）")
                sc += 1
            elif m20 > 0.15 and p20 > 0.10:
                lines.append(f"融資 20 日大增 {m20:.0%} 追價 → 留意過熱")
        if "mlimit" in d and d["mlimit"].iloc[-1] > 0:
            out["usage"] = mb.iloc[-1] / d["mlimit"].iloc[-1]
        if mb.iloc[-1] > 0 and "sbal" in d:
            ratio = d["sbal"].ffill().iloc[-1] / mb.iloc[-1]
            out["short_ratio"] = ratio
            if ratio > 0.3:
                lines.append(f"券資比 {ratio:.0%} 偏高（軋空題材／放空壓力並存）")
    # 大戶（集保，每週）
    if "big400" in d and d["big400"].notna().sum() >= 2:
        w = d["big400"].dropna()
        w = w[w.diff().fillna(1) != 0]                     # one value per published week
        if len(w) >= 2:
            ch = w.iloc[-1] - w.iloc[-2]
            out["big400"], out["big400_ch"] = w.iloc[-1], ch
            if ch >= 0.3:
                lines.append(f"大戶（≥400 張）持股 {w.iloc[-1]:.1f}%，較上週 +{ch:.1f}pp → 大戶加碼")
                sc += 1
            elif ch <= -0.3:
                lines.append(f"大戶（≥400 張）持股 {w.iloc[-1]:.1f}%，較上週 {ch:.1f}pp → 大戶減碼")
                sc -= 1
        elif len(w) == 1:
            out["big400"] = w.iloc[-1]
    out["lines"] = lines or ["籌碼無明顯方向"]
    out["score"] = sc
    out["tone"] = "偏多" if sc >= 2 else ("偏空" if sc <= -2 else "中性")
    return out


# ------------------------------------------------------------------ 資金配置
PROFILES = {
    "保守型": {"0050": 0.50, "mid": 0.20, "long": 0.30},
    "穩健型": {"0050": 0.20, "mid": 0.40, "long": 0.40},
    "積極型": {"short": 0.60, "long": 0.40},
}
PROFILE_NOTE = {
    "保守型": "一半放 0050 當核心，另一半用中、長線模型增強；每月看一次即可。",
    "穩健型": "中線＋長線為主、0050 兩成；每週花 10 分鐘調整中線的一份資金。",
    "積極型": "短線每週檢查換股＋長線抱趨勢；報酬與回撤都最大。",
}


def blend_nav(navs: dict, weights: dict, rebalance: str = "ME") -> pd.Series:
    """各子策略日淨值依權重組合，每月再平衡。navs: {'short'|'mid'|'long'|'0050': Series}."""
    keys = [k for k, w in weights.items() if w > 0 and k in navs]
    if not keys:
        return pd.Series(dtype=float)
    df = pd.concat({k: navs[k] for k in keys}, axis=1).dropna()
    rets = df.pct_change().fillna(0)
    w = np.array([weights[k] for k in keys], float)
    w = w / w.sum()
    val, hold = 1.0, w.copy()
    out = []
    months = df.index.to_period("M")
    for t in range(len(df)):
        if t > 0 and months[t] != months[t - 1]:
            hold = val * w
        hold = hold * (1 + rets.iloc[t].to_numpy())
        val = hold.sum()
        out.append(val)
    return pd.Series(out, index=df.index)


def stock_weights(n: int, horizon: str) -> np.ndarray:
    """短中線等權；長線名次加權 1.5→0.5（與回測一致）。"""
    if n <= 0:
        return np.zeros(0)
    if horizon == "long" and n > 1:
        w = np.linspace(1.5, 0.5, n)
    else:
        w = np.ones(n)
    return w / w.sum()


def lots(amount: float, price: float) -> tuple[int, int]:
    """金額 → (整張, 零股)。"""
    if not (np.isfinite(price) and price > 0 and amount > 0):
        return 0, 0
    sh = int(amount // price)
    return sh // 1000, sh % 1000
