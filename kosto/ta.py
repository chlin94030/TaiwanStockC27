"""技術指標（台股慣用參數）與日／週／月 K 重組。所有函式只用 pandas，輸入為單一個股的日資料 DataFrame。

欄位：open high low close volume(張) foreign trust dealer(張，買超為正) mbal sbal mlimit(張)
      fr(外資持股%) big400 big1000 retail50(集保持股%) holders(股東人數)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from decimal import Decimal,ROUND_FLOOR,ROUND_CEILING,ROUND_HALF_UP

MA_LINES = {5: "週線(5)", 10: "雙週線(10)", 20: "月線(20)", 60: "季線(60)", 120: "半年線(120)", 240: "年線(240)"}
FIELDS = {"Q_O": "open", "Q_H": "high", "Q_L": "low", "Q_C": "close", "Q_V": "volume", "Q_FI": "foreign",
          "Q_TR": "trust", "Q_DL": "dealer", "Q_MB": "mbal", "Q_SB": "sbal", "Q_ML": "mlimit", "Q_FR": "fr",
          "Q_H400": "big400", "Q_H1000": "big1000", "Q_HR": "retail50", "Q_HN": "holders"}


def stock_frame(B, j: int) -> pd.DataFrame:
    """One ticker's daily chart data from the bundle's 看盤資料包 (empty frame if the bundle has none)."""
    if "q_days" not in B.meta or "Q_C" not in B.a:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(B.meta["q_days"])
    df = pd.DataFrame({col: B.a[k][:, j].astype(float) for k, col in FIELDS.items() if k in B.a}, index=idx)
    for key,col in [('Q_HDATE','holder_date'),('Q_MDATE','margin_date')]:
        if key in B.a:df[col]=pd.to_datetime(B.a[key][:,j].astype(float),unit='D',origin='unix',errors='coerce')
    if 'Q_ADJ' in B.a:df['adj_index']=B.a['Q_ADJ'][:,j].astype(float)
    return df[df["close"].notna() & (df["close"] > 0)]


def resample(df: pd.DataFrame, period: str = "D") -> pd.DataFrame:
    """D 日K／W 週K／M 月K。價：開高低收；量與法人：加總；餘額與持股比例：期末值。"""
    if period == "D" or df.empty:
        return df
    rule = {"W": "W-FRI", "M": "ME"}[period]
    sum_known=lambda x:x.sum(min_count=1)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": sum_known,
           "foreign": sum_known, "trust": sum_known, "dealer": sum_known, "mbal": "last", "sbal": "last", "mlimit": "last",
           "fr": "last", "big400": "last", "big1000": "last", "retail50": "last", "holders": "last",
           'holder_date':'last','margin_date':'last','broker_net':sum_known}
    out = df.resample(rule).agg({k: v for k, v in agg.items() if k in df})
    out = out[out["close"].notna()]
    actual=pd.Series(df.index,index=df.index).resample(rule).last()
    out.index=pd.DatetimeIndex(actual.reindex(out.index))
    return out


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def kd(df: pd.DataFrame, n: int = 9) -> tuple[pd.Series, pd.Series]:
    """台股 KD(9,3,3)：RSV=(C−9日低)/(9日高−9日低)×100；K=⅔K前+⅓RSV；D=⅔D前+⅓K；起始 50。"""
    lo, hi = df["low"].rolling(n, min_periods=1).min(), df["high"].rolling(n, min_periods=1).max()
    rsv = ((df["close"] - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50).to_numpy()
    k = np.empty(len(rsv))
    d = np.empty(len(rsv))
    kp = dp = 50.0
    for i, r in enumerate(rsv):
        kp = kp * 2 / 3 + r / 3
        dp = dp * 2 / 3 + kp / 3
        k[i], d[i] = kp, dp
    return pd.Series(k, df.index), pd.Series(d, df.index)


def macd(close: pd.Series, fast=12, slow=26, sig=9):
    dif = ema(close, fast) - ema(close, slow)
    dem = ema(dif, sig)
    return dif, dem, dif - dem


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    total=up+dn
    return (100*up/total.where(total>0)).mask(total.eq(0),50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    c = out["close"]
    for n in MA_LINES:
        out[f"ma{n}"] = c.rolling(n, min_periods=n).mean()
    out["k"], out["d"] = kd(out)
    out["dif"], out["dem"], out["osc"] = macd(c)
    mid, sd = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std(ddof=0)
    out["bb_mid"], out["bb_up"], out["bb_dn"] = mid, mid + 2 * sd, mid - 2 * sd
    out["rsi"] = rsi(c)
    out["atr"] = atr(out)
    out["vma5"] = out["volume"].rolling(5, min_periods=1).mean()
    out["vma20"] = out["volume"].rolling(20, min_periods=1).mean()
    return out


# ------------------------------------------------------------------ 台股升降單位
def tick(p: float) -> float:
    if p < 10:
        return 0.01
    if p < 50:
        return 0.05
    if p < 100:
        return 0.1
    if p < 500:
        return 0.5
    if p < 1000:
        return 1.0
    return 5.0


def round_tick(p: float, mode: str = "near") -> float:
    if p is None or not np.isfinite(p) or p <= 0:
        return np.nan
    rounding={'near':ROUND_HALF_UP,'down':ROUND_FLOOR,'up':ROUND_CEILING}[mode]
    value=Decimal(str(p));step=Decimal(str(tick(p)))
    result=(value/step).to_integral_value(rounding=rounding)*step
    next_step=Decimal(str(tick(float(result))))
    if next_step!=step:result=(value/next_step).to_integral_value(rounding=rounding)*next_step
    return float(result.quantize(Decimal('0.01')))
