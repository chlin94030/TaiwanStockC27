"""精選母池 (candidate pools). Reconstituted monthly with date-t information only.

Rules
  ADV     近一年日均成交金額（舊版 Alpha Radar 的對照組）
  MCAP    市值
  INST2Y  近2年三大法人累計買賣超金額（使用者指定的「近2年法人買賣超排行」）
  KOSTO   科斯托蘭尼「固執者」母池：流動性 35% + 法人2年累計買超/市值 25%
          + 股票已付清(融資占股本低) 20% + 獲利品質(ROE) 20%
  BLEND   市值 40% + 成交金額 30% + 法人2年買超 30%
Sizes 100..500 are tested per horizon; the research picks the best on the development period only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import Ctx
from .optimize import avg_rank_cols

RULES = ("ADV", "MCAP", "INST2Y", "KOSTO", "BLEND")
RULE_ZH = {"ADV": "成交金額", "MCAP": "市值", "INST2Y": "近2年法人買超", "KOSTO": "科斯托蘭尼固執者池", "BLEND": "綜合"}


def _pct(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile (0..1) among mask; NaN elsewhere. Row-wise."""
    out = np.full(a.shape, np.nan)
    for i in range(a.shape[0]):
        m = mask[i] & np.isfinite(a[i])
        if m.sum() < 2:
            continue
        r = avg_rank_cols(a[i, m][:, None].astype(float))[:, 0]
        out[i, m] = r / (m.sum() - 1)
    return out


def pool_inputs(ctx: Ctx, sig_idx: np.ndarray, cfg: dict) -> dict:
    P = ctx.P
    r = cfg["research"]
    S = lambda df: df.to_numpy(dtype=float)[sig_idx]  # noqa: E731
    close = S(P["close"])
    amt = P["amount"].astype(float).where(P["tradable"])
    amt20 = S(amt.rolling(20, min_periods=10).mean())
    adv = S(amt.rolling(252, min_periods=60).mean())
    mcap = S(P["mcap"])
    inst_val = (P["total"].astype(float) * P["close"].astype(float).ffill()).rolling(500, min_periods=60).sum()
    inst2y = S(inst_val)
    mratio = S((P["mbal"].astype(float) * 1000 / P["shares"].astype(float)).fillna(0))
    pe, pb = S(P["pe"]), S(P["pb"])
    roe = np.where(np.isfinite(pe), pb / pe, np.where(np.isfinite(pb), -0.05, np.nan))
    listed = S(P["listed_days"])
    tradable = S(P["tradable"].astype(float)) > 0
    elig = (tradable & (close >= r["min_price"]) & (amt20 >= r["min_adv"]) & (listed >= r["min_history"]))
    return {"close": close, "amt20": amt20, "adv": adv, "mcap": mcap, "inst2y": inst2y, "mratio": mratio,
            "roe": roe, "elig": elig}


def pool_ranks(inp: dict, sig_dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """rank[rule] -> int16 [n_sig, n_tk]: 1 = best, 0 = not eligible. Fixed within each month."""
    elig = inp["elig"]
    scores = {
        "ADV": inp["adv"],
        "MCAP": inp["mcap"],
        "INST2Y": inp["inst2y"],
        "KOSTO": 0.35 * _pct(inp["adv"], elig)
                 + 0.25 * _pct(inp["inst2y"] / np.where(inp["mcap"] > 0, inp["mcap"], np.nan), elig)
                 + 0.20 * _pct(-inp["mratio"], elig)
                 + 0.20 * np.nan_to_num(_pct(inp["roe"], elig), nan=0.5),
        "BLEND": 0.4 * _pct(inp["mcap"], elig) + 0.3 * _pct(inp["adv"], elig) + 0.3 * _pct(inp["inst2y"], elig),
    }
    months = sig_dates.to_period("M")
    recon = np.r_[True, months[1:] != months[:-1]]  # first signal date of each month
    out = {}
    for rule, sc in scores.items():
        rk = np.zeros(sc.shape, np.int16)
        cur = None
        for i in range(sc.shape[0]):
            if recon[i] or cur is None:
                v = np.where(elig[i] & np.isfinite(sc[i]), sc[i], -np.inf)
                order = np.argsort(-v, kind="stable")
                cur = np.zeros(sc.shape[1], np.int16)
                n_ok = int(np.isfinite(v).sum())
                cur[order[:n_ok]] = np.arange(1, n_ok + 1, dtype=np.int16)
            rk[i] = cur
        out[rule] = rk
    return out


def membership(ranks: dict, rule: str, size: int) -> np.ndarray:
    rk = ranks[rule]
    return (rk >= 1) & (rk <= size)
