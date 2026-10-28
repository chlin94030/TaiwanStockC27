"""科斯托蘭尼 (André Kostolany) concepts translated into testable, point-in-time quantities.

Book idea (一個投機者的告白)                          → quantity used here
---------------------------------------------------------------------------------------------------
雞蛋理論 A1..B3：成交量 × 股票持有人數量                → volume ratio (20d/120d) × margin balance
  (持有人數 ≈ 融資餘額：借錢買股的「猶豫的投資者」)         (level vs own 1-yr history, 60d change)
A1、B3 買進；A2 續抱；A3、B1 賣出；B2 持現金等待           → stage gates + market exposure profiles
固執的投資者 vs 猶豫的投資者（股票已付清 vs 融資）          → k_hands: 法人買超 − 融資增加 ; k_paid
成交量小價漲＝有利；量大價漲＝危險；量增價跌＝接近底部；
  量小價跌＝前景堪慮                                      → k_vp = −sign(ret20)·log(vol20/vol120)
利空不跌＝超賣近底；利多不漲＝超買近頂                      → k_resilience (excess return on market's
                                                            worst / best days)
M 頭（多次攻不過）＝上限、W 底（多次跌不破）＝有人護盤        → k_mw, double-top gate, B3 confirmation
戒1 不跟著建議跑 / 媒體樂觀＝過熱                          → k_calm (−crowding: volume spike, price
                                                            spike, margin surge)
Holder counts, news reactions and money supply are NOT directly observed; the proxies are disclosed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STAGES
from .factors import Ctx, factor

F32 = np.float32


def clip01(x):
    return x.clip(0, 1) if isinstance(x, pd.DataFrame) else np.clip(x, 0, 1)


def wavg(parts, weights):
    """NaN-aware weighted average of component DataFrames (missing margin data → renormalise)."""
    num = None
    den = None
    for p, w in zip(parts, weights):
        p = p.astype(float)
        ok = p.notna()
        term = p.fillna(0) * w
        wt = ok.astype(float) * w
        num = term if num is None else num + term
        den = wt if den is None else den + wt
    return num / den.where(den > 0)


# ------------------------------------------------------------------ stock-level components
def components(c: Ctx) -> dict:
    def build():
        a = c.adj
        mx, mn = a.rolling(252, min_periods=120).max(), a.rolling(252, min_periods=120).min()
        pos = ((a - mn) / (mx - mn).where(mx > mn)).clip(0, 1)
        dd = a / mx - 1
        ma20, ma60 = c.ma(20), c.ma(60)
        ret20 = c.mom(20)
        vr = c.vma(20) / c.vma(120)
        bias20 = a / ma20 - 1
        mratio = c.mbal * 1000 / c.shares
        mz = (mratio - mratio.rolling(250, min_periods=120).mean()) / mratio.rolling(250, min_periods=120).std()
        mchg60 = c.mbal / c.mbal.shift(60).where(c.mbal.shift(60) > 0) - 1
        up_trend = clip01((a / ma60 - 1) / 0.08) * (ma60 > ma60.shift(10)).astype(float)
        down_trend = clip01((1 - a / ma60) / 0.08) * (ma60 < ma60.shift(10)).astype(float)
        return {"pos": pos, "dd": dd, "ret20": ret20, "vr": vr, "bias20": bias20, "mz": mz, "mchg60": mchg60,
                "up_trend": up_trend, "down_trend": down_trend, "ma20": ma20}
    return c.get("k_components", build)


def stage_scores(c: Ctx) -> dict:
    def build():
        k = components(c)
        pos, dd, ret20, vr, mz, mchg = k["pos"], k["dd"], k["ret20"], k["vr"], k["mz"], k["mchg60"]
        vol_low = clip01((1.0 - vr) / 0.4)
        vol_up = clip01((vr - 1.0) / 0.5)
        vol_hot = clip01((vr - 1.3) / 0.7)
        vol_mod = clip01(1 - (vr - 1.15).abs() / 0.5)
        m_high, m_low = clip01(mz / 1.5), clip01(-mz / 1.5)
        m_up, m_down = clip01(mchg / 0.3), clip01(-mchg / 0.3)
        m_flush = clip01((-mchg - 0.15) / 0.25)
        turn_up = clip01(ret20 / 0.08) * (c.adj > k["ma20"]).astype(float)
        falling = clip01(-ret20 / 0.08)
        s = {
            # A1 修正：量小、持有人少，價格自低檔回升
            "A1": wavg([clip01((0.45 - pos) / 0.35), turn_up, vol_low, wavg([m_low, m_down], [1, 1])], [.3, .3, .2, .2]),
            # A2 相隨：量與持有人增加，趨勢向上
            "A2": wavg([k["up_trend"], clip01(1 - (pos - 0.65).abs() / 0.35), vol_mod,
                        clip01(mchg / 0.15) * (1 - m_high)], [.35, .2, .25, .2]),
            # A3 過熱：量異常活躍、持有人最多、價格高檔乖離大
            "A3": wavg([clip01((pos - 0.8) / 0.2), clip01((k["bias20"] - 0.08) / 0.12), vol_hot,
                        wavg([m_high, m_up], [1, 1])], [.3, .25, .25, .2]),
            # B1 修正下跌：量小、持有人仍多但開始減少，自高點回落
            "B1": wavg([clip01((-dd - 0.05) / 0.15) * (pos > 0.4).astype(float), falling, vol_low,
                        wavg([m_high, 1 - m_down], [1, 1])], [.3, .3, .2, .2]),
            # B2 相隨下跌：量增加、持有人繼續減少
            "B2": wavg([k["down_trend"], clip01((0.6 - pos) / 0.4), vol_up, m_down], [.35, .15, .25, .25]),
            # B3 過熱下跌：量很大、持有人最少（融資斷頭出清）、深度回檔
            "B3": wavg([clip01((-dd - 0.25) / 0.25), vol_hot, m_flush, clip01((0.2 - pos) / 0.2)], [.3, .3, .25, .15]),
        }
        return s
    return c.get("k_stage_scores", build)


def stage_codes(c: Ctx, threshold: float = 0.35) -> pd.DataFrame:
    """0 = UNKNOWN, 1..6 = A1..B3 (argmax of stage scores, needs max >= threshold)."""
    def build():
        s = stage_scores(c)
        arr = np.stack([s[k].to_numpy(dtype=float) for k in STAGES], axis=0)
        best = np.nanargmax(np.where(np.isfinite(arr), arr, -1), axis=0)
        top = np.nanmax(np.where(np.isfinite(arr), arr, -1), axis=0)
        code = np.where(top >= threshold, best + 1, 0).astype(np.int8)
        return pd.DataFrame(code, index=c.P.dates, columns=c.P.tickers)
    return c.get("k_stage_codes", build)


def mw_patterns(c: Ctx, W: int = 120, band: float = 0.03, dip: float = 0.06) -> dict:
    """Count separate visits to the window high (M ceiling) and low (W floor)."""
    def build():
        hi = c.hi.to_numpy(dtype=float); lo = c.lo.to_numpy(dtype=float); cl = c.adj.to_numpy(dtype=float)
        H = c.hi.rolling(W, min_periods=W // 2).max().to_numpy(dtype=float)
        L = c.lo.rolling(W, min_periods=W // 2).min().to_numpy(dtype=float)
        T, N = hi.shape
        cnt_t = np.zeros((T, N), np.int16); cnt_b = np.zeros((T, N), np.int16)
        arm_t = np.ones((T, N), bool); arm_b = np.ones((T, N), bool)
        top_lvl, dip_lvl = H * (1 - band), H * (1 - dip)
        bot_lvl, reb_lvl = L * (1 + band), L * (1 + dip)
        with np.errstate(invalid="ignore"):
            for k in range(W - 1, -1, -1):
                hk = np.full((T, N), np.nan); lk = np.full((T, N), np.nan)
                if k:
                    hk[k:] = hi[:-k]; lk[k:] = lo[:-k]
                else:
                    hk, lk = hi, lo
                nt = hk >= top_lvl
                cnt_t += (nt & arm_t)
                arm_t = np.where(nt, False, arm_t) | (lk < dip_lvl)
                nb = lk <= bot_lvl
                cnt_b += (nb & arm_b)
                arm_b = np.where(nb, False, arm_b) | (hk > reb_lvl)
            below_top = cl < top_lvl
            m_score = np.clip(cnt_t - 1, 0, 3) * below_top
            breakout = (cnt_t >= 2) & (cl >= H * 0.995)
            w_score = np.clip(cnt_b - 1, 0, 3) * (cl > bot_lvl)
        hi_df = c.hi
        b0 = hi_df.rolling(20, min_periods=15).max(); b1 = b0.shift(20); b2 = b0.shift(40)
        lo_df = c.lo
        l0 = lo_df.rolling(20, min_periods=15).min(); l1 = l0.shift(20); l2 = l0.shift(40)
        hh = ((b0 > b1).astype(float) + (b1 > b2).astype(float)) / 2
        ll = ((l0 < l1).astype(float) + (l1 < l2).astype(float)) / 2
        idx, cols = c.P.dates, c.P.tickers
        mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)  # noqa: E731
        mom20 = c.mom(20)
        return {"m": mk(m_score.astype(float)), "w": mk(w_score.astype(float)), "breakout": mk(breakout.astype(float)),
                "hh": hh, "ll": ll,
                "double_top": mk((m_score >= 1) & (cl < H * 0.95)) & (mom20 < 0),
                "w_support": mk(w_score >= 1)}
    return c.get("k_mw", build)


# ------------------------------------------------------------------ K factors
@factor("k_egg", "K", 1, ("short", "mid", "long"), "雞蛋理論個股階段分數(A1/B3買、A3/B1賣)")
def _k_egg(c):
    s = stage_scores(c)
    return s["A1"] + s["B3"] + 0.5 * s["A2"] - s["A3"] - s["B1"] - 0.7 * s["B2"]


@factor("k_vp", "K", 1, ("short", "mid", "long"), "量價關係(量小價漲有利、量增價跌近底)")
def _k_vp(c):
    k = components(c)
    lv = np.log(k["vr"].where(k["vr"] > 0)).clip(-1.5, 1.5)
    return -np.sign(k["ret20"]) * lv * clip01(k["ret20"].abs() / 0.05)


@factor("k_hands", "K", 1, ("short", "mid", "long"), "籌碼轉移：法人買超−融資增加(猶豫者→固執者)")
def _k_hands(c):
    vol20 = c.vma(20) * 20
    inst = c.rsum("total", 20) / vol20
    m = c.mbal
    marg = ((m - m.shift(20)) * 1000 / vol20).fillna(0)
    return inst - marg


@factor("k_resilience", "K", 1, ("short", "mid", "long"), "利空不跌／利多不漲")
def _k_res(c):
    P = c.P
    r = P["ret"].astype(float).where(P["tradable"])
    mret = market_return(c)
    q_lo = mret.rolling(250, min_periods=60).quantile(0.15)
    q_hi = mret.rolling(250, min_periods=60).quantile(0.85)
    down = (mret <= q_lo).astype(float)
    up = (mret >= q_hi).astype(float)
    ex = r.sub(mret, axis=0)
    dn_ex = ex.mul(down, axis=0).rolling(120, min_periods=40).sum() / down.rolling(120, min_periods=40).sum().clip(lower=1).values[:, None]
    up_ex = ex.mul(up, axis=0).rolling(120, min_periods=40).sum() / up.rolling(120, min_periods=40).sum().clip(lower=1).values[:, None]
    pos = components(c)["pos"]
    return dn_ex + np.minimum(up_ex, 0) * (pos > 0.8).astype(float)


@factor("k_mw", "K", 1, ("short", "mid", "long"), "W底護盤／M頭上限(雙重底頂)")
def _k_mw(c):
    p = mw_patterns(c)
    return p["w"] - p["m"] + 1.5 * p["breakout"] + (p["hh"] - p["ll"])


@factor("k_paid", "K", 1, ("mid", "long"), "股票已付清(融資占股本低)")
def _k_paid(c):
    ratio = (c.mbal * 1000 / c.shares).where(c.adj.notna())
    return -ratio.fillna(0).where(c.adj.notna())


@factor("k_calm", "K", 1, ("short", "mid"), "不追熱門(避開爆量、急漲、融資暴增)")
def _k_calm(c):
    vz = np.log(c.vma(5) / c.vma(60)).clip(0, 3)
    spike = (c.mom(3) - 0.08).clip(lower=0) * 5
    m = c.mbal
    surge = (m / m.shift(5).where(m.shift(5) > 0) - 1).clip(0, 1).fillna(0)
    return -(vz + spike + surge)


# ------------------------------------------------------------------ market level egg (regime)
def market_return(c: Ctx) -> pd.Series:
    def build():
        P = c.P
        if P.taiex is not None and P.taiex.notna().sum() > 200:
            return (P.taiex / P.taiex.shift(1) - 1).fillna(0)
        if P.bench is not None:
            return P.bench.fillna(0)
        r = P["ret"].astype(float).where(P["tradable"])
        return r.mean(axis=1).fillna(0)
    return c.get("mkt_ret", build)


def market_regime(c: Ctx, smooth: int = 10) -> pd.DataFrame:
    """Daily market egg stage. Columns: A1..B3 scores, stage, confidence, and the inputs."""
    def build():
        P = c.P
        mr = market_return(c)
        idx = (1 + mr).cumprod()
        mx, mn = idx.rolling(250, min_periods=120).max(), idx.rolling(250, min_periods=120).min()
        pos = ((idx - mn) / (mx - mn)).clip(0, 1)
        dd = idx / mx - 1
        ret20 = idx / idx.shift(20) - 1
        ma60, ma200 = idx.rolling(60).mean(), idx.rolling(200, min_periods=150).mean()
        amt = P["amount"].astype(float).where(P["tradable"]).sum(axis=1)
        vr = amt.rolling(20).mean() / amt.rolling(250, min_periods=120).mean()
        mval = (P["mbal"].astype(float) * 1000 * P["close"].astype(float).ffill()).sum(axis=1, min_count=50)
        has_margin = mval.notna().mean() > 0.5
        if has_margin:
            mz = (mval - mval.rolling(500, min_periods=200).mean()) / mval.rolling(500, min_periods=200).std()
            mchg = mval / mval.shift(60) - 1
        else:
            mz = pd.Series(np.nan, index=P.dates); mchg = pd.Series(np.nan, index=P.dates)
        adj = c.adj
        breadth = (adj > c.ma(60)).where(adj.notna()).mean(axis=1)
        S = lambda x: x.clip(0, 1)  # noqa: E731

        def w(parts, ws):
            num = sum(p.fillna(0) * q for p, q in zip(parts, ws))
            den = sum(p.notna().astype(float) * q for p, q in zip(parts, ws))
            return num / den.where(den > 0)

        up = S((idx / ma60 - 1) / 0.05) * (ma60 > ma60.shift(10)).astype(float)
        down = S((1 - idx / ma60) / 0.05) * (ma60 < ma60.shift(10)).astype(float)
        sc = pd.DataFrame({
            "A1": w([S((0.45 - pos) / 0.35), S(ret20 / 0.05), S((1 - vr) / 0.3), S(-mz / 1.5), S((breadth - 0.3) / 0.3)],
                    [.25, .25, .2, .15, .15]),
            "A2": w([up, S((idx / ma200 - 1) / 0.08), S(1 - (vr - 1.1).abs() / 0.4), S(mchg / 0.15), S((breadth - 0.5) / 0.3)],
                    [.3, .2, .2, .15, .15]),
            "A3": w([S((pos - 0.85) / 0.15), S((vr - 1.25) / 0.5), S(mz / 1.5), S((ret20 - 0.05) / 0.08), S((breadth - 0.7) / 0.2)],
                    [.25, .25, .2, .15, .15]),
            "B1": w([S((-dd - 0.04) / 0.10) * (pos > 0.4), S(-ret20 / 0.05), S((1 - vr) / 0.3), S(mz / 1.5)],
                    [.3, .3, .2, .2]),
            "B2": w([down, S((1 - idx / ma200) / 0.08), S((vr - 1.0) / 0.4), S(-mchg / 0.2), S((0.45 - breadth) / 0.3)],
                    [.3, .2, .15, .2, .15]),
            "B3": w([S((-dd - 0.15) / 0.2), S((vr - 1.3) / 0.6), S((-mchg - 0.1) / 0.2), S((0.25 - breadth) / 0.2)],
                    [.3, .25, .25, .2]),
        }, index=P.dates)
        sm = sc.ewm(span=smooth, adjust=False, min_periods=3).mean()
        arr = sm[list(STAGES)].to_numpy(dtype=float)
        ok = np.isfinite(arr).all(axis=1)
        best = np.where(ok, np.nanargmax(np.nan_to_num(arr, nan=-1), axis=1), -1)
        srt = np.sort(np.nan_to_num(arr, nan=0), axis=1)
        conf = np.where(ok, (srt[:, -1] - srt[:, -2]) / np.maximum(srt[:, -1], 1e-9), 0)
        out = sm.copy()
        out["stage"] = [STAGES[i] if i >= 0 else "UNKNOWN" for i in best]
        out["confidence"] = conf
        out["index"] = idx
        out["drawdown"] = dd
        out["turnover_ratio"] = vr
        out["margin_z"] = mz
        out["breadth"] = breadth
        return out
    return c.get("mkt_regime", build)
