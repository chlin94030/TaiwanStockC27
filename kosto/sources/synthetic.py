"""Synthetic Taiwan-like market for demo mode and end-to-end tests.

Writes the SAME raw schema as the official/FinMind builders, so every downstream step
(panel → factors → pools → weights → backtest → report) is exercised exactly as with real data.
Planted, known effects let tests check that the research engine recovers real signal:
  * quality (revenue growth) drift            -> fundamental factors should get positive IC
  * informed foreign/trust flows               -> chip factors positive IC
  * margin (weak hands) chasing then reversal  -> Kostolany hands-transfer factor positive IC
  * 1-5 day reversal                           -> short-term reversal positive IC (5d)
  * hot volume-price spikes mean-revert        -> Kostolany volume-price factor positive IC
  * egg-cycle market regimes (A1..B3)          -> market stage exposure matters
NOTHING produced here is a statement about the real market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = {  # mu, sigma, turnover multiplier, margin growth/day, duration range
    "A1": (0.0010, 0.008, 0.70, -0.0002, (40, 90)),
    "A2": (0.0014, 0.009, 1.00, 0.0012, (120, 280)),
    "A3": (0.0016, 0.013, 1.60, 0.0030, (30, 70)),
    "B1": (-0.0008, 0.011, 0.85, 0.0003, (25, 60)),
    "B2": (-0.0009, 0.013, 1.10, -0.0020, (50, 130)),
    "B3": (-0.0030, 0.024, 1.90, -0.0070, (10, 25)),
}
ORDER = ["A1", "A2", "A3", "B1", "B2", "B3"]
SECTORS = ["半導體業", "電子零組件業", "電腦及週邊設備業", "光電業", "通信網路業", "其他電子業", "金融保險業",
           "航運業", "鋼鐵工業", "塑膠工業", "生技醫療業", "食品工業", "建材營造業", "電機機械", "汽車工業",
           "化學工業", "紡織纖維", "觀光餐旅"]


def _calendar(start, end, rng):
    days = pd.bdate_range(start, end)
    drop = set()
    for y in range(days[0].year, days[-1].year + 1):  # lunar new year + a few random holidays
        lny = pd.Timestamp(y, 2, 1) + pd.Timedelta(days=int(rng.integers(-12, 8)))
        drop.update(pd.bdate_range(lny, lny + pd.Timedelta(days=6)))
        for _ in range(5):
            drop.add(pd.Timestamp(y, int(rng.integers(1, 13)), int(rng.integers(1, 28))))
    return pd.DatetimeIndex([d for d in days if d not in drop])


def _regimes(n, rng):
    seq, i, k = [], 0, int(rng.integers(0, 6))
    while i < n:
        ph = ORDER[k % 6]
        lo, hi = PHASES[ph][4]
        L = int(rng.integers(lo, hi))
        seq += [ph] * L
        i += L
        k += 1
    return np.array(seq[:n])


def generate(n_stocks=600, start="2017-01-02", end="2026-10-23", seed=7):
    rng = np.random.default_rng(seed)
    cal = _calendar(start, end, rng)
    T, N = len(cal), n_stocks
    reg = _regimes(T, rng)
    mu = np.array([PHASES[p][0] for p in reg]); sig = np.array([PHASES[p][1] for p in reg])
    turn = np.array([PHASES[p][2] for p in reg]); mgro = np.array([PHASES[p][3] for p in reg])

    # ---- static stock attributes
    codes = sorted(rng.choice(np.arange(1101, 9960), size=N, replace=False))
    tickers = np.array([f"{c:04d}" for c in codes])
    board = np.where(rng.random(N) < 0.55, "TWSE", "TPEX")
    sector = rng.integers(0, len(SECTORS), N)
    beta = np.clip(rng.normal(1.0, 0.25, N), 0.3, 1.8)
    isig = rng.uniform(0.011, 0.028, N)
    mcap0 = np.exp(rng.normal(np.log(12e9), 1.5, N)); mcap0[:3] = [8e12, 2e12, 1.2e12]
    price0 = np.exp(rng.normal(np.log(45), 0.8, N)).clip(8, 900)
    shares = (mcap0 / price0).round(-3)
    base_turnover = np.exp(rng.normal(np.log(0.004), 0.7, N))  # daily volume / shares
    dy = rng.uniform(0.0, 0.065, N)
    start_i = np.where(rng.random(N) < 0.15, rng.integers(60, T - 300, N), 0)       # IPOs
    stop_i = np.where(rng.random(N) < 0.05, rng.integers(T // 3, T - 30, N), T)       # delistings
    margin_able = rng.random(N) < 0.85
    ex_day = rng.integers(105, 175, N)  # day-of-year index window for ex-dividend (~Jun-Aug)

    # ---- latent monthly quality (revenue growth driver)
    months = pd.period_range(cal[0] - pd.DateOffset(months=15), cal[-1], freq="M")
    M = len(months)
    q = np.zeros((M, N)); q[0] = rng.normal(0, 1, N)
    for m in range(1, M):
        q[m] = 0.92 * q[m - 1] + np.sqrt(1 - 0.92 ** 2) * rng.normal(0, 1, N)
    m_of_day = np.searchsorted(months.end_time.values, cal.values)  # month index for each day
    q_day = q[np.clip(m_of_day, 0, M - 1)]
    # revenue level: grows with quality; YoY ~ 0.12*q + noise; seasonality
    rev = np.zeros((M, N)); rev[:12] = (mcap0 / 25 / 12) * np.exp(rng.normal(0, 0.1, (12, N)))
    for m in range(12, M):
        yoy = 0.12 * q[m] + rng.normal(0, 0.10, N)
        rev[m] = rev[m - 12] * (1 + yoy).clip(0.3, 3)

    # ---- idiosyncratic shocks known up-front (so informed flows can "see" the future)
    E = rng.normal(0, 1, (T, N)) * isig
    csum = np.vstack([np.zeros(N), np.cumsum(E, 0)])
    fut20 = (csum[np.minimum(np.arange(T) + 21, T)] - csum[np.minimum(np.arange(T) + 1, T)]) / (isig * np.sqrt(20))
    fut10 = (csum[np.minimum(np.arange(T) + 11, T)] - csum[np.minimum(np.arange(T) + 1, T)]) / (isig * np.sqrt(10))
    sec_shock = rng.normal(0, 0.006, (T, len(SECTORS)))

    close = np.full((T, N), np.nan); ref = np.full((T, N), np.nan); openp = np.full((T, N), np.nan)
    high = np.full((T, N), np.nan); low = np.full((T, N), np.nan); vol = np.full((T, N), np.nan)
    event = np.zeros((T, N), dtype=np.int8)
    mbal = np.full((T, N), np.nan); fnet = np.zeros((T, N)); tnet = np.zeros((T, N)); dnet = np.zeros((T, N))
    px = price0.copy(); mb = shares * 0.002 / 1000 * margin_able  # margin balance in lots
    hist = np.zeros((25, N)); vhist = np.ones((120, N)) * base_turnover * shares
    peak = px.copy(); mb_hist = np.tile(mb, (60, 1))
    alive_prev = np.zeros(N, bool)
    doy = cal.dayofyear.values
    year = cal.year.values
    paid = {}
    for t in range(T):
        alive = (t >= start_i) & (t < stop_i)
        m = mu[t] + sig[t] * rng.normal()
        r20 = hist[-20:].sum(0)
        vr = vhist[-20:].mean(0) / np.maximum(vhist.mean(0), 1)
        hot = (r20 > 0.12) & (vr > 1.5)
        dd = px / np.maximum(peak, 1e-9) - 1
        mchg = mb / np.maximum(mb_hist[0], 1e-9) - 1
        flush = (dd < -0.35) & (vr > 1.4) & (mchg < -0.15)
        rev5 = -0.03 * hist[-5:].sum(0)
        r = (beta * m + sec_shock[t, sector] + E[t] + 0.00012 * q_day[t] + rev5
             - 0.0005 * hot + 0.0006 * flush - 0.0004 * np.clip(mchg, -1, 1))
        r = np.clip(r, -0.095, 0.095)
        prev = px.copy()
        # ex-dividend once a year
        exd = alive & alive_prev & (doy[t] >= ex_day) & np.array([(year[t], i) not in paid for i in range(N)]) & (dy > 0.003)
        rf = prev.copy()
        if exd.any():
            div = prev * dy
            rf = np.where(exd, np.round(prev - div, 2), prev)
            for i in np.where(exd)[0]:
                paid[(year[t], i)] = 1
        px = np.where(alive, rf * (1 + r), px)
        newly = alive & ~alive_prev
        px = np.where(newly, price0, px)
        rf = np.where(newly, price0 / (1 + r), rf)
        gap = np.clip(r * rng.uniform(0.1, 0.6, N) + rng.normal(0, 0.003, N), -0.095, 0.095)
        o = rf * (1 + gap)
        hi = np.maximum(o, px) * (1 + np.abs(rng.normal(0, 0.006, N)))
        lo = np.minimum(o, px) * (1 - np.abs(rng.normal(0, 0.006, N)))
        v = shares * base_turnover * turn[t] * np.exp(rng.normal(0, 0.35, N)) * (1 + 10 * np.abs(r)) * (1 + 0.8 * hot)
        # institutional & margin flows (shares)
        f = v * (0.0035 * fut20[t] + 0.12 * rng.normal(0, 1, N)) * (np.log(mcap0) / 25)
        tr = v * (0.003 * fut10[t] + 0.06 * rng.normal(0, 1, N)) * (mcap0 < 5e10)
        de = v * 0.05 * rng.normal(0, 1, N)
        mb = np.where(margin_able & alive,
                      mb * (1 + mgro[t] + 0.15 * hist[-5:].sum(0) - 0.0003 * fut20[t] + rng.normal(0, 0.01, N)), mb)
        mb = np.maximum(mb, 0)
        close[t] = np.where(alive, np.round(px, 2), np.nan); ref[t] = np.where(alive, rf, np.nan)
        openp[t] = np.where(alive, np.round(o, 2), np.nan); high[t] = np.where(alive, np.round(hi, 2), np.nan)
        low[t] = np.where(alive, np.round(lo, 2), np.nan); vol[t] = np.where(alive, np.round(v, -3), np.nan)
        event[t] = exd
        fnet[t] = np.where(alive, np.round(f, -3), 0); tnet[t] = np.where(alive, np.round(tr, -3), 0)
        dnet[t] = np.where(alive, np.round(de, -3), 0); mbal[t] = np.where(alive & margin_able, np.round(mb), np.nan)
        hist = np.vstack([hist[1:], np.where(alive, r, 0)])
        vhist = np.vstack([vhist[1:], np.where(alive, v, vhist[-1])])
        peak = np.where(newly, px, np.maximum(peak * 0.9995, px))
        mb_hist = np.vstack([mb_hist[1:], mb])
        alive_prev = alive

    return _to_tables(cal, tickers, board, sector, shares, close, ref, openp, high, low, vol, event,
                      fnet, tnet, dnet, mbal, dy, q_day, months, rev, mcap0, reg)


def _to_tables(cal, tickers, board, sector, shares, close, ref, openp, high, low, vol, event,
               fnet, tnet, dnet, mbal, dy, q_day, months, rev, mcap0, reg):
    T, N = close.shape
    D = np.repeat(cal.values, N); TK = np.tile(tickers, T); BD = np.tile(board, T)
    valid = np.isfinite(close.ravel())
    chg = (close - np.round(ref, 2)).ravel()
    ev = event.ravel().astype(bool)
    ev_label = np.where(BD == "TWSE", "X", "除息")
    price = pd.DataFrame({"date": D, "ticker": TK, "board": BD, "name": np.char.add("合成", TK),
                          "open": openp.ravel(), "high": high.ravel(), "low": low.ravel(), "close": close.ravel(),
                          "volume": vol.ravel(), "amount": (vol * (high + low + close) / 3).ravel().round(),
                          "change": np.where(ev, np.nan, np.round(chg, 2)), "event": np.where(ev, ev_label, ""),
                          "shares": np.where(BD == "TPEX", np.tile(shares, T), np.nan)})[valid]
    # benchmark 0050 = cap-weighted top-50 ; TAIEX = cap-weighted all
    rets = np.nan_to_num(close / np.round(ref, 2) - 1)
    mc = np.nan_to_num(close * shares)
    top = np.argsort(-mcap0)[:50]
    def idx(cols, base):
        w = mc[:, cols]; w = np.vstack([w[:1], w[:-1]])
        rr = (w * rets[:, cols]).sum(1) / np.maximum(w.sum(1), 1)
        return base * np.cumprod(1 + rr), rr
    p50, r50 = idx(top, 75.0)
    ptw, rtw = idx(np.arange(N), 9300.0)
    bench = pd.DataFrame({"date": cal, "ticker": "0050", "board": "TWSE", "name": "元大台灣50(合成)",
                          "close": np.round(p50, 2), "change": np.round(p50 - p50 / (1 + r50), 2), "event": ""})
    bench["open"] = bench["close"]; bench["high"] = bench["close"]; bench["low"] = bench["close"]
    bench["volume"] = 1e7; bench["amount"] = bench["volume"] * bench["close"]; bench["shares"] = np.nan
    taiex = pd.DataFrame({"date": cal, "ticker": "TAIEX", "board": "INDEX", "name": "加權指數(合成)",
                          "close": np.round(ptw, 2), "change": np.round(ptw - ptw / (1 + rtw), 2), "event": ""})
    price = pd.concat([price, bench, taiex], ignore_index=True)

    exr = pd.DataFrame({"date": D, "ticker": TK, "board": BD, "before": np.vstack([close[:1], close[:-1]]).ravel(),
                        "ref": np.round(ref, 2).ravel()})[ev]
    inst = pd.DataFrame({"date": D, "ticker": TK, "foreign": fnet.ravel(), "trust": tnet.ravel(),
                         "dealer": dnet.ravel()})[valid]
    inst["total"] = inst["foreign"] + inst["trust"] + inst["dealer"]
    mlim = np.tile(shares / 1000 * 0.25, T)
    margin = pd.DataFrame({"date": D, "ticker": TK, "mbal": mbal.ravel(), "mlimit": mlim,
                           "sbal": (mbal * 0.05).ravel().round()})[valid & np.isfinite(mbal.ravel())]
    # valuation: eps ~ quality; pe blank when eps <= 0
    eps = (np.tile(mcap0 / shares, (T, 1)) / 16) * (1 + 0.25 * q_day) - np.tile(mcap0 / shares, (T, 1)) * 0.02
    bvps = np.tile(mcap0 / shares, (T, 1)) / 1.6
    pe = np.where(eps > 0, close / eps, np.nan)
    val = pd.DataFrame({"date": D, "ticker": TK, "pe": np.round(pe, 2).ravel(), "pb": np.round(close / bvps, 2).ravel(),
                        "dy": np.round(np.tile(dy, T) * 100, 2)})[valid]
    sh = pd.DataFrame({"date": D, "ticker": TK, "shares": np.tile(shares, T), "foreign_ratio": 20.0})[valid & (BD == "TWSE")]
    sh = sh[sh["date"].dt.day <= 7].groupby([sh["date"].dt.to_period("M"), "ticker"]).head(1)
    ym = (months.year * 100 + months.month).values
    revenue = pd.DataFrame({"ticker": np.tile(tickers, len(months)), "ym": np.repeat(ym, N), "revenue": rev.ravel().round()})
    from .official import revenue_available_date
    avail = {y: revenue_available_date(y) for y in ym}
    revenue["avail_date"] = revenue["ym"].map(avail)
    revenue = revenue[revenue["avail_date"] <= cal[-1]]
    info = pd.DataFrame({"ticker": tickers, "name": np.char.add("合成", tickers), "sector": np.array(SECTORS)[sector],
                         "board": board})
    info = pd.concat([info, pd.DataFrame([{"ticker": "0050", "name": "元大台灣50(合成)", "sector": "ETF", "board": "TWSE"}])])
    truth = pd.DataFrame({"date": cal, "regime": reg})
    return {"price": price, "exright": exr, "institutional": inst, "margin": margin, "valuation": val,
            "shares": sh, "revenue": revenue, "info": info, "_regime_truth": truth}


def write_demo(store, n_stocks=600, seed=7, start="2017-01-02", end="2026-10-23") -> dict:
    tables = generate(n_stocks, start, end, seed)
    for k, f in tables.items():
        if k.startswith("_"):
            f.to_parquet(store.root / f"{k.strip('_')}.parquet", index=False)
            continue
        store.append(k, f)
    (store.root / "SYNTHETIC_DEMO_DATA").write_text("synthetic demo data — not real market data\n", encoding="utf-8")
    return {k: len(v) for k, v in tables.items()}
