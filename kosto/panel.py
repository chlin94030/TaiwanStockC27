"""Wide date × ticker matrices built from the raw store.

Total-return construction (verified against real TWSE/TPEx/FinMind payloads):
    reference price  ref_t = close_t - signed change_t         (normal days, splits, capital reductions)
                     ref_t = 除權息參考價 from TWT49U / exDailyQ   (ex-right / ex-dividend days: 'X', '除息')
    daily return     r_t   = close_t / ref_t - 1                (dividend reinvested, split-safe)
    entry-day return oc_t  = close_t / open_t - 1               (buy at the open after the signal)
    limit-up open    lu_t  = open_t / ref_t - 1 >= 9.5%          (cannot be bought → no fill)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sources.official import revenue_available_date
from .store import Store

STOCK_RE = r"^[1-9]\d{3}$"
F32 = np.float32


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: pd.Index
    m: dict = field(default_factory=dict)            # name -> DataFrame (dates × tickers)
    info: pd.DataFrame = None                        # index ticker: name, sector, board
    bench: pd.Series = None                          # 0050 daily total return
    taiex: pd.Series = None                          # TAIEX close
    rev_m: pd.DataFrame = None                       # monthly revenue (index ym int, columns ticker)
    synthetic: bool = False
    quality: dict = field(default_factory=dict)

    def __getitem__(self, k):
        return self.m[k]

    def has(self, k) -> bool:
        return k in self.m and np.isfinite(self.m[k].to_numpy(dtype=float)).any()


def _pivot(f: pd.DataFrame, col: str, dates, tickers) -> pd.DataFrame:
    if f is None or f.empty or col not in f:
        return pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=F32)
    g = f[["date", "ticker", col]].drop_duplicates(["date", "ticker"], keep="last")
    p = g.set_index(["date", "ticker"])[col].astype(F32).unstack("ticker")
    p.columns = p.columns.astype(str)
    return p.reindex(index=dates, columns=tickers).astype(F32)


def build_panel(store: Store, start=None, end=None, min_names_per_day: int = 50) -> Panel:
    price = store.read("price", start, end)
    if price.empty:
        raise ValueError("沒有價格資料：請先執行資料建置（或 demo）")
    price["ticker"] = price["ticker"].astype(str)
    for c in ("board", "event", "name"):
        if c in price:
            price[c] = price[c].astype("category")   # memory: ~4.5M rows on the full market
    stocks = price[price["ticker"].str.match(STOCK_RE)]
    counts = stocks.dropna(subset=["close"]).groupby("date").size()
    dates = pd.DatetimeIndex(sorted(counts[counts >= min(min_names_per_day, counts.max() * 0.5)].index))
    tickers = pd.Index(sorted(stocks["ticker"].unique()))
    P = Panel(dates=dates, tickers=tickers)
    P.synthetic = (store.root / "SYNTHETIC_DEMO_DATA").exists()

    for c in ("open", "high", "low", "close", "volume", "amount", "change"):
        P.m[c] = _pivot(stocks, c, dates, tickers)
    for c in ("open", "high", "low", "close"):
        P.m[c] = P.m[c].where(P.m[c] > 0)
    ev = stocks.assign(ev=(stocks["event"].astype(str).replace({"nan": ""}) != "").astype(F32))
    event = _pivot(ev, "ev", dates, tickers).fillna(0) > 0

    exr = store.read("exright", start, end)
    exref = _pivot(exr, "ref", dates, tickers) if not exr.empty else _pivot(None, "ref", dates, tickers)
    close = P.m["close"]
    prev_close = close.ffill().shift(1)
    ref = close - P.m["change"]
    ref = ref.where(exref.isna(), exref)                      # official ex-rights reference wins
    # TWSE also prints 'X' on non-dividend days (創新板, 全額交割, first day after a halt…). Real dividend days are
    # in the ex-rights table; for the rest the previous close IS the reference (verified: 97% move < ±10%).
    ref = ref.where(ref.notna() | ~close.notna(), prev_close)  # missing change → previous close
    ratio = ref / prev_close
    ref = ref.where((ratio > 0.02) & (ratio < 50) | prev_close.isna())
    ret = close / ref - 1
    first_days = close.notna().cumsum() <= 5                  # IPO honeymoon: no price limit
    bad = (ret.abs() > 0.105) & ~first_days & exref.isna()
    P.quality["suspicious_returns"] = int(bad.to_numpy().sum())
    P.quality["x_days_using_prev_close"] = int((event & exref.isna() & close.notna()).to_numpy().sum())
    ret = ret.where(~bad)
    P.m["ref"] = ref.astype(F32)
    P.m["ret"] = ret.fillna(0).astype(F32)                    # non-trading day = 0 return
    P.m["oc"] = (close / P.m["open"] - 1).astype(F32)
    P.m["gap"] = (P.m["open"] / ref - 1).astype(F32)
    P.m["tradable"] = (close.notna() & (P.m["volume"] > 0)).astype(bool)
    P.m["adj"] = (1 + P.m["ret"].astype(float)).cumprod().where(close.ffill().notna()).astype(F32)
    P.m["listed_days"] = close.notna().cumsum().astype(F32)

    # benchmark / index
    b = price[price["ticker"] == "0050"].set_index("date").reindex(dates)
    if not b.empty and b["close"].notna().any():
        bref = b["close"] - b["change"]
        bex = exr[exr["ticker"] == "0050"].set_index("date")["ref"].reindex(dates) if not exr.empty else None
        if bex is not None:
            bref = bref.where(bex.isna(), bex)
        bref = bref.fillna(b["close"].ffill().shift(1))      # 'X' without a dividend → previous close
        P.bench = (b["close"] / bref - 1).fillna(0).clip(-0.2, 0.2)
    t = price[price["ticker"] == "TAIEX"].set_index("date").reindex(dates)
    if not t.empty and t["close"].notna().any():
        P.taiex = t["close"].ffill()

    # shares: TPEx from daily quotes, TWSE from MI_QFIIS monthly (or FinMind) → forward fill
    sh = store.read("shares", None, end)
    sh_daily = _pivot(stocks, "shares", dates, tickers) if "shares" in stocks else None
    sh_m = None
    if not sh.empty:
        sh_m = sh.pivot_table(index="date", columns="ticker", values="shares", aggfunc="last")
        sh_m = sh_m.reindex(columns=tickers)
        sh_m = sh_m.reindex(sh_m.index.union(dates)).ffill().reindex(dates).astype(F32)
        fr = sh.pivot_table(index="date", columns="ticker", values="foreign_ratio", aggfunc="last").reindex(columns=tickers)
        P.m["foreign_ratio"] = fr.reindex(fr.index.union(dates)).ffill(limit=40).reindex(dates).astype(F32)
    shares = sh_daily if sh_daily is not None else pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=F32)
    if sh_m is not None:
        shares = shares.fillna(sh_m)
    shares = shares.ffill().bfill(limit=60)
    P.m["shares"] = shares.astype(F32)
    P.m["mcap"] = (close.ffill() * shares).astype(F32)

    # chips
    inst = store.read("institutional", start, end)
    for c in ("foreign", "trust", "dealer", "total"):
        observed=_pivot(inst,c,dates,tickers).where(P.m['tradable'])
        P.m[c+'_observed']=observed
        P.m[c] = observed.fillna(0).astype(F32)
    P.quality["institutional_coverage"] = float((inst["date"].nunique() / len(dates)) if not inst.empty else 0)
    mg = store.read("margin", start, end)
    for c in ("mbal", "mlimit", "sbal"):
        P.m[c] = _pivot(mg, c, dates, tickers).ffill(limit=5).astype(F32)
    P.quality["margin_coverage"] = float((mg["date"].nunique() / len(dates)) if not mg.empty else 0)

    # TDCC weekly big holders (optional; forward-filled up to 3 weeks)
    ho = store.read("holders", start, end)
    for c in ("big400", "big1000", "retail50", "holders"):
        P.m[c] = _pivot(ho, c, dates, tickers).reindex(dates)
        if not ho.empty:
            hp = ho[["date", "ticker", c]].drop_duplicates(["date", "ticker"], keep="last")
            hp = hp.set_index(["date", "ticker"])[c].astype(F32).unstack("ticker").reindex(columns=tickers)
            P.m[c] = hp.reindex(hp.index.union(dates)).ffill(limit=25).reindex(dates).astype(F32)
    # Preserve observation dates for display; repeated weekly values are still
    # distinct weekly observations, while a forward-fill is not a fresh release.
    for name,source,limit in [('holder_date',ho,25),('margin_date',mg,5)]:
        if not source.empty:
            src=source[['date','ticker']].drop_duplicates().copy()
            src['epoch_day']=(pd.to_datetime(src.date)-pd.Timestamp('1970-01-01')).dt.total_seconds()/86400
            p=src.pivot(index='date',columns='ticker',values='epoch_day').reindex(columns=tickers)
            P.m[name]=p.reindex(p.index.union(dates)).ffill(limit=limit).reindex(dates).astype(F32)
    P.quality['chart_chips_preserve_missing']=True

    # valuation
    va = store.read("valuation", start, end)
    for c in ("pe", "pb", "dy"):
        P.m[c] = _pivot(va, c, dates, tickers).ffill(limit=5).astype(F32)
    P.m["pe"] = P.m["pe"].where(P.m["pe"] > 0)
    P.quality["valuation_coverage"] = float((va["date"].nunique() / len(dates)) if not va.empty else 0)

    # monthly revenue (point-in-time mapping happens in factors)
    rv = store.read("revenue")
    if not rv.empty:
        rv["ticker"] = rv["ticker"].astype(str)
        rm = rv.pivot_table(index="ym", columns="ticker", values="revenue", aggfunc="last").reindex(columns=tickers)
        per = pd.period_range(pd.Period(f"{rm.index.min() // 100}-{rm.index.min() % 100:02d}", "M"),
                              pd.Period(f"{rm.index.max() // 100}-{rm.index.max() % 100:02d}", "M"), freq="M")
        P.rev_m = rm.reindex([p.year * 100 + p.month for p in per])  # contiguous months for YoY shifts
    P.quality["revenue_months"] = 0 if rv.empty else int(rv["ym"].nunique())

    info = store.read("info")
    names = price.dropna(subset=["name"]).groupby("ticker")["name"].last() if "name" in price else pd.Series(dtype=str)
    boards = price.groupby("ticker")["board"].last()
    P.info = pd.DataFrame(index=tickers)
    # plain strings (categoricals from the memory-saving read break fillna on pandas 2.x)
    P.info["name"] = names.astype(object).reindex(tickers)
    P.info["board"] = boards.astype(object).reindex(tickers)
    P.info["sector"] = "NA"
    if not info.empty:
        inf = info.drop_duplicates("ticker", keep="last").set_index("ticker")
        P.info["name"] = P.info["name"].fillna(inf["name"].astype(object).reindex(tickers))
        P.info["sector"] = inf["sector"].astype(object).reindex(tickers).fillna("NA")
    P.info["name"] = P.info["name"].fillna(pd.Series(tickers, index=tickers))
    P.info = P.info.astype(object)
    return P


def revenue_daily(P: Panel, func) -> pd.DataFrame:
    """Map a monthly revenue-derived matrix (index ym) to trading days using the filing deadline."""
    if P.rev_m is None or P.rev_m.empty:
        return pd.DataFrame(np.nan, index=P.dates, columns=P.tickers, dtype=F32)
    monthly = func(P.rev_m.sort_index())
    avail = pd.DatetimeIndex([revenue_available_date(ym) for ym in monthly.index])
    monthly.index = avail
    monthly = monthly[~monthly.index.duplicated(keep="last")]
    # a value is usable on the first trading day >= its availability date
    out = monthly.reindex(monthly.index.union(P.dates)).ffill(limit=None).reindex(P.dates)
    last_avail = pd.Series(monthly.index, index=monthly.index).reindex(monthly.index.union(P.dates)).ffill().reindex(P.dates)
    fresh = (last_avail >= P.dates - pd.Timedelta(days=75)).to_numpy()
    out = out.where(np.repeat(fresh[:, None], out.shape[1], axis=1))
    return out.astype(F32)
