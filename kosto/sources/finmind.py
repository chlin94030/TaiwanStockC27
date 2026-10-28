"""FinMind v4 API (fallback / per-stock backfill). Field names verified against live responses.

Free registered token: ~600 requests/hour. One request returns a stock's full date range, so a
dataset for ~1,900 stocks costs ~1,900 requests (≈3.5 h). Use it when the official bulk
endpoints are blocked, or for gaps. Date-wide queries (no data_id) are sponsor-only.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"


class QuotaExhausted(RuntimeError):
    pass


class FinMind:
    def __init__(self, token: str = "", hourly_quota: int = 550, log=print, stop_at: float | None = None):
        self.token = token.strip()
        self.quota = hourly_quota if self.token else min(hourly_quota, 280)
        self.log = log
        self.stop_at = stop_at  # epoch seconds: raise QuotaExhausted instead of sleeping past this
        self.calls = deque()
        self.s = requests.Session()
        if self.token:
            self.s.headers["Authorization"] = f"Bearer {self.token}"

    def _throttle(self):
        now = time.time()
        while self.calls and now - self.calls[0] > 3600:
            self.calls.popleft()
        if len(self.calls) >= self.quota:
            wait = 3600 - (now - self.calls[0]) + 5
            if self.stop_at and now + wait > self.stop_at:
                raise QuotaExhausted("FinMind 每小時額度用完，下次排程會自動續傳")
            self.log(f"  FinMind 額度暫滿，等待 {wait/60:.0f} 分鐘")
            time.sleep(wait)
        self.calls.append(time.time())

    def get(self, dataset: str, data_id: str | None, start: str, end: str | None = None) -> pd.DataFrame:
        params = {"dataset": dataset, "start_date": str(pd.Timestamp(start).date())}
        if data_id:
            params["data_id"] = data_id
        if end:
            params["end_date"] = str(pd.Timestamp(end).date())
        for attempt in range(3):
            self._throttle()
            try:
                r = self.s.get(API, params=params, timeout=60)
                j = r.json()
            except (requests.RequestException, ValueError) as e:
                self.log(f"  ! FinMind {dataset} {data_id}: {e}")
                time.sleep(5 * (attempt + 1))
                continue
            status = int(j.get("status", r.status_code))
            if status == 402:
                if self.stop_at:
                    raise QuotaExhausted(j.get("msg", "quota"))
                time.sleep(600)
                continue
            if status != 200:
                raise RuntimeError(f"FinMind {dataset} {data_id}: {j.get('msg')}")
            return pd.DataFrame(j.get("data") or [])
        raise RuntimeError(f"FinMind {dataset} {data_id}: retries exhausted")

    # ------------------------------------------------------------------ typed helpers
    def info(self) -> pd.DataFrame:
        f = self.get("TaiwanStockInfo", None, "2000-01-01")
        if f.empty:
            return f
        f = f[f["stock_id"].astype(str).str.fullmatch(r"[1-9]\d{3}|0050")]
        f = f.rename(columns={"stock_id": "ticker", "stock_name": "name", "industry_category": "sector"})
        f["board"] = f["type"].map({"twse": "TWSE", "tpex": "TPEX"}).fillna("OTHER")
        f = f[f["board"].isin(["TWSE", "TPEX"])]
        return f.drop_duplicates("ticker", keep="last")[["ticker", "name", "sector", "board"]]

    def price(self, ticker, start, end=None) -> pd.DataFrame:
        f = self.get("TaiwanStockPrice", ticker, start, end)
        if f.empty:
            return f
        out = pd.DataFrame({"date": pd.to_datetime(f["date"]), "ticker": ticker,
                            "open": f["open"], "high": f["max"], "low": f["min"], "close": f["close"],
                            "volume": f["Trading_Volume"], "amount": f["Trading_money"],
                            "change": f["spread"], "event": ""})
        for c in ("open", "high", "low", "close"):
            out.loc[out[c] <= 0, c] = np.nan  # FinMind uses 0 for no-trade days
        return out

    def exright(self, ticker, start, end=None) -> pd.DataFrame:
        f = self.get("TaiwanStockDividendResult", ticker, start, end)
        if f.empty:
            return f
        return pd.DataFrame({"date": pd.to_datetime(f["date"]), "ticker": ticker,
                             "before": f["before_price"], "ref": f["reference_price"]})

    def institutional(self, ticker, start, end=None) -> pd.DataFrame:
        f = self.get("TaiwanStockInstitutionalInvestorsBuySell", ticker, start, end)
        if f.empty:
            return f
        f["net"] = f["buy"] - f["sell"]
        p = f.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)

        def col(*names):
            s = pd.Series(0.0, index=p.index)
            for n in names:
                if n in p:
                    s = s + p[n].astype(float)
            return s.to_numpy()

        out = pd.DataFrame({"date": pd.to_datetime(p.index), "ticker": ticker})
        out["foreign"] = col("Foreign_Investor", "Foreign_Dealer_Self")
        out["trust"] = col("Investment_Trust")
        out["dealer"] = col("Dealer_self", "Dealer_Hedging", "Dealer")
        out["total"] = out["foreign"] + out["trust"] + out["dealer"]
        return out.reset_index(drop=True)

    def margin(self, ticker, start, end=None) -> pd.DataFrame:
        f = self.get("TaiwanStockMarginPurchaseShortSale", ticker, start, end)
        if f.empty:
            return f
        return pd.DataFrame({"date": pd.to_datetime(f["date"]), "ticker": ticker,
                             "mbal": f["MarginPurchaseTodayBalance"], "mlimit": f["MarginPurchaseLimit"],
                             "sbal": f["ShortSaleTodayBalance"]})

    def valuation(self, ticker, start, end=None) -> pd.DataFrame:
        f = self.get("TaiwanStockPER", ticker, start, end)
        if f.empty:
            return f
        out = pd.DataFrame({"date": pd.to_datetime(f["date"]), "ticker": ticker,
                            "pe": f["PER"], "pb": f["PBR"], "dy": f["dividend_yield"]})
        out.loc[out["pe"] <= 0, "pe"] = np.nan
        return out

    def shares(self, ticker, start, end=None) -> pd.DataFrame:
        f = self.get("TaiwanStockShareholding", ticker, start, end)
        if f.empty:
            return f
        out = pd.DataFrame({"date": pd.to_datetime(f["date"]), "ticker": ticker,
                            "shares": f["NumberOfSharesIssued"], "foreign_ratio": f["ForeignInvestmentSharesRatio"]})
        # monthly sample is enough for shares; keep month-ends to stay small
        out["m"] = out["date"].dt.to_period("M")
        return out.groupby("m").tail(1).drop(columns="m").reset_index(drop=True)

    def revenue(self, ticker, start, end=None) -> pd.DataFrame:
        f = self.get("TaiwanStockMonthRevenue", ticker, start, end)
        if f.empty:
            return f
        out = pd.DataFrame({"ticker": ticker, "ym": f["revenue_year"] * 100 + f["revenue_month"],
                            "revenue": f["revenue"].astype(float)})
        # 'date' is the 1st of the month after the revenue month; filing deadline is the 10th
        out["avail_date"] = pd.to_datetime(f["date"]) + pd.Timedelta(days=9)
        return out


FINMIND_METHODS = {
    "price": "price", "exright": "exright", "institutional": "institutional", "margin": "margin",
    "valuation": "valuation", "shares": "shares", "revenue": "revenue",
}
