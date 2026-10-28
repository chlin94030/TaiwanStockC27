"""Head-to-head comparison with ANY external recommendation list (e.g. the old Alpha Radar V22 app).

Accepted inputs (CSV, or a ZIP containing CSVs):
  * Alpha Radar V22 exports: monthly_decisions.csv / final_confirmation_records.csv
    (columns: horizon, date, selected='2330.TW|2317.TW|...')
  * Generic wide: date, [horizon], tickers/selected/picks  (separators | , ; or space)
  * Generic long: date, [horizon], ticker   (one row per stock)
Both models are measured with the SAME engine: buy at the open of the first trading day after the
signal date, equal weight, hold H days, sell at the open, same round-trip costs. Our model's picks are
taken from its own walk-forward record on the latest signal date ≤ the external signal date.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

TICK_RE = re.compile(r"(\d{4})(?:\.TWO?|\.TW)?")


@dataclass
class Daily:
    days: pd.DatetimeIndex
    tickers: list
    R: np.ndarray
    OC: np.ndarray
    OK: np.ndarray
    bench: np.ndarray

    @staticmethod
    def from_bundle(B) -> "Daily":
        return Daily(B.days, list(B.tickers), B.a["R"], B.a["OC"], B.a["OK"], B.a["BENCH"])

    @staticmethod
    def from_panel(P, start=None) -> "Daily":
        d0 = 0 if start is None else int(P.dates.searchsorted(pd.Timestamp(start)))
        OC = P["oc"].to_numpy(np.float32)[d0:]
        gap = P["gap"].to_numpy(np.float32)[d0:]
        OK = P["tradable"].to_numpy()[d0:] & np.isfinite(OC) & ~(gap >= 0.095)
        bench = P.bench.to_numpy(np.float32)[d0:] if P.bench is not None else np.zeros(len(P.dates) - d0, np.float32)
        return Daily(P.dates[d0:], list(P.tickers), P["ret"].to_numpy(np.float32)[d0:], OC, OK, bench)


def _tickers(cell) -> list:
    if isinstance(cell, (list, tuple)):
        cell = "|".join(map(str, cell))
    return [m.group(1) for m in TICK_RE.finditer(str(cell))]


def parse_external(name: str, raw: bytes) -> pd.DataFrame:
    frames = []
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            prefer = [n for n in z.namelist() if n.lower().endswith(".csv") and
                      any(k in n.lower() for k in ("decision", "confirmation", "record", "pick", "select"))]
            for n in prefer or [n for n in z.namelist() if n.lower().endswith(".csv")]:
                try:
                    frames.append(_parse_csv(z.read(n), n))
                except ValueError:
                    continue
    else:
        frames.append(_parse_csv(raw, name))
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("找不到含 date 與 股票代號 欄位的 CSV")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "horizon", "source"]).sort_values(["horizon", "date"])
    return out.reset_index(drop=True)


def _parse_csv(raw: bytes, name: str) -> pd.DataFrame:
    f = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    cols = {c.lower().strip(): c for c in f.columns}
    dcol = next((cols[c] for c in ("date", "signal_date", "日期", "訊號日") if c in cols), None)
    if dcol is None:
        raise ValueError("no date column")
    hcol = next((cols[c] for c in ("horizon", "h", "持有天數") if c in cols), None)
    tcol = next((cols[c] for c in ("selected", "tickers", "picks", "stocks", "推薦") if c in cols), None)
    lcol = next((cols[c] for c in ("ticker", "stock_id", "代號") if c in cols), None)
    if tcol is None and lcol is None:
        raise ValueError("no ticker column")
    f["_date"] = pd.to_datetime(f[dcol]).dt.normalize()
    f["_h"] = pd.to_numeric(f[hcol], errors="coerce").fillna(20).astype(int) if hcol else 20
    if tcol is not None:
        f["_t"] = f[tcol].map(_tickers)
        g = f[["_date", "_h", "_t"]]
    else:
        f["_t"] = f[lcol].map(lambda x: _tickers(x)[:1])
        g = f.groupby(["_date", "_h"])["_t"].sum().reset_index()
    g = g.rename(columns={"_date": "date", "_h": "horizon", "_t": "tickers"})
    g["source"] = name.split("/")[-1]
    return g


def holding_return(daily: Daily, date, H: int, tickers: list, buy_cost: float, sell_cost: float) -> dict:
    days = daily.days
    t = int(days.searchsorted(pd.Timestamp(date), side="right")) - 1  # signal day = last trading day ≤ date
    e, x = t + 1, t + H + 1
    if t < 0 or x >= len(days):
        return {"entry": None, "ret": np.nan, "bench": np.nan, "n": len(tickers), "covered": 0, "filled": 0}
    tk = {s: k for k, s in enumerate(daily.tickers)}
    js = [tk[s] for s in tickers if s in tk]
    fill = [j for j in js if daily.OK[e, j]]
    rets = []
    for j in fill:
        v = (1 + np.nan_to_num(daily.OC[e, j])) * np.prod(1 + daily.R[e + 1: x + 1, j].astype(float))
        oc_x = daily.OC[x, j]
        v = v / (1 + oc_x) if np.isfinite(oc_x) else v / (1 + daily.R[x, j])
        rets.append(v * (1 - sell_cost) / (1 + buy_cost) - 1)
    n = max(len(tickers), 1)
    gross = float(np.sum(rets) / n) if tickers else 0.0          # unfilled / unknown names = cash
    b = float(np.prod(1 + daily.bench[e: x].astype(float)) - 1)
    return {"entry": str(days[e].date()), "ret": gross, "bench": b, "n": len(tickers), "covered": len(js),
            "filled": len(fill)}


def head_to_head(daily: Daily, ext: pd.DataFrame, ours: dict, costs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ours: {H(int): tranches DataFrame with columns date, picks}. Returns (per-signal, summary)."""
    buy = costs["fee"] * costs["fee_discount"] + costs["slippage"]
    sell = costs["fee"] * costs["fee_discount"] + costs["tax"] + costs["slippage"]
    rows = []
    for _, r in ext.iterrows():
        H = int(r["horizon"])
        a = holding_return(daily, r["date"], H, r["tickers"], buy, sell)
        row = {"date": r["date"].date(), "horizon": H, "source": r.get("source", ""), "ext_n": a["n"],
               "ext_covered": a["covered"], "ext_ret": a["ret"], "bench": a["bench"], "entry": a["entry"]}
        tr = ours.get(H)
        if tr is not None and len(tr):
            prior = tr[pd.to_datetime(tr["date"]) <= r["date"]]
            if len(prior) and (r["date"] - pd.Timestamp(prior.iloc[-1]["date"])).days <= 10:
                o = holding_return(daily, r["date"], H, list(prior.iloc[-1]["picks"]), buy, sell)
                row.update({"our_ret": o["ret"], "our_signal": str(pd.Timestamp(prior.iloc[-1]["date"]).date())})
        rows.append(row)
    per = pd.DataFrame(rows)
    if per.empty:
        return per, pd.DataFrame()
    summ = []
    for H, g in per.groupby("horizon"):
        g = g.dropna(subset=["ext_ret"])
        both = g.dropna(subset=["our_ret"]) if "our_ret" in g else g.iloc[0:0]
        d = (both["our_ret"] - both["ext_ret"]) if len(both) else pd.Series(dtype=float)
        tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
        summ.append({"horizon": H, "signals": len(g), "paired": len(both),
                     "舊模型平均報酬": g["ext_ret"].mean(), "新模型平均報酬": both["our_ret"].mean() if len(both) else np.nan,
                     "0050平均": g["bench"].mean(), "舊模型勝率": (g["ext_ret"] > 0).mean(),
                     "新模型勝率": (both["our_ret"] > 0).mean() if len(both) else np.nan,
                     "舊模型勝0050": (g["ext_ret"] > g["bench"]).mean(),
                     "新模型勝0050": (both["our_ret"] > both["bench"]).mean() if len(both) else np.nan,
                     "新減舊(平均)": d.mean() if len(d) else np.nan, "差異t值": tstat,
                     "舊模型涵蓋率": (g["ext_covered"] / g["ext_n"].clip(lower=1)).mean()})
    return per, pd.DataFrame(summ)
