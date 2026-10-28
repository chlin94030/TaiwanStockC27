"""Research bundle: everything the optimiser/backtester needs, precomputed once (GitHub Actions)
and loaded by Streamlit for seconds-level re-tuning.

Arrays (S = signal dates, D = trading days from first signal, N = tickers ever in a pool, F = factors)
  U      uint16 [S,N,F]  universe percentile rank of each factor among eligible stocks (0 = missing)
  FWD    float32[3,S,N]  forward return open(t+1) -> open(t+H+1) for H = 5/20/60 (NaN = not matured)
  R      float32[D,N]    daily total return (close/ref - 1)
  OC     float32[D,N]    open -> close return (NaN = no open)
  OK     bool   [D,N]    buyable at the open (traded, not limit-up open)
  RANKS  int16  [P,S,N]  pool rank per rule (1 = best, 0 = not eligible)
  STAGE  int8   [S,N]    stock egg stage (0 unknown, 1..6 = A1..B3)
  DTOP   bool   [S,N]    M double-top flag ; WSUP bool [S,N] W support / stabilisation flag
  TRADE  bool   [S,N]    traded at the signal close ; CAP float32[S,N] 20-day average traded value
  BENCH  float32[D]      0050 daily return ; MKT float32[S,7] market stage scores + code
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import HORIZON_KEYS, STAGE_CODES, STAGES
from .factors import REGISTRY, compute, rsi14
from .kostolany import market_regime, mw_patterns, stage_codes
from .panel import Panel
from .optimize import avg_rank_cols
from .universe import RULES, pool_inputs, pool_ranks


@dataclass
class Bundle:
    meta: dict
    a: dict = field(default_factory=dict)

    # convenience
    @property
    def sig_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.meta["sig_dates"])

    @property
    def days(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.meta["days"])

    @property
    def tickers(self) -> list:
        return self.meta["tickers"]

    @property
    def factors(self) -> list:
        return self.meta["factors"]

    @property
    def sig_pos(self) -> np.ndarray:
        return np.asarray(self.meta["sig_pos"], int)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, __meta__=np.frombuffer(json.dumps(self.meta, ensure_ascii=False).encode(), np.uint8),
                            **self.a)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(path)

    @staticmethod
    def load(path) -> "Bundle":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["__meta__"]).decode())
            a = {k: z[k] for k in z.files if k != "__meta__"}
        return Bundle(meta, a)


def signal_positions(dates: pd.DatetimeIndex, start, step: int) -> np.ndarray:
    last = len(dates) - 1
    first = int(dates.searchsorted(pd.Timestamp(start)))
    pos = np.arange(last, first - 1, -step)[::-1]
    return pos[pos >= 0]


def _universe_rank(x: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """Percentile rank among eligible & finite values → uint16 1..65535 (0 = missing)."""
    out = np.zeros(x.shape, np.uint16)
    for i in range(x.shape[0]):
        m = elig[i] & np.isfinite(x[i])
        n = int(m.sum())
        if n < 5:
            continue
        r = avg_rank_cols(x[i, m][:, None].astype(float))[:, 0]
        out[i, m] = (1 + np.round(r * 65534 / max(n - 1, 1))).astype(np.uint16)
    return out


def forward_returns(R: np.ndarray, OC: np.ndarray, sig_rel: np.ndarray, H: int) -> np.ndarray:
    """open(t+1) → open(t+H+1) total return for each signal (rows) and ticker (cols)."""
    D, N = R.shape
    L = np.vstack([np.zeros((1, N)), np.cumsum(np.log1p(np.nan_to_num(R, nan=0.0).astype(float)), axis=0)])
    out = np.full((len(sig_rel), N), np.nan, np.float32)
    for k, t in enumerate(sig_rel):
        if t + H >= D:
            continue
        e = t + 1
        oc_in = OC[e].astype(float)
        if t + H + 1 < D:
            x = t + H + 1
            oc_out = np.where(np.isfinite(OC[x]), OC[x], np.nan).astype(float)
            grow = np.exp(L[x + 1] - L[e + 1])               # R[e+1..x]
            val = (1 + oc_in) * grow / np.where(np.isfinite(oc_out), 1 + oc_out, 1 + R[x])
            # not traded at exit open → hold to previous close (gap 0)
            val = np.where(np.isfinite(oc_out), val, (1 + oc_in) * np.exp(L[x] - L[e + 1]))
        else:  # exit open not observed yet → not matured
            continue
        out[k] = (val - 1).astype(np.float32)
    return out


def build_bundle(P: Panel, cfg: dict, log=print) -> Bundle:
    t0 = time.time()
    r = cfg["research"]
    pos = signal_positions(P.dates, r["start"], r["signal_step"])
    sig_dates = P.dates[pos]
    log(f"訊號日 {len(pos)} 個：{sig_dates[0].date()} → {sig_dates[-1].date()}")
    vals, ctx = compute(P, sample_idx=pos)
    log(f"因子計算 {len(vals)} 個 ({time.time()-t0:.0f}s)")
    inp = pool_inputs(ctx, pos, cfg)
    ranks = pool_ranks(inp, sig_dates)
    maxsize = max(cfg["pools"]["sizes"])
    ever = np.zeros(len(P.tickers), bool)
    for rk in ranks.values():
        ever |= ((rk >= 1) & (rk <= maxsize)).any(axis=0)
    cols = np.where(ever)[0]
    log(f"曾進入任一母池的股票 {len(cols)} 檔")
    elig = inp["elig"]
    fnames = list(vals.keys())
    U = np.zeros((len(pos), len(cols), len(fnames)), np.uint16)
    for j, f in enumerate(fnames):
        U[:, :, j] = _universe_rank(vals[f], elig)[:, cols]

    d0 = pos[0]
    days = P.dates[d0:]
    R = P["ret"].to_numpy(dtype=np.float32)[d0:, cols]
    OC = P["oc"].to_numpy(dtype=np.float32)[d0:, cols]
    gap = P["gap"].to_numpy(dtype=np.float32)[d0:, cols]
    trad = P["tradable"].to_numpy()[d0:, cols]
    OK = trad & np.isfinite(OC) & ~(gap >= 0.095)
    sig_rel = pos - d0
    FWD = np.stack([forward_returns(R, OC, sig_rel, cfg["horizons"][h]["days"]) for h in HORIZON_KEYS])

    stg = stage_codes(ctx).to_numpy()[pos][:, cols]
    mw = mw_patterns(ctx)
    dtop = mw["double_top"].to_numpy(dtype=bool)[pos][:, cols]
    adj = ctx.adj.to_numpy(dtype=float)
    ma5 = ctx.ma(5).to_numpy(dtype=float)
    mom5 = ctx.mom(5).to_numpy(dtype=float)
    stab = (adj > ma5) & (mom5 > 0)
    wsup = (mw["w_support"].to_numpy(dtype=bool) | stab)[pos][:, cols]
    mk = market_regime(ctx)
    mkt = mk[list(STAGES)].to_numpy(dtype=np.float32)[pos]
    mcode = np.array([STAGE_CODES.get(s, 0) for s in mk["stage"].to_numpy()[pos]], np.float32)
    bench = P.bench.reindex(days).fillna(0).to_numpy(np.float32) if P.bench is not None else np.zeros(len(days), np.float32)

    info = P.info.reindex(P.tickers[cols]).astype(object)
    meta = {
        "version": __version__, "built": pd.Timestamp.now(tz="Asia/Taipei").isoformat(timespec="seconds"),
        "synthetic": bool(P.synthetic), "sig_dates": [str(d.date()) for d in sig_dates],
        "days": [str(d.date()) for d in days], "sig_pos": sig_rel.tolist(), "tickers": list(P.tickers[cols]),
        "names": info["name"].fillna("").tolist(), "sectors": info["sector"].fillna("NA").tolist(),
        "boards": info["board"].fillna("").tolist(),
        "factors": fnames, "families": [REGISTRY[f].family for f in fnames],
        "directions": [REGISTRY[f].direction for f in fnames], "horizons": [list(REGISTRY[f].horizons) for f in fnames],
        "factor_zh": [REGISTRY[f].zh for f in fnames], "rules": list(RULES),
        "horizon_days": {h: cfg["horizons"][h]["days"] for h in HORIZON_KEYS},
        "data_quality": P.quality, "config": cfg,
        "coverage": {f: float((U[:, :, j] > 0).mean()) for j, f in enumerate(fnames)},
    }
    a = {"U": U, "FWD": FWD, "R": R, "OC": OC, "OK": OK,
         "RANKS": np.stack([ranks[rl][:, cols] for rl in RULES]).astype(np.int16),
         "STAGE": stg.astype(np.int8), "DTOP": dtop, "WSUP": wsup,
         "TRADE": trad[sig_rel], "CAP": inp["amt20"][:, cols].astype(np.float32),
         "CLOSE": inp["close"][:, cols].astype(np.float32),
         "MCAP": inp["mcap"][:, cols].astype(np.float32),
         # raw overheat indicators at each signal date (v1.3 不追高閘門: RSI14 and 20-day bias)
         "RSI14": rsi14(ctx).to_numpy(dtype=np.float32)[pos][:, cols],
         "BIAS20": (ctx.adj / ctx.ma(20) - 1).to_numpy(dtype=np.float32)[pos][:, cols],
         "BENCH": bench, "MKT": np.column_stack([mkt, mcode]).astype(np.float32),
         "MKT_DAILY": mk[list(STAGES) + ["confidence", "index", "drawdown", "turnover_ratio", "breadth"]]
         .reindex(days).to_numpy(np.float32),
         "MKT_DAILY_CODE": np.array([STAGE_CODES.get(s, 0) for s in mk["stage"].reindex(days).fillna("UNKNOWN")], np.int8)}
    # ---- 看盤資料包（App 的 K 線／籌碼圖用）：最近 chart_days 個交易日的原始價量與籌碼，單位：價(元)、量與籌碼(張)
    nq = int((cfg.get("app") or {}).get("chart_days", 500))
    q0 = max(0, len(P.dates) - nq)
    qdays = P.dates[q0:]

    def qa(name, scale=1.0):
        if name not in P.m:
            return np.full((len(qdays), len(cols)), np.nan, np.float32)
        return (P.m[name].to_numpy(dtype=np.float32)[q0:, cols] * scale).astype(np.float32)
    for key, name, sc in (("Q_O", "open", 1), ("Q_H", "high", 1), ("Q_L", "low", 1), ("Q_C", "close", 1),
                          ("Q_V", "volume", 1e-3), ("Q_FI", "foreign", 1e-3), ("Q_TR", "trust", 1e-3),
                          ("Q_DL", "dealer", 1e-3), ("Q_MB", "mbal", 1), ("Q_SB", "sbal", 1), ("Q_ML", "mlimit", 1),
                          ("Q_FR", "foreign_ratio", 1), ("Q_H400", "big400", 1), ("Q_H1000", "big1000", 1),
                          ("Q_HR", "retail50", 1), ("Q_HN", "holders", 1)):
        chart_name=name+'_observed' if name in ('foreign','trust','dealer') and name+'_observed' in P.m else name
        a[key] = qa(chart_name, sc)
    a['Q_ADJ']=qa('adj')
    a['Q_HDATE']=qa('holder_date')
    a['Q_MDATE']=qa('margin_date')
    a["Q_TAIEX"] = (P.taiex.reindex(qdays).to_numpy(np.float32) if P.taiex is not None
                    else np.full(len(qdays), np.nan, np.float32))
    meta["q_days"] = [str(d.date()) for d in qdays]
    # raw values of factors at the latest signal date (for explaining today's picks)
    a["LATEST_RAW"] = np.stack([vals[f][-1, cols] for f in fnames], axis=1).astype(np.float32)
    log(f"Bundle 完成 ({time.time()-t0:.0f}s)：S={len(pos)} D={len(days)} N={len(cols)} F={len(fnames)}")
    return Bundle(meta, a)
