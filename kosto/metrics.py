"""Performance statistics, robustness tests and multiple-testing corrections."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TDAYS = 245  # Taiwan trading days per year (approx.)


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p):
    # Acklam's rational approximation (enough for DSR thresholds)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - pl:
        return -_norm_ppf(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def perf(nav: pd.Series, bench: pd.Series | None = None) -> dict:
    nav = nav.dropna()
    if len(nav) < 20:
        return {}
    r = nav.pct_change().dropna()
    yrs = len(r) / TDAYS
    total = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + total) ** (1 / yrs) - 1 if yrs > 0 and total > -1 else np.nan
    vol = r.std() * np.sqrt(TDAYS)
    sharpe = r.mean() / r.std() * np.sqrt(TDAYS) if r.std() > 0 else np.nan
    down = r[r < 0].std() * np.sqrt(TDAYS)
    sortino = r.mean() * TDAYS / down if down > 0 else np.nan
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    yearly = nav.groupby(nav.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    # first partial year measured from first value
    out = {"total_return": total, "cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino, "mdd": mdd,
           "calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "worst_year": yearly.min() if len(yearly) else np.nan,
           "best_year": yearly.max() if len(yearly) else np.nan, "years": yrs, "skew": r.skew(), "kurt": r.kurt()}
    if bench is not None:
        b = bench.reindex(nav.index).ffill().dropna()
        rb = b.pct_change().reindex(r.index).fillna(0)
        ex = r - rb
        btot = b.iloc[-1] / b.iloc[0] - 1
        bcagr = (1 + btot) ** (1 / yrs) - 1 if yrs > 0 and btot > -1 else np.nan
        te = ex.std() * np.sqrt(TDAYS)
        out.update({"bench_cagr": bcagr, "excess_cagr": cagr - bcagr, "tracking_error": te,
                    "info_ratio": ex.mean() * TDAYS / te if te > 0 else np.nan,
                    "beta": np.cov(r, rb)[0, 1] / rb.var() if rb.var() > 0 else np.nan,
                    "win_days_vs_bench": float((ex > 0).mean())})
        yb = b.groupby(b.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
        out["worst_year_excess"] = (yearly - yb.reindex(yearly.index)).min()
    return out


def yearly_table(nav: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    f = pd.DataFrame({"策略": nav.groupby(nav.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)})
    if bench is not None:
        b = bench.reindex(nav.index)
        f["0050"] = b.groupby(b.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
        f["超額"] = f["策略"] - f["0050"]
    return f


def tranche_stats(tr: pd.DataFrame) -> dict:
    t = tr.dropna(subset=["ret"])
    if t.empty:
        return {}
    ex = t["ret"] - t["bench"]
    return {"tranches": len(t), "avg_tranche_ret": t["ret"].mean(), "hit_rate": float((t["ret"] > 0).mean()),
            "beat_bench_rate": float((ex > 0).mean()), "avg_excess": ex.mean(),
            "avg_exposure": t["exposure"].mean(), "avg_filled": t["n_filled"].mean()}


def probabilistic_sharpe(r: pd.Series, sr_bench: float = 0.0) -> float:
    """PSR (Bailey & López de Prado) on daily returns; sr_bench in daily units."""
    r = r.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()
    sk, ku = r.skew(), r.kurt() + 3
    den = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return _norm_cdf((sr - sr_bench) * math.sqrt(n - 1) / den)


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sr_std: float) -> float:
    """DSR: PSR against the expected max Sharpe of n_trials unskilled strategies (daily units)."""
    if n_trials < 2 or not np.isfinite(trial_sr_std) or trial_sr_std <= 0:
        return probabilistic_sharpe(r)
    g = 0.5772156649
    emax = trial_sr_std * ((1 - g) * _norm_ppf(1 - 1 / n_trials) + g * _norm_ppf(1 - 1 / (n_trials * math.e)))
    return probabilistic_sharpe(r, emax)


def block_bootstrap_excess(nav: pd.Series, bench: pd.Series, n: int = 1000, block: int = 21, seed: int = 0) -> dict:
    """Stationary block bootstrap of annualised excess return → 90% CI and P(excess > 0)."""
    r = nav.pct_change().dropna()
    rb = bench.reindex(nav.index).pct_change().reindex(r.index).fillna(0)
    ex = (r - rb).to_numpy()
    T = len(ex)
    if T < block * 3:
        return {}
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for k in range(n):
        idx = []
        while len(idx) < T:
            s = rng.integers(0, T)
            idx.extend(range(s, min(s + rng.geometric(1 / block), T)))
        stats[k] = ex[np.array(idx[:T])].mean() * TDAYS
    return {"excess_ann_mean": float(ex.mean() * TDAYS), "ci05": float(np.quantile(stats, 0.05)),
            "ci95": float(np.quantile(stats, 0.95)), "p_positive": float((stats > 0).mean())}


def selection_score(table: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted average of percentile ranks (higher is better) across candidate rows."""
    total = pd.Series(0.0, index=table.index)
    wsum = 0.0
    for k, w in weights.items():
        if k in table and table[k].notna().any():
            total += table[k].rank(pct=True).fillna(0) * w
            wsum += w
    return total / wsum if wsum else total
