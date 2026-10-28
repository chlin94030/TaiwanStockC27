"""IC / IC-IR based hierarchical factor weighting with a Kostolany family floor.

Level 1 (inside each family T/C/F/K):  w ∝ Σ⁻¹·IC̄  (shrunk covariance of daily IC series),
         non-negative after aligning every factor with its economic prior; a factor whose
         training IC contradicts its prior gets weight 0 (never flipped).
Level 2 (across families):  same IC-IR rule on the family composites, then the Kostolany family
         is lifted to at least λ ("權重加大"), and everything is shrunk toward the horizon prior
         with intensity α = n_eff / (n_eff + n0)  (few independent samples → stay near the prior).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAMILIES = ("T", "C", "F", "K")


@dataclass
class PoolView:
    """Within-pool centred ranks for one pool: per signal date, member indices and Z[n_mem, F]."""
    members: list          # list of int arrays (ticker column indices)
    Z: list                # list of float32 arrays [n_mem, F] in [-0.5, 0.5], 0 = missing/neutral


def avg_rank_cols(u: np.ndarray) -> np.ndarray:
    """Column-wise 0-based average ranks with ties sharing the mean position (NaN/inf sort last)."""
    n, F = u.shape
    order = np.argsort(u, axis=0, kind="stable")
    us = np.take_along_axis(u, order, 0)
    pos = np.arange(n)[:, None].repeat(F, 1)
    new = np.vstack([np.ones((1, F), bool), us[1:] != us[:-1]])
    start = np.maximum.accumulate(np.where(new, pos, 0), axis=0)
    endm = np.vstack([us[1:] != us[:-1], np.ones((1, F), bool)])
    end = np.minimum.accumulate(np.where(endm, pos, n - 1)[::-1], axis=0)[::-1]
    out = np.empty((n, F))
    np.put_along_axis(out, order, (start + end) / 2.0, 0)
    return out


def pool_view(U: np.ndarray, member_mask: np.ndarray, trade: np.ndarray) -> PoolView:
    members, Zs = [], []
    for i in range(U.shape[0]):
        idx = np.where(member_mask[i] & trade[i])[0]
        members.append(idx)
        if len(idx) < 3:
            Zs.append(np.zeros((len(idx), U.shape[2]), np.float32))
            continue
        u = U[i, idx, :].astype(np.float64)
        miss = u == 0
        # tie-aware rank among members (missing sorts last), centre to [-0.5, 0.5]; missing → 0
        rk = avg_rank_cols(np.where(miss, np.inf, u))
        n_ok = (~miss).sum(0)
        z = np.where(miss, 0.0, (rk + 0.5) / np.maximum(n_ok, 1) - 0.5).astype(np.float32)
        Zs.append(z)
    return PoolView(members, Zs)


def ic_matrix(view: PoolView, fwd: np.ndarray) -> np.ndarray:
    """Spearman IC per signal date (rows) and factor (cols). NaN when too few matured names."""
    S, F = len(view.Z), (view.Z[0].shape[1] if view.Z else 0)
    out = np.full((S, F), np.nan, np.float32)
    for i in range(S):
        idx = view.members[i]
        if len(idx) < 10:
            continue
        y = fwd[i, idx]
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        z = view.Z[i][ok]
        yr = avg_rank_cols(y[ok][:, None].astype(float))[:, 0]
        yr = (yr / (len(yr) - 1) - 0.5).astype(np.float32)
        num = (z * yr[:, None]).sum(0)
        den = np.sqrt((z ** 2).sum(0) * (yr ** 2).sum())
        out[i] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def composite_ic(view: PoolView, fwd: np.ndarray, w: np.ndarray, rows) -> np.ndarray:
    out = np.full(len(rows), np.nan)
    for k, i in enumerate(rows):
        idx = view.members[i]
        if len(idx) < 10:
            continue
        y = fwd[i, idx]
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        s = view.Z[i][ok] @ w
        if np.std(s) == 0:
            continue
        sr = avg_rank_cols(s[:, None].astype(float))[:, 0]
        yr = avg_rank_cols(y[ok][:, None].astype(float))[:, 0]
        out[k] = np.corrcoef(sr, yr)[0, 1]
    return out


def _shrunk_solve(X: np.ndarray, m: np.ndarray, delta: float = 0.5) -> np.ndarray:
    """w = (δ·diag + (1-δ)·Σ)⁻¹ m, made non-negative by iterative removal."""
    k = len(m)
    active = np.ones(k, bool)
    w = np.zeros(k)
    for _ in range(k):
        Xa = X[:, active]
        if Xa.shape[0] < 3 or not active.any():
            break
        C = np.cov(Xa, rowvar=False) if Xa.shape[1] > 1 else np.atleast_2d(np.var(Xa, ddof=1))
        C = delta * np.diag(np.diag(C)) + (1 - delta) * C + 1e-6 * np.eye(C.shape[0])
        sol = np.linalg.solve(C, m[active])
        w = np.zeros(k)
        w[active] = sol
        if (sol >= 0).all():
            break
        active &= w > 0
    w = np.clip(w, 0, None)
    return w


def _cap(w: np.ndarray, cap: float) -> np.ndarray:
    w = w / w.sum() if w.sum() > 0 else w
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any() or (w > 0).sum() * cap < 1:
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = (~over) & (w > 0)
        if not under.any():
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


@dataclass
class FitResult:
    weights: np.ndarray            # signed, per factor (sums of |w| = 1)
    family: dict                   # family weights
    stats: dict                    # per factor: mean IC, ICIR, t, used
    alpha: float
    n_rows: int


def fit_sparse(ic: np.ndarray, rows: np.ndarray, meta: dict, horizon: str, step: int, H: int,
               k: int = 5, tmin: float = 1.0, kK: int = 0, lamK: float = 0.0, exclude=(),
               anchor: float = 0.0, anchor_weights: dict | None = None,
               growth_blend: float = 0.0, growth_weights: dict | None = None,
               hyp_blend: float = 0.0, hyp_weights: dict | None = None) -> FitResult:
    """v1.2 精簡因子模型：只用訓練期 IC-IR t 值最高的 k 個因子（方向須與先驗一致、t > tmin），權重 ∝ |IC-IR|。
    kK > 0 時另外保留 K 族群最強的 kK 個因子，合計權重 lamK（科斯托蘭尼加強版）。
    實測（真實資料 2021-2025 開發期）：中線 30 個母池中 28 個、長線 29 個勝過階層式全因子權重。"""
    fams = np.array(meta["families"])
    dirs = np.array(meta["directions"])
    F = len(fams)
    allowed = np.array([horizon in h for h in meta["horizons"]])
    if exclude:
        allowed &= ~np.isin(np.array(meta["factors"]), list(exclude))
    X = ic[rows] if len(rows) else np.zeros((0, F))
    if len(X):
        X = np.nan_to_num(X[np.isfinite(X).any(axis=1)], nan=0.0)
    n = len(X)
    n_eff = n * step / max(H, step)
    mean = X.mean(0) if n else np.zeros(F)
    sd = X.std(0, ddof=1) if n > 2 else np.full(F, np.nan)
    icir = np.where(sd > 0, mean / np.where(sd > 0, sd, 1), 0)
    tstat = icir * np.sqrt(max(n_eff, 1))
    sign = np.where(dirs != 0, dirs, np.where(np.abs(tstat) > 1.0, np.sign(mean), 0)).astype(float)
    sign = np.where(allowed, sign, 0)
    at = tstat * sign
    stats = {"mean_ic": mean, "icir": icir, "t": tstat, "sign": sign, "allowed": allowed}
    if n < 8:
        return FitResult(np.zeros(F, np.float32), {}, stats, 0.0, int(n))
    isK = fams == "K"
    ok = (at > tmin) & (sign != 0)
    base_ok = ok & ~isK if kK else ok
    sel = np.zeros(F, bool)
    sel[np.argsort(-np.where(base_ok, at, -np.inf))[:k]] = True
    sel &= base_ok
    w = np.where(sel, np.abs(icir), 0.0)
    w = w / w.sum() if w.sum() > 0 else w
    if kK and w.sum() > 0:
        okK = (at > 0) & isK & (sign != 0)
        selK = np.zeros(F, bool)
        selK[np.argsort(-np.where(okK, at, -np.inf))[:kK]] = True
        selK &= okK
        if selK.any():
            wk = np.where(selK, np.abs(icir), 0.0)
            w = (1 - lamK) * w + lamK * wk / wk.sum()
    w = w * sign
    if anchor > 0 and anchor_weights and np.abs(w).sum() > 0:
        # v1.3 趨勢錨：固定權重的相對強度/動能組合（取自 V23.1 PRICE_RS），不參與估計 → 不會被資料過度配適
        names = list(meta["factors"])
        wa = np.zeros(F)
        for f, x in anchor_weights.items():
            if f in names and allowed[names.index(f)]:
                wa[names.index(f)] = x * (dirs[names.index(f)] or 1)
        if np.abs(wa).sum() > 0:
            w = (1 - anchor) * w / np.abs(w).sum() + anchor * wa / np.abs(wa).sum()
    if growth_blend > 0 and growth_weights and np.abs(w).sum() > 0:
        # v1.3 成長動能假說（V23.1 GROWTH_NO_FLOW 的固定權重，研究前即宣告、未用我方資料調整）與精簡模型做模型平均
        names = list(meta["factors"])
        wg = np.zeros(F)
        for f, x in growth_weights.items():
            if f in names:
                wg[names.index(f)] = x * (dirs[names.index(f)] or 1)
        if np.abs(wg).sum() > 0:
            w = (1 - growth_blend) * w / np.abs(w).sum() + growth_blend * wg / np.abs(wg).sum()
    if hyp_blend > 0 and hyp_weights and np.abs(w).sum() > 0:
        # v1.4 與 ChatGPT V24 預先宣告的價格領先假說做 50/50 模型平均（固定權重，不用我方資料估計）
        names = list(meta["factors"])
        wh = np.zeros(F)
        for f, x in hyp_weights.items():
            if f in names:
                wh[names.index(f)] = x * (dirs[names.index(f)] or 1)
        if np.abs(wh).sum() > 0:
            w = (1 - hyp_blend) * w / np.abs(w).sum() + hyp_blend * wh / np.abs(wh).sum()
    fam_w = {f: float(np.abs(w[fams == f]).sum()) for f in FAMILIES if np.abs(w[fams == f]).sum() > 0}
    return FitResult(w.astype(np.float32), fam_w, stats, 1.0, int(n))


def fit_weights(ic: np.ndarray, rows: np.ndarray, meta: dict, horizon: str, prior_family: dict,
                lam: float, mode: str, n0: float, max_w: float, step: int, H: int,
                view: PoolView | None = None, fwd: np.ndarray | None = None, exclude=(),
                k: int = 5, tmin: float = 1.0, kK: int = 0, lamK: float = 0.0,
                anchor: float = 0.0, anchor_weights: dict | None = None,
                growth_blend: float = 0.0, growth_weights: dict | None = None,
                hyp_blend: float = 0.0, hyp_weights: dict | None = None) -> FitResult:
    """mode: 'sparse' (精簡因子), 'opt' (IC-IR + λ floor), 'data' (no K family), 'kpure' (K only), 'prior' (no fitting)."""
    if mode == "sparse":
        return fit_sparse(ic, rows, meta, horizon, step, H, k, tmin, kK, lamK, exclude, anchor, anchor_weights,
                          growth_blend, growth_weights, hyp_blend, hyp_weights)
    fams = np.array(meta["families"])
    dirs = np.array(meta["directions"])
    allowed = np.array([horizon in h for h in meta["horizons"]])
    if exclude:
        allowed &= ~np.isin(np.array(meta["factors"]), list(exclude))
    if mode == "data":
        allowed &= fams != "K"
    if mode == "kpure":
        allowed &= fams == "K"
    F = len(fams)
    X = ic[rows] if len(rows) else np.zeros((0, F))
    if len(X):
        X = np.nan_to_num(X[np.isfinite(X).any(axis=1)], nan=0.0)  # factor without data → IC 0
    n = len(X)
    n_eff = n * step / max(H, step)
    alpha = 0.0 if mode == "prior" else n_eff / (n_eff + n0)
    mean = X.mean(0) if n else np.zeros(F)
    sd = X.std(0, ddof=1) if n > 2 else np.full(F, np.nan)
    icir = np.where(sd > 0, mean / np.where(sd > 0, sd, 1), 0)
    tstat = icir * np.sqrt(max(n_eff, 1))
    sign = np.where(dirs != 0, dirs, np.where(np.abs(tstat) > 1.0, np.sign(mean), 0)).astype(float)
    sign = np.where(allowed, sign, 0)

    w_in = np.zeros(F)
    fam_list = [f for f in FAMILIES if (allowed & (fams == f) & (sign != 0)).any()]
    for fam in fam_list:
        sel = np.where(allowed & (fams == fam) & (sign != 0))[0]
        prior = np.ones(len(sel)) / len(sel)
        if mode == "prior" or n < 8:
            w = prior
        else:
            aligned = X[:, sel] * sign[sel]
            m = aligned.mean(0)
            keep = m > 0
            w = np.zeros(len(sel))
            if keep.any():
                w[keep] = _shrunk_solve(aligned[:, keep], m[keep])
            w = _cap(w, max_w) if w.sum() > 0 else w
            # shrink toward equal prior; factors contradicting their prior stay at 0
            w = alpha * w + (1 - alpha) * prior * keep
            w = w / w.sum() if w.sum() > 0 else prior * 0
        w_in[sel] = w

    # family level
    fam_prior = {f: prior_family.get(f, 0) for f in fam_list}
    if mode == "data":
        fam_prior.pop("K", None)
    if mode == "kpure":
        fam_prior = {"K": 1.0} if "K" in fam_list else {}
    tot = sum(fam_prior.values())
    fam_prior = {f: v / tot for f, v in fam_prior.items()} if tot > 0 else {}
    fam_w = dict(fam_prior)
    if mode == "opt" and n >= 8 and view is not None and fwd is not None and len(fam_list) > 1:
        comp = []
        for fam in fam_list:
            wf = np.where(fams == fam, w_in * sign, 0)
            comp.append(composite_ic(view, fwd, wf, rows) if wf.any() else np.full(len(rows), np.nan))
        C = np.column_stack(comp)
        C = C[np.isfinite(C).all(axis=1)]
        if len(C) >= 8:
            v = _shrunk_solve(C, C.mean(0))
            if v.sum() > 0:
                v = v / v.sum()
                fam_w = {f: alpha * v[k] + (1 - alpha) * fam_prior.get(f, 0) for k, f in enumerate(fam_list)}
    if mode == "opt" and "K" in fam_w and lam > 0 and fam_w["K"] < lam:
        rest = sum(v for f, v in fam_w.items() if f != "K")
        fam_w = {f: (lam if f == "K" else (v / rest * (1 - lam) if rest > 0 else 0)) for f, v in fam_w.items()}
    s = sum(fam_w.values())
    fam_w = {f: v / s for f, v in fam_w.items()} if s > 0 else fam_w

    weights = np.zeros(F)
    for f, v in fam_w.items():
        weights += np.where(fams == f, w_in * v, 0)
    weights = weights * sign
    stats = {"mean_ic": mean, "icir": icir, "t": tstat, "sign": sign, "allowed": allowed}
    return FitResult(weights.astype(np.float32), fam_w, stats, float(alpha), int(n))
