"""Research protocol (pre-registered, chronological):

  1. 母池測試  every rule × size, same default model (λ=0.40, gates on, full exposure), walk-forward
              out-of-sample on the DEVELOPMENT period only → pick the pool per horizon.
  2. 模型測試  on the chosen pool: Kostolany λ × market-exposure profile, plus K-only; baselines
              DATA (no Kostolany), PRIOR (no fitting), POOL_EW, 0050. Pick on development only.
  3. 盲測      the last `holdout_months` are reported for the chosen model but never used to choose.
  4. 消融      switch each Kostolany component off to measure its contribution.
  5. 今日推薦  refit on all matured data and rank the latest signal date.
Weights are re-estimated every `retrain_months` using only labels that had matured by then
(purged: a training label must end before the retrain date).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import choose, market_stage, run, spec_from_cfg
from .bundle import Bundle
from .config import HORIZON_KEYS, STAGE_CODES, STAGE_NAMES
from .metrics import (block_bootstrap_excess, deflated_sharpe, perf, probabilistic_sharpe, selection_score,
                      tranche_stats, yearly_table)
from .optimize import PoolView, fit_weights, ic_matrix, pool_view


HKEY = {"short": 0, "mid": 1, "long": 2}
DEFAULT_BLOCK = ("A3", "B1", "B2")


OPPOSITE = {"f_size": "大市值", "t_lowvol": "波動較大", "t_lowidio": "個股波動較大", "t_rev5": "近5日強勢",
            "c_short_ratio": "券資比低", "k_paid": "融資比重高", "k_calm": "交易熱絡", "t_high252": "離一年新高較遠",
            "t_rs120": "相對大盤偏弱", "t_mom120x20": "中期動能偏弱", "f_rev_accel": "營收成長放緩", "f_rev_high": "營收未創新高",
            "f_ep": "本益比偏高", "f_dy": "殖利率偏低", "f_bp": "股價淨值比偏高", "f_rev_yoy": "月營收年減",
            "f_rev_yoy3": "近3月營收衰退", "f_eps_yoy": "EPS衰退", "f_roe": "ROE偏低", "f_rev_streak": "營收成長中斷"}


def factor_phrase(B, f: int, z: float) -> str:
    """因子在該股是「高」還是「低」：z<0 時改用反義描述（例：小市值 → 大市值）。"""
    name, zh = B.factors[f], B.meta["factor_zh"][f]
    if z >= 0:
        return zh
    return OPPOSITE.get(name, f"{zh}偏低")


@dataclass
class Variant:
    name: str
    mode: str = "opt"          # opt | data | kpure | prior
    lam: float = 0.40
    exposure: str = "none"
    gates: bool = True
    hysteresis: float | None = None   # None → horizon default
    stop: str = "default"             # default | off | on (= horizons.*.stop_loss_test)
    exclude: tuple = ()               # factors forced to weight 0 (ablation)
    stage_gate: bool | None = None    # 個股雞蛋階段閘門：None=依設定檔，True=強制開(A3/B1/B2)，False=關
    dtop_gate: bool = True            # M頭閘門
    k: int = 5                        # sparse：使用的因子數
    tmin: float = 1.0                 # sparse：t 值門檻
    kK: int = 0                       # sparse：另外保留的科斯托蘭尼因子數
    lamK: float = 0.0                 # sparse：科斯托蘭尼因子合計權重
    anchor: float = 0.0               # sparse：趨勢錨（固定權重相對強度組合）占比
    growth_blend: float = 0.0         # sparse：與「成長動能假說」(V23.1) 模型平均的占比
    hyp_blend: float = 0.0            # sparse：與 V24 價格領先假說平均的占比
    hypothesis: str | None = None     # research.score_hypotheses 的名稱
    label_h: str | None = None        # 用哪個期間的報酬估計權重（v1.5 短線：用中線標籤＝「快調慢α」）
    weighting: str | None = None      # None → 依設定檔 horizons.*.weighting
    overheat: bool | None = None      # 不追高閘門（RSI/乖離）：None=依設定檔
    sleeve: float | None = None       # 個股部位比例：None=依設定檔
    residual: str | None = None       # 未投入資金：cash | bench(0050)；None=依設定檔

    def label(self) -> str:
        return self.name


def smooth_pool_scores(t: pd.DataFrame) -> pd.Series:
    """Average each pool's score with its neighbouring sizes of the same rule (0.5 / 0.25 / 0.25).
    A pool that only wins by itself while its neighbours lose is more likely luck than a real effect."""
    out = pd.Series(np.nan, index=t.index)
    for _, g in t.groupby("rule"):
        g = g.sort_values("size")
        sc = g["raw_score"].to_numpy()
        for k, ix in enumerate(g.index):
            nb = [sc[j] for j in (k - 1, k + 1) if 0 <= j < len(sc)]
            out[ix] = 0.5 * sc[k] + 0.5 * float(np.mean(nb)) if nb else sc[k]
    return out


class Research:
    def __init__(self, B: Bundle, cfg: dict, log=print):
        self.B, self.cfg, self.log = B, cfg, log
        r = cfg["research"]
        sd = B.sig_dates
        self.S = len(sd)
        self.first_oos = int(sd.searchsorted(sd[0] + pd.DateOffset(months=r["train_min_months"])))
        self.holdout = int(sd.searchsorted(sd[-1] - pd.DateOffset(months=r["holdout_months"])))
        self.retrain = []
        d = sd[min(self.first_oos, self.S - 1)]
        while True:
            i = int(sd.searchsorted(d))
            if i >= self.S:
                break
            self.retrain.append(i)
            d = d + pd.DateOffset(months=r["retrain_months"])
        self._views, self._ics, self._sched = {}, {}, {}
        if "SECTOR" not in B.a:   # integer sector codes for the per-sector cap (NA → -1 = no cap)
            secs = B.meta.get("sectors") or ["NA"] * len(B.tickers)
            uniq = {x: k for k, x in enumerate(sorted({x for x in secs if x not in ("NA", "", None)}))}
            B.a["SECTOR"] = np.array([uniq.get(x, -1) for x in secs], np.int32)

    # ------------------------------------------------------------------ cached pieces
    def view(self, rule: str, size: int) -> PoolView:
        key = (rule, size)
        if key not in self._views:
            if len(self._views) > 6:
                self._views.pop(next(iter(self._views)))
                self._ics = {k: v for k, v in self._ics.items() if (k[0], k[1]) in self._views}
            ri = self.B.meta["rules"].index(rule)
            rk = self.B.a["RANKS"][ri]
            mask = (rk >= 1) & (rk <= size)
            self._views[key] = pool_view(self.B.a["U"], mask, self.B.a["TRADE"])
        return self._views[key]

    def ic(self, rule, size, h) -> np.ndarray:
        key = (rule, size, h)
        if key not in self._ics:
            self._ics[key] = ic_matrix(self.view(rule, size), self.B.a["FWD"][HKEY[h]])
        return self._ics[key]

    def H(self, h) -> int:
        return self.cfg["horizons"][h]["days"]

    def train_rows(self, upto_i: int, h: str) -> np.ndarray:
        sp = self.B.sig_pos
        cut = sp[upto_i] if upto_i < self.S else sp[-1] + 1
        return np.where(sp + self.H(h) + 1 <= cut)[0]

    # ------------------------------------------------------------------ weights
    def schedule(self, rule, size, h, v: Variant):
        r0 = self.cfg["research"]
        hz = v.label_h or h
        key = (rule, size, h, hz, v.mode, round(v.lam, 4), tuple(sorted(self.cfg["horizons"][hz]["family_prior"].items())),
               r0["n0_shrink"], r0["max_factor_weight"], self._excl(v), v.k, v.tmin, v.kK, v.lamK, v.anchor, v.growth_blend,
               v.hyp_blend, v.hypothesis)
        if key in self._sched:
            return self._sched[key]
        if len(self._sched) > 64:
            self._sched.clear()
        view, ic = self.view(rule, size), self.ic(rule, size, hz)
        r = self.cfg["research"]
        W = [None] * self.S
        hist = []
        pts = self.retrain + [self.S]
        for a, b in zip(pts[:-1], pts[1:]):
            rows = self.train_rows(a, hz)
            fit = fit_weights(ic, rows, self.B.meta, hz, self.cfg["horizons"][hz]["family_prior"], v.lam, v.mode,
                              r["n0_shrink"], r["max_factor_weight"], r["signal_step"], self.H(hz), view,
                              self.B.a["FWD"][HKEY[hz]], exclude=self._excl(v), **self._sp(v))
            for i in range(a, b):
                W[i] = fit.weights
            hist.append({"from": str(self.B.sig_dates[a].date()), "n_train": fit.n_rows, "alpha": round(fit.alpha, 3),
                         **{f"fam_{k}": round(val, 4) for k, val in fit.family.items()},
                         "weights": {f: float(w) for f, w in zip(self.B.factors, fit.weights) if abs(w) > 1e-6}})
        self._sched[key] = (W, hist)
        return W, hist

    def final_fit(self, rule, size, h, v: Variant):
        r = self.cfg["research"]
        hz = v.label_h or h
        rows = self.train_rows(self.S, hz)
        return fit_weights(self.ic(rule, size, hz), rows, self.B.meta, hz, self.cfg["horizons"][hz]["family_prior"],
                           v.lam, v.mode, r["n0_shrink"], r["max_factor_weight"], r["signal_step"], self.H(hz),
                           self.view(rule, size), self.B.a["FWD"][HKEY[hz]], exclude=self._excl(v), **self._sp(v))

    def _sp(self, v: Variant) -> dict:
        return {"k": v.k, "tmin": v.tmin, "kK": v.kK, "lamK": v.lamK, "anchor": v.anchor,
                "anchor_weights": self.cfg["research"].get("momentum_anchor"), "growth_blend": v.growth_blend,
                "growth_weights": self.cfg["research"].get("growth_hypothesis"), "hyp_blend": v.hyp_blend,
                "hyp_weights": (self.cfg["research"].get("score_hypotheses") or {}).get(v.hypothesis or "")}

    def _excl(self, v: Variant) -> tuple:
        """Factors a sparse model may not *select* (they can still enter through the fixed anchor/hypothesis)."""
        extra = tuple(self.cfg["research"].get("sparse_exclude") or ()) if v.mode == "sparse" else ()
        return tuple(v.exclude) + extra

    # ------------------------------------------------------------------ evaluation
    def spec(self, h, v: Variant):
        kw = {"use_gates": v.gates}
        hc, kc = self.cfg["horizons"][h], self.cfg["kostolany"]
        kw["weighting"] = v.weighting or hc.get("weighting", "equal")
        oh = kc.get("overheat_gate", False) if v.overheat is None else v.overheat
        if oh:
            kw["max_rsi"] = kc.get("max_rsi", 84)
            kw["max_bias20"] = kc.get("max_bias20", 0.14)
        kw["sleeve"] = float(hc.get("sleeve", 1.0) if v.sleeve is None else v.sleeve)
        kw["residual"] = v.residual or hc.get("residual", "cash")
        if self.cfg["research"].get("max_per_sector"):
            kw["max_per_sector"] = int(self.cfg["research"]["max_per_sector"])
        if v.stage_gate is False:
            kw.update(block_stages=(), b3_confirm=False)
        elif v.stage_gate is True:
            kw.update(block_stages=DEFAULT_BLOCK, b3_confirm=True)
        if not v.dtop_gate:
            kw["block_dtop"] = False
        if v.hysteresis is not None:
            kw["hysteresis"] = v.hysteresis
        if v.stop == "off":
            kw["stop_loss"] = None
        elif v.stop == "on":
            kw["stop_loss"] = self.cfg["horizons"][h].get("stop_loss_test") or self.cfg["horizons"][h]["stop_loss"]
        return spec_from_cfg(self.cfg, h, v.exposure, **kw)

    def segments(self, res):
        sd = self.B.sig_dates
        hd = sd[min(self.holdout, self.S - 1)]
        nav, bench = res.nav, res.bench
        dev = nav[nav.index < hd]
        hol = nav[nav.index >= nav.index[nav.index.searchsorted(hd) - 1]] if hd > nav.index[0] else nav
        return {"dev": perf(dev / dev.iloc[0], bench.reindex(dev.index)) if len(dev) > 20 else {},
                "holdout": perf(hol / hol.iloc[0], bench.reindex(hol.index)) if len(hol) > 20 else {},
                "oos": perf(nav, bench)}

    def evaluate(self, rule, size, h, v: Variant, keep=False) -> dict:
        W, hist = self.schedule(rule, size, h, v)
        res = run(self.B, self.view(rule, size), W, self.spec(h, v), start_i=self.first_oos)
        seg = self.segments(res)
        out = {"rule": rule, "size": size, "horizon": h, "variant": v.name, "mode": v.mode, "lam": v.lam,
               "exposure": v.exposure, "gates": v.gates, **{f"dev_{k}": val for k, val in seg["dev"].items()},
               **{f"hold_{k}": val for k, val in seg["holdout"].items()},
               **{f"oos_{k}": val for k, val in seg["oos"].items()},
               "turnover": res.turnover, **{f"tr_{k}": val for k, val in tranche_stats(
                   res.tranches[res.tranches["date"] < self.B.sig_dates[min(self.holdout, self.S - 1)]]).items()}}
        if keep:
            out["_res"], out["_hist"], out["_W"] = res, hist, W
        return out

    def pool_ew(self, rule, size, h):
        spec = spec_from_cfg(self.cfg, h, "none", equal_weight_all=True, use_gates=False, stop_loss=None)
        res = run(self.B, self.view(rule, size), [None] * self.S, spec, start_i=self.first_oos)
        return res

    # ------------------------------------------------------------------ studies
    def pool_study(self, h, rules=None, sizes=None, progress=None) -> pd.DataFrame:
        rules = rules or self.cfg["pools"]["rules"]
        sizes = sizes or self.cfg["pools"]["sizes"]
        v = self.model_variant(h, name="POOL_TEST")
        rows = []
        for rule in rules:
            for size in sizes:
                t0 = time.time()
                row = self.evaluate(rule, size, h, v)
                ew = self.pool_ew(rule, size, h)
                seg = self.segments(ew)["dev"]
                row.update({"pool_ew_dev_cagr": seg.get("cagr"), "pool_ew_dev_sharpe": seg.get("sharpe"),
                            "avg_members": float(np.mean([len(m) for m in self.view(rule, size).members[self.first_oos:]])),
                            "ic_composite_dev": self._composite_dev_ic(rule, size, h)})
                rows.append(row)
                if progress:
                    progress(f"{h} 母池 {rule}{size}: 開發期CAGR {row.get('dev_cagr', np.nan):.1%} "
                             f"Sharpe {row.get('dev_sharpe', np.nan):.2f} ({time.time()-t0:.1f}s)")
        t = pd.DataFrame(rows)
        t["raw_score"] = selection_score(t.rename(columns=lambda c: c.replace("dev_", "")), self.cfg["research"]["selection_weights"])
        t["score"] = smooth_pool_scores(t) if self.cfg["research"].get("pool_smoothing", True) else t["raw_score"]
        return t.sort_values("score", ascending=False).reset_index(drop=True)

    def _composite_dev_ic(self, rule, size, h):
        """Descriptive only (not used to choose): mean of the 5 largest |IC| in the development period."""
        ic = self.ic(rule, size, h)
        rows = np.arange(self.first_oos, self.holdout)
        rows = rows[np.isin(rows, self.train_rows(self.holdout, h))] if len(rows) else rows
        if not len(rows):
            return np.nan
        ok = np.array([h in hz for hz in self.B.meta["horizons"]])
        m = np.abs(np.nanmean(ic[rows][:, ok], 0))
        return float(np.nanmean(np.sort(m)[-5:]))

    def model_variant(self, h, name=None, **kw) -> Variant:
        """The horizon's default model from config (horizons.*.model)."""
        m = dict(self.cfg["horizons"][h].get("model") or {"mode": "opt", "lam": 0.25})
        m.update(kw)
        mode = m.get("mode", "opt")
        v = Variant(name or "", mode, float(m.get("lam", 0.0 if mode == "sparse" else 0.25)), m.get("exposure", "none"), True,
                    k=int(m.get("k", 5)), tmin=float(m.get("tmin", 1.0)), kK=int(m.get("kK", 0)), lamK=float(m.get("lamK", 0.0)),
                    anchor=float(m.get("anchor", 0.0)), growth_blend=float(m.get("growth_blend", 0.0)),
                    hyp_blend=float(m.get("hyp_blend", 0.0)), hypothesis=m.get("hypothesis"), label_h=m.get("label_horizon"))
        if not name:
            v.name = self.variant_label(v)
        return v

    @staticmethod
    def variant_label(v: Variant) -> str:
        if v.mode == "sparse":
            extra = ((f"＋K{v.kK}({v.lamK:.0%})" if v.kK else "") + (f"＋趨勢錨{v.anchor:.0%}" if v.anchor else "")
                     + (f"＋成長動能假說{v.growth_blend:.0%}" if v.growth_blend else "")
                     + (f"＋{v.hypothesis}{v.hyp_blend:.0%}" if v.hyp_blend else ""))
            if v.growth_blend >= 0.999:
                return f"成長動能假說(V23.1 GROWTH，我方資料) 水位={v.exposure}"
            return f"精簡{v.k}因子 t>{v.tmin:g}{extra} 水位={v.exposure}"
        return f"KOSTO λ={v.lam:.2f} 水位={v.exposure}"

    def variants(self, h=None) -> list[Variant]:
        r = self.cfg["research"]
        vs = []
        for ex in r["exposure_profiles"]:
            if h is not None and (self.cfg["horizons"][h].get("model") or {}).get("mode") == "sparse":
                base = self.model_variant(h, exposure=ex)
                vs.append(base)
                for kk in r.get("sparse_k_alternatives", []):
                    if kk != base.k:
                        vs.append(self.model_variant(h, exposure=ex, k=kk))
                vs.append(self.model_variant(h, exposure=ex, kK=2, lamK=0.3))      # 科斯托蘭尼加強版
                alt = 0.0 if base.anchor else 0.3                                    # 有/無趨勢錨對照
                vs.append(self.model_variant(h, exposure=ex, anchor=alt))
                gb = 0.0 if base.growth_blend else 0.3                               # 有/無成長動能假說對照
                vs.append(self.model_variant(h, exposure=ex, growth_blend=gb))
                vs.append(self.model_variant(h, exposure=ex, growth_blend=0.5))
                vs.append(self.model_variant(h, exposure=ex, growth_blend=1.0))      # V23.1 GROWTH 原假說
                if base.hyp_blend:
                    vs.append(self.model_variant(h, exposure=ex, hyp_blend=0.0))       # v1.3（無 V24 假說）對照
            for lam in r["lambdas"]:
                vs.append(Variant(f"KOSTO λ={lam:.2f} 水位={ex}", "opt", lam, ex, True))
        vs.append(Variant("KOSTO_PURE 只用科斯托蘭尼因子", "kpure", 1.0, "soft", True))
        for v in vs:
            if v.mode == "sparse":
                v.name = self.variant_label(v)
        return vs

    def baselines(self) -> list[Variant]:
        return [Variant("DATA 純數據(無科斯托蘭尼)", "data", 0.0, "none", False),
                Variant("PRIOR 先驗權重(不最佳化)", "prior", 0.40, "soft", True)]

    def variant_study(self, h, rule, size, progress=None) -> pd.DataFrame:
        rows = []
        for v in self.variants(h) + self.baselines():
            row = self.evaluate(rule, size, h, v)
            row["is_baseline"] = v.mode in ("data", "prior")
            rows.append(row)
            if progress:
                progress(f"{h} 模型 {v.name}: 開發期CAGR {row.get('dev_cagr', np.nan):.1%}")
        t = pd.DataFrame(rows)
        t["score"] = selection_score(t.rename(columns=lambda c: c.replace("dev_", "")), self.cfg["research"]["selection_weights"])
        return t

    def ablation(self, h, rule, size, v: Variant) -> pd.DataFrame:
        """Switch each Kostolany rule to the opposite state. 貢獻 = 績效(規則開) − 績效(規則關)：正＝這條規則有幫助."""
        from dataclasses import replace
        kf = [f for f, fam, hz in zip(self.B.factors, self.B.meta["families"], self.B.meta["horizons"])
              if fam == "K" and h in hz]
        zh = dict(zip(self.B.factors, self.B.meta["factor_zh"]))
        stage_on = bool(self.cfg["kostolany"]["block_stages"]) if v.stage_gate is None else bool(v.stage_gate)
        stop_on = (v.stop == "on") or (v.stop == "default" and bool(self.cfg["horizons"][h]["stop_loss"]))
        can_stop = bool(self.cfg["horizons"][h].get("stop_loss_test") or self.cfg["horizons"][h]["stop_loss"])
        # (key, label, variant, rule_is_on_in_base)
        cases = [("base", "目前選定模型", v, None),
                 ("stage_gate", "移除個股雞蛋階段閘門" if stage_on else "加回個股雞蛋階段閘門(A3/B1/B2不買)",
                  replace(v, stage_gate=not stage_on), stage_on),
                 ("dtop", "移除M頭閘門" if v.gates else "加回M頭閘門", replace(v, gates=True, dtop_gate=not v.gates) if not v.gates
                  else replace(v, dtop_gate=False), v.gates),
                 ("exposure", "移除市場階段持股水位" if v.exposure != "none" else "加入市場階段持股水位(soft)",
                  replace(v, exposure="none" if v.exposure != "none" else "soft"), v.exposure != "none"),
                 ("K", "移除科斯托蘭尼因子(K族群)", replace(v, exclude=tuple(v.exclude) + tuple(kf), kK=0) if v.mode == "sparse"
                  else replace(v, mode="data", lam=0.0), v.mode in ("opt", "kpure", "sparse")),
                 ("hyst", "移除耐心續抱(遲滯)", replace(v, hysteresis=1.0), True)]
        if can_stop:
            cases.append(("stop", "移除停損(戒3)" if stop_on else "加回停損(戒3)", replace(v, stop="off" if stop_on else "on"), stop_on))
        if v.mode in ("opt", "kpure", "sparse"):
            cases += [(f, f"移除 {f}（{zh[f]}）", replace(v, exclude=tuple(v.exclude) + (f,)), True) for f in kf if f not in v.exclude]
        rows = []
        for key, label, vv, on in cases:
            e = self.evaluate(rule, size, h, vv)
            rows.append({"key": key, "case": label, "rule_on": on, "dev_cagr": e.get("dev_cagr"), "dev_sharpe": e.get("dev_sharpe"),
                         "dev_mdd": e.get("dev_mdd"), "hold_cagr": e.get("hold_cagr"), "hold_mdd": e.get("hold_mdd"),
                         "turnover": e.get("turnover")})
        t = pd.DataFrame(rows)
        base = t.iloc[0]
        sign = t["rule_on"].map({True: 1.0, False: -1.0}).fillna(0.0)
        t["貢獻(開發期CAGR)"] = (base["dev_cagr"] - t["dev_cagr"]) * sign
        t["貢獻(盲測CAGR)"] = (base["hold_cagr"] - t["hold_cagr"]) * sign
        t.loc[0, ["貢獻(開發期CAGR)", "貢獻(盲測CAGR)"]] = np.nan
        t["目前"] = t["rule_on"].map({True: "啟用", False: "關閉"}).fillna("")
        return t.drop(columns=["rule_on"])

    # ------------------------------------------------------------------ latest recommendations
    def latest(self, h, rule, size, v: Variant, res=None) -> dict:
        B = self.B
        fit = self.final_fit(rule, size, h, v)
        spec = self.spec(h, v)
        i = self.S - 1
        nt = max(1, spec.H // spec.step)
        prev = []
        if res is not None and len(res.tranches) > nt:
            prev_row = res.tranches[res.tranches["i"] == i - nt]
            if len(prev_row):
                prev = [B.tickers.index(t) for t in prev_row.iloc[0]["picks"] if t in B.tickers]
        view = self.view(rule, size)
        picks, order, smap = choose(B, view, i, fit.weights, prev, spec)
        stage = market_stage(B, i)
        idx = view.members[i]
        pos_of = {j: k for k, j in enumerate(idx.tolist())}
        Z = view.Z[i]
        fz = B.meta["factor_zh"]

        def describe(j):
            k = pos_of[j]
            contrib = Z[k] * fit.weights
            top = np.argsort(-contrib)[:3]
            neg = np.argsort(contrib)[:1]
            st = int(B.a["STAGE"][i, j])
            close = float(B.a["CLOSE"][i, j])
            stop = spec.stop_loss or self.cfg["horizons"][h].get("alert_loss")
            return {"ticker": B.tickers[j], "name": B.meta["names"][j], "sector": B.meta["sectors"][j],
                    "close": close, "score": float(smap.get(j, np.nan)), "stage": STAGE_NAMES[st],
                    "kept": j in prev, "w_support": bool(B.a["WSUP"][i, j]),
                    "reasons": [f"{factor_phrase(B, f, Z[k][f])}(+{contrib[f]:.3f})" for f in top if contrib[f] > 0],
                    "risk": [f"{factor_phrase(B, f, Z[k][f])}({contrib[f]:.3f})" for f in neg if contrib[f] < 0],
                    "stop_price": round(close * (1 - stop), 2) if stop else None}

        blocked = []
        top_all = idx[np.argsort(-(Z @ fit.weights))][:30] if len(idx) else []
        chosen = set(picks) | set(order[: 2 * spec.top_k])
        for j in top_all:
            if j in chosen:
                continue
            st = STAGE_NAMES[int(B.a["STAGE"][i, j])]
            why = []
            if st in spec.block_stages:
                why.append(f"個股階段{st}")
            if spec.block_dtop and B.a["DTOP"][i, j]:
                why.append("M頭(多次攻不過)")
            if st == "B3" and spec.b3_confirm and not B.a["WSUP"][i, j]:
                why.append("B3未見止跌確認")
            if np.nan_to_num(B.a["CAP"][i, j]) * spec.max_participation < spec.capital / spec.top_k:
                why.append("流動性不足")
            if why:
                blocked.append({"ticker": B.tickers[j], "name": B.meta["names"][j], "why": "、".join(why)})
        return {"date": str(B.sig_dates[i].date()), "market_stage": stage,
                "exposure": float(spec.exposure.get(stage, 1.0)), "pool": f"{rule}{size}", "variant": v.name,
                "picks": sorted([describe(j) for j in picks], key=lambda d: -d["score"]),
                "alternates": [describe(j) for j in order if j not in picks][:10],
                "blocked": blocked[:10],
                "weights": {f: float(w) for f, w in zip(B.factors, fit.weights) if abs(w) > 1e-6},
                "family": fit.family}

    # ------------------------------------------------------------------ full ranking (個股診斷用)
    def rank_table(self, h, rule, size, v: Variant) -> pd.DataFrame:
        """最新訊號日：母池內每一檔的模型分數、名次（只計可買標的）、被排除原因與前三大加分因子。"""
        B = self.B
        fit = self.final_fit(rule, size, h, v)
        spec = self.spec(h, v)
        i = self.S - 1
        view = self.view(rule, size)
        idx = view.members[i]
        cols = ["ticker", "j", "score", "pct", "rank", "n_valid", "valid", "why", "reasons", "risk"]
        if fit.weights is None or len(idx) == 0:
            return pd.DataFrame(columns=cols)
        Z = view.Z[i]
        score = Z @ fit.weights
        why = [[] for _ in idx]
        cap = np.nan_to_num(B.a["CAP"][i, idx])
        for k in np.where(cap * spec.max_participation < spec.capital / spec.top_k)[0]:
            why[k].append("成交量太小")
        if spec.max_rsi is not None and "RSI14" in B.a:
            for k in np.where(B.a["RSI14"][i, idx] > spec.max_rsi)[0]:
                why[k].append("RSI 過熱")
        if spec.max_bias20 is not None and "BIAS20" in B.a:
            for k in np.where(B.a["BIAS20"][i, idx] > spec.max_bias20)[0]:
                why[k].append("乖離過大")
        if spec.use_gates:
            st = B.a["STAGE"][i, idx]
            for s_ in spec.block_stages:
                for k in np.where(st == STAGE_CODES[s_])[0]:
                    why[k].append(f"個股階段{s_}")
            if spec.b3_confirm:
                for k in np.where((st == STAGE_CODES["B3"]) & ~B.a["WSUP"][i, idx])[0]:
                    why[k].append("B3未止跌")
            if spec.block_dtop:
                for k in np.where(B.a["DTOP"][i, idx])[0]:
                    why[k].append("M頭（多次攻不過）")
        valid = np.array([not w for w in why])
        rank = np.full(len(idx), np.nan)
        vo = np.where(valid)[0]
        rank[vo[np.argsort(-score[vo], kind="stable")]] = np.arange(1, len(vo) + 1)
        pct = pd.Series(score).rank(pct=True).to_numpy()
        fz = B.meta["factor_zh"]
        rows = []
        for k, j in enumerate(idx.tolist()):
            contrib = Z[k] * fit.weights
            top = np.argsort(-contrib)[:3]
            neg = np.argsort(contrib)[:2]
            rows.append({"ticker": B.tickers[j], "j": j, "score": float(score[k]), "pct": float(pct[k]),
                         "rank": rank[k], "n_valid": len(vo), "valid": bool(valid[k]), "why": "、".join(why[k]),
                         "reasons": [factor_phrase(B, f, Z[k][f]) for f in top if contrib[f] > 0],
                         "risk": [factor_phrase(B, f, Z[k][f]) for f in neg if contrib[f] < 0]})
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------ factor table
    def factor_table(self, h, rule, size, weights) -> pd.DataFrame:
        ic = self.ic(rule, size, h)
        B = self.B
        dev = np.arange(self.first_oos if self.first_oos < self.S else 0, self.holdout)
        allr = self.train_rows(self.S, h)
        hol = np.arange(self.holdout, self.S)
        hol = hol[np.isin(hol, allr)]
        step, H = self.cfg["research"]["signal_step"], self.H(h)

        def st(rows):
            X = ic[rows]
            m = np.nanmean(X, 0) if len(rows) else np.full(ic.shape[1], np.nan)
            sd = np.nanstd(X, 0, ddof=1) if len(rows) > 2 else np.full(ic.shape[1], np.nan)
            neff = max(len(rows) * step / max(H, step), 1)
            return m, m / np.where(sd > 0, sd, np.nan), m / np.where(sd > 0, sd, np.nan) * np.sqrt(neff)
        all_rows = self.train_rows(self.S, h)
        m_all, ir_all, t_all = st(all_rows)
        m_dev, ir_dev, t_dev = st(dev[np.isin(dev, all_rows)])
        m_h, _, _ = st(hol)
        rows = []
        for k, f in enumerate(B.factors):
            rows.append({"factor": f, "中文": B.meta["factor_zh"][k], "family": B.meta["families"][k],
                         "prior_dir": B.meta["directions"][k],
                         "horizon_ok": (h in B.meta["horizons"][k]) or abs(weights.get(f, 0.0)) > 0,
                         "coverage": B.meta["coverage"][f], "ic_all": m_all[k], "icir_all": ir_all[k], "t_all": t_all[k],
                         "ic_dev": m_dev[k], "t_dev": t_dev[k], "ic_holdout": m_h[k], "weight": weights.get(f, 0.0)})
        return pd.DataFrame(rows).sort_values(["family", "weight"], ascending=[True, False])

    # ------------------------------------------------------------------ stage performance
    def stage_table(self, res) -> pd.DataFrame:
        tr = res.tranches.dropna(subset=["ret"]).copy()
        if tr.empty:
            return pd.DataFrame()
        tr["excess"] = tr["ret"] - tr["bench"]
        g = tr.groupby("stage").agg(n=("ret", "size"), avg_ret=("ret", "mean"), avg_bench=("bench", "mean"),
                                    avg_excess=("excess", "mean"), avg_exposure=("exposure", "mean"))
        return g.reset_index()

    # ------------------------------------------------------------------ full pipeline
    def run_all(self, horizons=HORIZON_KEYS, quick=False, progress=print) -> dict:
        t0 = time.time()
        out = {"meta": {k: self.B.meta[k] for k in ("version", "built", "synthetic", "data_quality")},
               "protocol": {"first_oos": str(self.B.sig_dates[min(self.first_oos, self.S - 1)].date()),
                            "holdout_start": str(self.B.sig_dates[min(self.holdout, self.S - 1)].date()),
                            "last_date": str(self.B.sig_dates[-1].date()), "signals": self.S,
                            "retrain_points": [str(self.B.sig_dates[i].date()) for i in self.retrain],
                            "quick": quick},
               "horizons": {}}
        rules = self.cfg["pools"]["rules"]
        sizes = self.cfg["pools"]["sizes"]
        if quick:
            rules = [r for r in rules if r in ("ADV", "KOSTO", "INST2Y")] or rules
            sizes = [s for s in sizes if s in (150, 300)] or sizes[:2]
        for h in horizons:
            progress(f"=== {self.cfg['horizons'][h]['label']} ===")
            pt = self.pool_study(h, rules, sizes, progress)
            best = pt.iloc[0]
            rule, size = best["rule"], int(best["size"])
            progress(f"{h} 選定母池 {rule}{size}")
            vt = self.variant_study(h, rule, size, progress)
            kv = vt[~vt["is_baseline"]].sort_values("score", ascending=False)
            if self.cfg["research"].get("lock_model", False):
                # v1.3：模型由多窗口研究決定並寫在設定檔（見 docs）；每次研究只重估權重、重挑母池，模型表僅供比較
                v = self.model_variant(h, exposure=(self.cfg["horizons"][h].get("model") or {}).get("exposure", "none"))
            else:
                v = next(x for x in self.variants(h) if x.name == kv.iloc[0]["variant"])
            out["horizons"][h] = self.horizon_block(h, rule, size, v, pt, vt)
            ev = out["horizons"][h]["eval"]
            progress(f"{h} 完成：開發期CAGR {ev.get('dev_cagr', np.nan):.1%}，盲測CAGR {ev.get('hold_cagr', np.nan):.1%} "
                     f"({time.time()-t0:.0f}s)")
        out["market"] = self.market_frame()
        out["elapsed"] = time.time() - t0
        return out

    def market_frame(self) -> pd.DataFrame:
        md = self.B.a["MKT_DAILY"]
        cols = ["A1", "A2", "A3", "B1", "B2", "B3", "confidence", "index", "drawdown", "turnover_ratio", "breadth"]
        mk = pd.DataFrame(md, index=self.B.days, columns=cols)
        mk["stage"] = [STAGE_NAMES[int(c)] for c in self.B.a["MKT_DAILY_CODE"]]
        return mk

    def horizon_block(self, h, rule, size, v: Variant, pt=None, vt=None, ablation=True) -> dict:
        """Everything the report needs for one horizon / pool / model."""
        if pt is None:
            row = self.evaluate(rule, size, h, v)
            row["score"] = np.nan
            pt = pd.DataFrame([row])
        if vt is None:
            rows = [dict(self.evaluate(rule, size, h, v), is_baseline=False)]
            rows += [dict(self.evaluate(rule, size, h, bv), is_baseline=True) for bv in self.baselines()]
            vt = pd.DataFrame(rows)
            vt["score"] = selection_score(vt.rename(columns=lambda c: c.replace("dev_", "")),
                                          self.cfg["research"]["selection_weights"])
        ev = self.evaluate(rule, size, h, v, keep=True)
        res = ev.pop("_res")
        hist = ev.pop("_hist")
        ev.pop("_W")
        base = {}
        for bv in self.baselines():
            be = self.evaluate(rule, size, h, bv, keep=True)
            base[bv.name] = be.pop("_res")
        ew = self.pool_ew(rule, size, h)
        daily = res.nav.pct_change().dropna()
        trials = pd.concat([pt, vt], ignore_index=True)
        sr_std = float(np.nanstd(trials["oos_sharpe"] / np.sqrt(245))) if "oos_sharpe" in trials else np.nan
        robust = {"psr": probabilistic_sharpe(daily), "dsr": deflated_sharpe(daily, len(trials), sr_std),
                  "n_trials": int(len(trials)), **block_bootstrap_excess(res.nav, res.bench)}
        own = vt[vt["variant"] == v.name]["score"]
        data_wins = bool(len(own) and vt[vt["mode"] == "data"]["score"].max() > own.max())
        ab = self.ablation(h, rule, size, v) if ablation else pd.DataFrame(
            columns=["case", "dev_cagr", "dev_sharpe", "dev_mdd", "hold_cagr", "hold_mdd", "turnover", "貢獻(開發期CAGR)"])
        mix = self.mix_table(res, ew)
        val = self.validation(h, rule, size, v, res, ev, robust)
        lt = self.latest(h, rule, size, v, res)
        fit = self.final_fit(rule, size, h, v)
        ft = self.factor_table(h, rule, size, lt["weights"])
        return {
            "label": self.cfg["horizons"][h]["label"], "days": self.H(h),
            "pool_table": pt, "chosen_pool": {"rule": rule, "size": size}, "variant_table": vt,
            "chosen": {"variant": v.name, "mode": v.mode, "lam": v.lam, "exposure": v.exposure, "gates": v.gates,
                       "k": v.k, "tmin": v.tmin, "kK": v.kK, "lamK": v.lamK, "weighting": v.weighting, "anchor": v.anchor,
                       "growth_blend": v.growth_blend, "hyp_blend": v.hyp_blend, "hypothesis": v.hypothesis,
                       "label_h": v.label_h},
            "data_beats_kostolany": data_wins, "eval": ev, "fit_history": hist, "factor_table": ft,
            "family_final": fit.family, "robust": robust, "ablation": ab, "stage_table": self.stage_table(res),
            "yearly": yearly_table(res.nav, res.bench), "latest": lt,
            "nav": pd.DataFrame({"策略": res.nav, "0050": res.bench, "母池等權": ew.nav.reindex(res.nav.index),
                                 **{k.split()[0]: r.nav.reindex(res.nav.index) for k, r in base.items()}}),
            "tranches": res.tranches, "turnover": res.turnover, "cost_drag": res.cost_drag, "mix": mix,
            "vs_pool_ew": dict(mix.attrs.get("vs_pool_ew", {})), "validation": val,
        }

    # ------------------------------------------------------------------ v1.3 validation (acceptance gates)
    def folds_table(self, res, months=6) -> pd.DataFrame:
        nav, bench = res.nav, res.bench
        hd = self.B.sig_dates[min(self.holdout, self.S - 1)]
        out, a = [], nav.index[0]
        while a < nav.index[-1]:
            b = a + pd.DateOffset(months=months)
            s = nav[(nav.index >= a) & (nav.index < b)]
            if len(s) > 20:
                bb = bench.reindex(s.index)
                out.append({"區塊": f"{s.index[0].date()}～{s.index[-1].date()}", "期間": "盲測" if s.index[0] >= hd else "開發",
                            "策略": s.iloc[-1] / s.iloc[0] - 1, "0050": bb.iloc[-1] / bb.iloc[0] - 1})
            a = b
        f = pd.DataFrame(out)
        if len(f):
            f["超額"] = f["策略"] - f["0050"]
        return f

    def presample(self, h, rule, size, v: Variant) -> dict:
        """Weights estimated ONLY on signals from the first OOS date onward, applied backwards to the earlier
        years (2019-2020 on the default config). No pool, model or parameter choice ever looked at this period."""
        if self.first_oos < 20:
            return {}
        r = self.cfg["research"]
        sp = self.B.sig_pos
        hz = v.label_h or h
        rows = np.where((np.arange(self.S) >= self.first_oos) & (sp + self.H(hz) + 1 <= sp[-1]))[0]
        fit = fit_weights(self.ic(rule, size, hz), rows, self.B.meta, hz, self.cfg["horizons"][hz]["family_prior"], v.lam, v.mode,
                          r["n0_shrink"], r["max_factor_weight"], r["signal_step"], self.H(hz), self.view(rule, size),
                          self.B.a["FWD"][HKEY[hz]], exclude=self._excl(v), **self._sp(v))
        W = [fit.weights] * self.S
        res = run(self.B, self.view(rule, size), W, self.spec(h, v), start_i=0)
        end = self.B.sig_dates[self.first_oos]
        nav = res.nav[res.nav.index < end]
        if len(nav) < 60:
            return {}
        m = perf(nav / nav.iloc[0], res.bench.reindex(nav.index))
        return {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()), **m,
                "nav": pd.DataFrame({"策略": nav / nav.iloc[0], "0050": res.bench.reindex(nav.index) / res.bench.reindex(nav.index).iloc[0]})}

    def validation(self, h, rule, size, v: Variant, res, ev, robust) -> dict:
        acc = self.cfg["research"].get("acceptance") or {}
        folds = self.folds_table(res, acc.get("fold_months", 6))
        st = acc.get("cost_stress", 0.005)
        sp = self.spec(h, v)
        sp.buy_cost += st / 2
        sp.sell_cost += st / 2
        W, _ = self.schedule(rule, size, h, v)
        rs = run(self.B, self.view(rule, size), W, sp, start_i=self.first_oos)
        stress = perf(rs.nav, rs.bench)
        pre = self.presample(h, rule, size, v) if self.cfg["research"].get("presample_test", True) else {}
        g = []

        def gate(name, value, thr, ok, fmt="pct"):
            g.append({"閘門": name, "數值": value, "門檻": thr, "通過": bool(ok) if value is not None and np.isfinite(value) else False, "fmt": fmt})
        ex = ev.get("oos_excess_cagr", np.nan)
        gate("全樣本外年化超額 ≥ 門檻", ex, acc.get("excess_cagr_min", 0.05), ex >= acc.get("excess_cagr_min", 0.05))
        ir = ev.get("oos_info_ratio", np.nan)
        gate("資訊比率 ≥ 門檻", ir, acc.get("ir_min", 0.5), ir >= acc.get("ir_min", 0.5), "num")
        mdd_gap = ev.get("oos_mdd", np.nan) - min(res.bench / res.bench.cummax() - 1)
        gate("最大回撤比 0050 多不超過", mdd_gap, -acc.get("max_extra_mdd", 0.10), mdd_gap >= -acc.get("max_extra_mdd", 0.10))
        fw = float((folds["超額"] > 0).mean()) if len(folds) else np.nan
        gate("半年區塊勝率 ≥ 門檻", fw, acc.get("min_fold_win", 0.5), fw >= acc.get("min_fold_win", 0.5))
        se = stress.get("excess_cagr", np.nan)
        gate(f"成本加壓（來回 +{st:.1%}）後超額 > 0", se, 0.0, se > 0)
        pp = robust.get("p_positive", np.nan)
        gate("自助法 超額>0 機率 ≥ 90%", pp, 0.9, pp >= 0.9)
        he = ev.get("hold_excess_cagr", np.nan)
        gate("盲測期超額 > 0", he, 0.0, he > 0)
        if pre:
            gate(f"未用期間 {pre['start'][:4]}-{pre['end'][:4]} 超額 > 0", pre.get("excess_cagr"), 0.0, pre.get("excess_cagr", -1) > 0)
        gates = pd.DataFrame(g)
        return {"gates": gates, "passed": int(gates["通過"].sum()), "total": len(gates), "folds": folds,
                "stress_excess": se, "presample": pre}

    def mix_table(self, res, ew=None) -> pd.DataFrame:
        """0050 core + model satellite (daily constant mix), and the model vs its own pool equal-weighted."""
        rm = res.nav.pct_change().fillna(0)
        rb = res.bench.reindex(res.nav.index).pct_change().fillna(0)
        hd = self.B.sig_dates[min(self.holdout, self.S - 1)]
        rows = []
        cands = [("本模型 100%", 0.0)] + [(f"0050 {x:.0%} ＋ 本模型 {1 - x:.0%}", float(x))
                                          for x in self.cfg["research"].get("core_mix", [0.5])] + [("0050 100%", 1.0)]
        for label, x in cands:
            nav = (1 + (1 - x) * rm + x * rb).cumprod()
            seg = self.segments(type("R", (), {"nav": nav, "bench": res.bench})())
            rows.append({"組合": label, **{f"{k}_{m}": seg[k].get(m) for k in ("dev", "holdout", "oos")
                                          for m in ("cagr", "excess_cagr", "sharpe", "mdd")}})
        t = pd.DataFrame(rows)
        if ew is not None:
            e = ew.nav.reindex(res.nav.index)
            seg_m = self.segments(res)
            seg_e = self.segments(type("R", (), {"nav": e / e.iloc[0], "bench": res.bench})())
            t.attrs["vs_pool_ew"] = {k: (seg_m[k].get("cagr", np.nan) - seg_e[k].get("cagr", np.nan)) for k in ("dev", "holdout", "oos")}
        return t

    def single(self, h, rule, size, v: Variant, ablation=True) -> dict:
        """Report-ready result for one user-chosen configuration (Streamlit quick test)."""
        t0 = time.time()
        out = {"meta": {k: self.B.meta[k] for k in ("version", "built", "synthetic", "data_quality")},
               "protocol": {"first_oos": str(self.B.sig_dates[min(self.first_oos, self.S - 1)].date()),
                            "holdout_start": str(self.B.sig_dates[min(self.holdout, self.S - 1)].date()),
                            "last_date": str(self.B.sig_dates[-1].date()), "signals": self.S,
                            "retrain_points": [str(self.B.sig_dates[i].date()) for i in self.retrain], "quick": False},
               "horizons": {h: self.horizon_block(h, rule, size, v, ablation=ablation)}, "market": self.market_frame()}
        out["elapsed"] = time.time() - t0
        return out
