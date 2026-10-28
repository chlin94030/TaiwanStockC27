import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto.backtest import Spec, run
from kosto.bundle import Bundle, build_bundle, forward_returns
from kosto.config import load_config
from kosto.factors import compute
from kosto.optimize import PoolView, avg_rank_cols
from kosto.panel import build_panel
from kosto.sources.synthetic import write_demo
from kosto.store import Store


# ------------------------------------------------------------------ total returns
def test_panel_total_return_on_exdividend_and_split(tmp_path):
    st = Store(tmp_path)
    d = pd.to_datetime(["2024-06-11", "2024-06-12", "2024-06-13", "2024-06-14"])
    price = pd.DataFrame([
        # ex-dividend: 06-13 shows 'X' with no change; official reference 96 (dividend 4)
        [d[0], "1111", "TWSE", 99, 99, 99, 99, 1e6, 1e8, 0.0, ""],
        [d[1], "1111", "TWSE", 100, 100, 100, 100, 1e6, 1e8, 1.0, ""],
        [d[2], "1111", "TWSE", 96.5, 97, 96, 97, 1e6, 1e8, np.nan, "X"],
        [d[3], "1111", "TWSE", 97, 98, 97, 98, 1e6, 1e8, 1.0, ""],
        # 1:4 split: prev close 188.64, new close 47.57 with change +0.41 vs adjusted reference 47.16
        [d[0], "2222", "TWSE", 188, 189, 187, 188.64, 1e6, 1e8, 0.0, ""],
        [d[1], "2222", "TWSE", 188, 189, 187, 188.64, 1e6, 1e8, 0.0, ""],
        [d[2], "2222", "TWSE", 47.5, 47.8, 47.3, 47.57, 1e6, 1e8, 0.41, ""],
        [d[3], "2222", "TWSE", 47.57, 47.6, 47.5, 47.57, 1e6, 1e8, 0.0, ""],
        # 'X' printed on a non-dividend day (no ex-rights record, no change) → previous close is the reference
        [d[0], "3333", "TWSE", 50, 50, 50, 50, 1e6, 1e8, 0.0, ""],
        [d[1], "3333", "TWSE", 51, 52, 50, 52, 1e6, 1e8, np.nan, "X"],
        [d[2], "3333", "TWSE", 52, 53, 52, 53, 1e6, 1e8, 1.0, ""],
        [d[3], "3333", "TWSE", 53, 53, 53, 53, 1e6, 1e8, 0.0, ""],
    ], columns=["date", "ticker", "board", "open", "high", "low", "close", "volume", "amount", "change", "event"])
    st.append("price", price)
    st.append("exright", pd.DataFrame([[d[2], "1111", "TWSE", 100.0, 96.0]], columns=["date", "ticker", "board", "before", "ref"]))
    P = build_panel(st)
    r = P["ret"]
    assert r.loc[d[1], "1111"] == pytest.approx(100 / 99 - 1, rel=1e-5)
    assert r.loc[d[2], "1111"] == pytest.approx(97 / 96 - 1, rel=1e-5)          # dividend reinvested
    assert r.loc[d[3], "1111"] == pytest.approx(98 / 97 - 1, rel=1e-5)
    assert r.loc[d[2], "2222"] == pytest.approx(47.57 / 47.16 - 1, rel=1e-5)    # split-safe
    assert r.loc[d[1], "3333"] == pytest.approx(52 / 50 - 1, rel=1e-5)          # v1.1: not zeroed
    assert P.quality["x_days_using_prev_close"] == 1
    assert P.quality["suspicious_returns"] == 0
    assert P["oc"].loc[d[2], "1111"] == pytest.approx(97 / 96.5 - 1, rel=1e-5)


# ------------------------------------------------------------------ small synthetic world shared by tests
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    st = Store(root)
    write_demo(st, 160, 3, start="2017-01-02", end="2023-12-29")
    cfg = load_config(overrides={"data": {"dir": str(root)}, "research": {"start": "2019-01-01", "holdout_months": 6},
                                 "pools": {"sizes": [60, 100], "rules": ["ADV", "KOSTO"]}})
    P = build_panel(st)
    B = build_bundle(P, cfg, log=lambda *a: None)
    return SimpleNamespace(st=st, cfg=cfg, P=P, B=B)


def test_no_lookahead_in_factors(world):
    """Scramble everything after date T; every factor value on or before T must be unchanged."""
    P = world.P
    T = int(len(P.dates) * 0.6)
    idx = np.arange(T - 200, T + 1, 25)
    base, _ = compute(P, sample_idx=idx)
    P2 = copy.copy(P)
    P2.m = {k: v.copy() for k, v in P.m.items()}
    rng = np.random.default_rng(0)
    for k, v in P2.m.items():
        if v.dtypes.iloc[0] == bool:
            continue
        a = v.to_numpy(dtype=float).copy()
        a[T + 1:] = a[T + 1:] * rng.uniform(0.5, 1.5, a[T + 1:].shape)
        P2.m[k] = pd.DataFrame(a.astype(v.dtypes.iloc[0]), index=v.index, columns=v.columns)
    P2.bench = P.bench.copy(); P2.bench.iloc[T + 1:] = 0.05
    P2.taiex = P.taiex.copy(); P2.taiex.iloc[T + 1:] *= 3
    rm = P.rev_m.copy()
    from kosto.sources.official import revenue_available_date
    late = [ym for ym in rm.index if revenue_available_date(ym) > P.dates[T]]
    rm.loc[late] = rm.loc[late] * 7
    P2.rev_m = rm
    after, _ = compute(P2, sample_idx=idx)
    for f in base:
        np.testing.assert_allclose(base[f], after[f], rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=f)


def test_forward_returns_open_to_open():
    R = np.array([[0, 0], [0.01, 0.02], [0.02, -0.01], [0.03, 0.0], [-0.01, 0.01], [0.0, 0.0]], np.float32)
    OC = np.array([[0, 0], [0.005, 0.01], [0.01, 0.0], [0.01, 0.0], [0.0, 0.005], [0.002, 0.0]], np.float32)
    f = forward_returns(R, OC, np.array([0]), H=3)
    # open(1)->close(1): 1.005 ; R2,R3 ; close(3)->open(4): (1+R4)/(1+OC4)
    exp = 1.005 * 1.02 * 1.03 * (0.99 / 1.0) - 1
    assert f[0, 0] == pytest.approx(exp, rel=1e-5)


def test_avg_rank_ties():
    r = avg_rank_cols(np.array([[3.0], [1.0], [1.0], [np.inf]]))
    assert r[:, 0].tolist() == [2.0, 0.5, 0.5, 3.0]


def _toy_bundle(D=31, step=5):
    rng = np.random.default_rng(1)
    R = rng.normal(0.001, 0.01, (D, 2)).astype(np.float32)
    OC = (R * 0.4).astype(np.float32)
    sig = np.arange(0, D - 1, step)
    S = len(sig)
    B = Bundle(meta={"sig_dates": [str(d.date()) for d in pd.bdate_range("2024-01-01", periods=D)[sig]],
                     "days": [str(d.date()) for d in pd.bdate_range("2024-01-01", periods=D)],
                     "sig_pos": sig.tolist(), "tickers": ["A", "B"]},
               a={"R": R, "OC": OC, "OK": np.ones((D, 2), bool), "BENCH": np.zeros(D, np.float32),
                  "CAP": np.full((S, 2), 1e12, np.float32), "STAGE": np.zeros((S, 2), np.int8),
                  "WSUP": np.ones((S, 2), bool), "DTOP": np.zeros((S, 2), bool), "MKT": np.zeros((S, 7), np.float32)})
    view = PoolView([np.array([0, 1])] * S, [np.array([[0.5], [-0.5]], np.float32)] * S)
    return B, view, [np.array([1.0], np.float32)] * S


def test_backtest_matches_buy_and_hold_when_always_same_stock():
    B, view, W = _toy_bundle()
    spec = Spec(H=5, step=5, top_k=1, buy_cost=0.0, sell_cost=0.0)
    res = run(B, view, W, spec)
    R, OC = B.a["R"].astype(float), B.a["OC"].astype(float)
    expect = (1 + OC[1, 0]) * np.prod(1 + R[2:, 0])
    assert res.nav.iloc[-1] == pytest.approx(expect, rel=1e-6)
    # with costs: only the very first purchase is charged (later tranches keep the same stock)
    spec_c = Spec(H=5, step=5, top_k=1, buy_cost=0.002, sell_cost=0.005)
    res_c = run(B, view, W, spec_c)
    assert res_c.nav.iloc[-1] == pytest.approx(expect * (1 - 0.002), rel=1e-6)


def test_backtest_overlapping_slots_and_limit_up_no_fill():
    B, view, W = _toy_bundle()
    B.a["OK"][1, 0] = False                      # first entry day: stock A opens limit-up → stays cash
    spec = Spec(H=10, step=5, top_k=1, buy_cost=0.0, sell_cost=0.0)
    res = run(B, view, W, spec)
    t0 = res.tranches.iloc[0]
    assert t0["n_filled"] == 0 and t0["ret"] == pytest.approx(0.0)
    assert res.tranches["slot"].tolist()[:4] == [0, 1, 0, 1]


def test_optimizer_finds_planted_signals_and_respects_prior(world):
    from kosto.research import Research, Variant
    R = Research(world.B, world.cfg)
    fit = R.final_fit("ADV", 100, "mid", Variant("t", "opt", 0.4))
    w = dict(zip(world.B.factors, fit.weights))
    assert w["k_hands"] > 0 and w["c_foreign20"] > 0                     # planted informed flows
    assert fit.family.get("K", 0) >= 0.4 - 1e-9                          # Kostolany floor
    signs = np.sign(fit.weights) * np.array(world.B.meta["directions"])
    assert (signs >= 0).all()                                            # never flips an economic prior
    data = R.final_fit("ADV", 100, "mid", Variant("d", "data", 0.0))
    assert all(abs(v) < 1e-12 for f, v in zip(world.B.factors, data.weights) if f.startswith("k_"))


def test_sparse_model_selects_few_prior_consistent_factors(world):
    from kosto.research import Research, Variant
    R = Research(world.B, world.cfg)
    fit = R.final_fit("ADV", 100, "mid", Variant("s", "sparse", 0.0, k=3, tmin=0.5))
    w = fit.weights
    assert 0 < (np.abs(w) > 0).sum() <= 3
    assert (np.sign(w) * np.array(world.B.meta["directions"]) >= 0).all()
    assert abs(np.abs(w).sum() - 1) < 1e-5
    fk = R.final_fit("ADV", 100, "mid", Variant("s", "sparse", 0.0, k=3, tmin=0.5, kK=2, lamK=0.3))
    kw = sum(abs(x) for f, x in zip(world.B.factors, fk.weights) if f.startswith("k_"))
    assert kw >= 0.3 - 1e-6                                              # Kostolany boost keeps ≥30% in K


def test_walk_forward_uses_only_matured_labels(world):
    from kosto.research import Research
    R = Research(world.B, world.cfg)
    sp = world.B.sig_pos
    for i in R.retrain:
        rows = R.train_rows(i, "long")
        assert (sp[rows] + 60 + 1 <= sp[i]).all()


def test_research_run_all_and_report(world, tmp_path):
    from kosto.report import build_zip
    from kosto.research import Research
    R = Research(world.B, world.cfg)
    res = R.run_all(horizons=["mid"], progress=lambda *a: None)
    hb = res["horizons"]["mid"]
    assert len(hb["pool_table"]) == 4 and len(hb["latest"]["picks"]) <= world.cfg["research"]["top_k"]
    ab = hb["ablation"]
    assert {"key", "目前", "貢獻(開發期CAGR)", "貢獻(盲測CAGR)"} <= set(ab.columns)
    assert ab.loc[ab["key"] == "stage_gate", "目前"].iloc[0] == "關閉"            # v1.1 default: stage gate off
    assert len(hb["mix"]) == len(world.cfg["research"]["core_mix"]) + 2
    assert set(hb["vs_pool_ew"]) == {"dev", "holdout", "oos"}
    val = hb["validation"]
    assert val["total"] >= 7 and {"閘門", "數值", "門檻", "通過"} <= set(val["gates"].columns)
    assert len(val["folds"]) >= 2
    z = build_zip(res, world.cfg)
    import io
    import zipfile
    names = zipfile.ZipFile(io.BytesIO(z)).namelist()
    assert {"report.html", "report.xlsx", "config_used.yaml", "picks_history_mid.csv"} <= set(names)


def test_pool_smoothing_and_sqrt_cap_weighting():
    from kosto.research import smooth_pool_scores
    t = pd.DataFrame({"rule": ["A"] * 3 + ["B"] * 2, "size": [100, 200, 300, 100, 200],
                      "raw_score": [1.0, 0.0, 0.0, 0.6, 0.6]})
    sm = smooth_pool_scores(t)
    assert sm[0] == pytest.approx(0.5) and sm[3] == pytest.approx(0.6)    # lone lucky cell is discounted
    B, view, W = _toy_bundle()
    B.a["MCAP"] = np.ones((len(B.sig_pos), B.a["R"].shape[1]), np.float32)
    eq = run(B, view, W, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0))
    sq = run(B, view, W, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0, weighting="sqrt_cap"))
    assert sq.nav.iloc[-1] == pytest.approx(eq.nav.iloc[-1], rel=1e-9)  # one stock → identical to equal weight


def test_benchmark_residual_and_overheat_gate():
    B, view, W = _toy_bundle()
    B.a["BENCH"] = np.full(B.a["R"].shape[0], 0.001, np.float32)
    cash = run(B, view, W, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0, sleeve=0.5))   # half the book idle
    ben = run(B, view, W, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0, sleeve=0.5, residual="bench", etf_cost=0.0))
    assert ben.nav.iloc[-1] > cash.nav.iloc[-1]                                               # idle half earns 0050
    B.a["RSI14"] = np.full((len(B.sig_pos), B.a["R"].shape[1]), 90, np.float32)
    B.a["BIAS20"] = np.zeros_like(B.a["RSI14"])
    hot = run(B, view, W, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0, max_rsi=84))
    assert hot.tranches["n_picks"].sum() == 0                                                  # all blocked as overheated


def test_short_model_can_use_mid_labels(world):
    """v1.5 快調慢α: a short model with label_h='mid' gets exactly the weights a mid model would get."""
    from kosto.research import Research, Variant
    R = Research(world.B, world.cfg)
    a = R.final_fit("ADV", 100, "short", Variant("s", "sparse", 0.0, k=3, tmin=0.5, label_h="mid")).weights
    b = R.final_fit("ADV", 100, "mid", Variant("m", "sparse", 0.0, k=3, tmin=0.5)).weights
    assert np.allclose(a, b)
