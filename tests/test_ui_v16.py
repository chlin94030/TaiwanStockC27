"""v1.6 看盤介面：產業鏈標籤、集保大戶、技術指標、交易計畫、籌碼短評、即時報價解析、看盤資料包、全排名。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto import chain, plan, ta
from kosto.bundle import build_bundle
from kosto.config import load_config
from kosto.panel import build_panel
from kosto.quotes import channel, market_phase, parse_mis
from kosto.sources.official import parse_tdcc_holders
from kosto.sources.synthetic import write_demo
from kosto.store import Store


# ------------------------------------------------------------------ 產業鏈
def test_chain_labels():
    assert chain.label("2330", "半導體業")["label"] == "電子中游-半導體｜IC/晶圓製造"
    assert chain.label("2454", "電子工業")["group"] == "電子上游-半導體"
    assert chain.label("2603", "航運業")["group"] == "傳產-交通運輸及航運"
    assert chain.label("2881", "金融保險")["group"].startswith("金融")
    lb = chain.label("2308", "電子零組件業")
    assert lb["group"] == "電子上游-電腦週邊" and "AI" in lb["themes"]
    fb = chain.label("9999", "半導體業")                     # 平台沒有 → 用官方產業別
    assert fb["group"] == "電子-半導體業" and fb["themes"] == []
    assert chain.label("9998", "")["label"] == "其他"


# ------------------------------------------------------------------ 集保戶股權分散表
TDCC = """﻿資料日期,證券代號,持股分級,人數,股數,占集保庫存數比例%
20261022,2330  ,1,800000,100000000,0.40
20261022,2330  ,2,500000,900000000,3.50
20261022,2330  ,8,10000,500000000,1.10
20261022,2330  ,11,2000,600000000,2.00
20261022,2330  ,12,900,450000000,1.50
20261022,2330  ,15,1500,22000000000,85.00
20261022,2330  ,17,2000000,25000000000,100.00
20261022,00988B,15,10,100,100.00
"""


def test_parse_tdcc_holders():
    f = parse_tdcc_holders(TDCC)
    assert list(f["ticker"]) == ["2330"]                      # ETF/債券代號被過濾
    r = f.iloc[0]
    assert r["date"] == pd.Timestamp("2026-10-22")
    assert r["big400"] == pytest.approx(86.5) and r["big1000"] == pytest.approx(85.0)
    assert r["retail50"] == pytest.approx(5.0) and r["holders"] == 2000000


# ------------------------------------------------------------------ 技術指標
def _bars(n=300, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": rng.uniform(1e3, 5e3, n),
                         "foreign": rng.normal(0, 100, n), "trust": rng.normal(0, 30, n), "dealer": rng.normal(0, 20, n),
                         "mbal": np.linspace(10000, 12000, n), "sbal": np.full(n, 500.0), "mlimit": np.full(n, 1e6)}, index=idx)


def test_kd_matches_definition():
    df = _bars(30)
    k, d = ta.kd(df)
    lo, hi = df["low"].iloc[:9].min(), df["high"].iloc[:9].max()
    kp = dp = 50.0
    for i in range(9):
        lo_i, hi_i = df["low"].iloc[:i + 1].min(), df["high"].iloc[:i + 1].max()
        rsv = (df["close"].iloc[i] - lo_i) / (hi_i - lo_i) * 100
        kp = kp * 2 / 3 + rsv / 3
        dp = dp * 2 / 3 + kp / 3
    assert k.iloc[8] == pytest.approx(kp) and d.iloc[8] == pytest.approx(dp)
    assert ((k >= 0) & (k <= 100)).all()


def test_indicators_and_resample():
    df = _bars(300)
    ind = ta.indicators(df)
    assert ind["ma240"].notna().sum() == 300 - 239
    assert ind["ma20"].iloc[-1] == pytest.approx(df["close"].tail(20).mean())
    assert (ind["bb_up"].dropna() >= ind["bb_dn"].dropna()).all()
    w = ta.resample(df, "W")
    assert w["high"].iloc[3] >= w["open"].iloc[3] and w.index[-1] == df.index[-1]
    assert w["volume"].sum() == pytest.approx(df["volume"].sum())


def test_tick_rounding():
    assert ta.tick(9.99) == 0.01 and ta.tick(49.9) == 0.05 and ta.tick(99) == 0.1 and ta.tick(499) == 0.5
    assert ta.tick(999) == 1 and ta.tick(1500) == 5
    assert ta.round_tick(37.77, "down") == 37.75 and ta.round_tick(2408, "down") == 2405 and ta.round_tick(118.7, "up") == 119.0


# ------------------------------------------------------------------ 交易計畫與盤中狀態
def test_trade_plan_and_status():
    ind = ta.indicators(_bars(300))
    p = plan.trade_plan(ind, "short")
    c = ind["close"].iloc[-1]
    assert p.buy_low <= p.buy_high <= p.chase
    assert p.alert < c and p.alert >= c * 0.93 - ta.tick(c)
    assert plan.trade_plan(ind, "long").alert <= p.alert              # 長線警戒較寬
    assert plan.live_status(p.chase * 1.02, p)[0] == "wait"
    assert plan.live_status(p.buy_low, p)[0] == "dip"
    assert plan.live_status(p.alert * 0.99, p)[0] == "warn"
    assert plan.live_status(p.buy_high, p, limit_up=p.buy_high)[0] == "lock"
    assert plan.live_status(c, p, role="keep")[0] == "hold"


def test_chip_summary_sync_and_kostolany():
    df = _bars(120)
    df[["foreign", "trust", "dealer"]] = 0.0
    df.iloc[-5:, df.columns.get_loc("foreign")] = 500
    df.iloc[-5:, df.columns.get_loc("trust")] = 200
    df.iloc[-5:, df.columns.get_loc("dealer")] = 50
    df["close"] = np.linspace(100, 120, len(df))                      # 股價漲
    df["mbal"] = np.linspace(20000, 15000, len(df))                   # 融資減 → 籌碼流向固執者
    cs = plan.chip_summary(df)
    assert any("同步買超" in x for x in cs["lines"])
    assert any("固執者" in x for x in cs["lines"])
    assert cs["tone"] == "偏多" and cs["table"].iloc[0]["連續"] == 5
    df["big400"] = np.r_[np.full(len(df) - 5, 60.0), np.full(5, 61.0)]
    assert plan.chip_summary(df)["big400_ch"] == pytest.approx(1.0)


def test_blend_nav_and_weights():
    idx = pd.bdate_range("2024-01-01", periods=60)
    a = pd.Series(np.linspace(1, 2, 60), index=idx)
    b = pd.Series(np.ones(60), index=idx)
    nav = plan.blend_nav({"mid": a, "0050": b}, {"mid": 0.5, "0050": 0.5})
    assert 1.0 < nav.iloc[-1] < 2.0 and nav.iloc[0] == pytest.approx(1.0)
    w = plan.stock_weights(10, "long")
    assert w.sum() == pytest.approx(1) and w[0] == pytest.approx(3 * w[-1])
    assert plan.stock_weights(10, "mid")[0] == pytest.approx(0.1)
    assert plan.lots(100000, 37.75) == (2, 649)


# ------------------------------------------------------------------ 即時報價
MIS = {"msgArray": [
    {"c": "2330", "n": "台積電", "z": "2385.0000", "y": "2420.0000", "o": "2435.0000", "h": "2435.0000", "l": "2385.0000",
     "v": "22651", "a": "2390.0000_2395.0000_", "b": "2385.0000_2380.0000_", "u": "2660.0000", "w": "2180.0000", "t": "13:30:00",
     "d": "20261022"},
    {"c": "6488", "n": "環球晶", "z": "-", "y": "880.0000", "a": "826.0000_", "b": "825.0000_", "u": "968", "w": "792", "t": "10:01:02"}]}


def test_parse_mis_and_channels():
    q = parse_mis(MIS).set_index("ticker")
    assert q.loc["2330", "price"] == 2385 and q.loc["2330", "chg"] == pytest.approx(2385 / 2420 - 1)
    assert q.loc["6488", "price"] == pytest.approx(825.5)            # 未成交：買賣中價
    assert channel("6488", "TPEX") == "otc_6488.tw" and channel("2330", "TWSE") == "tse_2330.tw"
    assert market_phase(pd.Timestamp("2026-10-24 10:00", tz="Asia/Taipei")) == "休市"
    assert market_phase(pd.Timestamp("2026-10-26 10:00", tz="Asia/Taipei")) == "盤中"
    assert market_phase(pd.Timestamp("2026-10-26 15:00", tz="Asia/Taipei")) == "盤後"


# ------------------------------------------------------------------ 看盤資料包 + 全排名
@pytest.fixture(scope="module")
def small(tmp_path_factory):
    root = tmp_path_factory.mktemp("v16")
    st = Store(root)
    write_demo(st, 90, 5, start="2018-01-02", end="2022-12-30")
    t0 = st.read("price")["ticker"].astype(str)
    tk = sorted(t0[t0.str.match(r"^[1-9]\d{3}$")].unique())[:5]
    st.append("holders", pd.DataFrame([{"date": pd.Timestamp("2022-12-23"), "ticker": t, "big400": 55.0, "big1000": 40.0,
                                        "retail50": 20.0, "holders": 1000.0} for t in tk]))
    cfg = load_config(overrides={"data": {"dir": str(root)}, "research": {"start": "2019-06-01", "holdout_months": 6},
                                 "pools": {"sizes": [40], "rules": ["ADV"]}, "app": {"chart_days": 300}})
    P = build_panel(st)
    B = build_bundle(P, cfg, log=lambda *a: None)
    return SimpleNamespace(cfg=cfg, P=P, B=B, held=tk)


def test_bundle_chart_pack(small):
    B = small.B
    assert len(B.meta["q_days"]) == 300 and B.a["Q_C"].shape == (300, len(B.tickers))
    j = B.tickers.index(small.held[0]) if small.held[0] in B.tickers else None
    df = ta.stock_frame(B, 0)
    assert {"open", "close", "volume", "foreign", "mbal"} <= set(df.columns) and len(df) > 200
    if j is not None:
        h = ta.stock_frame(B, j)["big400"]
        assert h.iloc[-1] == pytest.approx(55.0)                    # 週資料往後補值
    assert np.isfinite(B.a["Q_TAIEX"]).any()


def test_rank_table_consistent_with_latest(small):
    from kosto.research import Research
    R = Research(small.B, small.cfg, log=lambda m: None)
    v = R.model_variant("mid")
    rt = R.rank_table("mid", "ADV", 40, v)
    lt = R.latest("mid", "ADV", 40, v)
    top = rt[rt["valid"]].sort_values("rank")["ticker"].tolist()
    assert [p["ticker"] for p in lt["picks"]][:3] == top[:3] or set(p["ticker"] for p in lt["picks"]) <= set(top[:25])
    assert rt["rank"].dropna().min() == 1 and (rt.loc[~rt["valid"], "rank"].isna()).all()
    assert all(isinstance(x, list) for x in rt["reasons"])


def test_kline_figure_builds():
    pytest.importorskip("plotly")
    from kosto.charts import kline, nav_figure
    ind = ta.indicators(_bars(300))
    p = plan.trade_plan(ind, "mid")
    fig = kline(ind, mas=(5, 20, 60, 240), panels=("volume", "kd", "macd", "inst", "margin", "holders"), plan=p, dark=False)
    names = [t.name for t in fig.data]
    assert "K線" in names and "週線(5)" in names and "外資(張)" in names
    assert "大戶≥400張(%)" not in names                               # 沒有集保資料時自動略過
    assert len(nav_figure({"a": ind["close"]}).data) == 1
