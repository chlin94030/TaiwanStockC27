import pandas as pd

from kosto.build import Builder
from kosto.config import load_config
from kosto.sources import official as O
from kosto.sources.finmind import FinMind
from kosto.store import Store
from tests.fixtures import official_samples as S


def test_builder_resumes_and_skips_holidays(tmp_path, monkeypatch):
    calls = {"twse": 0, "tpex": 0}

    def twse_price(self, d):
        calls["twse"] += 1
        if d == pd.Timestamp("2024-06-12"):
            return O.parse_twse_price(S.TWSE_HOLIDAY, d)          # holiday → empty
        return O.parse_twse_price(S.TWSE_MI_INDEX_20240613, d)

    def tpex_price(self, d):
        calls["tpex"] += 1
        return O.parse_tpex_price(S.TPEX_PRICE_2024, d)

    monkeypatch.setattr(O.Official, "twse_price", twse_price)
    monkeypatch.setattr(O.Official, "tpex_price", tpex_price)
    cfg = load_config(overrides={"data": {"dir": str(tmp_path), "start": "2024-06-10", "pause_sec": 0}})
    b = Builder(cfg, "price", end="2024-06-28")
    b.recent_cut = pd.Timestamp("2024-06-26")
    rep = b.run()
    assert rep["price"]["fetched"] == 2 * 15
    f = b.store.read("price")
    assert set(f["ticker"]) >= {"2330", "6488", "TAIEX"}
    assert "2024-06-12" in b.store.done_set("price:TWSE")          # holiday remembered
    n1 = dict(calls)
    b2 = Builder(cfg, "price", end="2024-06-28")
    b2.recent_cut = pd.Timestamp("2024-06-26")
    b2.run()
    # second run only retries nothing older than the recent window
    assert calls["twse"] - n1["twse"] == 0


def test_finmind_transforms(monkeypatch):
    fm = FinMind("token", log=lambda *a: None)
    inst = pd.DataFrame([
        {"date": "2024-01-02", "stock_id": "2330", "buy": 19034488, "name": "Foreign_Investor", "sell": 11202763},
        {"date": "2024-01-02", "stock_id": "2330", "buy": 0, "name": "Foreign_Dealer_Self", "sell": 0},
        {"date": "2024-01-02", "stock_id": "2330", "buy": 869000, "name": "Investment_Trust", "sell": 109685},
        {"date": "2024-01-02", "stock_id": "2330", "buy": 80000, "name": "Dealer_self", "sell": 641052},
        {"date": "2024-01-02", "stock_id": "2330", "buy": 25585, "name": "Dealer_Hedging", "sell": 472504}])
    rev = pd.DataFrame([{"date": "2024-02-01", "stock_id": "2330", "country": "Taiwan", "revenue": 215785127000,
                         "revenue_month": 1, "revenue_year": 2024, "create_time": ""}])
    price = pd.DataFrame([{"date": "2024-06-13", "stock_id": "2330", "Trading_Volume": 59656092, "Trading_money": 54980784829,
                           "open": 923, "max": 935, "min": 911, "close": 919, "spread": 0, "Trading_turnover": 98777}])
    data = {"TaiwanStockInstitutionalInvestorsBuySell": inst, "TaiwanStockMonthRevenue": rev, "TaiwanStockPrice": price}
    monkeypatch.setattr(FinMind, "get", lambda self, ds, *a, **k: data[ds].copy())
    i = fm.institutional("2330", "2024-01-01").iloc[0]
    assert i["foreign"] == 7831725 and i["trust"] == 759315 and i["dealer"] == -1007971 and i["total"] == 7583069
    r = fm.revenue("2330", "2024-01-01").iloc[0]
    assert r["ym"] == 202401 and r["avail_date"] == pd.Timestamp("2024-02-10")
    p = fm.price("2330", "2024-06-01").iloc[0]
    assert p["close"] == 919 and p["amount"] == 54980784829


def test_auto_build_two_threads_and_progress(tmp_path, monkeypatch):
    from kosto.build import auto_build, progress
    from kosto.sources.finmind import FinMind

    def fake(parser, sample):
        return lambda self, d: parser(sample, d)
    for name, (parser, sample) in {
        "twse_price": (O.parse_twse_price, S.TWSE_MI_INDEX_20240613), "tpex_price": (O.parse_tpex_price, S.TPEX_PRICE_2024),
        "twse_inst": (O.parse_twse_inst, S.TWSE_T86_2024), "tpex_inst": (O.parse_tpex_inst, S.TPEX_INST_2024),
        "twse_margin": (O.parse_twse_margin, S.TWSE_MARGIN), "tpex_margin": (O.parse_tpex_margin, S.TPEX_MARGIN),
        "twse_valuation": (O.parse_twse_valuation, S.TWSE_BWIBBU_2024), "tpex_valuation": (O.parse_tpex_valuation, S.TPEX_PE),
        "twse_shares": (O.parse_twse_shares, S.TWSE_QFIIS)}.items():
        monkeypatch.setattr(O.Official, name, fake(parser, sample))
    monkeypatch.setattr(O.Official, "twse_exright", lambda self, s, e: O.parse_twse_exright(S.TWSE_TWT49U))
    monkeypatch.setattr(O.Official, "tpex_exright", lambda self, s, e: O.parse_tpex_exright(S.TPEX_EXRIGHT))
    monkeypatch.setattr(O.Official, "revenue", lambda self, y, m: O.parse_mops_revenue(S.MOPS_HTML, y, m)
                        .assign(avail_date=O.revenue_available_date(y * 100 + m)))
    monkeypatch.setattr(FinMind, "info", lambda self: pd.DataFrame(
        [{"ticker": "2330", "name": "台積電", "sector": "半導體業", "board": "TWSE"}]))
    cfg = load_config(overrides={"data": {"dir": str(tmp_path), "start": "2024-04-01", "end": "2024-06-28", "pause_sec": 0}})
    assert progress(cfg)["percent"] < 0.1
    p = auto_build(cfg, log=lambda *a: None)
    assert p["complete"], p
    assert not p["errors"]
    st = Store(tmp_path, "price").state["done"]
    assert len(st["price:TWSE"]) > 50 and len(st["price:TPEX"]) > 50   # both threads' progress survived the merge


def test_state_merge_and_repo_detection(tmp_path, monkeypatch):
    from kosto import setup_page
    from kosto.store import _merge
    assert _merge({"done": {"a": ["1", "2"]}, "x": 1}, {"done": {"a": ["3"], "b": ["9"]}}) == \
        {"done": {"a": ["1", "2", "3"], "b": ["9"]}, "x": 1}
    g = tmp_path / ".git"
    g.mkdir()
    (g / "config").write_text('[remote "origin"]\n\turl = https://github.com/yangty/kosto-lab.git\n')
    monkeypatch.setattr(setup_page, "ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    assert setup_page.detect_repo("", "") == ("yangty/kosto-lab", "自動偵測")
    assert setup_page.detect_repo("me/x", "") == ("me/x", "secrets")
