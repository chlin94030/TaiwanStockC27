import numpy as np
import pandas as pd

from kosto.sources import official as O
from tests.fixtures import official_samples as S

D = pd.Timestamp("2024-06-13")


def row(f, t):
    return f[f["ticker"] == t].iloc[0]


def test_twse_price_signs_events_and_filters():
    f = O.parse_twse_price(S.TWSE_MI_INDEX_20240613, D)
    assert "00634R" not in set(f["ticker"])                      # ETFs other than 0050 are dropped
    tsmc = row(f, "2330")
    assert tsmc["event"] == "X" and np.isnan(tsmc["change"])      # ex-dividend: change unknown
    assert tsmc["volume"] == 59656092 and tsmc["amount"] == 54980784829
    assert row(f, "1101")["change"] == 0.05
    assert row(f, "2317")["change"] == -1.50
    assert np.isnan(row(f, "2303")["close"])                      # '--' = no trade
    tx = row(f, "TAIEX")
    assert tx["close"] == 22000.0 and tx["change"] == -50.0


def test_twse_split_day_uses_signed_change():
    f = O.parse_twse_price(S.TWSE_MI_INDEX_20250618, pd.Timestamp("2025-06-18"))
    r = row(f, "0050")
    assert r["change"] == 0.41 and r["close"] == 47.57


def test_twse_holiday_empty():
    assert O.parse_twse_price(S.TWSE_HOLIDAY, D).empty
    assert O.parse_twse_inst(S.TWSE_HOLIDAY, D).empty


def test_twse_exright():
    f = O.parse_twse_exright(S.TWSE_TWT49U)
    r = row(f, "2330")
    assert r["date"] == D and r["before"] == 909.0 and r["ref"] == 905.5


def test_t86_both_layouts():
    n = O.parse_twse_inst(S.TWSE_T86_2024, D)
    r = row(n, "2330")
    assert r["foreign"] == 7831725 and r["trust"] == 759315 and r["dealer"] == -1007971 and r["total"] == 7583069
    o = O.parse_twse_inst(S.TWSE_T86_2017, D)
    r = row(o, "1101")
    assert r["foreign"] == -323985 and r["trust"] == -150000 and r["dealer"] == 85000 and r["total"] == -388985


def test_twse_margin_positional():
    r = row(O.parse_twse_margin(S.TWSE_MARGIN, D), "2330")
    assert (r["mbal"], r["mlimit"], r["sbal"]) == (12844, 6483017, 208)


def test_bwibbu_both_orders_and_negative_eps():
    f = O.parse_twse_valuation(S.TWSE_BWIBBU_2024, D)
    assert row(f, "2330")["pe"] == 17.17 and row(f, "2330")["pb"] == 4.59 and row(f, "2330")["dy"] == 1.85
    assert np.isnan(row(f, "9999")["pe"])
    g = O.parse_twse_valuation(S.TWSE_BWIBBU_2017, D)
    assert row(g, "1101")["pe"] == 22.05 and row(g, "1101")["dy"] == 3.53 and row(g, "1101")["pb"] == 1.32


def test_qfiis_shares():
    r = row(O.parse_twse_shares(S.TWSE_QFIIS, D), "2330")
    assert r["shares"] == 25932070992 and r["foreign_ratio"] == 73.08


def test_tpex_price_formats():
    a = row(O.parse_tpex_price(S.TPEX_PRICE_2017, D), "5347")
    assert a["change"] == 0.80 and a["shares"] == 1638982267        # '+ 0.80' with a space
    f = O.parse_tpex_price(S.TPEX_PRICE_2024, D)
    assert "00679B" not in set(f["ticker"])
    assert row(f, "6488")["change"] == -7.0
    ex = row(f, "3171")
    assert ex["event"] == "除息" and np.isnan(ex["change"])
    assert np.isnan(row(f, "2740")["close"])
    assert O.parse_tpex_price(S.TPEX_HOLIDAY, D).empty


def test_tpex_inst_both_layouts():
    a = row(O.parse_tpex_inst(S.TPEX_INST_2017, D), "5347")
    assert (a["foreign"], a["trust"], a["dealer"], a["total"]) == (481153, 0, 109000, 590153)
    b = row(O.parse_tpex_inst(S.TPEX_INST_2024, D), "5347")
    assert (b["foreign"], b["trust"], b["dealer"], b["total"]) == (-2220406, -164000, -160617, -2545023)
    assert b["foreign"] + b["trust"] + b["dealer"] == b["total"]


def test_tpex_margin_pe_exright():
    m = row(O.parse_tpex_margin(S.TPEX_MARGIN, D), "6488")
    assert (m["mbal"], m["mlimit"], m["sbal"]) == (2208, 109028, 61)
    p = row(O.parse_tpex_valuation(S.TPEX_PE, D), "6488")
    assert p["pe"] == 11.95 and p["pb"] == 3.81 and p["dy"] == 2.76
    e = row(O.parse_tpex_exright(S.TPEX_EXRIGHT), "3171")
    assert e["date"] == pd.Timestamp("2024-07-10") and e["ref"] == 28.0


def test_mops_revenue_and_pit_date():
    f = O.parse_mops_revenue(S.MOPS_HTML, 2024, 1)
    assert row(f, "2330")["revenue"] == 215785127 * 1000
    assert len(f) == 2
    assert O.revenue_available_date(202401) == pd.Timestamp("2024-02-10")
    assert O.revenue_available_date(202412) == pd.Timestamp("2025-01-10")


def test_roc_dates():
    assert O.roc_to_date("113年06月13日") == pd.Timestamp("2024-06-13")
    assert O.roc_to_date("108/01/10") == pd.Timestamp("2019-01-10")
