import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from kosto.compare import Daily, head_to_head, holding_return, parse_external
from kosto.config import load_config


V22 = """horizon,date,pool,stage,stage_confidence,action,candidate,selected,state,net,benchmark,excess_0050,stock_exposure
20,2024-01-02,KOSTO300,A2,0.5,HOLD,x,2330.TW|6488.TWO|1101.TW,SELECTED,0.01,0.0,0.01,1.0
5,2024-01-02,KOSTO300,A2,0.5,HOLD,x,2330.TW,SELECTED,0.01,0.0,0.01,1.0
"""


def test_parse_v22_and_zip_and_long():
    a = parse_external("monthly_decisions.csv", V22.encode())
    assert a.loc[a["horizon"] == 20, "tickers"].iloc[0] == ["2330", "6488", "1101"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("V22/monthly_decisions.csv", V22)
        z.writestr("REPORT.txt", "x")
    b = parse_external("V22_TEST_REPORT.zip", buf.getvalue())
    assert len(b) == 2
    long = "date,ticker\n2024-01-02,2330\n2024-01-02,2317\n"
    c = parse_external("x.csv", long.encode())
    assert c["tickers"].iloc[0] == ["2330", "2317"] and c["horizon"].iloc[0] == 20


def _daily():
    days = pd.bdate_range("2024-01-01", periods=12)
    R = np.full((12, 2), 0.01, np.float32)
    OC = np.full((12, 2), 0.004, np.float32)
    return Daily(days, ["2330", "6488"], R, OC, np.ones((12, 2), bool), np.zeros(12, np.float32))


def test_holding_return_and_unknown_names_are_cash():
    d = _daily()
    r = holding_return(d, "2024-01-01", 3, ["2330", "9999"], 0.0, 0.0)
    one = 1.004 * 1.01 ** 3 / 1.004 - 1
    assert r["ret"] == pytest.approx(one / 2, rel=1e-5) and r["covered"] == 1


def test_head_to_head_pairs_with_latest_prior_signal():
    d = _daily()
    ext = pd.DataFrame({"date": [pd.Timestamp("2024-01-03")], "horizon": [3], "tickers": [["6488"]], "source": ["x"]})
    ours = {3: pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "picks": [["2330"]]})}
    per, summ = head_to_head(d, ext, ours, {"fee": 0, "fee_discount": 1, "tax": 0, "slippage": 0})
    assert per["our_signal"].iloc[0] == "2024-01-02" and summ["paired"].iloc[0] == 1


def test_config_validation():
    with pytest.raises(ValueError):
        load_config(overrides={"horizons": {"mid": {"days": 22}}})
    with pytest.raises(ValueError):
        load_config(overrides={"horizons": {"mid": {"family_prior": {"T": 0.5, "C": 0.5, "F": 0.5, "K": 0.0}}}})
    cfg = load_config()
    assert cfg["horizons"]["long"]["days"] == 60 and cfg["research"]["top_k"] == 10


def test_visible_workflow_copy_matches_real_workflow():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert (root / "pipeline_workflow_COPY.txt").read_text(encoding="utf-8") == \
        (root / ".github" / "workflows" / "pipeline.yml").read_text(encoding="utf-8")
