"""Time-valid cross-market context for execution, never a rewritten closed signal."""
from __future__ import annotations
from dataclasses import dataclass
from urllib.parse import quote
import numpy as np
import pandas as pd
import requests

YAHOO="https://query1.finance.yahoo.com/v8/finance/chart/"
ASSETS={
    "sox":("^SOX","費半",.04),"nasdaq":("^IXIC","Nasdaq",.03),
    "nq_fut":("NQ=F","Nasdaq期貨",.03),"nikkei":("^N225","日經",.01),
    "kospi":("^KS11","韓國綜合",.01)}
TOTAL_WEIGHT=sum(v[2] for v in ASSETS.values())

@dataclass(frozen=True)
class GlobalSnapshot:
    observed_at:str;moves:dict;ages:dict;available_weight:float;score:float;label:str;note:str
    @property
    def confidence(self):return min(1.,self.available_weight/TOTAL_WEIGHT) if TOTAL_WEIGHT else 0.

def _read_symbol(symbol,session,now):
    r=session.get(YAHOO+quote(symbol,safe=""),params={"range":"5d","interval":"1d"},
        headers={"User-Agent":"Mozilla/5.0 AlphaRadar/27"},timeout=5)
    r.raise_for_status();result=r.json()["chart"]["result"][0];meta=result.get("meta",{})
    current=float(meta.get("regularMarketPrice",np.nan));previous=float(meta.get("chartPreviousClose",meta.get("previousClose",np.nan)))
    stamp=int(meta.get("regularMarketTime",0) or 0)
    if not np.isfinite(current+previous) or previous<=0 or stamp<=0:raise ValueError("quote fields unavailable")
    return current/previous-1,max(0.,(now.timestamp()-stamp)/3600)

def fetch_global(now=None,session=None):
    now=pd.Timestamp.now(tz="Asia/Taipei") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:now=now.tz_localize("Asia/Taipei")
    session=session or requests.Session();moves={};ages={};available=weighted=0.
    for key,(symbol,_,weight) in ASSETS.items():
        try:
            move,age=_read_symbol(symbol,session,now)
            if age>96:continue
            moves[key]=float(move);ages[key]=float(age);available+=weight;weighted+=weight*np.clip(move/.04,-1,1)
        except (requests.RequestException,ValueError,KeyError,TypeError,IndexError):pass
    score=float(np.clip(weighted/available if available else 0.,-1,1)*100)
    label="偏多" if score>=25 else "偏空" if score<=-25 else "中性"
    missing=[name for key,(_,name,_) in ASSETS.items() if key not in moves]
    note=("跨市場資料完整" if not missing else "缺少"+"、".join(missing)) if available else "海外行情暫不可用，維持中性"
    return GlobalSnapshot(now.isoformat(),moves,ages,available,score,label,note)

def group_sensitivity(group):
    text=str(group)
    if "半導體" in text:return 1.
    if text.startswith("電子"):return .75
    if any(x in text for x in ("汽車","化學","塑膠","鋼鐵","航運")):return .40
    if any(x in text for x in ("食品","電信","金融","生技")):return .20
    return .35

def linked_score(base_score,group,snapshot):
    """88% closed model plus up to 12% available overseas context."""
    if snapshot is None or snapshot.available_weight<=0:return float(base_score)
    support=50+.5*snapshot.score*group_sensitivity(group);w=TOTAL_WEIGHT*snapshot.confidence
    return float((1-w)*base_score+w*support)

def display_rows(snapshot):
    return pd.DataFrame([{"市場":name,"漲跌%":None if key not in snapshot.moves else snapshot.moves[key]*100,
        "模型權重":weight,"資料狀態":"已取得" if key in snapshot.moves else "未取得"}
        for key,(_,name,weight) in ASSETS.items()])
