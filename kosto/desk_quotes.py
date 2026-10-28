"""Verified trade quotes and execution freshness. Reference quotes never become trades."""
from __future__ import annotations
from dataclasses import dataclass
import threading
import numpy as np
import pandas as pd
from .quotes import Quotes,channel

TZ='Asia/Taipei'


def taipei(now=None):
    t=pd.Timestamp.now(tz=TZ) if now is None else pd.Timestamp(now)
    return t.tz_localize(TZ) if t.tzinfo is None else t.tz_convert(TZ)


def phase(now=None,holidays=()):
    t=taipei(now);day=t.strftime('%Y-%m-%d');minute=t.hour*60+t.minute
    if t.weekday()>=5 or day in holidays:return '休市'
    if minute<510:return '盤前'
    if minute<540:return '盤前試撮'
    if minute<805:return '盤中'
    if minute<=810:return '收盤競價'
    return '盤後'


def _number(value,positive=False):
    try:v=float(str(value).split('_')[0])
    except (ValueError,TypeError):return np.nan
    return v if np.isfinite(v) and (not positive or v>0) else np.nan


def strict_mis(payload):
    rows=[]
    for item in payload.get('msgArray',[]) or []:
        px=_number(item.get('z'),True);bid=_number(item.get('b'),True);ask=_number(item.get('a'),True)
        date=str(item.get('d',''));tm=str(item.get('t',''))
        at=pd.NaT
        if len(date)==8 and ':' in tm:
            try:at=pd.Timestamp(f'{date[:4]}-{date[4:6]}-{date[6:8]} {tm}',tz=TZ)
            except (ValueError,TypeError):pass
        prev=_number(item.get('y'),True)
        rows.append({'ticker':str(item.get('c','')),'name':str(item.get('n','')),'price':px,
            'trade_price':px,'reference_price':(bid+ask)/2 if np.isfinite(bid+ask) else np.nan,
            'kind':'TRADE' if np.isfinite(px) else 'NO_TRADE','trade_at':at,'date':date,'time':tm,
            'prev':prev,'open':_number(item.get('o'),True),'high':_number(item.get('h'),True),
            'low':_number(item.get('l'),True),'bid':bid,'ask':ask,
            'limit_up':_number(item.get('u'),True),'limit_down':_number(item.get('w'),True),
            'volume':_number(item.get('v')),'chg':px/prev-1 if np.isfinite(px+prev) and prev>0 else np.nan})
    return pd.DataFrame(rows,columns=['ticker','name','price','trade_price','reference_price','kind','trade_at','date','time',
                                     'prev','open','high','low','bid','ask','limit_up','limit_down','volume','chg'])


def quote_check(quote,now=None,max_age=180):
    t=taipei(now)
    if quote is None:return False,'尚無即時成交報價'
    px=_number(quote.get('trade_price',quote.get('price')),True)
    if quote.get('kind')!='TRADE' or not np.isfinite(px):return False,'無成交價；買賣委託價僅供參考'
    at=quote.get('trade_at')
    if at is None or pd.isna(at):return False,'報價未附可核對的成交時間'
    at=taipei(at)
    age=(t-at).total_seconds()
    if at.date()!=t.date():return False,'成交日期不是今天'
    if age< -10 or age>max_age:return False,'報價已逾時，等待更新'
    if phase(t)!='盤中':return False,'目前不在盤中逐筆交易時段'
    return True,'成交時間已核對'


class LiveFeed(Quotes):
    def __init__(self,timeout=6,cooldown=60):
        super().__init__(timeout,cooldown);self._lock=threading.RLock()

    def fetch(self,channels):
        # Reuse source transport/session, with the strict parser installed for
        # this client only. No global monkey-patch and no midpoint fallback.
        import time,requests
        from .quotes import MIS
        with self._lock:
            if not channels:return strict_mis({})
            if time.time()-self._fail<self.cooldown:
                self.error='報價來源暫停回應，稍後再試';return strict_mis({})
            self.error='';self._warmup();chunks=[]
            for start in range(0,min(len(channels),150),50):
                try:
                    r=self.s.get(MIS+'/api/getStockInfo.jsp',params={'ex_ch':'|'.join(channels[start:start+50]),
                                   'json':'1','delay':'0','_':int(time.time()*1000)},timeout=self.timeout)
                    r.raise_for_status();chunks.append(strict_mis(r.json()))
                except (requests.RequestException,ValueError,TypeError):
                    self.error='即時報價暫不可用；收盤資料仍可查看';self._fail=time.time();break
            out=pd.concat(chunks,ignore_index=True) if chunks else strict_mis({})
            return out.drop_duplicates('ticker',keep='last')
