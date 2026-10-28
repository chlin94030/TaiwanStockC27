"""Intraday execution priority over a bounded model pool, with frozen base scores."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .desk_plan import decision,signal_current
from .desk_quotes import quote_check,taipei


def scan_rows(snapshot,plans,quotes,frames,now=None,synthetic=False,top_k=10,global_snapshot=None):
    rows=[];t=taipei(now)
    for rec in snapshot.table.loc[snapshot.table.valid].to_dict('records'):
        ticker=rec['ticker'];q=quotes.get(ticker);p=plans.get(ticker)
        fresh,_=quote_check(q,t);data_ready=snapshot.ready and signal_current(snapshot.date,t)
        role='keep' if rec.get('held') and rec['rank']<=snapshot.keep_rank else 'pick'
        dec=decision(p,q,role,t,'live',data_ready,synthetic)
        price=q.get('price',np.nan) if q is not None else np.nan
        frame=frames.get(ticker)
        ratio=np.nan
        if fresh and frame is not None and len(frame)>=20:
            typical=frame.volume.tail(20).mean();elapsed=max(5,min(270,t.hour*60+t.minute-540))
            if typical>0 and np.isfinite(q.get('volume',np.nan)):ratio=q['volume']/(typical*elapsed/270)
        near=0.
        if p is not None and np.isfinite(price):
            distance=abs(price-p.entry)/max(p.entry-p.alert,p.entry*.005)
            near=max(0.,1-distance)
        strength=0.
        if fresh and q.get('open',0)>0:strength=float(np.clip((price/q['open']-1)/.03,-1,1))
        volume=float(np.clip(ratio-1,0,2)/2) if np.isfinite(ratio) else 0.
        from .global_markets import linked_score
        linked=linked_score(rec['score'],rec.get('group',''),global_snapshot)
        priority=.70*linked+15*near+10*max(0,strength)+5*volume
        rows.append({**rec,'price':price,'change':q.get('chg',np.nan) if q else np.nan,
                     'quote_time':str(q.get('time','')) if q else '', 'quote_fresh':fresh,
                     'priority':priority,'global_score':linked,'volume_ratio':ratio,'code':dec.code,'status':dec.label,
                     'reason':dec.reason,'can_enter':dec.can_enter,'buy_low':p.buy_low if p else np.nan,
                     'buy_high':p.buy_high if p else np.nan,'chase':p.chase if p else np.nan,
                     'alert':p.alert if p else np.nan,'take1':p.take1 if p else np.nan})
    if not rows:return pd.DataFrame()
    out=pd.DataFrame(rows).sort_values(['can_enter','priority','ticker'],ascending=[False,False,True]).reset_index(drop=True)
    out['live_rank']=np.arange(1,len(out)+1)
    beyond=out.live_rank>top_k
    out.loc[beyond & out.can_enter,['code','status','reason']]=['watch','順位觀察','盤中順位尚未進入前'+str(top_k)+'名']
    out.loc[beyond,'can_enter']=False
    out['horizon']=snapshot.horizon
    return out
