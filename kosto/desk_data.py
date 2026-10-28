"""Safe bundle/report access, reusable chart frames and a labelled offline demo."""
from __future__ import annotations
import copy,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .bundle import Bundle,build_bundle
from . import ta
from .config import load_config


def load_bundle(value):
    data=Path(value).read_bytes() if isinstance(value,(str,Path)) else bytes(value)
    if len(data)>600*1024*1024:raise ValueError('看盤資料包超過600MB上限')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if len(z.infolist())>128 or sum(f.file_size for f in z.infolist())>800_000_000:raise ValueError('資料包過大；請先縮小母池或看盤歷史長度')
    B=Bundle.load(io.BytesIO(data));n=len(B.tickers);s=len(B.sig_dates);f=len(B.factors)
    if not 1<=n<=6000 or not 1<=s<=6000 or not 1<=f<=200:raise ValueError('資料包維度不合法')
    for key,shape in [('U',(s,n,f)),('STAGE',(s,n)),('CAP',(s,n)),('TRADE',(s,n))]:
        if key not in B.a or B.a[key].shape!=shape:raise ValueError('資料包欄位維度不符：'+key)
    if B.sig_dates.has_duplicates or not B.sig_dates.is_monotonic_increasing:raise ValueError('訊號日不合法')
    if len(B.tickers)!=len(set(B.tickers)):raise ValueError('股票代號重複')
    for name in ('names','sectors'):
        if len(B.meta.get(name,[]))!=n:raise ValueError('股票名稱或族群欄位長度不符')
        B.meta[name]=[str(x) if x is not None else '' for x in B.meta[name]]
    if 'q_days' in B.meta:
        q=len(B.meta['q_days'])
        for key,value in B.a.items():
            if key.startswith('Q_') and key!='Q_TAIEX' and value.shape!=(q,n):raise ValueError('看盤陣列維度不符：'+key)
    B.meta['desk_content_id']=hashlib.sha256(data).hexdigest()[:20]
    return B


def report_summary(data):
    if not data:return None
    if isinstance(data,(str,Path)):data=Path(data).read_bytes()
    if bytes(data[:2])==b'PK':
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name=next((n for n in z.namelist() if Path(n).name=='results_summary.json'),None)
            if name is None:return None
            info=z.getinfo(name)
            if info.file_size>10*1024*1024:raise ValueError('摘要過大')
            result=json.loads(z.read(name))
    else:result=json.loads(data)
    if not isinstance(result,dict) or not isinstance(result.get('horizons'),dict):raise ValueError('不支援的研究摘要')
    return result


def chart_frame(B,j,adjusted=True):
    frame=ta.stock_frame(B,j).copy()
    if frame.empty or not adjusted:return frame
    index=frame.get('adj_index')
    if index is None and 'R' in B.a:
        # Legacy bundles can reconstruct a total-return display index; anchor
        # to the LAST raw close so current tradable prices retain their units.
        returns=pd.Series(B.a['R'][:,j],index=B.days)
        index=(1+returns).cumprod().reindex(frame.index)
    if index is not None and index.notna().all() and np.isfinite(index.iloc[-1]) and index.iloc[-1]>0:
        ac=index/index.iloc[-1]*frame.close.iloc[-1];ratio=ac/frame.close
        for c in ('open','high','low','close'):frame[c]=frame[c]*ratio
    return frame


def build_demo(root,n=120):
    from .sources.synthetic import write_demo
    from .store import Store
    from .panel import build_panel
    root=Path(root);path=root/'demo_bundle.npz'
    if path.exists():return load_bundle(path)
    store=Store(root/'synthetic')
    write_demo(store,n,26,start='2018-01-02',end='2026-10-27')
    cfg=load_config(overrides={'data':{'dir':str(store.root)},'research':{'start':'2019-06-01'},'app':{'chart_days':520}})
    B=build_bundle(build_panel(store),cfg,log=lambda *a:None)
    B.meta['synthetic']=True;B.save(path)
    return load_bundle(path)


def content_settings(B):
    # Recipe comes from current checked-in settings, never stale results.pkl.
    cfg=load_config();original=B.meta.get('config') or {}
    if B.meta.get('legacy_alpha_import'):
        cfg['desk']={'pools':{h:{'rule':'ADV','size':300} for h in ('short','mid','long')}}
    if original.get('research',{}).get('start'):cfg['research']['start']=original['research']['start']
    return cfg
