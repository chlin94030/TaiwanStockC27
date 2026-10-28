"""Bounded latest-session refresh; existing raw history is reused."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import json,threading
import pandas as pd
from .store import Store
from .config import load_config
from .bundle import build_bundle
from .panel import build_panel
from .desk_quotes import taipei

_LOCK=threading.RLock();_WORKERS={}


def _state_path(root):return Path(root)/'artifacts'/'desk_update.json'


def update_state(root):
    try:return json.loads(_state_path(root).read_text())
    except (ValueError,OSError):return {}


def _stamp(root,status,message):
    p=_state_path(root);p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'status':status,'message':message,'at':datetime.now(timezone.utc).isoformat()},ensure_ascii=False));tmp.replace(p)


def local_store(root):
    root=Path(root)
    for p in [root/'data',root/'artifacts'/'alpha_bridge'/'alpha_prices']:
        if (p/'raw'/'price').exists():return p
    raise ValueError('目前只有看盤資料包，沒有本機原始行情。請沿用原每日更新流程，完成後重新載入雲端資料。')


def rebuild(root,progress=lambda m:None):
    root=Path(root);path=local_store(root);store=Store(path);prices=store.read('price')
    if prices.empty:raise ValueError('本機行情為空')
    cfg=load_config(overrides={'data':{'dir':str(path),'start':str(prices.date.min().date())}})
    cfg['research']['start']=str(max(pd.Timestamp(cfg['research']['start']),prices.date.min()+pd.Timedelta(days=365)).date())
    B=build_bundle(build_panel(store),cfg,log=progress)
    if path.name=='alpha_prices':B.meta.update(legacy_alpha_import=True,preferred_profile='price',legacy_alpha_note='沿用V25價量；基本面及籌碼仍依實際補齊情形顯示')
    output=root/'artifacts'/'bundle.npz';output.parent.mkdir(exist_ok=True);B.save(output)
    return output


def latest_refresh(root,now=None,provider=None):
    root=Path(root);store=Store(local_store(root))
    if provider is None:
        from .sources.official import Http,Official
        provider=Official(Http(pause=2.6,retries=2,timeout=18,log=lambda m:_stamp(root,'RUNNING',m)))
    t=taipei(now);target=t.normalize().tz_localize(None)
    if t.hour<14:target-=pd.offsets.BDay(1)
    dates=pd.bdate_range(end=target,periods=5)[::-1];selected=None;prices=[]
    for date in dates:
        parts=[]
        for board in ['twse','tpex']:
            _stamp(root,'RUNNING',f'確認{date.date()} {board}收盤價量')
            parts.append(getattr(provider,board+'_price')(date))
        good=[p is not None and len(p)>=50 for p in parts]
        if all(good):selected=date;prices=parts;break
        if any(good):raise ValueError('上市與上櫃收盤資料尚未同步齊全，保留前次看盤資料')
    if selected is None:raise ValueError('最近交易日尚未取得完整收盤資料')
    store.append('price',pd.concat(prices,ignore_index=True));gaps=[]
    for dataset,suffix in [('institutional','inst'),('margin','margin'),('valuation','valuation')]:
        _stamp(root,'RUNNING','更新 '+dataset+'｜'+str(selected.date()))
        parts=[]
        for board in ['twse','tpex']:
            try:
                part=getattr(provider,board+'_'+suffix)(selected)
                if part is None or part.empty:gaps.append(dataset+' '+board)
                else:parts.append(part)
            except Exception:gaps.append(dataset+' '+board)
        if parts:store.append(dataset,pd.concat(parts,ignore_index=True))
    for board in ['twse','tpex']:
        try:
            part=getattr(provider,board+'_exright')(selected,selected)
            if part is not None and not part.empty:store.append('exright',part)
        except Exception:gaps.append('exright '+board)
    try:
        holders=provider.tdcc_holders()
        if holders is not None and not holders.empty:store.append('holders',holders)
    except Exception:gaps.append('holders')
    _stamp(root,'RUNNING','沿用既有歷史，重建看盤快取')
    output=rebuild(root,lambda m:_stamp(root,'RUNNING',m))
    message='價量已至'+str(selected.date())
    if gaps:message+='；部分籌碼尚缺：'+','.join(gaps)
    _stamp(root,'COMPLETE_WITH_GAPS' if gaps else 'COMPLETE',message)
    return output


def start_latest(root):
    root=Path(root).resolve();local_store(root)
    with _LOCK:
        if str(root) in _WORKERS and _WORKERS[str(root)].is_alive():raise ValueError('更新已在執行')
        def worker():
            try:latest_refresh(root)
            except Exception as e:_stamp(root,'ERROR',str(e))
        thread=threading.Thread(target=worker,daemon=True,name='AlphaRadar-latest');_WORKERS[str(root)]=thread;thread.start()
