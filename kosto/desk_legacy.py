"""Read-only V21--V25 SQLite bridge. Does not invent missing chips or fundamentals."""
from __future__ import annotations
import json,sqlite3,io,zipfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .store import Store
from .config import load_config
from .panel import build_panel
from .bundle import build_bundle


def discover_alpha(base):
    base=Path(base);active=base/'studio'/'active_dataset.json'
    if active.exists():
        try:
            key=json.loads(active.read_text())['dataset_id'];meta=base/'studio'/'datasets'/key/'dataset.json'
            obj=json.loads(meta.read_text());db=(base/obj['daily_cache']/'market.sqlite').resolve()
            if db.is_relative_to(base.resolve()) and db.exists():return [{'path':str(db),'label':'原V25固定行情資料','meta':obj}]
        except (ValueError,KeyError,OSError):pass
    return [{'path':str(p),'label':str(p.relative_to(base)),'meta':{}} for p in sorted(base.rglob('market.sqlite'))][:10] if base.exists() else []


def restore_alpha_zip(data,destination):
    """Extract only read-only price and metadata inputs, verifying known checksums."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        info=z.infolist()
        if len(info)>12000 or sum(x.file_size for x in info)>6_000_000_000:raise ValueError('備份大小或檔案數異常')
        if any(Path(x.filename).is_absolute() or '..' in Path(x.filename).parts for x in info):raise ValueError('備份含不安全路徑')
        manifest={}
        if 'SNAPSHOT.json' in z.namelist():
            if z.getinfo('SNAPSHOT.json').file_size>10_000_000:raise ValueError('備份清單異常')
            manifest=json.loads(z.read('SNAPSHOT.json')).get('files',{})
        selected=[x for x in info if Path(x.filename).name in {'market.sqlite','roster.json','active_dataset.json','dataset.json'}]
        if not any(Path(x.filename).name=='market.sqlite' for x in selected):raise ValueError('這份ZIP沒有逐日行情資料庫；績效報告不能取代行情備份')
        for member in selected:
            target=destination/member.filename;target.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256();tmp=target.with_suffix(target.suffix+'.tmp')
            with z.open(member) as src,tmp.open('wb') as out:
                while True:
                    part=src.read(1024*1024)
                    if not part:break
                    out.write(part);digest.update(part)
            expected=manifest.get(member.filename)
            if isinstance(expected,dict):expected=expected.get('sha256') or expected.get('sha')
            if expected and expected!=digest.hexdigest():tmp.unlink(missing_ok=True);raise ValueError('備份SHA核對不符：'+member.filename)
            tmp.replace(target)
    return discover_alpha(destination)


def import_alpha(db,destination,source_root=None,limit=500,progress=lambda m:None):
    db=Path(db).resolve();root=Path(destination);store=Store(root/'alpha_prices')
    progress('讀取原行情庫；原檔保持唯讀')
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
        con.execute('BEGIN')
        tables={x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'raw','adj'}<=tables:raise ValueError('這份SQLite不含原V25價量表')
        latest=con.execute('SELECT MAX(date) FROM raw').fetchone()[0]
        if not latest:raise ValueError('原行情庫沒有價格資料')
        since=str((pd.Timestamp(latest)-pd.Timedelta(days=90)).date())
        tickers=[x[0] for x in con.execute('SELECT ticker FROM raw WHERE date>=? AND close>0 GROUP BY ticker ORDER BY AVG(amount) DESC LIMIT ?',
                                             (since,int(limit))) if str(x[0]).split('.')[0]!='0050']
        tickers+=['0050.TW'];placeholders=','.join('?' for _ in tickers)
        raw=pd.read_sql_query('SELECT r.*,a.close AS adj_close FROM raw r LEFT JOIN adj a ON r.ticker=a.ticker AND r.date=a.date WHERE r.ticker IN ('+placeholders+')',con,params=tickers)
    raw['date']=pd.to_datetime(raw.date);raw=raw.sort_values(['ticker','date'])
    ratio=raw.groupby('ticker').adj_close.pct_change(fill_method=None)+1
    if raw.loc[raw.ticker=='0050.TW','adj_close'].notna().sum()<60:raise ValueError('0050調整價不足，不能建立可比基準')
    raw['change']=raw.close-raw.close/ratio
    raw.loc[raw.groupby('ticker').cumcount()==0,'change']=0.
    raw['board']=np.where(raw.ticker.str.endswith('.TWO'),'TPEX','TWSE')
    raw['ticker']=raw.ticker.str.split('.').str[0];raw['name']=raw.ticker;raw['event']=''
    info=[];base=Path(source_root) if source_root else db.parent.parent
    rosters=sorted(base.glob('runs/*/roster.json'),key=lambda p:p.stat().st_mtime,reverse=True) if base.exists() else []
    if rosters:
        try:
            entries=json.loads(rosters[0].read_text())
            for x in entries:
                info.append({'ticker':str(x['ticker']).split('.')[0],'name':x.get('name',''),'sector':x.get('sector','NA')})
        except (ValueError,KeyError,OSError):pass
    if info:
        names={x['ticker']:x['name'] for x in info};raw['name']=raw.ticker.map(names).fillna(raw.ticker)
        store.append('info',pd.DataFrame(info))
    progress(f'沿用{raw.ticker.nunique()-1}檔歷史價量；建立技術因子與看盤資料')
    store.append('price',raw[['date','ticker','board','name','open','high','low','close','volume','amount','change','event']])
    start=raw.date.min();cfg=load_config(overrides={'data':{'dir':str(store.root),'start':str(start.date())},
            'research':{'start':str(max(pd.Timestamp('2019-01-01'),start+pd.Timedelta(days=365)).date())}})
    B=build_bundle(build_panel(store),cfg,log=progress)
    B.meta.update(legacy_alpha_import=True,source_scope='V25_READ_ONLY_PRICE_SUBSET',preferred_profile='price',
                  legacy_alpha_note='沿用原價量；法人、資券、營收與市值資料尚待補齊，先採價格確認模式')
    root.mkdir(exist_ok=True,parents=True);path=root/'bundle.npz';B.save(path)
    return path
