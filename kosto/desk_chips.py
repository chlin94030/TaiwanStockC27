"""Chip summaries preserve missing observations and the original data period."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _sum(series,n):
    s=pd.to_numeric(series.tail(n),errors='coerce')
    return float(s.sum()) if len(s)==n and s.notna().all() else np.nan


def _streak(series):
    a=pd.to_numeric(series,errors='coerce').to_numpy(float)
    if not len(a) or not np.isfinite(a[-1]) or a[-1]==0:return 0
    sign=np.sign(a[-1]);n=0
    for x in a[::-1]:
        if not np.isfinite(x) or np.sign(x)!=sign:break
        n+=1
    return int(n*sign)


def chips(frame):
    rows=[];notes=[];score=0;last_date=None
    if frame is None or frame.empty:return {'table':pd.DataFrame(),'notes':['尚無籌碼資料'],'tone':'資料不足'}
    names={'foreign':'外資','trust':'投信','dealer':'自營商'}
    for col,name in names.items():
        s=frame[col] if col in frame else pd.Series(np.nan,index=frame.index)
        known=s.dropna();date=str(known.index[-1].date()) if len(known) else None
        rows.append({'法人':name,'最新日(張)':float(s.iloc[-1]) if pd.notna(s.iloc[-1]) else np.nan,
                     '近5日(張)':_sum(s,5),'近20日(張)':_sum(s,20),'連續買賣超':_streak(s),'資料日':date or '未取得'})
        if date:last_date=max(last_date or date,date)
    table=pd.DataFrame(rows)
    f=table['近5日(張)'].to_numpy(float)
    if np.isfinite(f).all():
        if (f>0).all():notes.append('外資、投信、自營商近5日同步買超');score+=2
        elif (f<0).all():notes.append('三大法人近5日同步賣超');score-=2
        elif f[0]>0 and f[1]>0:notes.append('外資與投信同買，自營商不同步');score+=1
        elif f[0]*f[1]<0:notes.append('外資與投信方向分歧，觀察後續持續性')
        else:notes.append('法人方向混合')
    else:notes.append('法人近5日尚有缺值，暫不判定同步方向')
    balances={}
    for col,name in [('mbal','融資餘額'),('sbal','融券餘額')]:
        source=frame
        if 'margin_date' in frame:source=frame.dropna(subset=['margin_date']).drop_duplicates('margin_date',keep='last')
        s=source[col].dropna() if col in source else pd.Series(dtype=float)
        date=str(pd.Timestamp(source.loc[s.index[-1],'margin_date']).date()) if len(s) and 'margin_date' in source else str(s.index[-1].date()) if len(s) else None
        balances[name]={'value':float(s.iloc[-1]) if len(s) else None,
                        'change':float(s.iloc[-1]-s.iloc[-2]) if len(s)>1 else None,
                        'date':date}
    holders={'value':None,'change_pp':None,'date':None,'exact_date':False}
    if 'big400' in frame:
        d=frame[['big400']].dropna().copy()
        if 'holder_date' in frame:
            d['source_date']=frame.loc[d.index,'holder_date'];d=d.dropna(subset=['source_date']).drop_duplicates('source_date',keep='last')
            holders['exact_date']=True
        else:
            d=d.resample('W-FRI').last().dropna();d['source_date']=d.index
        if len(d):
            holders['value']=float(d.big400.iloc[-1]);holders['date']=str(pd.Timestamp(d.source_date.iloc[-1]).date())
            if len(d)>1:holders['change_pp']=float(d.big400.iloc[-1]-d.big400.iloc[-2])
    if holders['change_pp'] is not None:
        notes.append(f"大戶≥400張持股較前一週樣本{holders['change_pp']:+.2f}百分點")
    if holders['value'] is None:notes.append('大戶週資料尚未取得')
    return {'table':table,'notes':notes,'tone':'偏多' if score>=2 else '偏空' if score<=-2 else '中性',
            'date':last_date,'balances':balances,'holders':holders,'broker_available':False}


def validate_broker(frame):
    columns=['date','ticker','buy_lots','sell_lots','net_lots','source']
    if not set(columns)<=set(frame):raise ValueError('主力檔案須含日期、代號、買進張數、賣出張數、淨買超、來源')
    x=frame[columns].copy();x['date']=pd.to_datetime(x.date,errors='coerce');x['ticker']=x.ticker.astype(str)
    if x.date.isna().any() or not x.ticker.str.fullmatch(r'\d{4}').all():raise ValueError('日期或股票代號格式錯誤')
    if x.source.fillna('').astype(str).str.strip().eq('').any():raise ValueError('須註明分點資料來源')
    for c in ['buy_lots','sell_lots','net_lots']:x[c]=pd.to_numeric(x[c],errors='coerce')
    if not np.isfinite(x[['buy_lots','sell_lots','net_lots']]).all().all() or (x[['buy_lots','sell_lots']]<0).any().any():raise ValueError('分點數值須有效且買賣張數非負')
    if ((x.buy_lots-x.sell_lots-x.net_lots).abs()>.011).any():raise ValueError('買進減賣出與淨買超不一致')
    if x.duplicated(['date','ticker']).any():raise ValueError('同股同日不得重複')
    return x.sort_values(['ticker','date'])
