"""Decision desk invariants: quotes, independent prices, missing chips, capital and model identity."""
from dataclasses import replace
from pathlib import Path
import copy,json,sqlite3,hashlib
import numpy as np
import pandas as pd
import pytest
from kosto import ta
from kosto.desk_quotes import strict_mis,quote_check,phase
from kosto.desk_plan import build_plan,decision,allocation,signal_current
from kosto.desk_chips import chips,validate_broker
from kosto.desk_models import DeskEngine,model_key,search_stocks,Snapshot
from kosto.desk_data import build_demo,content_settings,load_bundle
from kosto.desk_radar import scan_rows


def bars(n=280):
    dates=pd.bdate_range(end='2026-10-27',periods=n);k=np.arange(n)
    close=100+np.sin(k/8)+k*.01
    return pd.DataFrame({'open':close-.3,'high':close+1,'low':close-1,'close':close,'volume':2000.,
                         'foreign':100.,'trust':30.,'dealer':10.,'mbal':10000.,'sbal':500.},index=dates)


def quote(price,now='2026-10-28T10:10:00+08:00',**over):
    q={'ticker':'2330','price':price,'trade_price':price,'kind':'TRADE','trade_at':pd.Timestamp(now),
       'time':'10:10:00','prev':price,'open':price*.995,'high':price*1.01,'low':price*.99,'volume':1000.,
       'bid':price-.05,'ask':price+.05,'limit_up':price*1.1,'limit_down':price*.9,'chg':0.}
    q.update(over);return q


@pytest.mark.parametrize('value',[10,50,100,500,1000])
def test_tick_down_keeps_exact_valid_boundary(value):
    assert ta.round_tick(value,'down')==value
    assert ta.round_tick(value,'up')==value


def test_rsi_monotone_and_flat_are_defined():
    assert ta.rsi(pd.Series(np.arange(100,dtype=float))).iloc[-1]==100
    assert ta.rsi(pd.Series([12.]*100)).iloc[-1]==50


def test_no_trade_quote_never_uses_midpoint_as_price():
    q=strict_mis({'msgArray':[{'c':'2330','z':'-','a':'100.5_','b':'100_','d':'20261028','t':'10:10:00'}]}).iloc[0].to_dict()
    assert np.isnan(q['price']) and q['reference_price']==100.25 and q['kind']=='NO_TRADE'
    assert not quote_check(q,'2026-10-28T10:10:10+08:00')[0]


@pytest.mark.parametrize('trade_at,now',[('2026-10-27T10:10:00+08:00','2026-10-28T10:10:00+08:00'),
    ('2026-10-28T10:00:00+08:00','2026-10-28T10:10:00+08:00'),
    ('2026-10-28T10:20:00+08:00','2026-10-28T10:10:00+08:00')])
def test_stale_previous_day_and_future_quotes_cannot_be_live(trade_at,now):
    assert not quote_check(quote(100,trade_at),now)[0]


def test_closing_auction_and_after_hours_do_not_offer_immediate_entry():
    p=build_plan(bars(),'mid')
    for now in ['2026-10-28T08:50:00+08:00','2026-10-28T13:25:00+08:00','2026-10-28T15:00:00+08:00']:
        assert not decision(p,quote(p.entry,now),now=now).can_enter
    assert phase('2026-10-31T10:10:00+08:00')=='休市'
    assert phase('2026-10-28T10:10:00+08:00',holidays={'2026-10-28'})=='休市'


def test_plan_prices_and_risk_targets_are_ordered_and_tick_valid():
    p=build_plan(bars(),'mid')
    assert p.alert<p.buy_low<=p.entry<=p.buy_high<=p.chase<p.take1<=p.take2
    assert p.take1>=p.entry+2*(p.entry-p.alert)
    assert build_plan(bars(59),'mid') is None
    for name in ['buy_low','buy_high','chase','alert','take1']:
        value=getattr(p,name);assert ta.round_tick(value,'down')==value


def test_entry_requires_real_quote_zone_and_current_inputs():
    p=build_plan(bars(),'mid');now='2026-10-28T10:10:00+08:00';q=quote(p.entry)
    assert decision(p,q,now=now).can_enter
    assert not decision(p,q,now=now,synthetic=True).can_enter
    assert not decision(p,q,now=now,data_ready=False).can_enter
    assert not decision(p,None,now=now).can_enter
    assert not decision(p,q,now=now,role='keep').can_enter
    assert not decision(p,q,now=now,mode='close').can_enter
    assert decision(p,quote(p.buy_low-.1),now=now).code=='weak'
    assert decision(p,quote(p.chase+.5),now=now).code=='chase'


def test_missing_institutional_data_stays_missing_not_zero():
    f=bars();f[['foreign','trust','dealer']]=np.nan
    c=chips(f)
    assert c['table']['近5日(張)'].isna().all()
    assert not any('同步買超' in x for x in c['notes'])
    f.iloc[-5:,f.columns.get_loc('foreign')]=20
    f.iloc[-5:,f.columns.get_loc('trust')]=10
    f.iloc[-5:,f.columns.get_loc('dealer')]=5
    assert any('同步買超' in x for x in chips(f)['notes'])


def test_unchanged_holder_week_is_not_skipped_in_weekly_delta():
    f=bars(30);f['big400']=np.r_[np.full(20,60.),np.full(10,61.)]
    f['holder_date']=pd.NaT
    for start,date in [(0,'2026-10-02'),(20,'2026-10-16'),(25,'2026-10-23')]:f.iloc[start:start+5 if start else 20,f.columns.get_loc('holder_date')]=pd.Timestamp(date)
    f['holder_date']=f.holder_date.ffill()
    h=chips(f)['holders'];assert h['change_pp']==0. and h['date']=='2026-10-23' and h['exact_date']


def test_weekly_chart_does_not_turn_missing_flows_into_zero():
    f=bars(20);f['foreign']=np.nan
    w=ta.resample(f,'W')
    assert w.foreign.isna().all() and w.volume.sum()==f.volume.sum()
    assert set(w.index)<=set(f.index)


def test_allocation_deduplicates_caps_risk_fees_and_cash():
    p=build_plan(bars(),'mid');capital=1_000_000
    selections={'short':['2330','2454'],'mid':['2330','2454'],'long':['2330']}
    weights={'cash':.1,'0050':.2,'short':.3,'mid':.2,'long':.2}
    rows,ledger=allocation(capital,weights,selections,{'2330':p,'2454':p},{'2330':p.entry,'2454':p.entry},
                         groups={'2330':'半導體','2454':'半導體'},max_name=.1,max_group=.15)
    assert rows.ticker.is_unique
    assert rows.weight.max()<=.100001 and rows.weight.sum()<=.150001
    assert rows.risk_amount.max()<=5000
    assert ledger['cash']>=ledger['cash_floor']
    assert ledger['stocks_including_fees']+ledger['etf']+ledger['cash']==pytest.approx(capital)
    with pytest.raises(ValueError,match='100%'):allocation(capital,{'cash':.5},selections,{}, {})
    no_short,unused=allocation(capital,{'cash':.1,'0050':.4,'mid':.3,'long':.2},selections,{'2330':p,'2454':p},{'2330':p.entry,'2454':p.entry})
    assert all('短線' not in x for x in no_short.sources)


def test_alpha_backup_rejects_traversal_and_preserves_verified_bytes(tmp_path):
    import io,zipfile
    from kosto.desk_legacy import restore_alpha_zip
    data=b'SQLite test bytes';buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('cache/market.sqlite',data)
        z.writestr('SNAPSHOT.json',json.dumps({'files':{'cache/market.sqlite':hashlib.sha256(data).hexdigest()}}))
    result=restore_alpha_zip(buf.getvalue(),tmp_path/'good')
    assert result and Path(result[0]['path']).read_bytes()==data
    bad=io.BytesIO()
    with zipfile.ZipFile(bad,'w') as z:z.writestr('../market.sqlite',data)
    with pytest.raises(ValueError,match='不安全'):restore_alpha_zip(bad.getvalue(),tmp_path/'bad')


def test_broker_import_requires_consistent_net_and_source():
    f=pd.DataFrame([{'date':'2026-10-27','ticker':'2330','buy_lots':100,'sell_lots':60,'net_lots':40,'source':'licensed test fixture'}])
    assert validate_broker(f).net_lots.iloc[0]==40
    f.loc[0,'net_lots']=50
    with pytest.raises(ValueError,match='不一致'):validate_broker(f)


@pytest.fixture(scope='module')
def demo(tmp_path_factory):return build_demo(tmp_path_factory.mktemp('desk-demo'),100)


def test_recipe_identity_keeps_sparse_blend_and_label_horizon(demo):
    E=DeskEngine(demo,content_settings(demo));v=E.R.model_variant('short');cfg=E.cfg
    assert v.label_h=='mid' and v.hyp_blend==.5 and v.growth_blend==.3
    key=model_key(v,cfg,'BLEND',500,'short')
    assert model_key(replace(v,k=v.k+1),cfg,'BLEND',500,'short')!=key
    assert model_key(replace(v,label_h='long'),cfg,'BLEND',500,'short')!=key
    assert model_key(replace(v,hyp_blend=.25),cfg,'BLEND',500,'short')!=key


def test_three_horizon_snapshots_and_returned_copies(demo):
    E=DeskEngine(demo,content_settings(demo))
    for h,batches,threshold in [('short',1,25),('mid',4,20),('long',12,25)]:
        s=E.snapshot(h);assert s.ready and len(s.picks)>0 and s.batch_count==batches and s.keep_rank==threshold
        first=s.table.score.iloc[0];s.table.loc[s.table.index[0],'score']=-999
        assert E.snapshot(h).table.score.iloc[0]==first


def test_search_by_name_ticker_and_unknown_never_fabricates_score(demo):
    t=demo.tickers[0];name=demo.meta['names'][0]
    assert search_stocks(demo,t)[0]['ticker']==t
    assert search_stocks(demo,name)[0]['name']==name
    assert search_stocks(demo,'不存在的公司名稱')==[]


def test_live_price_change_reorders_radar_and_removes_buy_status():
    p=build_plan(bars(),'mid');now='2026-10-28T10:10:00+08:00'
    table=pd.DataFrame([{'ticker':t,'name':t,'group':'半導體','valid':True,'rank':r,'score':90.,'held':False} for t,r in [('2330',1),('2454',2)]])
    s=Snapshot('mid','test','test','2026-10-27','test',table,[],{},1.,20,4,True,[])
    plans={'2330':p,'2454':p};frames={'2330':bars(),'2454':bars()};quotes={'2330':quote(p.entry),'2454':quote(p.chase+1)}
    first=scan_rows(s,plans,quotes,frames,now=now)
    assert first.iloc[0].ticker=='2330' and bool(first.iloc[0].can_enter)
    quotes={'2330':quote(p.chase+1),'2454':quote(p.entry)}
    second=scan_rows(s,plans,quotes,frames,now=now)
    assert second.iloc[0].ticker=='2454' and not bool(second.loc[second.ticker=='2330','can_enter'].iloc[0])


def test_safe_bundle_loader_rejects_non_npz_and_reports_are_json_only():
    with pytest.raises(Exception):load_bundle(b'not an npz')
    from kosto.desk_data import report_summary
    with pytest.raises(Exception):report_summary(b'\x80\x04fake-pickle')


def test_latest_partial_market_does_not_publish_a_new_bundle(tmp_path):
    from kosto.store import Store
    from kosto.desk_update import latest_refresh
    Store(tmp_path/'data').append('price',pd.DataFrame([{'date':'2026-10-26','ticker':'2330','close':100.}]))
    class Partial:
        def twse_price(self,date):return pd.DataFrame({'ticker':[str(2000+i) for i in range(60)],'date':date,'close':100.})
        def tpex_price(self,date):return pd.DataFrame()
    with pytest.raises(ValueError,match='同步齊全'):latest_refresh(tmp_path,'2026-10-27T18:00:00+08:00',Partial())
    assert not (tmp_path/'artifacts'/'bundle.npz').exists()
    assert Store(tmp_path/'data').read('price').date.max()==pd.Timestamp('2026-10-26')


def test_legacy_readonly_prices_can_build_a_price_desk(tmp_path):
    from kosto.desk_legacy import import_alpha
    db=tmp_path/'market.sqlite';dates=pd.bdate_range('2022-01-03',periods=850);rows=[];adj=[]
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE raw(ticker TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL)')
        con.execute('CREATE TABLE adj(ticker TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL)')
        for j,t in enumerate(['0050.TW']+[str(2300+i)+'.TW' for i in range(12)]):
            k=np.arange(len(dates));c=100*np.exp(.0003*k+.02*np.sin(k/(8+j)))
            for d,px in zip(dates,c):
                rows.append((t,str(d.date()),px,px*1.01,px*.99,px,2e6,2e8));adj.append((t,str(d.date()),px,px*1.01,px*.99,px))
        con.executemany('INSERT INTO raw VALUES(?,?,?,?,?,?,?,?)',rows);con.executemany('INSERT INTO adj VALUES(?,?,?,?,?,?)',adj)
    before=hashlib.sha256(db.read_bytes()).hexdigest();output=import_alpha(db,tmp_path/'bridge',limit=20)
    assert hashlib.sha256(db.read_bytes()).hexdigest()==before
    B=load_bundle(output);assert B.meta['legacy_alpha_import'] and not B.meta['synthetic']
    E=DeskEngine(B,content_settings(B),'price');assert not E.snapshot('mid').table.empty
