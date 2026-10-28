"""Alpha Radar V27 Global: responsive decision desk with time-valid overseas context."""
from __future__ import annotations
import io,json,os,tempfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from . import ta
from .desk_style import css,esc,stats,title
from .desk_data import load_bundle,report_summary,chart_frame,build_demo,content_settings
from .desk_models import DeskEngine,search_stocks,MODEL_TITLES
from .desk_plan import HZ,PROFILES,build_plan,decision,allocation,signal_current
from .desk_quotes import LiveFeed,channel,phase,taipei,quote_check
from .desk_chips import chips,validate_broker
from .desk_radar import scan_rows
from .global_markets import fetch_global,display_rows
from .chain import label as chain_label
from . import ui as fmt

ROOT=Path(os.getenv('ALPHA_DESK_PROJECT_ROOT',Path(__file__).resolve().parent.parent)).resolve()
PAGES=['選股總覽','盤中雷達','明日布局','個股診斷','資金配置','資料與策略']


def secret(name):
    try:return str(st.secrets.get(name,os.getenv(name,''))).strip()
    except Exception:return os.getenv(name,'').strip()


@st.cache_resource(show_spinner=False)
def local_bundle(path,mtime):return load_bundle(path)


@st.cache_resource(show_spinner=False)
def demo_bundle():return build_demo(ROOT/'artifacts'/'demo')


@st.cache_resource(show_spinner=False)
def engine(content,profile,cfg_json,_B):return DeskEngine(_B,json.loads(cfg_json),profile)


@st.cache_data(show_spinner=False,max_entries=800)
def frame(content,j,adjusted,_B):return chart_frame(_B,j,adjusted)


@st.cache_resource(show_spinner=False)
def quote_client():return LiveFeed()


@st.cache_data(ttl=20,show_spinner=False,max_entries=40)
def quote_batch(channels):
    feed=quote_client();q=feed.fetch(list(channels));return q,feed.error


@st.cache_data(ttl=300,show_spinner=False)
def global_batch():return fetch_global()


def global_panel(B,expanded=False):
    if B.meta.get('synthetic'):
        st.markdown('<div class="desk-global neutral"><b>跨市場脈動｜操作示範</b><span>未連接海外行情，不調整合成排名</span></div>',unsafe_allow_html=True)
        return None
    snap=global_batch();tone='positive' if snap.label=='偏多' else 'negative' if snap.label=='偏空' else 'neutral'
    st.markdown(f'<div class="desk-global {tone}"><b>跨市場脈動｜{esc(snap.label)}</b><span>{esc(snap.note)} · 選股模型88%＋海外情境最多12%</span></div>',unsafe_allow_html=True)
    if expanded:
        view=display_rows(snap)
        st.dataframe(view.style.format({'漲跌%':'{:+.2f}%','模型權重':'{:.0%}'},na_rep='—'),hide_index=True,width='stretch')
        st.caption('費半4%、Nasdaq 3%、Nasdaq期貨3%、日經1%、韓股1%；依產業敏感度折算。海外資料缺漏會自動縮小權重。')
    return snap


@st.cache_data(ttl=300,show_spinner=False)
def cloud_stamp(repo,token):
    from .remote import release_info,RESEARCH_TAG
    return release_info(repo,RESEARCH_TAG,token)


@st.cache_resource(show_spinner=False)
def cloud_bundle(repo,token,stamp):
    from .remote import fetch_asset,RESEARCH_TAG
    B=load_bundle(fetch_asset(repo,RESEARCH_TAG,'bundle.npz',token))
    report=None
    try:report=report_summary(fetch_asset(repo,RESEARCH_TAG,'desk_summary.json',token))
    except Exception:
        try:report=report_summary(fetch_asset(repo,RESEARCH_TAG,'report.zip',token))
        except Exception:pass
    try:
        validation=json.loads(fetch_asset(repo,RESEARCH_TAG,'v27_validation.json',token))
        report=report or {'horizons':{}};report['_v27_validation']=validation
    except Exception:pass
    return B,report


def _json(path):
    try:return json.loads(Path(path).read_text())
    except (ValueError,OSError):return {}


def get_source():
    if st.session_state.get('desk_demo') or os.getenv('ALPHA_DESK_DEMO')=='1':return demo_bundle(),None,'操作示範'
    path=st.session_state.get('desk_source_path') or os.getenv('ALPHA_DESK_BUNDLE')
    if path and Path(path).is_file():return local_bundle(str(path),Path(path).stat().st_mtime_ns),None,'匯入的看盤資料'
    from .setup_page import detect_repo
    repo,_=detect_repo(secret('GITHUB_REPO'),st.query_params.get('repo',''))
    if repo:
        try:
            info=cloud_stamp(repo,secret('GITHUB_TOKEN'));stamp=(info.get('assets',{}).get('bundle.npz') or {}).get('updated_at')
            if stamp:
                with st.spinner('載入看盤資料；後續切換頁面沿用快取…'):
                    B,report=cloud_bundle(repo,secret('GITHUB_TOKEN'),stamp)
                return B,report,'每日更新資料'
        except Exception:st.session_state['desk_source_warning']='雲端資料暫未取得，可沿用本機快取或匯入資料包'
    for path in [ROOT/'artifacts'/'bundle.npz',ROOT/'artifacts'/'alpha_bridge'/'bundle.npz']:
        if path.is_file():
            report=None
            for r in [ROOT/'artifacts'/'desk_summary.json',ROOT/'artifacts'/'report.zip']:
                if r.exists():
                    try:report=report_summary(r);break
                    except Exception:pass
            validation=_json(ROOT/'artifacts'/'v27_validation.json')
            if validation:
                report=report or {'horizons':{}};report['_v27_validation']=validation
            return local_bundle(str(path),path.stat().st_mtime_ns),report,'本機看盤資料'
    return None,None,''


def header(B=None,source=''):
    when=taipei();date=str(B.sig_dates[-1].date()) if B is not None else '尚未載入'
    st.markdown('<div class="desk-top"><div><div class="desk-wordmark">ALPHA RADAR <span>PRO</span></div>'
                '<div class="desk-product">台股決策台 · 短中長線選股與交易計畫</div></div>'
                f'<div class="desk-meta">{when:%Y.%m.%d %H:%M}　{esc(phase(when))}<br>模型資料 {esc(date)}　·　V27.0 GLOBAL</div></div>',unsafe_allow_html=True)
    if B is None:return
    if B.meta.get('synthetic'):
        st.markdown('<div class="desk-demo">操作示範｜所有價格、排名與籌碼均為合成資料；沒有真實買入訊號。</div>',unsafe_allow_html=True)
    count=len(B.tickers);mode={'price':'價格確認','source':'來源定稿模型','v27':'V27十面向融合'}.get(st.session_state.get('desk_profile'),'V27十面向融合')
    st.markdown(f'<div class="desk-strip"><span>資料 <b>{esc(source)}</b></span><span>股票 <b>{count:,} 檔</b></span>'
                f'<span>策略 <b>{esc(mode)}</b></span><span>法人 <b>盤後日資料</b></span><span>大戶 <b>集保週資料</b></span>'
                '<span>盤中 <b>核對成交時間後才顯示可布局</b></span></div>',unsafe_allow_html=True)


def setup():
    st.markdown(title('連接你的選股資料','已有CLAUDE資料包或原Alpha行情庫，可直接沿用；也可先預覽完整操作畫面。'),unsafe_allow_html=True)
    a,b=st.columns([1.5,1])
    with a:
        supplied=st.file_uploader('匯入 CLAUDE 看盤資料包（bundle.npz）',type=['npz'],key='setup_bundle')
        if supplied is not None and st.button('採用這份資料包',type='primary'):
            try:
                load_bundle(supplied.getvalue());p=ROOT/'artifacts'/'imported_bundle.npz';p.parent.mkdir(exist_ok=True)
                p.write_bytes(supplied.getvalue());st.session_state['desk_source_path']=str(p);st.rerun()
            except Exception as e:st.error('資料包無法使用：'+str(e))
        st.caption('CLAUDE既有資料與每日更新流程可延續使用；本版不讀取上傳的執行型pickle檔。')
    with b:
        st.markdown('**先看介面與操作流程**')
        st.write('預覽盤中雷達、買進區、資金配置與K線；示範數字不代表真實台股。')
        if st.button('開啟操作示範',type='primary',width='stretch'):
            st.session_state['desk_demo']=True;st.rerun()
    from .desk_legacy import discover_alpha,import_alpha,restore_alpha_zip
    backup=st.file_uploader('或還原原Alpha的 V21_DATABASE.zip 備份',type=['zip'],key='setup_alpha_backup')
    if backup is not None and st.button('還原備份中的行情'):
        try:
            destination=ROOT/'artifacts'/'restored_alpha';restore_alpha_zip(backup.getvalue(),destination)
            st.session_state['desk_restored_base']=str(destination);st.rerun()
        except Exception as e:st.error('備份尚未還原：'+str(e))
    old_root=Path(st.session_state.get('desk_restored_base',ROOT/'data'/'v21_universe_lab'));old=discover_alpha(old_root)
    if old:
        st.markdown('**找到原Alpha歷史行情**')
        chosen=st.selectbox('沿用的資料庫',old,format_func=lambda x:x['label'])
        if st.button('沿用V25行情，建立新版看盤快取'):
            try:
                status=st.empty()
                path=import_alpha(chosen['path'],ROOT/'artifacts'/'alpha_bridge',old_root,progress=lambda m:status.info(m))
                st.session_state['desk_source_path']=str(path);st.session_state['desk_profile']='price';st.rerun()
            except Exception as e:st.error('資料轉入未完成：'+str(e))
    with st.expander('沒有現成資料包：建立／更新正式資料'):
        st.write('將程式部署到原CLAUDE儲存庫，可沿用每日資料更新。原Alpha資料在其他儲存庫時，先下載其備份保存；本版程式包不含行情庫。')
        repo=st.text_input('GitHub儲存庫（帳號／名稱）',value=st.query_params.get('repo',''))
        if repo and st.button('連接這個儲存庫'):
            from .setup_page import detect_repo
            value,_=detect_repo('',repo)
            if value:st.query_params['repo']=value;st.cache_data.clear();st.rerun()
            else:st.error('請使用帳號／儲存庫名稱格式')
        if st.button('在本機開始／接續正式資料更新'):
            from .setup_page import start_local_build
            start_local_build();st.info('已啟動背景更新；完成後重新載入。電腦須保持開機。')


def resolve(B,text):
    out=[]
    for word in str(text).replace('，',',').replace('、',',').replace(' ', ',').split(','):
        word=word.strip()
        if not word:continue
        found=search_stocks(B,word)
        exact=[r for r in found if word in (r['ticker'],r['name'])]
        if len(exact)==1:out.append(exact[0]['ticker'])
    return list(dict.fromkeys(out))


def fetch_quotes(B,tickers):
    idx={t:j for j,t in enumerate(B.tickers)}
    if B.meta.get('synthetic'):
        out={}
        for t in tickers:
            if t not in idx:continue
            f=frame(B.meta['desk_content_id'],idx[t],False,B)
            if f.empty:continue
            px=float(f.close.iloc[-1])*(1+(idx[t]%7-3)*.002)
            out[t]={'price':px,'trade_price':px,'prev':float(f.close.iloc[-1]),'open':px*.997,
                    'low':px*.99,'high':px*1.01,'volume':float(f.volume.iloc[-1]),'chg':px/f.close.iloc[-1]-1,
                    'kind':'DEMO','trade_at':taipei(),'time':'示範','bid':px*.999,'ask':px*1.001,
                    'limit_up':px*1.1,'limit_down':px*.9}
        return out,''
    channels=tuple(channel(t,(B.meta.get('boards') or ['']*len(idx))[idx[t]]) for t in tickers if t in idx)
    q,error=quote_batch(channels)
    return {str(r['ticker']):r for r in q.to_dict('records')},error


def stock_plan(B,t,h='mid',quote=None):
    j=B.tickers.index(t);f=frame(B.meta['desk_content_id'],j,True,B)
    reference=quote.get('prev') if quote is not None and signal_current(str(f.index[-1].date())) else None
    return build_plan(f,h,reference)


def _navigate_stock(t):
    st.session_state['desk_lookup_value']=t;st.session_state['desk_route_pending']='個股診斷';st.rerun()


def table(data,key,show_chips=False):
    if data is None or data.empty:st.info('目前沒有符合篩選條件的標的。');return
    view=data.copy()
    view['股票']=view.ticker.astype(str)+' '+view['name'].astype(str)
    view['族群']=view['group'].astype(str)
    view['買進區']=view.apply(lambda r:f"{fmt.price(r.get('buy_low'))}～{fmt.price(r.get('buy_high'))}",axis=1)
    fields={'股票':'股票','族群':'族群','horizon_label':'週期','score':'模型分數','price':'參考價',
            'change':'漲跌%','買進區':'買進區','status':'操作判斷','alert':'防守價','take1':'2R參考價'}
    if 'live_rank' in view:fields={'live_rank':'盤中順位',**fields,'volume_ratio':'預估量比','quote_time':'報價時間'}
    if show_chips:fields.update({'chip_tone':'籌碼','reasons':'主要依據'})
    keep=[k for k in fields if k in view];visible=view[keep].rename(columns=fields)
    if '漲跌%' in visible:visible['漲跌%']=visible['漲跌%']*100
    numeric={c:'{:.2f}' for c in ['模型分數','參考價','防守價','2R參考價','預估量比'] if c in visible}
    if '漲跌%' in visible:numeric['漲跌%']='{:+.2f}%'
    styled=visible.style.format(numeric,na_rep='—')
    if '漲跌%' in visible:styled=styled.map(lambda v:'color:#c83a40' if pd.notna(v) and v>0 else 'color:#148759' if pd.notna(v) and v<0 else '',subset=['漲跌%'])
    event=st.dataframe(styled,hide_index=True,width='stretch',height=min(450,75+len(visible)*36),on_select='rerun',selection_mode='single-row',key=key)
    selected=event.selection.rows
    if selected and st.button('查看選取個股的完整診斷',key=key+'_open'):_navigate_stock(str(view.iloc[selected[0]].ticker))
    st.download_button('下載這份清單 CSV',view.to_csv(index=False).encode('utf-8-sig'),file_name='AlphaRadar_'+key+'.csv',mime='text/csv',key=key+'_csv')


def close_rows(B,snapshots,horizons,holdings=()):
    out=[]
    for h in horizons:
        snap=snapshots[h]
        for r in snap.picks:
            t=r['ticker'];f=frame(B.meta['desk_content_id'],B.tickers.index(t),True,B);p=build_plan(f,h)
            if f.empty:continue
            role='keep' if t in holdings else 'pick';ready=snap.ready and signal_current(snap.date)
            d=decision(p,role=role,mode='close',data_ready=ready,synthetic=B.meta.get('synthetic',False))
            cs=chips(f)
            out.append({**r,'horizon':h,'horizon_label':HZ[h],'price':float(f.close.iloc[-1]),
                'change':float(f.close.iloc[-1]/f.close.iloc[-2]-1) if len(f)>1 else np.nan,
                'status':d.label,'code':d.code,'reason':d.reason,'can_enter':False,'chip_tone':cs['tone'],
                **{k:getattr(p,k) if p else np.nan for k in ['buy_low','buy_high','chase','alert','take1','take2']}})
    result=pd.DataFrame(out)
    if len(horizons)>1 and not result.empty:
        order={h:k for k,h in enumerate(['mid','long','short'])};result['_order']=result.horizon.map(order)
        labels=result.groupby('ticker').horizon_label.agg(lambda x:'／'.join(dict.fromkeys(x)))
        result=result.sort_values(['_order','rank']).drop_duplicates('ticker');result['horizon_label']=result.ticker.map(labels)
    return result.reset_index(drop=True)


def overview(B,E,holdings):
    st.markdown(title('選股總覽','先選操作週期，再看買進區與防守價；同一檔跨模型入選只列一次。'),unsafe_allow_html=True)
    global_panel(B)
    snaps={h:E.snapshot(h,holdings) for h in HZ};allrows=close_rows(B,snaps,list(HZ),holdings)
    counts=[len(snaps[h].picks) for h in HZ]
    st.markdown(stats([('綜合觀察',str(len(allrows))+' 檔','跨週期去除重複'),('短線候選',str(counts[0])+' 檔','每5個交易日檢查續抱'),
                      ('中線候選',str(counts[1])+' 檔','分4批、每週檢查一批'),('長線候選',str(counts[2])+' 檔','分12批、每週檢查一批')]),unsafe_allow_html=True)
    a,b,c=st.columns([1.3,1.6,1]);choice=a.segmented_control('操作週期',['綜合']+list(HZ.values()),default='綜合',key='overview_h') or '綜合'
    groups=sorted(allrows.group.unique()) if not allrows.empty else []
    group=b.selectbox('族群篩選',['全部族群']+groups,key='overview_group')
    show=c.toggle('顯示籌碼與依據',value=False,key='overview_chips')
    hs=list(HZ) if choice=='綜合' else [next(h for h,v in HZ.items() if v==choice)]
    rows=close_rows(B,snaps,hs,holdings)
    if group!='全部族群' and not rows.empty:rows=rows[rows.group==group]
    table(rows,'overview',show)
    st.caption('模型分數是母池相對百分位，並非勝率。買區依收盤技術面計算；進場前到盤中雷達核對成交價。2R為風險距離推算的減碼參考。')
    for h,s in snaps.items():
        if s.notes:st.caption(HZ[h]+'：'+'；'.join(s.notes))
    with st.expander('現在怎麼做',expanded=False):
        st.write('盤中：開「盤中雷達」找可分批布局。盤後：開「明日布局」確認下一交易日觀察名單。輸入個股代號或公司名可查三個週期的判斷。')


def radar(B,E,holdings,watch):
    st.markdown(title('盤中雷達','基本模型截至最近收盤；盤中以有效成交價、買區距離及價量狀態重排候選。'),unsafe_allow_html=True)
    global_snap=global_panel(B,expanded=True)
    c1,c2,c3,c4=st.columns([1.7,1,1,1]);hs=c1.multiselect('追蹤週期',list(HZ),default=list(HZ),format_func=HZ.get,key='radar_h')
    count=c2.selectbox('掃描上限',[50,100,150],index=1,key='radar_n');interval=c3.selectbox('更新間隔（秒）',[30,60,120],index=1,key='radar_interval')
    auto=c4.toggle('自動更新',value=phase()=='盤中' and not B.meta.get('synthetic'),key='radar_auto')
    @st.fragment(run_every=str(interval)+'s' if auto else None)
    def live_board():
        st.button('立即重新掃描',key='radar_refresh',on_click=quote_batch.clear)
        snaps={h:E.snapshot(h,holdings) for h in hs}
        universe=list(dict.fromkeys(holdings+watch))
        # Round-robin across horizons prevents a first-horizon-only scan cap.
        ranked={h:s.table.loc[s.table.valid].head(50).ticker.tolist() for h,s in snaps.items()}
        for k in range(50):
            for h in hs:
                if k<len(ranked[h]) and ranked[h][k] not in universe:universe.append(ranked[h][k])
        universe=universe[:count];quotes,error=fetch_quotes(B,universe)
        if error:st.caption(error)
        curves={t:frame(B.meta['desk_content_id'],B.tickers.index(t),True,B) for t in universe if t in E.idx}
        pieces=[]
        for h,s in snaps.items():
            s.table=s.table[s.table.ticker.isin(universe)].copy()
            plans={t:build_plan(curves[t],h,quotes.get(t,{}).get('prev')) for t in curves}
            part=scan_rows(s,plans,quotes,curves,synthetic=B.meta.get('synthetic',False),top_k=E.cfg['research']['top_k'],global_snapshot=global_snap)
            if not part.empty:pieces.append(part)
        rows=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
        if not rows.empty:
            labels=rows.groupby('ticker').horizon.agg(lambda x:'／'.join(HZ[h] for h in dict.fromkeys(x)))
            rows=rows.sort_values(['can_enter','priority'],ascending=[False,False]).drop_duplicates('ticker');rows['horizon_label']=rows.ticker.map(labels)
        ready=int(rows.can_enter.sum()) if not rows.empty else 0;fresh=sum(quote_check(q)[0] for q in quotes.values())
        st.markdown(stats([('符合布局條件',str(ready)+' 檔','需同時通過模型、價位與成交時間'),('有效成交報價',str(fresh)+' 檔',f'本次掃描{len(universe)}檔'),
                          ('等拉回',str(int(rows.code.eq('chase').sum()) if not rows.empty else 0)+' 檔','超過買區或追價上限'),('更新時間',taipei().strftime('%H:%M:%S'),f'每{interval}秒重新檢視')]),unsafe_allow_html=True)
        only=st.toggle('只看可分批布局',value=False,key='radar_only')
        table(rows[rows.can_enter] if only and not rows.empty else rows,'radar')
        if fresh==0 and not B.meta.get('synthetic'):st.info('尚無可用的盤中成交報價。收盤清單仍可查看；目前不顯示可立即買入。')
        st.caption('盤中順位屬執行排序；法人與營收使用已公布資料。預估量比按當日已交易時間線性換算，並非完整日成交量。')
    live_board()


def tomorrow(B,E,holdings):
    st.markdown(title('明日布局','把盤後名單整理成下一交易日的限價、配比與進出場檢查。'),unsafe_allow_html=True)
    global_panel(B,expanded=True)
    date=str(B.sig_dates[-1].date());now=taipei()
    if phase()=='盤後' and date!=str(now.date()) and not B.meta.get('synthetic'):
        st.warning('目前顯示前次名單（'+date+'）；本日盤後資料尚未更新完成，請更新資料後再確認明日布局。')
    hs=st.multiselect('布局週期',list(HZ),default=['mid','long'],format_func=HZ.get,key='tomorrow_h')
    snaps={h:E.snapshot(h,holdings) for h in hs};rows=close_rows(B,snaps,hs,holdings)
    table(rows,'tomorrow',True)
    st.markdown('### 執行順序')
    st.write('① 盤後確認名單日期與本期換倉金額。　② 開盤後只在買區內分3筆投入，先做1/3。　③ 跳空超過追價上限先等；低於買區先等止跌。　④ 13:25前仍未符合條件，當日保留資金。')
    st.markdown('### 出場檢查')
    st.write('跌破防守價：檢視減碼與基本面變化。到2R參考價：可評估先減1/3，剩餘部位依模型續抱門檻追蹤。跌出續抱門檻：在下次固定檢查日檢視換股。')
    thresholds='；'.join(HZ[h]+'前'+str(s.keep_rank)+'名' for h,s in snaps.items())
    st.caption('目前續抱門檻：'+thresholds+'。價位及分批規則為操作覆蓋層，尚未納入原模型績效。')


def stock_detail(B,E,t,h='mid',key='detail'):
    j=E.idx[t];group=chain_label(t,B.meta['sectors'][j]);raw=frame(B.meta['desk_content_id'],j,True,B)
    st.markdown(f'<div class="desk-stock"><span class="code">{esc(t)}</span><span class="name">{esc(B.meta["names"][j])}</span>'
                f'<span class="desk-tag">{esc(group["group"])}</span><span class="desk-tag">{esc(group["segment"])}</span></div>',unsafe_allow_html=True)
    global_panel(B)
    if raw.empty:st.info('尚無這檔股票的K線資料，請更新看盤資料包。');return
    quotes={}
    if phase()=='盤中' or B.meta.get('synthetic'):quotes,_=fetch_quotes(B,[t])
    q=quotes.get(t);p=stock_plan(B,t,h,q);holdings=resolve(B,st.session_state.get('desk_held',''))
    diagnoses=[];verdicts={}
    for hz in HZ:
        s=E.snapshot(hz,holdings);rec=s.table.loc[s.table.ticker.eq(t)]
        if rec.empty:diagnoses.append({'週期':HZ[hz],'模型排名':'母池外','判斷':'本期未評分','依據':'仍可檢視技術及籌碼'});continue
        r=rec.iloc[0];role=r.role if bool(r.valid) else 'none'
        pp=stock_plan(B,t,hz,q);mode='live' if phase()=='盤中' else 'close'
        d=decision(pp,q,role,mode=mode,data_ready=s.ready and signal_current(s.date),synthetic=B.meta.get('synthetic',False))
        verdicts[hz]=d
        rank=f"第{int(r['rank'])}名" if pd.notna(r['rank']) else '條件未通過'
        diagnoses.append({'週期':HZ[hz],'模型排名':rank,'判斷':d.label,'依據':r.why or r.reasons})
    st.dataframe(pd.DataFrame(diagnoses),hide_index=True,width='stretch')
    if h in verdicts:
        d=verdicts[h]
        st.markdown('<div class="desk-verdict"><b>'+esc(HZ[h]+'｜'+d.label)+'</b><br>'+esc(d.reason)+'</div>',unsafe_allow_html=True)
    px=q.get('price',np.nan) if q else float(raw.close.iloc[-1]);verified,why=quote_check(q)
    if not verified:st.caption('參考價 '+fmt.price(px)+'｜收盤資料 '+str(raw.index[-1].date())+'｜'+why)
    if p:
        st.markdown(stats([('買進區',fmt.price(p.buy_low)+'～'+fmt.price(p.buy_high),'成交價進入區間才檢查第一筆'),('追價上限',fmt.price(p.chase),'高於此價先等拉回'),
                          ('防守價',fmt.price(p.alert),'跌破時檢視減碼'),('2R參考減碼',fmt.price(p.take1),'依風險距離推算，非報酬預測')]),unsafe_allow_html=True)
    a,b,c,d=st.columns([1,1.4,2.3,1]);period=a.segmented_control('K線週期',['D','W','M'],format_func=lambda x:{'D':'日K','W':'週K','M':'月K'}[x],default='D',key=key+'_per') or 'D'
    ma=b.multiselect('均線',list(ta.MA_LINES),default=[5,20,60],format_func=lambda n:ta.MA_LINES[n] if period=='D' else str(n)+('週' if period=='W' else '月')+'均線',key=key+'_ma')
    options={'boll':'布林通道','volume':'成交量','kd':'KD','macd':'MACD','inst':'三大法人','margin':'資券餘額','holders':'大戶持股','broker':'主力分點'}
    panels=c.multiselect('指標與籌碼',list(options),default=['boll','volume','kd','macd'],format_func=options.get,key=key+'_ind')
    span=d.selectbox('顯示區間',['1個月','3個月','半年','1年','2年'],index=2,key=key+'_span')
    source=raw.copy();broker=st.session_state.get('desk_broker')
    if broker is not None:
        sub=broker[broker.ticker.eq(t)].set_index('date');source['broker_net']=sub.net_lots.reindex(source.index)
    if 'broker' in panels and ('broker_net' not in source or source.broker_net.notna().sum()==0):
        st.caption('主力分點尚未接入；大戶持股另外使用集保週資料。')
    ind=ta.indicators(ta.resample(source,period))
    days={'1個月':22,'3個月':66,'半年':132,'1年':252,'2年':520}[span]
    bars=max(8,int(days/{'D':1,'W':5,'M':21}[period]))
    from .charts import kline
    fig=kline(ind,mas=ma,boll='boll' in panels,panels=[x for x in panels if x!='boll'],plan=p if period=='D' else None,
              bars=bars,dark=st.session_state.get('desk_dark',False),ma_unit=period)
    st.plotly_chart(fig,width='stretch',theme=None,config={'displaylogo':False,'scrollZoom':True},key=key+'_chart')
    unavailable=[str(n)+('日' if period=='D' else '週' if period=='W' else '月') for n in ma if ind['ma'+str(n)].notna().sum()==0]
    if unavailable:st.caption('歷史長度不足，尚無 '+ '、'.join(unavailable)+'均線。')
    st.caption('K線以最新原始收盤價對齊的調整歷史顯示；日K的5／20／60／120／240均線依序為週／月／季／半年／年線。週月K以週月為計算單位。')
    cs=chips(raw);left,right=st.columns([1.45,1])
    with left:
        st.markdown('### 三大法人各自進出')
        st.dataframe(cs['table'].style.format({'最新日(張)':'{:+,.0f}','近5日(張)':'{:+,.0f}','近20日(張)':'{:+,.0f}'},na_rep='—'),hide_index=True,width='stretch')
        st.write('；'.join(cs['notes']))
        st.caption('法人為盤後公布的日資料，盤中不會把尚未公布數字當成當日買賣超。')
    with right:
        st.markdown('### 融資融券與大戶')
        for label,value in cs['balances'].items():
            if value['value'] is None:st.write(label+'：未取得')
            else:st.write(f"{label}：{value['value']:,.0f}張｜資料{value['date']}")
        holder=cs['holders']
        if holder['value'] is not None:
            change=f"{holder['change_pp']:+.2f}百分點" if holder['change_pp'] is not None else '前期資料不足'
            st.write(f"大戶≥400張：{holder['value']:.2f}%｜較前週樣本{change}");st.caption('基準日 '+str(holder['date'])+('' if holder['exact_date'] else '（舊資料包，原始週基準日未保留）'))
        else:st.write('大戶持股：未取得集保週資料')
        if broker is None:st.caption('主力分點：尚無資料')
        else:
            sub=broker[broker.ticker.eq(t)].sort_values('date')
            if not sub.empty:st.write(f"主力分點淨買超：{sub.net_lots.iloc[-1]:+,.0f}張｜{sub.date.iloc[-1].date()}");st.caption(str(sub.source.iloc[-1]))


def lookup(B,E):
    st.markdown(title('個股診斷','輸入股票代號或公司名稱，同時查看短中長線排名、可布局條件與價位。'),unsafe_allow_html=True)
    if 'desk_lookup_value' in st.session_state:st.session_state['desk_query']=st.session_state.pop('desk_lookup_value')
    value=st.text_input('股票代號／公司名稱',placeholder='例如 2330、台積電、聯發科',key='desk_query')
    matches=search_stocks(B,value)
    if not value:st.info('輸入代號或公司名稱開始查詢。');return
    if not matches:st.info('這檔股票尚未納入目前資料包；更新資料後可查模型，系統不替母池外股票編造分數。');return
    selected=matches[0] if len(matches)==1 else st.selectbox('符合的股票',matches,format_func=lambda r:r['ticker']+' '+r['name']+'｜'+chain_label(r['ticker'],B.meta['sectors'][r['j']])['group'],key='desk_match')
    h=st.segmented_control('交易計畫週期',list(HZ),format_func=HZ.get,default='mid',key='lookup_h') or 'mid'
    stock_detail(B,E,selected['ticker'],h)


def capital_page(B,E,holdings):
    st.markdown(title('資金配置','先看目標持倉，再看本期投入；跨模型同一檔合併計算，另保留費用與現金。'),unsafe_allow_html=True)
    a,b=st.columns([1,2]);capital=a.number_input('可用投資資金（元）',min_value=10000,max_value=1_000_000_000,value=1_000_000,step=10000,key='desk_capital')
    profile=b.segmented_control('配置範例',list(PROFILES)+['自訂'],default='均衡配置',key='capital_profile') or '均衡配置'
    if profile=='自訂':
        columns=st.columns(5);labels={'cash':'現金','0050':'0050','short':'短線','mid':'中線','long':'長線'}
        weights={k:col.number_input(labels[k]+' %',min_value=0,max_value=100,value=v,step=5,key='custom_'+k)/100 for col,(k,v) in zip(columns,{'cash':10,'0050':40,'short':0,'mid':30,'long':20}.items())}
    else:weights=PROFILES[profile].copy()
    controls=st.columns(3);max_name=controls[0].slider('單一個股上限 %',5,20,10,key='max_name')/100
    max_group=controls[1].slider('同族群上限 %',10,50,25,key='max_group')/100
    risk=controls[2].selectbox('每檔至防守價的風險預算',[.25,.5,1.0],index=1,format_func=lambda x:str(x)+'% 總資金',key='risk_budget')/100
    snaps={h:E.snapshot(h,holdings) for h in HZ};selections={h:[p['ticker'] for p in s.picks] if s.ready else [] for h,s in snaps.items()}
    plans={};prices={};groups={}
    for h,s in snaps.items():
        for r in s.picks:
            t=r['ticker'];p=stock_plan(B,t,h)
            if p is None:continue
            # Shared names use the tighter defense and one consolidated budget.
            if t not in plans or p.alert>plans[t].alert:plans[t]=p
            prices[t]=plans[t].entry;groups[t]=r['group']
    exposures={h:s.exposure if signal_current(s.date) else 0 for h,s in snaps.items()}
    try:target,ledger=allocation(capital,weights,selections,plans,prices,exposures=exposures,max_name=max_name,max_group=max_group,risk_fraction=risk,groups=groups)
    except ValueError as e:st.error(str(e));return
    etf_label='0050核心（選配）' if weights.get('0050',0)>0 else '0050核心（未採用）'
    etf_hint='已持有核心者不重複新增' if weights.get('0050',0)>0 else '0050不是必要條件'
    st.markdown(stats([('目標個股含費用',f"{ledger['stocks_including_fees']:,.0f} 元",'目標持倉，非每週新增投入'),(etf_label,f"{ledger['etf']:,.0f} 元",etf_hint),
                      ('目標保留現金',f"{ledger['cash']:,.0f} 元",'含風險及集中度限制後的餘款'),('目標資金核對','100% 相符' if ledger['reconciled'] else '待確認','個股＋費用＋ETF＋現金＝總資金')]),unsafe_allow_html=True)
    mode=st.segmented_control('顯示金額',['目標持倉','本週新一批'],default='本週新一批',key='alloc_mode') or '本週新一批'
    if mode=='本週新一批':
        batch_weights={**weights,**{h:weights.get(h,0)/snaps[h].batch_count for h in HZ}}
        batch_weights['cash']=1-batch_weights.get('0050',0)-sum(batch_weights.get(h,0) for h in HZ)
        target,batch=allocation(capital,batch_weights,selections,plans,prices,exposures=exposures,max_name=max_name,max_group=max_group,risk_fraction=risk,groups=groups)
        st.info('本週這一批的個股買入預算（含費用）：'+f"{batch['stocks_including_fees']:,.0f} 元。"+('0050核心及其他批次不在本週重複買入。' if weights.get('0050',0)>0 else '本配置未指定0050，其餘留待後續批次或現金。'))
        st.caption('本週僅顯示一批新投入：短線1批、中線1/4、長線1/12；其他資金留待後續批次。已有部位先計入目標，不代表每週再投入整份總額。')
    if target.empty:st.info('目前尚無可換算的合格股票或價位。')
    else:
        names={t:n for t,n in zip(B.tickers,B.meta['names'])};target['股票']=target.ticker.map(lambda t:t+' '+names.get(t,t))
        cols=['股票','group','sources','amount','fee_buffer','weight','shares','lots','odd_shares','price','buy_low','buy_high','alert','risk_amount']
        show=target[cols].rename(columns={'group':'族群','sources':'策略來源','amount':'買股金額','fee_buffer':'費用緩衝','weight':'占總資金','shares':'股數','lots':'整張','odd_shares':'零股','price':'掛單試算價','buy_low':'買區下緣','buy_high':'買區上緣','alert':'防守價','risk_amount':'至防守風險'})
        formats={'買股金額':'{:,.0f}','費用緩衝':'{:,.0f}','占總資金':'{:.1%}','至防守風險':'{:,.0f}'}
        formats.update({k:fmt.price for k in ['掛單試算價','買區下緣','買區上緣','防守價']})
        st.dataframe(show.style.format(formats,na_rep='—'),hide_index=True,width='stretch')
        st.download_button('下載配比與下單試算',show.to_csv(index=False).encode('utf-8-sig'),'AlphaRadar_allocation.csv','text/csv')
    st.caption('0050為選配，不是必要條件。範例配比供調整；此為本金試算，尚未扣除既有持股股數。含買進手續費與滑價緩衝，未建模券商最低費用；同族群上限僅計直接個股，未穿透ETF成分股。實際成交前再核對數量。')


def data_page(B,E,report,source):
    st.markdown(title('資料與策略','查看資料日期、模型依據、原始報告與更新狀態。'),unsafe_allow_html=True)
    a,b=st.columns([1.4,1]);
    with a:
        st.write('來源：'+source+'｜最近訊號 '+str(B.sig_dates[-1].date()))
        choices=['v27','source','price'];choice=st.selectbox('模型組合',choices,
            format_func=lambda x:{'v27':'V27十面向融合（預設候選）','source':'來源定稿模型（對照）','price':'價格確認（基本面尚未補齊）'}[x],
            index=choices.index(E.profile) if E.profile in choices else 0,key='data_model_choice')
        if choice!=E.profile and st.button('套用模型組合'):
            st.session_state['desk_profile']=choice;st.rerun()
        if st.button('重新載入最新資料',type='primary'):
            st.cache_data.clear();st.cache_resource.clear();st.rerun()
        if st.button('重建本機看盤快取（沿用現有行情）'):
            from .desk_update import rebuild
            try:
                path=rebuild(ROOT);st.session_state['desk_source_path']=str(path);st.rerun()
            except Exception as e:st.error(str(e))
        if st.button('補上最近交易日的正式價量與籌碼'):
            from .desk_update import start_latest
            try:start_latest(ROOT);st.info('已開始背景補齊最近交易日；完成後重新載入資料。')
            except Exception as e:st.error(str(e))
        from .desk_update import update_state
        @st.fragment(run_every='4s')
        def progress():
            state=update_state(ROOT)
            if state:st.caption(str(state.get('status'))+'｜'+str(state.get('message')))
        progress()
    with b:
        up=st.file_uploader('替換看盤資料包',type=['npz'],key='data_bundle')
        if up and st.button('採用新資料'):
            try:
                load_bundle(up.getvalue());p=ROOT/'artifacts'/'imported_bundle.npz';p.write_bytes(up.getvalue());st.session_state['desk_source_path']=str(p);st.rerun()
            except Exception as e:st.error(str(e))
        rfile=st.file_uploader('補入原績效報告（ZIP或JSON）',type=['zip','json'],key='data_report')
        if rfile:
            try:st.session_state['desk_report']=report_summary(rfile.getvalue());st.success('已載入報告，僅作既有績效描述。')
            except Exception as e:st.error(str(e))
        supplied=st.file_uploader('主力分點資料（選用CSV）',type=['csv'],key='data_broker')
        if supplied:
            try:
                broker=validate_broker(pd.read_csv(io.BytesIO(supplied.getvalue()),dtype={'ticker':str}))
                if st.button('採用主力分點資料'):st.session_state['desk_broker']=broker;st.success('已載入來源清楚的分點資料。')
            except Exception as e:st.error(str(e))
        template='date,ticker,buy_lots,sell_lots,net_lots,source\n'
        st.download_button('下載分點欄位範本',template,'broker_template.csv','text/csv')
    snaps=[E.snapshot(h) for h in HZ]
    st.dataframe(pd.DataFrame([{'週期':HZ[s.horizon],'模型':s.title,'資料日':s.date,'母池':s.pool,'續抱門檻':'前'+str(s.keep_rank)+'名','資金批數':s.batch_count,'合格':'可評分' if s.ready else '資料待補齊'} for s in snaps]),hide_index=True,width='stretch')
    with st.expander('使用方式與進出場規則',expanded=True):
        st.write('短線：用較慢的成長／動能訊號，每5個交易日檢查。中線：成長與相對強度融合，分4批。長線：趨勢錨與領先股融合，分12批。V27再以20%十面向分散先驗降低單一因子依賴；盤中另有最多12%海外情境。')
        st.write('十面向：營收成長、獲利品質、估值、相對強勢、趨勢結構、法人籌碼、融資與持股、下行風險、流動性容量、科斯托蘭尼市場行為。0050可設為0%。')
        st.write('盤中：核對當日成交時間、模型資料、買區及流動性，符合條件才顯示可分批布局。盤後：確認本日資料更新，整理下一交易日限價與防守價。')
        st.write('價格計畫：ATR與均線產生買區、防守與2R/3R參考；這些是操作規則，尚未納入原回測。看盤軟體不會替你下單。')
    rep=st.session_state.get('desk_report') or report
    with st.expander('既有模型績效與資料限制'):
        if rep:
            rows=[]
            for h,x in rep.get('horizons',{}).items():
                ev=x.get('eval',{});rows.append({'週期':HZ.get(h,h),'全期CAGR':ev.get('oos_cagr'),'同期0050':ev.get('oos_bench_cagr'),'最大回撤':ev.get('oos_mdd')})
            st.dataframe(pd.DataFrame(rows).style.format({'全期CAGR':'{:.1%}','同期0050':'{:.1%}','最大回撤':'{:.1%}'},na_rep='—'),hide_index=True,width='stretch')
        else:st.write('尚未載入這份資料的績效報告；不顯示預設報酬或虛構勝率。')
        st.write('先前已看過的歷史績效屬既有研究。新介面的盤中排序、價位與集中度限制仍需獨立驗證。營收代理公告日、價格修訂及歷史名冊可能影響結果。')
        st.json(B.meta.get('data_quality',{}))
    with st.expander('V27十面向模型驗證'):
        validation=(rep or {}).get('_v27_validation') if rep else None
        if validation:
            rows=[]
            for result in validation.get('results',[]):
                oos=result.get('windows',{}).get('oos',{})
                rows.append({'週期':HZ.get(result.get('horizon'),result.get('horizon')),'母池':result.get('pool'),
                    '候選驗收':'通過' if result.get('accepted') else '未通過',
                    '來源OOS CAGR':oos.get('source_cagr'),'V27 OOS CAGR':oos.get('v27_cagr'),
                    'CAGR差':oos.get('delta_cagr'),'來源MDD':oos.get('source_mdd'),'V27 MDD':oos.get('v27_mdd')})
            st.dataframe(pd.DataFrame(rows).style.format({c:'{:+.1%}' for c in ['來源OOS CAGR','V27 OOS CAGR','CAGR差','來源MDD','V27 MDD']},na_rep='—'),hide_index=True,width='stretch')
            st.caption(str(validation.get('protocol',''))+'；不自動把單次通過解讀為未來報酬保證。')
        else:st.write('尚未用這份正式資料驗證V27；資料流程可執行 `python cli.py validate-v27` 產生比較報告。')
    with st.expander('版本與計算依據'):
        st.json({'product':'Alpha Radar V27.0 Global','source_engine':'Claude Kosto 1.6 + Alpha V24/V25 price hypotheses',
                 'models':{s.horizon:{'id':s.model_id,'weights':s.weights} for s in snaps},'automatic_orders':False})
    st.download_button('下載簡要操作指南',(ROOT/'START_HERE.txt').read_text() if (ROOT/'START_HERE.txt').exists() else '先確認資料日期，再依盤中雷達或明日布局執行。','AlphaRadar_V26_START_HERE.txt','text/plain')


def main():
    st.set_page_config(page_title='Alpha Radar PRO Global｜台股決策台',page_icon='▥',layout='wide',initial_sidebar_state='expanded')
    dark=st.sidebar.toggle('深色模式',value=False,key='desk_dark');st.markdown(css(dark),unsafe_allow_html=True)
    try:B,report,source=get_source()
    except Exception as e:header();st.error('資料載入未完成：'+str(e));setup();return
    if 'desk_profile' not in st.session_state:st.session_state['desk_profile']='price' if B is not None and B.meta.get('legacy_alpha_import') else 'v27'
    if st.session_state.get('desk_profile')=='fusion':st.session_state['desk_profile']='v27'
    header(B,source)
    if st.session_state.get('desk_route_pending'):
        st.session_state['desk_page']=st.session_state.pop('desk_route_pending')
    page=st.radio('功能',PAGES,horizontal=True,key='desk_page',label_visibility='collapsed')
    if B is None:setup();return
    cfg=content_settings(B);E=engine(B.meta['desk_content_id'],st.session_state['desk_profile'],json.dumps(cfg,sort_keys=True),B)
    st.sidebar.markdown('**持股與自選**')
    held_text=st.sidebar.text_input('我的持股（代號或名稱，逗號分隔）',key='desk_held',placeholder='2330, 聯發科')
    watch_text=st.sidebar.text_input('自選觀察（不視為已持有）',key='desk_watch',placeholder='2382, 6488')
    holdings=resolve(B,held_text);watch=resolve(B,watch_text)
    st.sidebar.caption(f'持股{len(holdings)}檔・自選{len(watch)}檔；只用於本次操作判斷。')
    st.sidebar.caption('名單資料 '+str(B.sig_dates[-1].date())+'｜'+str(len(B.tickers))+' 檔')
    if B.meta.get('legacy_alpha_import'):st.sidebar.info(B.meta.get('legacy_alpha_note'))
    if B.meta.get('synthetic') and st.sidebar.button('回到正式資料'):
        st.session_state['desk_demo']=False;st.session_state.pop('desk_source_path',None);st.rerun()
    if page=='選股總覽':overview(B,E,holdings)
    elif page=='盤中雷達':radar(B,E,holdings,watch)
    elif page=='明日布局':tomorrow(B,E,holdings)
    elif page=='個股診斷':lookup(B,E)
    elif page=='資金配置':capital_page(B,E,holdings)
    else:data_page(B,E,report,source)
    st.markdown('<div class="desk-footer">ALPHA RADAR PRO · 決策依據與資料時間可查核。價位為操作參考，市場價格可能跳空或無法成交；系統不自動下單。</div>',unsafe_allow_html=True)
