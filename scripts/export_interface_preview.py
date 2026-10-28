"""Export a self-contained, explicitly synthetic UI preview. No network or live signals."""
from pathlib import Path
import sys,json,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from kosto.desk_data import build_demo,content_settings,chart_frame
from kosto.desk_models import DeskEngine
from kosto.desk_plan import build_plan,HZ,PROFILES
from kosto.desk_chips import chips
from kosto import ta
from kosto.desk_style import css

ROOT=Path(__file__).resolve().parents[1]


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(pd.Timestamp,)):return str(x.date())
    return x


def export(out):
    B=build_demo(ROOT/'artifacts'/'demo');E=DeskEngine(B,content_settings(B));snaps={h:E.snapshot(h) for h in HZ}
    selections={h:[p['ticker'] for p in s.picks] for h,s in snaps.items()}
    tickers=list(dict.fromkeys(t for h in HZ for t in selections[h]));stocks={}
    for t in tickers:
        j=B.tickers.index(t);f=chart_frame(B,j);p={h:build_plan(f,h).as_dict() for h in HZ}
        by_h={h:s.table.loc[s.table.ticker.eq(t)].iloc[0].to_dict() for h,s in snaps.items() if s.table.ticker.eq(t).any()}
        rec=next(iter(by_h.values()));charts={}
        for period in ['D','W','M']:
            q=ta.indicators(ta.resample(f,period)).tail({'D':252,'W':110,'M':36}[period])
            cols=['open','high','low','close','volume','k','d','dif','dem','osc','bb_up','bb_dn','foreign','trust','dealer','mbal','sbal','big400']+['ma'+str(n) for n in ta.MA_LINES]
            charts[period]={'x':[str(x.date()) for x in q.index],**{c:q[c].round(4).tolist() for c in cols if c in q}}
        cs=chips(f)
        stocks[t]={'ticker':t,'name':B.meta['names'][j],'group':rec['group'],'segment':rec['segment'],
                   'price':float(f.close.iloc[-1]),'change':float(f.close.iloc[-1]/f.close.iloc[-2]-1),
                   'scores':by_h,'plans':p,'charts':charts,'chips':{**cs,'table':cs['table'].to_dict('records')}}
    payload=clean({'stocks':stocks,'selections':selections,'profiles':PROFILES,'exposures':{h:s.exposure for h,s in snaps.items()},'date':str(B.sig_dates[-1].date())})
    script_data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    html=HTML.replace('%%CSS%%',css(False)).replace('%%PLOTLY%%',get_plotlyjs()).replace('%%DATA%%',script_data)
    Path(out).parent.mkdir(exist_ok=True,parents=True);Path(out).write_text(html,encoding='utf-8')
    return out


HTML='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpha Radar PRO Global V27｜互動介面預覽</title>%%CSS%%
<style>
body{margin:0;background:var(--desk-bg);color:var(--desk-text);font-family:Inter,"PingFang TC","Noto Sans CJK TC","Microsoft JhengHei","Heiti TC",sans-serif;font-size:14px}
aside{position:fixed;inset:0 auto 0 0;width:205px;background:white;border-right:1px solid var(--desk-line);padding:30px 20px;box-sizing:border-box}
main{margin-left:205px;padding:20px 35px 40px;max-width:1500px}.aside-label{font-size:12px;color:#64758b;margin-top:28px;margin-bottom:8px}
input,select,button{font:inherit;border:1px solid #dce3ec;background:white;border-radius:5px;padding:9px;color:#14253d;box-sizing:border-box}
button{cursor:pointer}button:hover{border-color:#087e76}.primary{background:#087e76;color:white;border:0}input[type=checkbox]{accent-color:#087e76}
nav{display:flex;gap:5px;border-bottom:1px solid #dce3ec;margin:10px 0 20px;flex-wrap:wrap}nav button{background:transparent;border:0;border-radius:0;border-bottom:3px solid transparent;padding:12px 15px}nav button.active{border-color:#087e76;color:#087e76;font-weight:700}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:15px 0}.wrap{background:white;border:1px solid #dce3ec;border-radius:5px;overflow:auto}
table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:12px}th,td{text-align:right;padding:11px 12px;border-bottom:1px solid #e8edf3;font-variant-numeric:tabular-nums}th{background:#f6f8fb;color:#64758b;font-weight:500}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}tbody tr{cursor:pointer}tbody tr:hover{background:#edf7f5}.up{color:#c83a40}.down{color:#148759}.tag{font-size:11px;border-radius:3px;padding:3px 6px;background:#e8f4f2;color:#087e76}
.panel{background:white;border:1px solid #dce3ec;border-radius:5px;padding:18px;margin:14px 0}.two{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}.muted{color:#64758b;font-size:12px;line-height:1.8}#chart{background:white;border:1px solid #dce3ec;border-radius:5px;margin-top:12px}.row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}h3{font-size:16px}section{display:none}section.active{display:block}.num{font-variant-numeric:tabular-nums}.note{background:#eaf4f5;border-left:3px solid #087e76;padding:12px;margin:12px 0;font-size:13px;line-height:1.8}
@media(max-width:760px){aside{display:none}main{margin:0;padding:14px}.two{grid-template-columns:1fr}nav button{padding:10px 7px;font-size:13px}.desk-top{gap:8px}.desk-meta{font-size:9px}}
</style>
<aside><div style="font-weight:800;letter-spacing:1.2px">ALPHA RADAR <span style="color:#087e76">PRO</span></div><p class="muted">台股決策台 V27.0 GLOBAL</p><div class="aside-label">快速個股查詢</div><input id="quick" placeholder="代號或公司名" style="width:100%"><p class="muted">可試輸入表格中的代號，或點選任一列。</p><div class="aside-label">資料狀態</div><b>操作示範</b><p class="muted">合成股價・合成籌碼<br>不連線、不產生真實買入訊號</p><div class="aside-label">正式版功能</div><p class="muted">60秒盤中掃描<br>盤後明日名單<br>短中長模型診斷<br>配比與風險預算<br>日／週／月K與籌碼</p></aside>
<main><div class="desk-top"><div><div class="desk-wordmark">ALPHA RADAR <span>PRO</span></div><div class="desk-product">台股決策台 · 短中長線選股與交易計畫</div></div><div class="desk-meta">V27.0 GLOBAL 互動預覽<br>2026.10.28 · 合成資料</div></div>
<div class="desk-demo">操作預覽｜所有價格、排名、交易價位與籌碼都是合成數據。正式資料及即時掃描請使用完整程式版。</div>
<div class="desk-strip"><span>資料 <b>合成示範</b></span><span>報價 <b>未接即時</b></span><span>法人 <b>盤後日資料</b></span><span>大戶 <b>集保週資料</b></span><span>進場 <b>正式版核對成交時間</b></span></div>
<nav id="nav"></nav>
<section id="overview" class="active"><div class="desk-title">選股總覽</div><div class="desk-sub">先選操作週期，再看買進區與防守價；跨週期入選只列一次。點選個股查看完整圖表。</div><div class="desk-grid" id="overviewStats"></div>
<div class="controls"><label>操作週期 <select id="hfilter"><option value="all">綜合</option><option value="short">短線</option><option value="mid">中線</option><option value="long">長線</option></select></label><label>族群 <select id="gfilter"><option value="all">全部族群</option></select></label></div><div class="wrap" id="overviewTable"></div><p class="muted">模型分數為相對百分位，並非勝率。預覽全部為合成資料；2R為風險距離推算。</p></section>
<section id="radar"><div class="desk-title">盤中雷達</div><div class="desk-sub">正式版由模型候選池掃描，核對當日成交時間與買區，再排序可布局標的。</div><div class="note">此頁為離線預覽，不接收即時成交。正式版可選30／60／120秒更新，報價缺漏或逾時會顯示等待，不以委託中價當成交價。</div><div class="controls"><label>掃描上限 <select><option>100檔</option><option>50檔</option><option>150檔</option></select></label><label>更新間隔 <select><option>60秒</option><option>30秒</option><option>120秒</option></select></label><button id="radarDemo">模擬刷新排序</button></div><div class="wrap" id="radarTable"></div></section>
<section id="tomorrow"><div class="desk-title">明日布局</div><div class="desk-sub">確認名單日期，整理下交易日限價、防守與減碼參考。</div><div class="wrap" id="tomorrowTable"></div><div class="panel"><h3>從盤後到進出場</h3><p>① 確認本日盤後資料已更新。② 在買區內分3筆，第一筆先做1/3。③ 跳空超過追價上限先等，低於買區先等止跌。④ 13:25前仍未符合條件，保留資金。</p><p>跌破防守價檢視減碼；到2R可評估先減1/3；跌出續抱門檻，在下次固定檢查日檢視換股。</p><p class="muted">短線每5個交易日檢查；中線分4批、長線分12批，每週檢查一批。這些價位與配比覆蓋層尚未納入原回測。</p></div></section>
<section id="stock"><div class="desk-title">個股診斷</div><div class="desk-sub">用代號或公司名查詢三個週期，切換K線、均線與技術／籌碼面板。</div><div class="controls"><input id="stockQuery" placeholder="股票代號／公司名"><select id="stockSelect"></select><select id="stockH"><option value="mid">中線交易計畫</option><option value="short">短線交易計畫</option><option value="long">長線交易計畫</option></select></div><div id="stockHeading"></div><div class="wrap" id="diagTable"></div><div class="desk-grid" id="planStats"></div><div class="controls"><label>K線 <select id="period"><option value="D">日K</option><option value="W">週K</option><option value="M">月K</option></select></label><span id="mas"></span><span id="panels"></span></div><div id="chart"></div><div class="two"><div class="panel"><h3>三大法人各自進出</h3><div class="wrap" id="chipsTable"></div><p class="muted" id="chipsNotes"></p></div><div class="panel"><h3>資券・大戶・主力</h3><div id="balances"></div><p class="muted">大戶為集保週資料；主力分點需匯入有來源的CSV。未取得資料時不顯示虛構數值。</p></div></div></section>
<section id="capital"><div class="desk-title">資金配置</div><div class="desk-sub">跨週期同股合併，保留現金、費用與風險預算。配比為可調整範例。</div><div class="controls"><label>投資資金 <input type="number" id="capitalInput" min="10000" step="10000" value="1000000"></label><label>配置範例 <select id="profile"></select></label><label>金額 <select id="batch"><option value="batch">本週新一批</option><option value="target">目標持倉</option></select></label></div><div class="desk-grid" id="capitalStats"></div><div class="note">單股上限10%、同族群直接個股上限25%、每檔至防守價風險預算0.5%。0050為選配；採用時屬核心預算，非每週再投入；此為本金試算，未扣既有股數；族群上限未穿透0050成分股。</div><div class="wrap" id="capitalTable"></div></section>
<section id="guide"><div class="desk-title">資料與策略</div><div class="desk-sub">正式版保留資料日期、模型依據、原績效報告與更新狀態。</div><div class="panel"><h3>預設模型組合</h3><p>短線：十面向分散先驗20%＋成長動能、用中線標籤估計慢變因子，每5日檢查，前25名內可續抱。</p><p>中線：十面向分散先驗20%＋精簡成長因子與Alpha相對強度假說融合，分4批，續抱門檻前20名。</p><p>長線：十面向分散先驗20%＋趨勢錨與領先股融合，分12批、名次加權，續抱門檻前25名。</p></div><div class="panel"><h3>正式資料如何使用</h3><p>可沿用CLAUDE的bundle.npz與每日更新；也可還原原Alpha的V21_DATABASE.zip作唯讀價量轉入。基本面尚未補齊時可明確選用價格確認模式。</p><p>盤中排序與價位屬操作層，尚未證明其真實市場績效最優；歷史報告與新規則分開說明。</p></div></section>
<div class="desk-footer">ALPHA RADAR PRO · 此檔僅供互動介面預覽。全部示範數據為合成資料；正式版不自動下單。</div></main>
<script>%%PLOTLY%%</script><script>const D=%%DATA%%;
const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
const hz={short:'短線',mid:'中線',long:'長線'}, stockKeys=Object.keys(D.stocks);let active=stockKeys[0];
const num=(x,d=2)=>x==null||!Number.isFinite(+x)?'—':(+x).toLocaleString('zh-TW',{minimumFractionDigits:d,maximumFractionDigits:d});
const price=x=>num(x,x<50?2:x<500?1:0), stat=(a,b,c)=>`<div class="desk-stat"><div class="label">${esc(a)}</div><div class="value">${esc(b)}</div><div class="hint">${esc(c)}</div></div>`;
const table=(heads,rows,ids=[])=>`<table><thead><tr>${heads.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map((r,i)=>`<tr ${ids[i]?`data-t="${ids[i]}"`:''}>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
const tabs={overview:'選股總覽',radar:'盤中雷達',tomorrow:'明日布局',stock:'個股診斷',capital:'資金配置',guide:'資料與策略'};
$('nav').innerHTML=Object.entries(tabs).map(([id,label])=>`<button data-page="${id}" class="${id==='overview'?'active':''}">${label}</button>`).join('');
function page(id){document.querySelectorAll('section').forEach(x=>x.classList.toggle('active',x.id===id));document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('active',x.dataset.page===id));if(id==='stock')drawStock();if(id==='capital')drawCapital();}
$('nav').onclick=e=>{if(e.target.dataset.page)page(e.target.dataset.page)};
document.addEventListener('click',e=>{let row=e.target.closest('tr[data-t]');if(row){active=row.dataset.t;$('stockSelect').value=active;page('stock');window.scrollTo(0,0)}});
const members=t=>Object.keys(hz).filter(h=>D.selections[h].includes(t));
function rowsFor(keys,h='mid'){return keys.map(t=>{let s=D.stocks[t],hs=members(t),hh=hs.includes(h)?h:hs[0],p=s.plans[hh],r=s.scores[hh];return [esc(t+' '+s.name),esc(s.group),esc(hs.map(x=>hz[x]).join('／')),num(r?.score,1),price(s.price),`<span class="${s.change>=0?'up':'down'}">${s.change>=0?'+':''}${num(s.change*100)}%</span>`,price(p.buy_low)+'～'+price(p.buy_high),'<span class="tag">合成預覽</span>',price(p.alert),price(p.take1)]})}
const heads=['股票','族群','週期','模型分數','參考價','漲跌','買進區','判斷','防守價','2R參考'];
function drawOverview(){let h=$('hfilter').value,g=$('gfilter').value,keys=h==='all'?stockKeys:D.selections[h];keys=keys.filter(t=>g==='all'||D.stocks[t].group===g);$('overviewTable').innerHTML=table(heads,rowsFor(keys,h==='all'?'mid':h),keys)}
$('overviewStats').innerHTML=stat('綜合觀察',stockKeys.length+' 檔','跨週期去除重複')+Object.keys(hz).map(h=>stat(hz[h]+'候選',D.selections[h].length+' 檔',{short:'每5日檢查續抱',mid:'分4批、每週一批',long:'分12批、每週一批'}[h])).join('');
$('gfilter').innerHTML+=[...new Set(stockKeys.map(t=>D.stocks[t].group))].sort().map(g=>`<option>${esc(g)}</option>`).join('');$('hfilter').onchange=drawOverview;$('gfilter').onchange=drawOverview;drawOverview();
$('tomorrowTable').innerHTML=table(heads,rowsFor(stockKeys),stockKeys);$('radarTable').innerHTML=table(heads,rowsFor(stockKeys),stockKeys);$('radarDemo').onclick=()=>{let k=[...stockKeys].sort(()=>Math.random()-.5);$('radarTable').innerHTML=table(heads,rowsFor(k),k)};
$('stockSelect').innerHTML=stockKeys.map(t=>`<option value="${t}">${esc(t+' '+D.stocks[t].name)}</option>`).join('');$('stockSelect').onchange=()=>{active=$('stockSelect').value;drawStock()};
function search(input){let q=input.value.trim();let t=stockKeys.find(t=>(t+' '+D.stocks[t].name).includes(q));if(t&&q){active=t;$('stockSelect').value=t;page('stock')}}
$('stockQuery').oninput=()=>search($('stockQuery'));$('quick').oninput=()=>search($('quick'));$('stockH').onchange=drawStock;
$('mas').innerHTML=[5,10,20,60,120,240].map(n=>`<label style="font-size:12px;margin-right:7px"><input type="checkbox" name="ma" value="${n}" ${[5,20,60].includes(n)?'checked':''}>${n}MA</label>`).join('');
$('panels').innerHTML=Object.entries({volume:'成交量',kd:'KD',macd:'MACD',inst:'法人',margin:'資券',holders:'大戶',boll:'布林'}).map(([k,n])=>`<label style="font-size:12px;margin-right:7px"><input type="checkbox" name="panel" value="${k}" ${['volume','kd','macd','boll'].includes(k)?'checked':''}>${n}</label>`).join('');
document.querySelectorAll('input[name=ma],input[name=panel]').forEach(x=>x.onchange=drawChart);$('period').onchange=drawChart;
function drawStock(){let s=D.stocks[active],h=$('stockH').value,p=s.plans[h];$('stockHeading').innerHTML=`<div class="desk-stock"><span class="code">${active}</span><span class="name">${esc(s.name)}</span><span class="desk-tag">${esc(s.group)}</span><span class="desk-tag">${esc(s.segment)}</span></div>`;
$('diagTable').innerHTML=table(['週期','模型排名','目前判斷','主要依據'],Object.keys(hz).map(h=>{let r=s.scores[h];return [hz[h],r&&r.rank?'第'+r.rank+'名':'未評分','合成資料預覽',esc(r?.reasons||'資料待補齊')]}));
$('planStats').innerHTML=stat('買進區',price(p.buy_low)+'～'+price(p.buy_high),'有效成交價進入區間再檢查')+stat('追價上限',price(p.chase),'高於此價先等')+stat('防守價',price(p.alert),'跌破時檢視減碼')+stat('2R參考減碼',price(p.take1),'風險距離推算，非報酬預測');
let c=s.chips; $('chipsTable').innerHTML=table(['法人','最新日(張)','近5日(張)','近20日(張)','連續買賣超'],c.table.map(r=>[esc(r['法人']),num(r['最新日(張)'],0),num(r['近5日(張)'],0),num(r['近20日(張)'],0),num(r['連續買賣超'],0)]));$('chipsNotes').innerText=c.notes.join('；');$('balances').innerHTML=Object.entries(c.balances).map(([k,v])=>`<p>${k}：${v.value==null?'未取得':num(v.value,0)+' 張'}<span class="muted">　${esc(v.date||'')}</span></p>`).join('')+'<p>大戶持股：'+(c.holders.value==null?'未取得':num(c.holders.value)+'%')+'</p><p>主力分點：尚無資料</p>';drawChart()}
function drawChart(){if(!$('stock').classList.contains('active'))return;let s=D.stocks[active],period=$('period').value,a=s.charts[period],p=s.plans[$('stockH').value],pan=[...document.querySelectorAll('input[name=panel]:checked')].map(x=>x.value),ma=[...document.querySelectorAll('input[name=ma]:checked')].map(x=>+x.value);let rows=pan.filter(x=>x!=='boll'&&(x!=='holders'||a.big400?.some(v=>v!=null))),parts=['price',...rows],tr=[],lay={height:420+rows.length*120,paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'Microsoft JhengHei, Noto Sans CJK TC, sans-serif',color:'#24374e'},margin:{l:10,r:65,t:35,b:35},showlegend:true,legend:{orientation:'h',y:1.06},hovermode:'x unified',barmode:'relative',dragmode:'pan',shapes:[]};let priceFrac=rows.length?.52:1,other=(1-priceFrac)/Math.max(rows.length,1),cursor=1;const colors={5:'#e3a41a',10:'#ce83ce',20:'#36a2d8',60:'#8ab63f',120:'#e58743',240:'#aa7ac6'};
parts.forEach((part,i)=>{let axis=i+1,suffix=i?' '+axis:'',xn='x'+(i?axis:''),yn='y'+(i?axis:''),size=i?other:priceFrac;let bottom=Math.max(0,cursor-size+.015);lay['xaxis'+(i?axis:'')]={type:'category',anchor:yn,showticklabels:i===parts.length-1,nticks:7,rangeslider:{visible:false},showgrid:false};lay['yaxis'+(i?axis:'')]={domain:[bottom,cursor],anchor:xn,side:'right',gridcolor:'#e5ebf2',zerolinecolor:'#dae2ec',title:{text:{price:'價格',volume:'成交量(張)',kd:'KD',macd:'MACD',inst:'法人(張)',margin:'資券(張)',holders:'大戶(%)'}[part],font:{size:10}}};cursor-=size;
const line=(key,name,color,dash)=>{if(a[key]?.some(v=>v!=null))tr.push({type:'scatter',mode:'lines',x:a.x,y:a[key],name,line:{color,width:1.3,dash},xaxis:xn,yaxis:yn})};
if(part==='price'){tr.push({type:'candlestick',x:a.x,open:a.open,high:a.high,low:a.low,close:a.close,name:'K線',showlegend:false,increasing:{line:{color:'#e44a51'},fillcolor:'#e44a51'},decreasing:{line:{color:'#16a065'},fillcolor:'#16a065'},xaxis:xn,yaxis:yn});ma.forEach(n=>line('ma'+n,n+({D:'日',W:'週',M:'月'}[period])+'均線',colors[n]));if(pan.includes('boll')){line('bb_up','布林上軌','#a1afbe','dot');line('bb_dn','布林下軌','#a1afbe','dot')}if(period==='D')[[p.buy_high,'#168bbc'],[p.alert,'#d94a4a'],[p.chase,'#d89b28']].forEach(([y,col])=>lay.shapes.push({type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:y,y1:y,line:{color:col,width:1,dash:'dash'}}))}
if(part==='volume')tr.push({type:'bar',x:a.x,y:a.volume,name:'成交量',marker:{color:a.close.map((v,k)=>v>=a.open[k]?'#e8565b':'#24a76f')},xaxis:xn,yaxis:yn,showlegend:false});
if(part==='kd'){line('k','K(9)','#e3a41a');line('d','D(9)','#36a2d8')}
if(part==='macd'){tr.push({type:'bar',x:a.x,y:a.osc,name:'OSC',marker:{color:a.osc.map(v=>v>=0?'#e8565b':'#24a76f')},xaxis:xn,yaxis:yn,showlegend:false});line('dif','DIF','#e3a41a');line('dem','MACD','#36a2d8')}
if(part==='inst')[['foreign','外資','#398cdd'],['trust','投信','#d878ae'],['dealer','自營商','#daa42c']].forEach(([key,n,c])=>tr.push({type:'bar',x:a.x,y:a[key],name:n,marker:{color:c},xaxis:xn,yaxis:yn}));if(part==='margin'){line('mbal','融資餘額','#d9697b');line('sbal','融券餘額','#24a76f')}if(part==='holders')line('big400','大戶≥400張','#daa42c')});Plotly.react('chart',tr,lay,{responsive:true,displaylogo:false,scrollZoom:true})}
$('profile').innerHTML=Object.keys(D.profiles).map(x=>`<option ${x==='均衡配置'?'selected':''}>${x}</option>`).join('');['capitalInput','profile','batch'].forEach(k=>$(k).onchange=drawCapital);
function drawCapital(){let cap=Math.max(10000,+$('capitalInput').value||1000000),w={...D.profiles[$('profile').value]},isBatch=$('batch').value==='batch';if(isBatch){w.short/=1;w.mid/=4;w.long/=12;w.cash=1-w['0050']-w.short-w.mid-w.long}let agg={};Object.keys(hz).forEach(h=>{if(!w[h])return;let list=D.selections[h],raw=list.map((_,i)=>h==='long'&&list.length>1?1.5-i/(list.length-1):1),sum=raw.reduce((a,b)=>a+b,0);list.forEach((t,i)=>{agg[t]??={target:0,sources:[]};agg[t].target+=cap*w[h]*(D.exposures[h]??1)*Math.min(1,list.length/10)*raw[i]/sum;agg[t].sources.push(h)})});let group={},spent=0,rows=[];Object.entries(agg).sort((a,b)=>b[1].target-a[1].target).forEach(([t,v])=>{let s=D.stocks[t],p=v.sources.map(h=>s.plans[h]).sort((a,b)=>b.alert-a.alert)[0],px=p.entry,risk=px-p.alert,budget=Math.min(v.target,cap*.1,Math.max(0,cap*.25-(group[s.group]||0))),q=Math.min(Math.floor(budget/(px*1.002425)),Math.floor(cap*.005/risk));if(q<=0)return;let cost=q*px*1.002425;spent+=cost;group[s.group]=(group[s.group]||0)+cost;rows.push([esc(t+' '+s.name),esc(s.group),v.sources.map(h=>hz[h]).join('／'),num(q*px,0),num(q*px*.002425,0),num(cost/cap*100,1)+'%',q,price(px),price(p.alert)])});let core=cap*w['0050'],cash=cap-spent-core;$('capitalStats').innerHTML=stat(isBatch?'本週個股含費用':'目標個股含費用',num(spent,0)+' 元','按同股合併及風險上限')+stat('0050核心（選配）',num(core,0)+' 元','可設0%；已持有者不重複買入')+stat('未投入／保留',num(cash,0)+' 元','含後續批次與保留資金')+stat('資金核對','100% 相符','股票＋費用＋核心＋餘款');$('capitalTable').innerHTML=table(['股票','族群','策略來源','買股金額','費用緩衝','占總資金','股數','試算價','防守價'],rows)}
drawStock();drawCapital();
</script></html>'''


if __name__=='__main__':
    output=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'artifacts'/'AlphaRadar_V27_Interface_Preview.html'
    print(export(output))
