"""Explicit entry, exit and capital plans. Price overlays are not backtested alpha."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import numpy as np
import pandas as pd
from . import ta,plan as source_plan
from .desk_quotes import quote_check,taipei,phase

HZ={'short':'短線','mid':'中線','long':'長線'}
PROFILES={
    '核心優先':{'cash':.20,'0050':.60,'short':0.,'mid':.10,'long':.10},
    '均衡配置':{'cash':.10,'0050':.40,'short':0.,'mid':.30,'long':.20},
    '主動操作':{'cash':.10,'0050':.20,'short':.30,'mid':0.,'long':.40},
    '純個股均衡':{'cash':.20,'0050':0.,'short':0.,'mid':.45,'long':.35},
    '純個股主動':{'cash':.15,'0050':0.,'short':.35,'mid':0.,'long':.50}}


@dataclass(frozen=True)
class TradePlan:
    close:float;buy_low:float;buy_high:float;chase:float;alert:float;support:float;resist:float
    atr_pct:float;rsi:float;bias20:float;hot:bool;entry:float;take1:float;take2:float;asof:str
    def as_dict(self):return asdict(self)


def build_plan(frame,horizon='mid',reference=None):
    if frame is None or len(frame)<60:return None
    ind=ta.indicators(frame)
    if not np.isfinite(ind.atr.iloc[-1]) or ind.atr.iloc[-1]<=0:return None
    p=source_plan.trade_plan(ind,horizon)
    if p is None:return None
    ratio=1.
    if reference is not None and np.isfinite(reference) and reference>0:ratio=float(reference)/p.close
    levels={k:ta.round_tick(getattr(p,k)*ratio,'down' if k not in ('resist',) else 'up')
            for k in ('close','buy_low','buy_high','chase','alert','support','resist')}
    if not 0<levels['alert']<levels['buy_low']<=levels['buy_high']<=levels['chase']:return None
    entry=ta.round_tick((levels['buy_low']+levels['buy_high'])/2,'near');risk=entry-levels['alert']
    return TradePlan(**levels,atr_pct=p.atr_pct,rsi=p.rsi,bias20=p.bias20,hot=p.hot,entry=entry,
        take1=ta.round_tick(entry+2*risk,'up'),take2=ta.round_tick(entry+3*risk,'up'),asof=str(frame.index[-1].date()))


@dataclass(frozen=True)
class Decision:
    code:str;label:str;can_enter:bool;reason:str


def signal_current(signal_date,now=None,max_sessions=1):
    today=taipei(now).normalize().tz_localize(None);d=pd.Timestamp(signal_date).normalize().tz_localize(None)
    if d>today:return False
    return len(pd.bdate_range(d,today))-1<=max_sessions


def decision(plan,quote=None,role='pick',now=None,mode='live',data_ready=True,synthetic=False):
    if synthetic:return Decision('demo','示範資料',False,'合成數據僅供操作預覽')
    if plan is None:return Decision('missing','資料不足',False,'至少需要60根有效日K建立價位')
    if not data_ready:return Decision('stale','等待資料更新',False,'模型或價格資料尚未更新至可用日期')
    if role=='none':return Decision('watch','未入選',False,'目前不在模型可新買名單')
    if mode=='close':
        return Decision('hold' if role=='keep' else 'tomorrow','續抱觀察' if role=='keep' else '下交易日觀察',False,
                        f'限價參考{plan.buy_low:g}～{plan.buy_high:g}；開盤後核對成交價')
    valid,reason=quote_check(quote,now)
    if not valid:return Decision('quote','等待有效報價',False,reason)
    px=float(quote['price'])
    if np.isfinite(quote.get('limit_down',np.nan)) and px<=quote['limit_down']:
        return Decision('weak','跌停觀察',False,'價格觸及跌停，先確認流動性及消息')
    if np.isfinite(quote.get('limit_up',np.nan)) and px>=quote['limit_up']:
        return Decision('chase','漲停不追',False,'等待回到買進區，漲停價不新增部位')
    if px<=plan.alert:return Decision('exit','跌破防守',False,'持有者檢視減碼；空手者等待型態修復')
    if role=='keep':return Decision('hold','持有可續抱',False,'仍在該週期續抱門檻內，空手不新買')
    if px>plan.chase:return Decision('chase','等拉回',False,'成交價超過追價上限')
    if px<plan.buy_low:return Decision('weak','等止跌',False,'低於買進區，先等止跌並回到區間')
    bid,ask=quote.get('bid',np.nan),quote.get('ask',np.nan)
    if np.isfinite(bid+ask) and (ask-bid)/px>.005:
        return Decision('spread','價差偏大',False,'最佳買賣價差超過0.5%，等待流動性改善')
    if px>plan.buy_high:return Decision('chase','接近上限',False,'先等成交價回到買進區，再考慮第一筆')
    return Decision('buy','可分批布局',True,'有效成交價位於買進區；先投入預定本批資金的1/3')


def allocation(capital,weights,selections,plans,prices,exposures=None,top_k=10,max_name=.10,max_group=.25,
               risk_fraction=.005,fee_rate=.001425,slippage=.001,groups=None):
    """Deduplicate sleeves, cap risk/concentration; every dollar reconciles to cash."""
    if not np.isfinite(capital) or capital<=0:raise ValueError('投資金額須大於0')
    if set(weights)-{'cash','0050','short','mid','long'}:raise ValueError('未知的配置欄位')
    if any(not np.isfinite(w) or w<0 for w in weights.values()) or abs(sum(weights.values())-1)>1e-6:
        raise ValueError('配置合計須為100%')
    exposures=exposures or {};groups=groups or {};aggregate={}
    for h in HZ:
        picks=list(dict.fromkeys(selections.get(h,[])))[:top_k]
        nominal=source_plan.stock_weights(len(picks),h)*min(1.,len(picks)/top_k)
        budget=capital*weights.get(h,0)*max(0,min(1,float(exposures.get(h,1))))
        if budget<=0:continue
        for t,w in zip(picks,nominal):
            a=aggregate.setdefault(t,{'target':0.,'sources':[]});a['target']+=budget*float(w);a['sources'].append(h)
    rows=[];group_used={};spent=0.;buy_cost=fee_rate+slippage
    for t,a in sorted(aggregate.items(),key=lambda x:(-x[1]['target'],x[0])):
        p=plans.get(t);px=prices.get(t,np.nan);group=groups.get(t,'未分類')
        if p is None or not np.isfinite(px) or px<=0:continue
        risk_unit=max(px-p.alert,0)
        if risk_unit<=0:continue
        cash_budget=min(a['target'],capital*max_name,max(0,capital*max_group-group_used.get(group,0)))
        quantity=min(math.floor(cash_budget/(px*(1+buy_cost))),math.floor(capital*risk_fraction/risk_unit))
        if quantity<=0:continue
        amount=quantity*px;cost=amount*buy_cost;spent+=amount+cost;group_used[group]=group_used.get(group,0)+amount+cost
        rows.append({'ticker':t,'group':group,'sources':'／'.join(HZ[h] for h in a['sources']),
                     'weight':(amount+cost)/capital,'amount':amount,'fee_buffer':cost,'shares':quantity,
                     'lots':quantity//1000,'odd_shares':quantity%1000,'price':px,'risk_amount':quantity*risk_unit,
                     'buy_low':p.buy_low,'buy_high':p.buy_high,'alert':p.alert})
    etf=max(0,capital*weights.get('0050',0));cash=capital-spent-etf
    if cash< -1e-5:raise ValueError('配置超出可用資金')
    return pd.DataFrame(rows),{'capital':capital,'stocks_including_fees':spent,'etf':etf,'cash':max(0,cash),
            'cash_floor':capital*weights.get('cash',0),'reconciled':abs(spent+etf+cash-capital)<1e-5}
