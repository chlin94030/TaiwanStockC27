"""Pinned short/mid/long recipes; immutable configuration and complete identity."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import copy,hashlib,json,threading,math
import numpy as np
import pandas as pd
from .research import Research
from .config import STAGE_NAMES
from .chain import label as chain_label

MODEL_TITLES={'short':'成長動能・週頻檢查','mid':'成長 × 相對強度','long':'趨勢錨 × 領先股'}
POOL_DEFAULT={'short':('BLEND',500),'mid':('BLEND',500),'long':('MCAP',500)}


def model_key(variant,cfg,rule,size,h):
    from .ten_dimension import PRIORS,BLEND
    value={'variant':asdict(variant),'horizon':h,'rule':rule,'size':size,
           'recipe':cfg['horizons'][h],'costs':cfg['costs'],'policy':cfg['kostolany'],
           'v27_prior':PRIORS[h],'v27_prior_blend':BLEND}
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()[:16]


@dataclass
class Snapshot:
    horizon:str;title:str;model_id:str;date:str;pool:str;table:pd.DataFrame;picks:list
    weights:dict;exposure:float;keep_rank:int;batch_count:int;ready:bool;notes:list


class DeskEngine:
    def __init__(self,bundle,cfg,profile='v27'):
        self.B=bundle;self.cfg=copy.deepcopy(cfg);self.profile=profile
        self.R=Research(bundle,self.cfg,log=lambda *a:None);self._lock=threading.RLock();self._cache={}
        self.idx={t:j for j,t in enumerate(bundle.tickers)}

    def snapshot(self,h,holdings=()):
        holds=tuple(sorted(set(holdings)));key=(h,holds)
        with self._lock:
            if key not in self._cache:self._cache[key]=self._compute(h,holds)
            result=self._cache[key]
            return Snapshot(**{**result.__dict__,'table':result.table.copy(deep=True),'picks':copy.deepcopy(result.picks),
                               'weights':result.weights.copy(),'notes':result.notes.copy()})

    def _compute(self,h,holdings):
        B=self.B;rule,size=POOL_DEFAULT[h]
        if rule not in B.meta['rules']:rule=B.meta['rules'][0]
        overrides=(self.cfg.get('desk') or {}).get('pools',{}).get(h,{})
        rule=overrides.get('rule',rule);size=int(overrides.get('size',size))
        if rule not in B.meta['rules']:raise ValueError('指定母池不在資料包')
        v=self.R.model_variant(h);spec=self.R.spec(h,v);notes=[]
        if self.profile=='price':
            hypothesis='PRICE_ALPHA_5_LEADERS' if h=='long' else 'PRICE_ALPHA_8_BALANCED'
            raw=self.cfg['research']['score_hypotheses'][hypothesis]
            w=np.array([raw.get(f,0.) for f in B.factors],float)
            w/=max(np.abs(w).sum(),1e-12);notes.append('已手動採用價格確認策略，並非完整成長模型')
            title='價格確認・'+MODEL_TITLES[h]
        else:
            fit=self.R.final_fit(rule,size,h,v);w=np.asarray(fit.weights,float)
            if self.profile=='v27':
                from .ten_dimension import blend_weights
                w=blend_weights(w,B.factors,h);title='十面向融合・'+MODEL_TITLES[h]
            else:title='來源定稿・'+MODEL_TITLES[h]
        view=self.R.view(rule,size);idx=view.members[-1];z=view.Z[-1];score=z@w
        known=B.a['U'][-1,idx,:]>0
        coverage=(known*np.abs(w)).sum(axis=1)/max(np.abs(w).sum(),1e-12)
        n=len(idx);why=[[] for _ in range(n)];top_k=int(self.cfg['research']['top_k'])
        for k,j in enumerate(idx):
            cap=B.a['CAP'][-1,j]
            if not np.isfinite(score[k]) or np.abs(w).sum()<1e-8:why[k].append('模型權重尚不可用')
            if coverage[k]<.90:why[k].append('模型加權資料不足90%')
            if not np.isfinite(cap) or cap*spec.max_participation<spec.capital/top_k:why[k].append('流動性不足')
            if not B.a['TRADE'][-1,j]:why[k].append('訊號日無有效交易')
            st=STAGE_NAMES.get(int(B.a['STAGE'][-1,j]),'UNKNOWN')
            if spec.use_gates:
                if st in spec.block_stages:why[k].append('市場型態'+st)
                if spec.block_dtop and B.a['DTOP'][-1,j]:why[k].append('M頭尚未修復')
                if spec.b3_confirm and st=='B3' and not B.a['WSUP'][-1,j]:why[k].append('尚無止跌確認')
            if spec.max_rsi is not None and ('RSI14' not in B.a or not np.isfinite(B.a['RSI14'][-1,j]) or B.a['RSI14'][-1,j]>spec.max_rsi):why[k].append('RSI條件未通過')
            if spec.max_bias20 is not None and ('BIAS20' not in B.a or not np.isfinite(B.a['BIAS20'][-1,j]) or B.a['BIAS20'][-1,j]>spec.max_bias20):why[k].append('乖離條件未通過')
        valid=np.array([not x for x in why]);ranks=np.full(n,np.nan)
        order=np.where(valid)[0];order=order[np.argsort(-score[order],kind='stable')];ranks[order]=np.arange(1,len(order)+1)
        pct=pd.Series(score).rank(pct=True).to_numpy()*100
        raw_weights={f:float(x) for f,x in zip(B.factors,w) if abs(x)>1e-6}
        keep_rank=int(math.ceil(top_k*(v.hysteresis if v.hysteresis is not None else self.cfg['horizons'][h].get('hysteresis',2.5))))
        chosen=[k for k in order if B.tickers[idx[k]] in holdings and ranks[k]<=keep_rank][:top_k]
        chosen+= [k for k in order if k not in chosen][:max(0,top_k-len(chosen))]
        chosen_set=set(chosen);rows=[]
        for k,j in enumerate(idx):
            ticker=B.tickers[j];contrib=z[k]*w
            positives=[B.meta['factor_zh'][f] for f in np.argsort(-contrib)[:3] if contrib[f]>0]
            negatives=[B.meta['factor_zh'][f] for f in np.argsort(contrib)[:2] if contrib[f]<0]
            chain=chain_label(ticker,B.meta['sectors'][j]);held=ticker in holdings
            rows.append({'ticker':ticker,'name':B.meta['names'][j],'group':chain['group'],'segment':chain['segment'],
                'themes':'／'.join(chain.get('themes',[])[:3]),'rank':ranks[k],'score':float(pct[k]),'raw_score':float(score[k]),
                'coverage':float(coverage[k]),'valid':bool(valid[k]),'why':'；'.join(why[k]),
                'reasons':'、'.join(positives),'risk':'、'.join(negatives),'selected':k in chosen_set,'held':held,
                'role':'pick' if k in chosen_set and (not held or ranks[k]<=top_k) else ('keep' if held and valid[k] and ranks[k]<=keep_rank else 'none')})
        table=pd.DataFrame(rows,columns=['ticker','name','group','segment','themes','rank','score','raw_score','coverage','valid','why','reasons','risk','selected','held','role'])
        if not table.empty:table=table.sort_values(['valid','rank','ticker'],ascending=[False,True,True],na_position='last')
        stage=STAGE_NAMES.get(int(B.a['MKT'][-1,-1]),'UNKNOWN')
        exposure=float(spec.exposure.get(stage,1.0))*float(getattr(spec,'sleeve',1.0))
        ready=bool(len(order)>=max(5,min(top_k,n)) and len(w)>0 and np.abs(w).sum()>0)
        if not ready:notes.append('合格股票或模型資料不足，暫緩新增部位')
        if size>n:notes.append(f'本次母池實際可得{n}檔')
        identity=model_key(v,self.cfg,rule,size,h)+'-'+self.profile
        return Snapshot(h,title,identity,str(B.sig_dates[-1].date()),f'{rule}{size}／實際{n}檔',table,
            table.loc[table.selected].to_dict('records') if not table.empty else [],raw_weights,
            exposure,keep_rank,max(1,int(spec.H)//int(spec.step)),ready,notes)


def search_stocks(bundle,query,limit=30):
    import unicodedata
    key=unicodedata.normalize('NFKC',str(query)).strip().casefold().replace('臺','台')
    if not key:return []
    out=[]
    for j,(ticker,name) in enumerate(zip(bundle.tickers,bundle.meta['names'])):
        text=(str(ticker)+' '+str(name)).casefold().replace('臺','台')
        if key in text:out.append((0 if key==ticker else 1 if key==str(name).casefold() else 2,ticker,name,j))
    return [{'ticker':t,'name':n,'j':j} for _,t,n,j in sorted(out)[:limit]]
