"""Walk-forward comparison of the source recipe and V27 ten-dimension blend."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .backtest import run
from .desk_models import POOL_DEFAULT
from .research import Research
from .ten_dimension import blend_weights,BLEND

def _metric(segment,key):
    value=segment.get(key)
    return None if value is None or not np.isfinite(value) else float(value)

def validate(bundle,cfg,progress=lambda m:None):
    research=Research(bundle,cfg,log=lambda *a:None);rows=[]
    for h in ('short','mid','long'):
        rule,size=POOL_DEFAULT[h]
        if rule not in bundle.meta['rules']:rule=bundle.meta['rules'][0]
        v=research.model_variant(h);weights,_=research.schedule(rule,size,h,v)
        enhanced=[None if w is None else blend_weights(w,bundle.factors,h) for w in weights]
        spec=research.spec(h,v);view=research.view(rule,size)
        source=research.segments(run(bundle,view,weights,spec,start_i=research.first_oos))
        v27=research.segments(run(bundle,view,enhanced,spec,start_i=research.first_oos))
        windows={}
        for name in ('dev','holdout','oos'):
            a,b=source[name],v27[name]
            windows[name]={"source_cagr":_metric(a,'cagr'),"v27_cagr":_metric(b,'cagr'),
                "delta_cagr":None if _metric(a,'cagr') is None or _metric(b,'cagr') is None else _metric(b,'cagr')-_metric(a,'cagr'),
                "source_mdd":_metric(a,'mdd'),"v27_mdd":_metric(b,'mdd'),
                "source_ir":_metric(a,'info_ratio'),"v27_ir":_metric(b,'info_ratio')}
        deltas=[x['delta_cagr'] for x in windows.values() if x['delta_cagr'] is not None]
        mdd_oos=windows['oos'];mdd_ok=(mdd_oos['source_mdd'] is None or mdd_oos['v27_mdd'] is None or mdd_oos['v27_mdd']>=mdd_oos['source_mdd']-.05)
        accepted=bool(deltas and np.mean(deltas)>0 and min(deltas)>=-.10 and mdd_ok)
        rows.append({"horizon":h,"pool":f'{rule}{size}',"prior_blend":BLEND,"accepted":accepted,"windows":windows})
        progress(f'{h}: '+('通過候選驗收' if accepted else '候選未通過'))
    return {"protocol":"固定20%十面向先驗；三窗口平均CAGR須改善、任一窗口不退步10pp、OOS回撤不得惡化5pp以上",
        "automatic_promotion":False,"results":rows}

def save(bundle,cfg,path='artifacts/v27_validation.json',progress=lambda m:None):
    result=validate(bundle,cfg,progress);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2));return p
