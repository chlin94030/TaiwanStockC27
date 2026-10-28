import pandas as pd
import pytest
from kosto.global_markets import fetch_global,linked_score,group_sensitivity,TOTAL_WEIGHT
from kosto.ten_dimension import blend_weights,prior_vector,BLEND
from kosto.desk_plan import PROFILES

class Response:
    def __init__(self,symbol):self.symbol=symbol
    def raise_for_status(self):pass
    def json(self):
        move={"%5ESOX":.04,"%5EIXIC":.02,"NQ%3DF":.01,"%5EN225":-.01,"%5EKS11":.0}[self.symbol]
        return {"chart":{"result":[{"meta":{"regularMarketPrice":100*(1+move),"chartPreviousClose":100,
            "regularMarketTime":pd.Timestamp('2026-10-28T09:00:00+08:00').timestamp()}}]}}

class Session:
    def get(self,url,**kwargs):return Response(url.rsplit('/',1)[-1])

def test_global_weights_are_explicit_and_sector_sensitive():
    snap=fetch_global('2026-10-28T10:00:00+08:00',Session())
    assert snap.available_weight==pytest.approx(TOTAL_WEIGHT)==pytest.approx(.12)
    assert snap.confidence==1 and snap.label=='偏多'
    assert group_sensitivity('電子上游-半導體')>group_sensitivity('金融-金融業')
    assert linked_score(70,'電子上游-半導體',snap)>linked_score(70,'金融-金融業',snap)

def test_missing_global_data_is_neutral_and_never_blocks_model():
    class Empty:
        def get(self,*a,**k):raise ValueError('offline')
    snap=fetch_global('2026-10-28T10:00:00+08:00',Empty())
    assert snap.available_weight==0 and snap.score==0
    assert linked_score(83,'電子上游-半導體',snap)==83

def test_ten_dimension_prior_and_blend_are_normalized():
    factors=['f_rev_yoy3','f_roe','f_ep','t_rs120','t_ma_align','c_foreign20','c_margin_chg20','t_lowidio','k_hands']
    prior=prior_vector(factors,'mid');out=blend_weights([1]*len(factors),factors,'mid')
    assert abs(abs(prior).sum()-1)<1e-12 and abs(abs(out).sum()-1)<1e-12
    assert BLEND==.20 and (out>=0).all()

def test_0050_is_optional_in_profiles():
    assert PROFILES['純個股均衡']['0050']==0
    assert sum(PROFILES['純個股均衡'].values())==pytest.approx(1)
