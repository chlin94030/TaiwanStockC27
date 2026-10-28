"""Predeclared V27 diversified prior. Promotion still requires real-data walk-forward evidence."""
from __future__ import annotations
import numpy as np

# Ten lenses: growth, quality, value, leadership, trend structure, institutions,
# financing/ownership, downside risk, liquidity/capacity, behavioral regime.
PRIORS={
 'short':{
  't_high20':.10,'t_ma_align':.10,'t_rsi_health':.05,'t_macd':.04,
  'c_foreign5':.08,'c_trust5':.08,'c_inst_streak':.05,'c_margin_chg20':.04,
  'k_hands':.12,'k_resilience':.08,'k_calm':.10,'k_vp':.08,'k_mw':.08},
 'mid':{
  'f_rev_yoy3':.07,'f_rev_accel':.08,'f_eps_yoy':.05,'f_roe':.05,'f_ep':.04,
  't_rs120':.10,'t_mom120x20':.07,'t_ma_align':.05,'t_lowidio':.07,'t_lowvol':.04,
  'c_foreign20':.05,'c_trust20':.05,'c_margin_chg20':.03,'k_hands':.07,
  'k_resilience':.04,'k_paid':.02,'k_egg':.02},
 'long':{
  'f_rev_yoy3':.06,'f_rev_accel':.06,'f_eps_yoy':.06,'f_roe':.07,'f_ep':.05,'f_dy':.03,
  't_rs120':.12,'t_mom120x20':.08,'t_high252':.06,'t_lowidio':.08,'t_lowvol':.04,
  'c_foreign20':.04,'c_trust20':.04,'c_inst60':.04,'c_foreign_hold':.03,
  'k_hands':.05,'k_resilience':.04,'k_paid':.03,'k_egg':.02}
}
BLEND=.20

def prior_vector(factors,horizon):
    mapping=PRIORS[horizon];out=np.array([mapping.get(name,0.) for name in factors],float)
    total=np.abs(out).sum()
    return out/total if total else out

def blend_weights(source,factors,horizon,blend=BLEND):
    source=np.asarray(source,float);den=np.abs(source).sum()
    source=source/den if den else source
    prior=prior_vector(factors,horizon)
    usable=np.isfinite(source)&np.isfinite(prior)
    out=np.where(usable,(1-blend)*source+blend*prior,0.)
    den=np.abs(out).sum()
    return out/den if den else out
