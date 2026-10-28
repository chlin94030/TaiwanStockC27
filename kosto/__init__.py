"""Kosto Lab — 科斯托蘭尼量化選股實驗室.

Architecture (deliberately different from Alpha Radar V22):
  GitHub Actions (data + heavy research)  ->  Parquet store  ->  wide date x ticker panel
  -> vectorised factor library (技術/籌碼/基本/科斯托蘭尼)  -> pool tests  -> IC-IR hierarchical weights
  -> tranche (overlapping) backtester  -> research bundle (.npz)  -> Streamlit (seconds-level re-tuning)
"""
__version__ = "27.0.0"
APP_NAME = "Alpha Radar PRO Global 台股決策台"
