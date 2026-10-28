"""Full downloadable test report: report.html + report.xlsx + CSVs + config_used.yaml → ZIP."""
from __future__ import annotations

import html
import io
import json
import zipfile

import numpy as np
import pandas as pd

from . import APP_NAME, __version__
from .config import STAGE_ZH, dump_yaml
from .svg import bar_chart, heat_table, line_chart, stage_strip
from .universe import RULE_ZH

CSS = """
:root{--bg:#ffffff;--fg:#1f2937;--mut:#6b7280;--line:#e5e7eb;--card:#f9fafb;--acc:#2563eb;--bad:#dc2626;--good:#16a34a}
@media (prefers-color-scheme: dark){:root{--bg:#111827;--fg:#e5e7eb;--mut:#9ca3af;--line:#374151;--card:#1f2937;--acc:#60a5fa}}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif;
max-width:1000px;margin:0 auto;padding:16px;line-height:1.55}
h1{font-size:1.5rem;margin:.2rem 0}h2{font-size:1.2rem;border-bottom:2px solid var(--acc);padding-bottom:4px;margin-top:2rem}
h3{font-size:1.05rem;margin-top:1.4rem}.mut{color:var(--mut);font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.82rem;margin:.5rem 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 6px;text-align:right;white-space:nowrap}th{background:var(--card)}
td.l,th.l{text-align:left;white-space:normal}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:.6rem 0}
.banner{background:#fef3c7;color:#92400e;border:1px solid #f59e0b;border-radius:8px;padding:10px;margin:10px 0;font-weight:600}
.pos{color:var(--good)}.neg{color:var(--bad)}svg.chart{width:100%;height:auto;margin:.4rem 0}
.ct{font-size:13px;font-weight:600;fill:var(--fg)}.ax{font-size:10px;fill:var(--mut)}.lg{font-size:11px;fill:var(--fg)}
.grid{stroke:var(--line);stroke-width:1}.legend{font-size:.75rem;color:var(--mut)}.sw{display:inline-block;width:10px;height:10px;margin:0 3px 0 8px}
.heat td{color:#111}ul{padding-left:1.2rem}.kpi{display:flex;flex-wrap:wrap;gap:8px}.kpi div{flex:1 1 140px;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:8px}.kpi b{display:block;font-size:1.2rem}
"""


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x*100:+.{d}f}%"


def num(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def cls(x):
    return "" if x is None or not np.isfinite(x) else ("pos" if x > 0 else "neg")


def table(df: pd.DataFrame, fmts: dict | None = None, left=()) -> str:
    fmts = fmts or {}
    out = ["<table><tr>" + "".join(f"<th class='{'l' if c in left else ''}'>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            f = fmts.get(c)
            if f == "pct":
                s, k = pct(v), cls(v)
            elif f == "pct0":
                s, k = ("–" if v is None or not np.isfinite(v) else f"{v*100:.1f}%"), ""
            elif f and isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                s, k = f.format(v), ""
            else:
                s, k = ("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)), ""
            cells.append(f"<td class='{k} {'l' if c in left else ''}'>{html.escape(s)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out) + "</table>"


KOSTO_MAP = [
    ("雞蛋理論 A1/B3 買、A2 等待、A3/B1 賣、B2 持現金（個股）", "個股雞蛋階段閘門＋k_egg 因子", ["stage_gate", "k_egg"]),
    ("雞蛋理論（大盤）：過熱減碼、恐慌加碼", "市場階段持股水位（none/soft 比較）", ["exposure"]),
    ("固執的投資者 vs 猶豫的投資者（股票是否已付清）", "k_hands 籌碼轉移、k_paid 融資占股本、KOSTO 母池", ["k_hands", "k_paid"]),
    ("量價四象限（量小價漲有利、量增價跌近底…）", "k_vp 量價因子", ["k_vp"]),
    ("利空不跌＝近底、利多不漲＝近頂", "k_resilience", ["k_resilience"]),
    ("M 頭 / W 底", "k_mw 因子 + M 頭禁止買進", ["k_mw", "dtop"]),
    ("戒1 不跟著建議跑、媒體過度樂觀", "k_calm 避開爆量急漲與融資暴增", ["k_calm"]),
    ("律3 耐心、戒6 不對風吹草動反應", "遲滯續抱（名次仍在前 K×遲滯倍數 就不換股）", ["hyst"]),
    ("戒3 不要想把賠掉的再賺回來", "持有期停損（v1.1 預設關閉，改為推薦表『警戒價』提醒）", ["stop"]),
    ("整體：科斯托蘭尼因子族群", "精簡模型中 K 因子與其他因子公平競爭；加強版保留最強 2 個 K 因子 30%；階層式用最低權重 λ", ["K"]),
    ("律2 足夠資金免受壓力、律9 考慮所有風險", "不融資、等權10檔、流動性容量限制（買進金額≤20日均量2%）", []),
    ("戒8 不要只想獲利就賣掉", "不設停利；只要分數仍佳就續抱", []),
]


ZH = {"variant": "模型", "dev_cagr": "開發期CAGR", "dev_sharpe": "開發期Sharpe", "dev_mdd": "開發期MDD",
      "dev_excess_cagr": "開發期超額", "hold_cagr": "盲測CAGR", "hold_mdd": "盲測MDD", "turnover": "周轉率",
      "score": "綜合分數", "case": "情境", "factor": "因子", "family": "族群", "coverage": "覆蓋率", "ic_dev": "開發期IC",
      "t_dev": "t值", "ic_holdout": "盲測IC", "ic_all": "全期IC", "weight": "權重", "stage": "市場階段", "n": "批數",
      "avg_ret": "平均報酬", "avg_bench": "0050同期", "avg_excess": "平均超額", "avg_exposure": "平均水位",
      "from": "生效日", "n_train": "訓練期數", "alpha": "資料權重α"}


def hints(H: dict) -> list[str]:
    out = []
    for h, r in H.items():
        lab = r["label"]
        ev = r["eval"]
        if r.get("data_beats_kostolany"):
            out.append(f"{lab}：純數據模型(DATA)在開發期綜合分數高於所有科斯托蘭尼版本 → 可試 λ=0.25 或檢視 K 因子 IC。")
        ab = r["ablation"].iloc[1:]
        for _, row in ab.sort_values("貢獻(開發期CAGR)").head(3).iterrows():
            c = row["貢獻(開發期CAGR)"]
            if np.isfinite(c) and c < -0.01 and row.get("目前", "啟用") == "啟用":
                out.append(f"{lab}：「{row['case']}」後開發期 CAGR 提高 {-c*100:.1f}% → 此規則在本期間拖累績效，可考慮關閉或放寬。")
        for _, row in ab.sort_values("貢獻(開發期CAGR)", ascending=False).head(2).iterrows():
            c = row["貢獻(開發期CAGR)"]
            if np.isfinite(c) and c > 0.01 and row.get("目前") == "關閉":
                out.append(f"{lab}：「{row['case']}」可讓開發期 CAGR 提高 {c*100:.1f}% → 可考慮在設定檔開啟。")
        dex, hex_ = ev.get("dev_excess_cagr"), ev.get("hold_excess_cagr")
        if dex is not None and hex_ is not None and np.isfinite(dex) and np.isfinite(hex_) and hex_ < dex - 0.10:
            vs = r.get("vs_pool_ew", {}).get("holdout", np.nan)
            if np.isfinite(vs) and vs >= 0:
                out.append(f"{lab}：盲測期超額報酬比開發期低 {(dex-hex_)*100:.0f} 個百分點，但選股仍勝過同母池等權 {vs*100:.1f}% → "
                           "落後主因是 0050 權值股（台積電等）集中上漲，而非選股失靈；想貼近大盤可用「0050 核心＋本模型衛星」。")
            else:
                out.append(f"{lab}：盲測期超額報酬比開發期低 {(dex-hex_)*100:.0f} 個百分點，且輸給同母池等權 → "
                           "可能過度配適或市場風格改變，建議提高 n0_shrink 或縮小候選。")
        rb = r["robust"]
        if rb.get("dsr") is not None and np.isfinite(rb["dsr"]) and rb["dsr"] < 0.5:
            out.append(f"{lab}：考慮 {rb['n_trials']} 次測試後的 Deflated Sharpe = {rb['dsr']:.2f}（<0.5）→ 優勢在統計上尚不顯著，請以小資金或紙上交易驗證。")
        ft = r["factor_table"]
        bad = ft[(ft["weight"].abs() > 0.02) & (ft["t_dev"] * np.sign(ft["weight"]) < -1.5)]
        for _, row in bad.iterrows():
            out.append(f"{lab}：因子「{row['中文']}」有權重但開發期 IC 方向相反(t={row['t_dev']:.1f}) → 檢查資料或移除。")
        st = r["stage_table"]
        if len(st) and {"A1", "B3", "A3", "B1"} & set(st["stage"]):
            buy = st[st["stage"].isin(["A1", "B3"])]["avg_bench"].mean()
            sell = st[st["stage"].isin(["A3", "B1"])]["avg_bench"].mean()
            if np.isfinite(buy) and np.isfinite(sell) and buy < sell:
                out.append(f"{lab}：市場判為 A1/B3(買點) 後的大盤報酬 {buy*100:.1f}% 低於 A3/B1(賣點) 後的 {sell*100:.1f}% → 市場雞蛋階段判斷在本資料不具預測力，建議水位用 none/soft。")
        if h == "short" and r["turnover"] > 0.5:
            out.append(f"{lab}：每次換股周轉率 {r['turnover']:.0%}，年交易成本約占資金 {r['cost_drag']:.0%} → "
                       "台股證交稅 0.3% 讓每週換股很難勝過 0050；建議短線清單只當中長線標的的『進場時機』參考，不單獨操作。")
    if "short" in H and "mid" in H:
        ov = _pick_overlap(H["short"], H["mid"])
        if np.isfinite(ov) and ov >= 0.5:
            out.append(f"短線與中線同一天的推薦平均重疊 {ov:.0%}（v1.5 短線＝中線的慢α分數、每 5 日檢視）→ "
                       "兩者擇一操作即可：能每週檢查用短線、每月檢查用中線；同時操作等於重複持有同一批股票。")
    return out or ["本次未偵測到需要優先調整的項目。"]


def _pick_overlap(a: dict, b: dict) -> float:
    """Average share of b's picks that also appear in a's picks on the same signal date."""
    try:
        ta = a["tranches"].set_index("date")["picks"]
        tb = b["tranches"].set_index("date")["picks"]
        v = [len(set(ta[d]) & set(p)) / len(p) for d, p in tb.items()
             if d in ta.index and isinstance(ta[d], list) and isinstance(p, list) and len(p)]
        return float(np.mean(v)) if v else np.nan
    except Exception:
        return np.nan


def _bench_sharpe(r: dict) -> float:
    try:
        b = r["nav"]["0050"].dropna().pct_change().dropna()
        return float(b.mean() / b.std() * np.sqrt(245))
    except Exception:
        return np.nan


def verdict(r: dict) -> str:
    """Plain-language answer to 'does it beat 0050?' for one horizon."""
    ev, rb = r["eval"], r["robust"]
    dex, hex_, p = ev.get("dev_excess_cagr", np.nan), ev.get("hold_excess_cagr", np.nan), rb.get("p_positive", np.nan)
    sh, bsh = ev.get("oos_sharpe", np.nan), np.nan
    if not np.isfinite(dex):
        return "資料不足"
    if np.isfinite(p) and p >= 0.95 and dex > 0 and hex_ > 0:
        return "顯著優於 0050（超額>0 機率≥95%，開發期與盲測期皆為正）"
    if np.isfinite(p) and p <= 0.05:
        return "顯著落後 0050（超額>0 機率≤5%）→ 不建議單獨使用"
    if dex > 0 and (not np.isfinite(hex_) or hex_ > 0):
        return "優於 0050，但統計上尚不顯著"
    if dex > 0:
        return "開發期優於 0050、盲測期落後 → 優勢不穩定，未達統計顯著"
    return "未優於 0050（差異未達統計顯著）"


def build_html(res: dict, cfg: dict, comparison: tuple | None = None) -> str:
    H = res["horizons"]
    meta, proto = res["meta"], res["protocol"]
    p = [f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Kosto Lab 回測報告</title><style>{CSS}</style></head><body>",
         f"<h1>{APP_NAME}｜完整測試報告</h1>",
         f"<div class='mut'>版本 {__version__}｜建置 {meta.get('built')}｜訊號 {res['protocol']['signals']} 期（每 {cfg['research']['signal_step']} 交易日）"
         f"｜樣本外起點 {proto['first_oos']}｜盲測起點 {proto['holdout_start']}｜最新資料 {proto['last_date']}</div>"]
    if meta.get("synthetic"):
        p.append("<div class='banner'>⚠ 示範模式：本報告使用「合成資料」驗證程式流程，數字不代表真實台股，請勿據以投資。"
                 "連上 GitHub Actions 的真實資料後重新產生報告。</div>")
    if proto.get("quick"):
        p.append("<div class='banner'>快速模式：只測試部分母池（ADV/KOSTO/INST2Y × 150/300）。完整測試請用 full 模式。</div>")
    # ---------------- summary
    p.append("<h2>1. 總覽：哪個模型、哪個母池</h2>")
    rows, vrows = [], []
    for h, r in H.items():
        ev, rb = r["eval"], r["robust"]
        rows.append({"期間": r["label"], "精選母池": f"{r['chosen_pool']['rule']}{r['chosen_pool']['size']}（{RULE_ZH[r['chosen_pool']['rule']]}）",
                     "選定模型": r["chosen"]["variant"], "開發期CAGR": ev.get("dev_cagr"), "0050同期": ev.get("dev_bench_cagr"),
                     "開發期Sharpe": ev.get("dev_sharpe"), "開發期MDD": ev.get("dev_mdd"), "盲測CAGR": ev.get("hold_cagr"),
                     "盲測0050": ev.get("hold_bench_cagr"), "盲測MDD": ev.get("hold_mdd"), "DSR": rb.get("dsr")})
        vrows.append({"期間": r["label"], "全樣本外CAGR": ev.get("oos_cagr"), "0050": ev.get("oos_bench_cagr"),
                      "Sharpe": ev.get("oos_sharpe"), "0050 Sharpe": _bench_sharpe(r), "最大回撤": ev.get("oos_mdd"),
                      "超額>0機率": rb.get("p_positive"), "選股力(勝母池等權/年)": r.get("vs_pool_ew", {}).get("oos"),
                      "是否顯著優於0050": verdict(r),
                      "驗收閘門": (f"{r['validation']['passed']}/{r['validation']['total']} 通過" if r.get("validation") else "—")})
    p.append(table(pd.DataFrame(rows), {"開發期CAGR": "pct", "0050同期": "pct", "開發期MDD": "pct", "盲測CAGR": "pct",
                                        "盲測0050": "pct", "盲測MDD": "pct", "開發期Sharpe": "{:.2f}", "PSR": "{:.2f}",
                                        "DSR": "{:.2f}"}, left=("期間", "精選母池", "選定模型")))
    p.append("<h3>與台灣50（0050）比較：全部樣本外期間（開發期＋盲測）</h3>")
    p.append(table(pd.DataFrame(vrows), {"全樣本外CAGR": "pct", "0050": "pct", "Sharpe": "{:.2f}", "0050 Sharpe": "{:.2f}",
                                         "最大回撤": "pct", "超額>0機率": "pct0", "選股力(勝母池等權/年)": "pct"},
                   left=("期間", "是否顯著優於0050")))
    p.append("<p class='mut'>開發期＝樣本外起點到盲測起點（用來挑母池與模型）；盲測＝最後 "
             f"{cfg['research']['holdout_months']} 個月，挑選時完全沒看。PSR＝夏普比大於 0 的機率；DSR＝扣除多次測試運氣後的機率；"
             "超額>0機率＝區塊自助法估計年化超額報酬為正的機率（≥95% 才算顯著優於 0050）。"
             "選股力＝本模型減去『同一母池全部等權持有』的年化報酬，衡量挑股本身的價值（扣除母池效果）。</p>")
    p.append("<h3>0050 核心＋本模型衛星（降低落後 0050 的風險）</h3>")
    mix_rows = []
    for h, r in H.items():
        if "mix" in r and len(r["mix"]):
            for _, m in r["mix"].iterrows():
                mix_rows.append({"期間": r["label"], **m.to_dict()})
    if mix_rows:
        mt = pd.DataFrame(mix_rows)
        keep = ["期間", "組合", "dev_cagr", "dev_sharpe", "dev_mdd", "holdout_cagr", "holdout_mdd", "oos_cagr", "oos_excess_cagr", "oos_sharpe", "oos_mdd"]
        mt = mt[[c for c in keep if c in mt]].rename(columns={
            "dev_cagr": "開發期CAGR", "dev_sharpe": "開發期Sharpe", "dev_mdd": "開發期MDD", "holdout_cagr": "盲測CAGR",
            "holdout_mdd": "盲測MDD", "oos_cagr": "全樣本外CAGR", "oos_excess_cagr": "全期超額", "oos_sharpe": "全期Sharpe", "oos_mdd": "全期MDD"})
        p.append(table(mt, {c: ("{:.2f}" if "Sharpe" in c else "pct") for c in mt.columns if c not in ("期間", "組合")}, left=("期間", "組合")))
        p.append("<p class='mut'>每日固定比例再平衡的近似值（實務上每月再平衡即可）。0050 權重越高越貼近大盤、回撤也越接近大盤。</p>")
    # ---------------- today's picks
    first = next(iter(H.values()))
    lt0 = first["latest"]
    p.append(f"<h2>2. 今日推薦（{lt0['date']} 收盤後）</h2>")
    p.append(f"<div class='card'>市場雞蛋階段：<b>{STAGE_ZH.get(lt0['market_stage'], lt0['market_stage'])}</b></div>")
    for h, r in H.items():
        lt = r["latest"]
        p.append(f"<h3>{r['label']}｜母池 {lt['pool']}｜建議持股水位 {lt['exposure']:.0%}</h3>")
        if lt["picks"]:
            df = pd.DataFrame([{"代號": x["ticker"], "名稱": x["name"], "產業": x["sector"], "收盤": x["close"],
                                "分數": x["score"], "個股階段": x["stage"], "續抱": "✓" if x["kept"] else "",
                                "主要理由": "；".join(x["reasons"]), "警戒價": x["stop_price"]} for x in lt["picks"]])
            p.append(table(df, {"收盤": "{:.2f}", "分數": "{:.3f}", "警戒價": "{:.2f}"}, left=("名稱", "產業", "主要理由")))
        if lt["blocked"]:
            p.append("<p class='mut'>高分但被科斯托蘭尼規則排除：" + "、".join(f"{b['ticker']}{b['name']}（{b['why']}）" for b in lt["blocked"][:8]) + "</p>")
    # ---------------- market regime
    mk = res["market"]
    p.append("<h2>3. 市場雞蛋階段（科斯托蘭尼）</h2>")
    p.append(line_chart({"市場指數": mk["index"]}, "大盤指數與階段", log=True))
    p.append(stage_strip(mk["stage"]))
    p.append("<p class='mut'>階段由「指數位置與趨勢、成交量(20/250日)、融資餘額(持有人數代理)、市場廣度」判定；持有人數與新聞反應無法直接觀察，以上為代理變數。</p>")
    # ---------------- per horizon
    for h, r in H.items():
        p.append(f"<h2>4-{r['label']}：回測細節</h2>")
        val = r.get("validation")
        if val:
            gt = val["gates"].copy()
            gt["數值"] = [("—" if x is None or not np.isfinite(x) else (f"{x:+.1%}" if f == "pct" else f"{x:.2f}"))
                         for x, f in zip(gt["數值"], gt["fmt"])]
            gt["門檻"] = [(f"{x:+.0%}" if f == "pct" else f"{x:.2f}") for x, f in zip(gt["門檻"], gt["fmt"])]
            gt["通過"] = gt["通過"].map({True: "✅", False: "❌"})
            p.append(f"<h3>驗收閘門（預先宣告，{val['passed']}/{val['total']} 通過）</h3>" + table(gt.drop(columns=["fmt"]), left=("閘門",)))
            p.append("<p class='mut'>閘門門檻寫在 config 的 research.acceptance，研究前就固定；未全數通過＝不能宣稱穩定勝過 0050。</p>")
            fd = val["folds"]
            if len(fd):
                p.append("<h3>時序區塊（每 6 個月）</h3>" + table(fd, {"策略": "pct", "0050": "pct", "超額": "pct"}, left=("區塊", "期間")))
            pre = val.get("presample") or {}
            if pre:
                p.append(f"<h3>未用期間測試 {pre['start']}～{pre['end']}（只用之後的資料訓練，回頭驗證）</h3>")
                p.append(line_chart({c: pre["nav"][c] for c in pre["nav"].columns}, "未用期間淨值", log=True))
                p.append(f"<p>策略年化 {pct(pre.get('cagr'))}、0050 {pct(pre.get('bench_cagr'))}、超額 {pct(pre.get('excess_cagr'))}、"
                         f"最大回撤 {pct(pre.get('mdd'))}。這段期間（含 2020 疫情崩跌）從未被任何母池、模型或參數選擇看過。</p>")
        nav = r["nav"]
        p.append(line_chart({c: nav[c] for c in nav.columns}, f"{r['label']} 淨值（樣本外，含交易成本）", log=True))
        yt = r["yearly"]
        p.append(bar_chart([str(i) for i in yt.index], {c: yt[c].tolist() for c in yt.columns if c != "超額"}, "年度報酬"))
        ev = r["eval"]
        kp = [("開發期 CAGR", pct(ev.get("dev_cagr"))), ("開發期超額", pct(ev.get("dev_excess_cagr"))),
              ("盲測 CAGR", pct(ev.get("hold_cagr"))), ("盲測超額", pct(ev.get("hold_excess_cagr"))),
              ("每次周轉率", f"{r['turnover']:.0%}"), ("年交易成本(占資金)", f"{r['cost_drag']:.1%}"),
              ("單批獲利率", f"{ev.get('tr_hit_rate', float('nan')):.0%}"),
              ("單批勝0050", f"{ev.get('tr_beat_bench_rate', float('nan')):.0%}")]
        p.append("<div class='kpi'>" + "".join(f"<div>{k}<b>{v}</b></div>" for k, v in kp) + "</div>")
        pt = r["pool_table"]
        if len(pt) <= 1:
            p.append(f"<p class='mut'>單一設定報告：母池固定為 {r['chosen_pool']['rule']}{r['chosen_pool']['size']}，未做母池比較。</p>")
            pt = pt.assign(score=pt["score"].fillna(1.0))
        hm = pt.pivot_table(index="rule", columns="size", values="score")
        hm.index = [f"{i} {RULE_ZH.get(i, '')}" for i in hm.index]
        p.append("<h3>母池測試（開發期綜合分數，越綠越好）</h3>" + heat_table(hm, "{:.2f}"))
        hc = pt.pivot_table(index="rule", columns="size", values="dev_cagr")
        hc.index = [f"{i} {RULE_ZH.get(i, '')}" for i in hc.index]
        p.append("<p class='mut'>同一模型在各母池的開發期 CAGR：</p>" + heat_table(hc))
        vt = r["variant_table"][["variant", "dev_cagr", "dev_sharpe", "dev_mdd", "dev_excess_cagr", "hold_cagr", "hold_mdd", "turnover", "score"]]
        p.append("<h3>模型比較（同一母池）</h3>" + table(vt.rename(columns=ZH), {ZH[k]: v for k, v in {
            "dev_cagr": "pct", "dev_mdd": "pct", "dev_excess_cagr": "pct", "hold_cagr": "pct", "hold_mdd": "pct",
            "dev_sharpe": "{:.2f}", "turnover": "pct0", "score": "{:.2f}"}.items()}, left=("模型",)))
        p.append("<p class='mut'>挑選規則：只在科斯托蘭尼版本中挑開發期綜合分數最高者；DATA/PRIOR 為對照組。</p>")
        p.append("<h3>科斯托蘭尼規則消融（把一條規則切到相反狀態，看績效變化）</h3>" + table(
            r["ablation"].drop(columns=["key"], errors="ignore").rename(columns=ZH), {ZH.get(k, k): v for k, v in {
                "dev_cagr": "pct", "dev_mdd": "pct", "hold_cagr": "pct", "hold_mdd": "pct", "dev_sharpe": "{:.2f}",
                "turnover": "pct0", "貢獻(開發期CAGR)": "pct", "貢獻(盲測CAGR)": "pct"}.items()}, left=("情境",)))
        p.append("<p class='mut'>貢獻＝規則開啟 − 規則關閉的 CAGR（正＝有幫助）。盲測欄只供檢查，從未用來挑選。</p>")
        fw = r["family_final"]
        p.append("<h3>因子與權重（最新一次估計）</h3><p>族群權重：" + "、".join(f"{k}={v:.0%}" for k, v in fw.items())
                 + "（T技術、C籌碼、F基本、K科斯托蘭尼）</p>")
        ft = r["factor_table"]
        ft = ft[ft["horizon_ok"]][["factor", "中文", "family", "coverage", "ic_dev", "t_dev", "ic_holdout", "ic_all", "weight"]]
        p.append(table(ft.rename(columns=ZH), {ZH[k]: v for k, v in {"coverage": "pct0", "ic_dev": "{:.3f}", "t_dev": "{:.1f}",
                 "ic_holdout": "{:.3f}", "ic_all": "{:.3f}", "weight": "{:+.3f}"}.items()}, left=("中文",)))
        p.append("<p class='mut'>IC＝因子排名與未來報酬排名的相關係數（台股常見 0.02~0.06 已屬有用）；t 值絕對值 >2 較可信。"
                 "權重 0 代表訓練資料與先驗方向相反或不顯著，系統不會硬把方向翻轉。</p>")
        st = r["stage_table"]
        if len(st):
            p.append("<h3>依市場階段分組的單批績效</h3>" + table(st.rename(columns=ZH), {ZH[k]: v for k, v in {
                "avg_ret": "pct", "avg_bench": "pct", "avg_excess": "pct", "avg_exposure": "pct0"}.items()}))
            p.append("<p class='mut'>檢查雞蛋階段是否有預測力：A1/B3（買點）之後的 0050 報酬應高於 A3/B1（賣點）之後。</p>")
        hist = pd.DataFrame(r["fit_history"]).drop(columns=["weights"]).rename(columns=ZH)
        p.append("<h3>權重重估歷程（走勢前進）</h3>" + table(hist, {c: "{:.2f}" for c in hist.columns if c.startswith("fam_") or c == "資料權重α"}))
        p.append("<p class='mut'>α 越接近 1 越相信資料、越接近 0 越貼近先驗權重；長線獨立樣本少，α 自動較低。</p>")
    # ---------------- comparison
    if comparison is not None and comparison[1] is not None and len(comparison[1]):
        p.append("<h2>5. 新舊模型比較（同一引擎、同一進出場規則）</h2>")
        s = comparison[1]
        p.append(table(s, {c: "pct" for c in s.columns if "報酬" in c or "平均" in c or "新減舊" in c} |
                       {c: "pct0" for c in s.columns if "勝" in c or "涵蓋" in c} | {"差異t值": "{:.2f}"}))
        p.append("<p class='mut'>新模型取「舊模型訊號日之前最近一次」的推薦，以相同進場日計算；t 值 > 2 才算顯著差異。</p>")
    # ---------------- Kostolany map
    p.append("<h2>6. 《一個投機者的告白》觀念 → 程式規則 → 實測貢獻</h2>")
    rows = []
    for idea, impl, cases in KOSTO_MAP:
        contrib = []
        for h, r in H.items():
            ab = r["ablation"]
            if "key" not in ab:
                continue
            ab = ab[ab["key"].isin(cases)]
            if len(ab):
                st = "/".join(sorted(set(ab["目前"]))) if "目前" in ab else ""
                contrib.append(f"{r['label'][:2]} 開發{pct(np.nansum(ab['貢獻(開發期CAGR)']))} 盲測{pct(np.nansum(ab.get('貢獻(盲測CAGR)', np.nan)))}"
                               + (f"（{st}）" if st else ""))
        rows.append({"書中觀念": idea, "程式實作": impl, "規則對CAGR的貢獻(正=有幫助)": "；".join(contrib) or "（不可單獨關閉）"})
    p.append(table(pd.DataFrame(rows), left=("書中觀念", "程式實作", "規則對CAGR的貢獻(正=有幫助)")))
    p.append("<p class='mut'>貢獻＝規則開啟時的 CAGR − 關閉時的 CAGR。括號為目前預設狀態；『關閉』代表實測後預設不用，但仍列出若加回的影響。</p>")
    # ---------------- hints
    p.append("<h2>7. 自動調整建議</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in hints(H)) + "</ul>")
    p.append("<p class='mut'>調整方式：修改 config/default.yaml（或 Streamlit「快速調整」頁），重新執行即可；每次報告都附 config_used.yaml 方便回溯。</p>")
    # ---------------- method
    p.append("<h2>8. 方法與限制</h2><ul>"
             "<li>資料：證交所/櫃買中心/公開資訊觀測站逐日全市場（或 FinMind）；含已下市股票，無存活者偏誤（FinMind 逐檔模式除外）。</li>"
             "<li>報酬：以官方『除權息參考價』還原，股利再投入；訊號日收盤決策、次日開盤成交、持有 H 日於開盤出場；開盤漲停買不到視為現金。</li>"
             f"<li>成本：手續費 {cfg['costs']['fee']*100:.4f}%×{cfg['costs']['fee_discount']} 折、證交稅 {cfg['costs']['tax']*100:.1f}%、滑價單邊 {cfg['costs']['slippage']*100:.1f}%。</li>"
             "<li>權重：每 6 個月以過去已到期資料重估。v1.2 中長線預設『精簡因子模型』：只用訓練期 IC-IR t 值最高的 k 個因子"
             "（方向須與經濟直覺一致），權重 ∝ IC-IR；另比較階層式 IC-IR（科斯托蘭尼族群下限 λ）與科斯托蘭尼加強版，以開發期分數挑選。</li>"
             "<li>月營收以法定申報期限（次月10日）後才可使用；本益比/淨值比為交易所每日公布值。</li>"
             "<li>雞蛋理論的「持有人數」以融資餘額代理、「媒體氣氛」以爆量急漲與融資暴增代理；新聞反應與貨幣供給未觀察。</li>"
             "<li>歷史回測不保證未來績效；本工具僅供研究，不構成投資建議，也不會自動下單。</li></ul>")
    p.append("<h2>附錄：本次設定</h2><pre style='font-size:.75rem;overflow-x:auto'>" + html.escape(dump_yaml(cfg)) + "</pre>")
    p.append("</body></html>")
    return "".join(p)


def _xls_name(s: str) -> str:
    for ch in "[]:*?/\\":
        s = s.replace(ch, "_")
    return s[:31]


def build_excel(res: dict, comparison=None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summ = []
        for h, r in res["horizons"].items():
            row = {"期間": r["label"], "母池": f"{r['chosen_pool']['rule']}{r['chosen_pool']['size']}", "模型": r["chosen"]["variant"]}
            row.update({k: v for k, v in r["eval"].items() if isinstance(v, (int, float, np.floating, str, bool))})
            row.update({f"robust_{k}": v for k, v in r["robust"].items()})
            summ.append(row)
        pd.DataFrame(summ).to_excel(xw, sheet_name="總覽", index=False)
        for h, r in res["horizons"].items():
            tag = {"short": "短", "mid": "中", "long": "長"}[h]
            lt = r["latest"]
            pd.DataFrame([dict(x, reasons="；".join(x["reasons"]), risk="；".join(x["risk"])) for x in lt["picks"]]).to_excel(
                xw, sheet_name=_xls_name(f"{tag}_今日推薦"), index=False)
            pd.DataFrame([dict(x, reasons="；".join(x["reasons"]), risk="；".join(x["risk"])) for x in lt["alternates"]]).to_excel(
                xw, sheet_name=_xls_name(f"{tag}_備選"), index=False)
            r["pool_table"].to_excel(xw, sheet_name=_xls_name(f"{tag}_母池測試"), index=False)
            r["variant_table"].to_excel(xw, sheet_name=_xls_name(f"{tag}_模型比較"), index=False)
            r["ablation"].to_excel(xw, sheet_name=_xls_name(f"{tag}_消融"), index=False)
            r["factor_table"].to_excel(xw, sheet_name=_xls_name(f"{tag}_因子IC權重"), index=False)
            wh = pd.DataFrame([{"from": x["from"], **x["weights"]} for x in r["fit_history"]])
            wh.to_excel(xw, sheet_name=_xls_name(f"{tag}_權重歷程"), index=False)
            r["stage_table"].to_excel(xw, sheet_name=_xls_name(f"{tag}_階段績效"), index=False)
            r["yearly"].to_excel(xw, sheet_name=_xls_name(f"{tag}_年度"))
            if r.get("validation"):
                r["validation"]["gates"].drop(columns=["fmt"]).to_excel(xw, sheet_name=_xls_name(f"{tag}_驗收閘門"), index=False)
                r["validation"]["folds"].to_excel(xw, sheet_name=_xls_name(f"{tag}_時序區塊"), index=False)
            tr = r["tranches"].copy()
            tr["picks"] = tr["picks"].map(lambda x: "|".join(x))
            tr.to_excel(xw, sheet_name=_xls_name(f"{tag}_每期推薦紀錄"), index=False)
            r["nav"].to_excel(xw, sheet_name=_xls_name(f"{tag}_淨值"))
        mk = res["market"].copy()
        mk.to_excel(xw, sheet_name="市場階段")
        if comparison is not None and comparison[0] is not None and len(comparison[0]):
            comparison[1].to_excel(xw, sheet_name="新舊比較_摘要", index=False)
            comparison[0].to_excel(xw, sheet_name="新舊比較_逐期", index=False)
    return buf.getvalue()


def summary_json(res: dict) -> dict:
    out = {"meta": res["meta"], "protocol": res["protocol"], "horizons": {}}
    for h, r in res["horizons"].items():
        out["horizons"][h] = {
            "label": r["label"], "chosen_pool": r["chosen_pool"], "chosen": r["chosen"],
            "eval": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r["eval"].items()
                     if isinstance(v, (int, float, np.floating, str, bool))},
            "robust": r["robust"], "family_final": r["family_final"], "latest": r["latest"],
            "gates": (r["validation"]["gates"].drop(columns=["fmt"]).to_dict("records") if r.get("validation") else None),
        }
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def build_zip(res: dict, cfg: dict, comparison=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.html", build_html(res, cfg, comparison))
        z.writestr("report.xlsx", build_excel(res, comparison))
        z.writestr("config_used.yaml", dump_yaml(cfg))
        z.writestr("results_summary.json", json.dumps(summary_json(res), ensure_ascii=False, indent=1))
        for h, r in res["horizons"].items():
            tr = r["tranches"][["date", "stage", "exposure", "ret", "bench", "picks"]].copy()
            tr.insert(1, "horizon", r["days"])
            tr["selected"] = tr.pop("picks").map(lambda x: "|".join(x))
            z.writestr(f"picks_history_{h}.csv", tr.to_csv(index=False).encode("utf-8-sig"))
            z.writestr(f"nav_{h}.csv", r["nav"].to_csv().encode("utf-8-sig"))
        z.writestr("market_stage.csv", res["market"].to_csv().encode("utf-8-sig"))
        z.writestr("README.txt", (
            "Kosto Lab 測試報告\n"
            "report.html          完整報告（手機/電腦瀏覽器開啟）\n"
            "report.xlsx          所有表格（母池測試、模型比較、因子IC、權重歷程、每期推薦…）\n"
            "config_used.yaml     本次設定；修改後貼回 config/default.yaml 即可重跑\n"
            "picks_history_*.csv  每期推薦清單（可匯入其他系統或 Kosto Lab 的「新舊比較」）\n"
            "nav_*.csv            每日淨值；market_stage.csv 市場雞蛋階段\n").encode("utf-8-sig"))
    return buf.getvalue()
