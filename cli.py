"""Kosto Lab command line (used by GitHub Actions and for local runs).

  python cli.py demo [--stocks 500] [--quick]            合成資料示範：建資料→bundle→研究→報告
  python cli.py update --group price [--budget 330]       官方/FinMind 資料增量更新（可續傳）
  python cli.py bundle                                    由資料庫建立研究 bundle（因子、母池、報酬）
  python cli.py research [--quick] [--compare old.zip]    完整研究 + 報告 ZIP
  python cli.py status                                    資料庫狀態
  python cli.py auto                                      本機一鍵：建資料(可續傳)→完成後自動研究+報告
  python cli.py progress [--write status/status.json]     資料庫完成度（GitHub 自動續跑用）
  python cli.py pack --group price --out x.tar.gz         打包某資料群組（Actions 用）
  python cli.py unpack x.tar.gz                           解開資料群組
"""
from __future__ import annotations

import argparse
import os
import warnings
import pickle
import sys
import time
from pathlib import Path

from kosto.config import HORIZON_KEYS, load_config

ART = Path("artifacts")
warnings.filterwarnings("ignore", category=RuntimeWarning)   # "mean of empty slice" etc. — harmless, keeps logs readable


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def cmd_update(a, cfg):
    from kosto.build import build
    token = os.getenv("FINMIND_TOKEN", "")
    rep = build(cfg, a.group, token, a.budget, log=log)
    log(f"完成：{rep}")


def cmd_bundle(a, cfg):
    from kosto.bundle import build_bundle
    from kosto.panel import build_panel
    from kosto.store import Store
    st = Store(cfg["data"]["dir"])
    log(st.summary().to_string())
    P = build_panel(st, start=cfg["data"]["start"], end=cfg["data"].get("end"))
    log(f"panel: {len(P.dates)} 日 × {len(P.tickers)} 檔；品質 {P.quality}")
    B = build_bundle(P, cfg, log=log)
    ART.mkdir(exist_ok=True)
    B.save(ART / "bundle.npz")
    if a.compare_panel:
        from kosto.compare import Daily
        import numpy as np
        d = Daily.from_panel(P, start=cfg["research"]["start"])
        np.savez_compressed(ART / "daily_all.npz", R=d.R, OC=d.OC, OK=d.OK, bench=d.bench,
                            days=np.array([str(x.date()) for x in d.days]), tickers=np.array(d.tickers))
    log(f"bundle 已存：{ART/'bundle.npz'} ({(ART/'bundle.npz').stat().st_size/1e6:.1f} MB)")


def cmd_research(a, cfg):
    from kosto.bundle import Bundle
    from kosto.report import build_zip
    from kosto.research import Research
    B = Bundle.load(a.bundle)
    R = Research(B, cfg, log=log)
    hz = a.horizons.split(",") if a.horizons else list(HORIZON_KEYS)
    res = R.run_all(hz, quick=a.quick, progress=log)
    comparison = None
    if a.compare:
        comparison = run_compare(a.compare, res, B, cfg)
    ART.mkdir(exist_ok=True)
    with open(ART / "results.pkl", "wb") as f:
        pickle.dump({"res": res, "cfg": cfg, "comparison": comparison}, f, protocol=4)
    (ART / "report.zip").write_bytes(build_zip(res, cfg, comparison))
    from kosto.report import summary_json
    import json
    (ART/'desk_summary.json').write_text(json.dumps(summary_json(res),ensure_ascii=False),encoding='utf-8')
    log(f"報告：{ART/'report.zip'}  ({res['elapsed']:.0f}s)")


def run_compare(paths, res, B, cfg):
    import numpy as np
    import pandas as pd
    from kosto.compare import Daily, head_to_head, parse_external
    frames = []
    for p in paths:
        frames.append(parse_external(p, Path(p).read_bytes()))
    ext = pd.concat(frames, ignore_index=True)
    daily = Daily.from_bundle(B)
    if (ART / "daily_all.npz").is_file():
        z = np.load(ART / "daily_all.npz")
        daily = Daily(pd.DatetimeIndex(z["days"]), list(z["tickers"]), z["R"], z["OC"], z["OK"], z["bench"])
    ours = {r["days"]: r["tranches"] for r in res["horizons"].values()}
    per, summ = head_to_head(daily, ext, ours, cfg["costs"])
    log(summ.to_string())
    return per, summ


def cmd_demo(a, cfg):
    from kosto.sources.synthetic import write_demo
    from kosto.store import Store
    d = Path(cfg["data"]["dir"])
    if a.fresh and d.exists():
        import shutil
        shutil.rmtree(d)
    st = Store(d)
    if not (d / "SYNTHETIC_DEMO_DATA").exists():
        log(f"產生合成資料 {a.stocks} 檔…")
        log(str(write_demo(st, a.stocks, a.seed)))
    a.compare_panel = False
    cmd_bundle(a, cfg)
    a.bundle = str(ART / "bundle.npz")
    a.compare = None
    cmd_research(a, cfg)


def cmd_summary(a, cfg):
    """Markdown summary (GitHub Actions job summary)."""
    with open(ART / "results.pkl", "rb") as f:
        res = pickle.load(f)["res"]
    lines = ["## Kosto Lab 研究摘要", "", f"資料至 {res['protocol']['last_date']}｜盲測起點 {res['protocol']['holdout_start']}", "",
             "| 期間 | 母池 | 模型 | 開發期CAGR | 0050 | 盲測CAGR | 0050 | 市場階段 | 今日推薦 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for h, r in res["horizons"].items():
        e, lt = r["eval"], r["latest"]
        f = lambda x: "–" if x is None or x != x else f"{x:+.1%}"  # noqa: E731
        picks = " ".join(f"{x['ticker']}{x['name']}" for x in lt["picks"][:10])
        lines.append(f"| {r['label']} | {r['chosen_pool']['rule']}{r['chosen_pool']['size']} | {r['chosen']['variant']} | "
                     f"{f(e.get('dev_cagr'))} | {f(e.get('dev_bench_cagr'))} | {f(e.get('hold_cagr'))} | "
                     f"{f(e.get('hold_bench_cagr'))} | {lt['market_stage']} | {picks} |")
    print("\n".join(lines))


def write_status(cfg, path, extra=None):
    import json
    from kosto.build import progress
    st = progress(cfg)
    st.update(extra or {})
    if (ART / "results.pkl").is_file():
        try:
            with open(ART / "results.pkl", "rb") as f:
                res = pickle.load(f)["res"]
            st["research"] = {"last_date": res["protocol"]["last_date"], "built": res["meta"].get("built"),
                              "synthetic": bool(res["meta"].get("synthetic"))}
        except Exception:  # noqa: BLE001
            pass
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    if "research" not in st and "research" in old:
        st["research"] = old["research"]
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return st


def cmd_progress(a, cfg):
    from kosto.build import progress
    st = write_status(cfg, a.write, {"stage": a.stage} if a.stage else None) if a.write else progress(cfg)
    log(f"資料完成度 {st['percent']:.1%}：" + "、".join(f"{k} {v:.0%}" for k, v in st["groups"].items()))
    for k, v in st.get("blocked", {}).items():
        log(f"  ⚠ {k}: {v}")
    if a.github_output and os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"complete={'true' if st['complete'] else 'false'}\n")
    print("COMPLETE" if st["complete"] else "INCOMPLETE")


def cmd_auto(a, cfg):
    """Local one-click: build/continue the database, then research + report when complete."""
    from kosto.build import auto_build
    token = os.getenv("FINMIND_TOKEN", "")
    status = Path("status/status.json")
    write_status(cfg, status, {"stage": "downloading"})
    p = auto_build(cfg, token, a.budget, log=log)
    write_status(cfg, status, {"stage": "downloading" if not p["complete"] else "researching"})
    if not p["complete"] and not a.force_research:
        log(f"資料尚未完整（{p['percent']:.0%}）。再執行一次即可續傳。")
        return
    a.compare_panel = True
    cmd_bundle(a, cfg)
    a.bundle = str(ART / "bundle.npz")
    a.compare = [str(x) for x in Path("compare").glob("*") if x.suffix.lower() in (".csv", ".zip")] or None
    a.horizons = None
    cmd_research(a, cfg)
    write_status(cfg, status, {"stage": "done"})


def cmd_status(a, cfg):
    from kosto.store import Store
    print(Store(cfg["data"]["dir"]).summary().to_string())


def cmd_validate_v27(a,cfg):
    from kosto.bundle import Bundle
    from kosto.validate_v27 import save
    path=save(Bundle.load(a.bundle),cfg,a.out,log);log('V27驗證報告：'+str(path))


def main(argv=None):
    p = argparse.ArgumentParser(description="Kosto Lab 科斯托蘭尼量化選股實驗室")
    p.add_argument("--config", default=None, help="自訂 YAML（會覆蓋 config/default.yaml 的同名欄位）")
    p.add_argument("--data-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update"); u.add_argument("--group", required=True); u.add_argument("--budget", type=float, default=330)
    b = sub.add_parser("bundle"); b.add_argument("--compare-panel", action="store_true")
    r = sub.add_parser("research")
    r.add_argument("--bundle", default=str(ART / "bundle.npz")); r.add_argument("--quick", action="store_true")
    r.add_argument("--horizons", default=None); r.add_argument("--compare", nargs="*", default=None)
    d = sub.add_parser("demo")
    d.add_argument("--stocks", type=int, default=500); d.add_argument("--seed", type=int, default=7)
    d.add_argument("--quick", action="store_true"); d.add_argument("--fresh", action="store_true")
    d.add_argument("--horizons", default=None)
    sub.add_parser("status")
    sub.add_parser("summary")
    vv=sub.add_parser("validate-v27");vv.add_argument("--bundle",default=str(ART/"bundle.npz"));vv.add_argument("--out",default=str(ART/"v27_validation.json"))
    pg = sub.add_parser("progress"); pg.add_argument("--write", default=None); pg.add_argument("--stage", default=None)
    pg.add_argument("--github-output", action="store_true")
    au = sub.add_parser("auto"); au.add_argument("--budget", type=float, default=600)
    au.add_argument("--quick", action="store_true"); au.add_argument("--force-research", action="store_true")
    pk = sub.add_parser("pack"); pk.add_argument("--group", required=True); pk.add_argument("--out", required=True)
    up = sub.add_parser("unpack"); up.add_argument("archive", nargs="+")
    a = p.parse_args(argv)
    over = {"data": {"dir": a.data_dir}} if a.data_dir else None
    if a.cmd == "demo" and not a.data_dir:
        over = {"data": {"dir": "data_demo"}}
    cfg = load_config(a.config, over)
    if a.cmd == "pack":
        from kosto.remote import pack
        print(pack(cfg["data"]["dir"], a.group, a.out))
        return
    if a.cmd == "unpack":
        from kosto.remote import unpack
        for x in a.archive:
            if Path(x).is_file():
                unpack(cfg["data"]["dir"], x)
                log(f"解開 {x}")
        return
    {"update": cmd_update, "bundle": cmd_bundle, "research": cmd_research, "demo": cmd_demo,
     "status": cmd_status, "summary": cmd_summary, "validate-v27":cmd_validate_v27,
     "progress": cmd_progress, "auto": cmd_auto}[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
