"""Incremental, resumable, time-budgeted data builder (designed for GitHub Actions 6-hour jobs).

Each download group (see store.GROUPS) keeps its own state file so the four daily groups can run
as parallel matrix jobs. Official whole-market endpoints are used first; if they are blocked for a
run, the builder falls back to FinMind per-stock requests (when a token is available).
"""
from __future__ import annotations

import threading
import time

import pandas as pd

from .sources.finmind import FinMind, QuotaExhausted
from .sources.official import DAILY_DATASETS, Http, Official, SourceError, revenue_available_date, today_tw, weekdays
from .store import GROUPS, Store

RECENT_DAYS = 4  # empty results this close to today are retried next run (data may not be published yet)


class Builder:
    def __init__(self, cfg: dict, group: str, token: str = "", budget_minutes: float = 330, log=print, end=None,
                 boards=("TWSE", "TPEX")):
        if group not in GROUPS:
            raise ValueError(f"group must be one of {list(GROUPS)}")
        self.cfg, self.group, self.log, self.boards = cfg, group, log, tuple(boards)
        self.store = Store(cfg["data"]["dir"], state_name=group)
        self.deadline = time.time() + budget_minutes * 60
        self.http = Http(cfg["data"]["pause_sec"], cfg["data"]["max_retries"], log=log)
        self.off = Official(self.http)
        self.fm = FinMind(token, cfg["data"]["finmind_hourly_quota"], log=log, stop_at=self.deadline) if token else None
        self.start = pd.Timestamp(cfg["data"]["start"])
        self.end = pd.Timestamp(end or cfg["data"].get("end") or today_tw())
        self.recent_cut = self.end - pd.Timedelta(days=RECENT_DAYS)

    def out_of_time(self) -> bool:
        return time.time() > self.deadline

    # ------------------------------------------------------------------ entry
    def run(self) -> dict:
        report = {}
        for ds in GROUPS[self.group]:
            if self.out_of_time():
                break
            t0 = time.time()
            if ds in DAILY_DATASETS:
                report[ds] = self.daily(ds)
            elif ds == "exright":
                report[ds] = self.exright()
            elif ds == "shares":
                report[ds] = self.shares()
            elif ds == "revenue":
                report[ds] = self.revenue()
            elif ds == "info":
                report[ds] = self.info()
            elif ds == "holders":
                report[ds] = self.holders()
            self.log(f"[{ds}] {report.get(ds)}  ({time.time()-t0:.0f}s, http={self.http.calls})")
        self.store.save_state()
        return report

    # ------------------------------------------------------------------ daily whole-market
    def daily(self, ds: str) -> dict:
        providers = self.cfg["data"]["providers"][ds]
        twse_m, tpex_m = DAILY_DATASETS[ds]
        stats = {"fetched": 0, "rows": 0, "blocked": False, "fallback": None}
        if "official" in providers:
            for board, method in (("TWSE", twse_m), ("TPEX", tpex_m)):
                if board not in self.boards:
                    continue
                key = f"{ds}:{board}"
                done = self.store.done_set(key)
                all_days = [d for d in weekdays(self.start, self.end)]
                todo = [d for d in all_days if d.strftime("%Y-%m-%d") not in done]
                # newest few first so a daily update is never starved by a long backfill
                todo = todo[-3:][::-1] + todo[:-3]
                buf, marks, fails = [], [], 0
                for i, d in enumerate(todo):
                    if self.out_of_time():
                        break
                    try:
                        f = getattr(self.off, method)(d)
                        fails = 0
                    except SourceError as e:
                        fails += 1
                        self.log(f"  {key} {d.date()} 失敗: {e}")
                        if fails >= 3:
                            stats["blocked"] = True
                            self._flag(key, str(e))
                            break
                        continue
                    if not f.empty:
                        buf.append(f)
                        marks.append(d.strftime("%Y-%m-%d"))
                    elif d < self.recent_cut:
                        marks.append(d.strftime("%Y-%m-%d"))  # holiday
                    stats["fetched"] += 1
                    if len(buf) >= 20 or (i == len(todo) - 1):
                        stats["rows"] += self._flush(ds, buf, key, marks)
                        buf, marks = [], []
                    if stats["fetched"] % 100 == 0:
                        self.log(f"  {key}: {stats['fetched']} 日完成，最新 {d.date()}")
                stats["rows"] += self._flush(ds, buf, key, marks)
                if not stats["blocked"]:
                    self._flag(key, None)
        if (stats["blocked"] or "official" not in providers) and "finmind" in providers and self.fm:
            stats["fallback"] = self.finmind_backfill(ds)
        return stats

    def _flag(self, key, err):
        b = self.store.state.setdefault("blocked", {})
        if err:
            b[key] = f"{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M} {err}"[:200]
        else:
            b[key] = ""
        self.store.save_state()

    def _flush(self, ds, buf, key, marks) -> int:
        n = 0
        if buf:
            n = sum(len(x) for x in buf)
            self.store.append(ds, pd.concat(buf, ignore_index=True))
        if marks:
            self.store.mark_done(key, marks)
            self.store.save_state()
        return n

    # ------------------------------------------------------------------ yearly ex-rights reference prices
    def exright(self) -> dict:
        stats = {"years": 0, "rows": 0}
        if "official" not in self.cfg["data"]["providers"]["exright"]:
            return self._fm_or_skip("exright")
        done = self.store.done_set("exright")
        for year in range(self.start.year, self.end.year + 1):
            if self.out_of_time():
                break
            if str(year) in done and year < self.end.year:
                continue
            s, e = pd.Timestamp(year, 1, 1), min(pd.Timestamp(year, 12, 31), self.end)
            try:
                f = pd.concat([self.off.twse_exright(s, e), self.off.tpex_exright(s, e)], ignore_index=True)
            except SourceError as err:
                self.log(f"  exright {year} 失敗: {err}")
                self._flag("exright", str(err))
                return dict(stats, fallback=self._fm_or_skip("exright"))
            self.store.append("exright", f)
            stats["rows"] += len(f)
            stats["years"] += 1
            if year < self.end.year:
                self.store.mark_done("exright", [str(year)])
        return stats

    # ------------------------------------------------------------------ shares (TWSE MI_QFIIS monthly; TPEx has shares in quotes)
    def shares(self) -> dict:
        stats = {"months": 0, "rows": 0}
        if "official" not in self.cfg["data"]["providers"]["shares"]:
            return self._fm_or_skip("shares")
        done = self.store.done_set("shares:TWSE")
        months = pd.period_range(self.start, self.end, freq="M")
        for m in months:
            if self.out_of_time():
                break
            key = str(m)
            if key in done and m < months[-1]:
                continue
            for d in pd.bdate_range(m.start_time, min(m.end_time, self.end))[:6]:
                try:
                    f = self.off.twse_shares(d)
                except SourceError as err:
                    self.log(f"  shares {d.date()} 失敗: {err}")
                    self._flag("shares:TWSE", str(err))
                    return dict(stats, fallback=self._fm_or_skip("shares"))
                if not f.empty:
                    self.store.append("shares", f)
                    stats["rows"] += len(f)
                    stats["months"] += 1
                    if m < months[-1]:
                        self.store.mark_done("shares:TWSE", [key])
                    break
        return stats

    # ------------------------------------------------------------------ monthly revenue (MOPS)
    def revenue(self) -> dict:
        stats = {"months": 0, "rows": 0}
        if "official" not in self.cfg["data"]["providers"]["revenue"]:
            return self._fm_or_skip("revenue")
        done = self.store.done_set("revenue")
        start = (self.start - pd.DateOffset(months=14)).to_period("M")  # YoY needs 12 months before start
        months = [m for m in pd.period_range(start, self.end, freq="M")
                  if revenue_available_date(m.year * 100 + m.month) <= self.end + pd.Timedelta(days=25)]
        for m in months:
            if self.out_of_time():
                break
            key = str(m)
            recent = revenue_available_date(m.year * 100 + m.month) > self.end - pd.Timedelta(days=40)
            if key in done and not recent:
                continue
            try:
                f = self.off.revenue(m.year, m.month)
            except SourceError as err:
                self.log(f"  revenue {key} 失敗: {err}")
                self._flag("revenue", str(err))
                return dict(stats, fallback=self._fm_or_skip("revenue"))
            if not f.empty:
                self.store.append("revenue", f)
                stats["rows"] += len(f)
                stats["months"] += 1
                if not recent:
                    self.store.mark_done("revenue", [key])
        return stats

    # ------------------------------------------------------------------ stock list / sector
    def info(self) -> dict:
        try:
            fm = self.fm or FinMind("", log=self.log)
            f = fm.info()
        except Exception as e:  # noqa: BLE001 - info is optional
            self.log(f"  info 失敗（不影響回測，只少了產業別）: {e}")
            return {"rows": 0}
        self.store.append("info", f)
        return {"rows": len(f)}

    # ------------------------------------------------------------------ TDCC weekly big-holder snapshot (選用)
    def holders(self) -> dict:
        """集保股權分散表只提供「最新一週」→ 每次執行抓一次、逐週累積歷史（抓不到不影響研究）。"""
        try:
            f = self.off.tdcc_holders()
        except Exception as e:  # noqa: BLE001 - optional dataset
            self.log(f"  holders 失敗（不影響研究，只少了大戶持股）: {e}")
            return {"rows": 0}
        self.store.append("holders", f)
        return {"rows": len(f), "date": str(f["date"].max().date()) if len(f) else None}

    # ------------------------------------------------------------------ FinMind per-stock fallback
    def _fm_or_skip(self, ds):
        if "finmind" in self.cfg["data"]["providers"].get(ds, []) and self.fm:
            return self.finmind_backfill(ds)
        return "skipped(no finmind token)"

    def _tickers(self) -> list[str]:
        info = self.store.read("info")
        if not info.empty:
            return sorted(info["ticker"].astype(str).unique())
        try:
            info = self.fm.info()
            self.store.append("info", info)
            return sorted(info["ticker"].unique())
        except Exception:  # noqa: BLE001
            return []

    def finmind_backfill(self, ds: str) -> dict:
        tickers = self._tickers()
        state = self.store.state.setdefault("finmind", {}).setdefault(ds, {})
        stats = {"tickers": 0, "rows": 0, "remaining": None}
        buf = []
        method = getattr(self.fm, ds)
        try:
            for i, t in enumerate(tickers):
                if self.out_of_time():
                    break
                last = state.get(t)
                s = (pd.Timestamp(last) + pd.Timedelta(days=1)) if last else self.start
                if ds == "revenue" and not last:
                    s = self.start - pd.DateOffset(months=14)
                if s > self.end:
                    continue
                f = method(t, s, self.end)
                if not f.empty:
                    buf.append(f)
                state[t] = str(self.end.date())
                stats["tickers"] += 1
                if len(buf) >= 50:
                    stats["rows"] += self._fm_flush(ds, buf)
                    buf = []
        except QuotaExhausted as e:
            self.log(f"  {e}")
        stats["rows"] += self._fm_flush(ds, buf)
        stats["remaining"] = sum(1 for t in tickers if state.get(t) != str(self.end.date()))
        return stats

    def _fm_flush(self, ds, buf) -> int:
        if not buf:
            return 0
        f = pd.concat(buf, ignore_index=True)
        self.store.append(ds, f)
        self.store.save_state()
        return len(f)


def build(cfg, group, token="", budget_minutes=330, log=print, end=None) -> dict:
    return Builder(cfg, group, token, budget_minutes, log, end).run()


# ====================================================================== progress (for the app & workflow)
def progress(cfg: dict, end=None) -> dict:
    """How complete is the database? Used to auto-continue the first build and to show a progress bar."""
    start = pd.Timestamp(cfg["data"]["start"])
    end = pd.Timestamp(end or cfg["data"].get("end") or today_tw())
    cut = end - pd.Timedelta(days=RECENT_DAYS)
    need_days = {d.strftime("%Y-%m-%d") for d in weekdays(start, cut)}
    out = {"groups": {}, "blocked": {}, "checked_at": str(pd.Timestamp.now(tz="Asia/Taipei").floor("s"))}
    info_n = None
    try:
        info = Store(cfg["data"]["dir"], "misc").read("info")
        info_n = len(info) if not info.empty else None
    except Exception:  # noqa: BLE001
        pass
    for g, dss in GROUPS.items():
        st = Store(cfg["data"]["dir"], state_name=g).state
        done = st.get("done", {})
        fmst = st.get("finmind", {})
        parts = []
        for ds in dss:
            if ds in DAILY_DATASETS:
                off = [len(need_days & set(done.get(f"{ds}:{b}", []))) / max(len(need_days), 1) for b in ("TWSE", "TPEX")]
                pct = sum(off) / 2
                if ds in fmst and info_n:
                    fm = sum(1 for v in fmst[ds].values() if pd.Timestamp(v) >= cut) / info_n
                    pct = max(pct, fm)
                parts.append(pct)
            elif ds == "exright":
                yrs = {str(y) for y in range(start.year, end.year)}
                parts.append(len(yrs & set(done.get("exright", []))) / max(len(yrs), 1) if yrs else 1.0)
            elif ds == "shares":
                ms = {str(m) for m in pd.period_range(start, end, freq="M")[:-1]}
                parts.append(len(ms & set(done.get("shares:TWSE", []))) / len(ms) if ms else 1.0)
            elif ds == "revenue":
                ms = {str(m) for m in pd.period_range(start - pd.DateOffset(months=14), end, freq="M")
                      if revenue_available_date(m.year * 100 + m.month) <= end - pd.Timedelta(days=40)}
                got = set(done.get("revenue", []))
                pct = len(ms & got) / len(ms) if ms else 1.0
                if "revenue" in fmst and info_n:
                    pct = max(pct, len(fmst["revenue"]) / info_n)
                parts.append(pct)
        pct = sum(parts) / len(parts) if parts else 0.0
        out["groups"][g] = round(min(pct, 1.0), 4)
        for k, v in st.get("blocked", {}).items():
            if v:
                out["blocked"][k] = v
    out["percent"] = round(sum(out["groups"].values()) / len(out["groups"]), 4)
    out["complete"] = all(v >= 0.995 for v in out["groups"].values())
    return out


def auto_build(cfg: dict, token: str = "", budget_minutes: float = 600, log=print) -> dict:
    """Local one-machine build: TWSE and TPEx run in two parallel threads (different websites,
    each still politely paced), then the small monthly/yearly tables. Safe to stop and rerun."""
    errors = []

    def board_worker(board):
        for ds in DAILY_DATASETS:
            try:
                Builder(cfg, ds, token, budget_minutes, log=lambda m, b=board: log(f"[{b}] {m}"), boards=(board,)).run()
            except Exception as e:  # noqa: BLE001
                errors.append(f"{board} {ds}: {e}")
                log(f"[{board}] {ds} 錯誤：{e}")

    th = [threading.Thread(target=board_worker, args=(b,), daemon=True) for b in ("TWSE", "TPEX")]
    for t in th:
        t.start()
    for t in th:
        t.join()
    try:
        Builder(cfg, "misc", token, budget_minutes, log=log).run()
    except Exception as e:  # noqa: BLE001
        errors.append(f"misc: {e}")
    p = progress(cfg)
    p["errors"] = errors
    return p
