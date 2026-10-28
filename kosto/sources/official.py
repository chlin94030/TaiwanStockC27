"""Free official whole-market sources: TWSE (上市), TPEx (上櫃), MOPS (月營收).

One HTTP request returns ALL stocks for one date (or one month / one date range), so a
full 2017→today build needs ~20k requests instead of ~2,000 stocks × datasets.
Every endpoint and field layout below was verified against live payloads (2017, 2019, 2024, 2026).

Parsing rules learned from the live data:
* TWSE MI_INDEX: 漲跌(+/-) is HTML ('<p style= color:red>+</p>'); ex-right/dividend days show 'X'
  with 漲跌價差 0.00 → the reference price must come from TWT49U (除權除息計算結果表).
* TPEx otc quotes: 漲跌 is '+0.32', '+ 0.80' (2017), '0.00', or '除息'/'除權'/'除權息'/'---'.
  TPEx ex-rights reference prices come from bulletin/exDailyQ. The quote table also has 發行股數.
* T86 and TPEx dailyTrade changed layout around 2017-12 (foreign dealer split out).
* Holidays: TWSE stat '很抱歉，沒有符合條件的資料!'; TPEx stat 'ok' with zero rows.
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests

TWSE = "https://www.twse.com.tw"
TPEX = "https://www.tpex.org.tw"
MOPS = "https://mopsov.twse.com.tw"
TDCC = "https://opendata.tdcc.com.tw"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}
STOCK_RE = re.compile(r"^[0-9]{4}$")          # common stocks
KEEP_RE = re.compile(r"^(?:[1-9][0-9]{3}|0050)$")  # stocks + 0050 benchmark


class SourceError(RuntimeError):
    """Endpoint failed (network / block / format). The builder may fall back to FinMind."""


# --------------------------------------------------------------------------- helpers
def num(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"<[^>]+>", "", str(x)).replace(",", "").replace(" ", "").strip()
    if s in ("", "-", "--", "---", "----", "N/A", "X"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean(x) -> str:
    return re.sub(r"<[^>]+>", "", str(x)).replace("　", " ").strip()


def norm(x) -> str:
    return re.sub(r"[\s()（）/]", "", clean(x))


def find_col(fields, *cands, exclude=()) -> int | None:
    nf = [norm(f) for f in fields]
    for c in cands:
        c = norm(c)
        for i, f in enumerate(nf):
            if f == c and not any(e in f for e in exclude):
                return i
    for c in cands:
        c = norm(c)
        for i, f in enumerate(nf):
            if c in f and not any(e in f for e in exclude):
                return i
    return None


def roc_to_date(s: str) -> pd.Timestamp | None:
    """'113年06月13日' / '113/07/01' / '108/01/10' -> Timestamp."""
    m = re.findall(r"\d+", str(s))
    if len(m) < 3:
        return None
    y, mo, d = int(m[0]), int(m[1]), int(m[2])
    if y < 1911:
        y += 1911
    try:
        return pd.Timestamp(y, mo, d)
    except ValueError:
        return None


class Http:
    """Polite session with pacing + retries. Raises SourceError when blocked."""

    def __init__(self, pause: float = 2.6, retries: int = 3, timeout: int = 30, log=print):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.pause, self.retries, self.timeout, self.log = pause, retries, timeout, log
        self._last = 0.0
        self.calls = 0

    def get(self, url: str, params=None, as_json=True, encoding=None):
        err = None
        for attempt in range(self.retries):
            wait = self.pause - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                self.calls += 1
                if r.status_code in (403, 429, 503):
                    raise SourceError(f"HTTP {r.status_code} (可能被暫時封鎖)")
                r.raise_for_status()
                if encoding:
                    r.encoding = encoding
                if not as_json:
                    return r.text
                txt = r.text.strip()
                if txt.startswith("<"):
                    raise SourceError("收到 HTML 而非 JSON（網站維護或封鎖）")
                return r.json()
            except (requests.RequestException, ValueError, SourceError) as e:
                err = e
                backoff = self.pause * (3 ** attempt) + 2
                host = url.split("/")[2] if "//" in url else url
                why = str(e) if isinstance(e, SourceError) else type(e).__name__
                self.log(f"  ! {host} 連線失敗（{why[:60]}），{backoff:.0f} 秒後重試")
                time.sleep(backoff)
        raise SourceError(str(err))


def _ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _slash(d) -> str:
    return pd.Timestamp(d).strftime("%Y/%m/%d")


# --------------------------------------------------------------------------- TWSE parsers
def _twse_tables(payload: dict) -> list[tuple[list, list]]:
    out = []
    for t in payload.get("tables") or []:
        if t.get("fields") and t.get("data") is not None:
            out.append((t["fields"], t["data"]))
    if payload.get("fields") and payload.get("data") is not None:
        out.append((payload["fields"], payload["data"]))
    for n in range(1, 12):  # legacy fieldsN/dataN shape
        if payload.get(f"fields{n}") and payload.get(f"data{n}") is not None:
            out.append((payload[f"fields{n}"], payload[f"data{n}"]))
    return out


def _twse_empty(payload: dict) -> bool:
    return str(payload.get("stat", "")).upper() != "OK"


def parse_twse_price(payload: dict, d) -> pd.DataFrame:
    if _twse_empty(payload):
        return pd.DataFrame()
    rows, taiex = [], None
    for fields, data in _twse_tables(payload):
        if norm(fields[0]) == "指數":
            for r in data:
                if clean(r[0]) == "發行量加權股價指數":
                    sign = clean(r[2])
                    ch = num(r[3])
                    ch = -ch if sign == "-" else ch
                    taiex = {"date": pd.Timestamp(d), "ticker": "TAIEX", "board": "INDEX", "name": "加權指數",
                             "close": num(r[1]), "change": ch, "event": ""}
            continue
        ic = find_col(fields, "證券代號")
        if ic is None or find_col(fields, "收盤價") is None:
            continue
        col = {k: find_col(fields, *v) for k, v in {
            "name": ("證券名稱",), "volume": ("成交股數",), "amount": ("成交金額",), "open": ("開盤價",),
            "high": ("最高價",), "low": ("最低價",), "close": ("收盤價",), "sign": ("漲跌(+/-)",),
            "chg": ("漲跌價差",)}.items()}
        for r in data:
            code = clean(r[ic])
            if not KEEP_RE.match(code):
                continue
            sign = clean(r[col["sign"]]) if col["sign"] is not None else ""
            ch = num(r[col["chg"]]) if col["chg"] is not None else np.nan
            event = ""
            if sign == "X":
                event, ch = "X", np.nan
            elif sign == "-":
                ch = -ch
            elif sign not in ("+",):
                ch = 0.0 if ch == 0 or np.isnan(ch) else ch  # blank sign = unchanged
            rows.append({"date": pd.Timestamp(d), "ticker": code, "board": "TWSE", "name": clean(r[col["name"]]),
                         "open": num(r[col["open"]]), "high": num(r[col["high"]]), "low": num(r[col["low"]]),
                         "close": num(r[col["close"]]), "volume": num(r[col["volume"]]),
                         "amount": num(r[col["amount"]]), "change": ch, "event": event})
    if taiex:
        rows.append(taiex)
    return pd.DataFrame(rows)


def parse_twse_exright(payload: dict) -> pd.DataFrame:
    if _twse_empty(payload):
        return pd.DataFrame()
    rows = []
    for fields, data in _twse_tables(payload):
        idt, ic = find_col(fields, "資料日期"), find_col(fields, "股票代號")
        ib, ir = find_col(fields, "除權息前收盤價"), find_col(fields, "除權息參考價")
        if None in (idt, ic, ib, ir):
            continue
        for r in data:
            code = clean(r[ic])
            if KEEP_RE.match(code):
                rows.append({"date": roc_to_date(r[idt]), "ticker": code, "board": "TWSE",
                             "before": num(r[ib]), "ref": num(r[ir])})
    f = pd.DataFrame(rows)
    return f.dropna(subset=["date"]) if not f.empty else f


def parse_twse_inst(payload: dict, d) -> pd.DataFrame:
    if _twse_empty(payload):
        return pd.DataFrame()
    rows = []
    for fields, data in _twse_tables(payload):
        ic = find_col(fields, "證券代號")
        if ic is None:
            continue
        nf = [norm(f) for f in fields]
        f_idx = [i for i, f in enumerate(nf) if "外" in f and "買賣超" in f]
        t_idx = [i for i, f in enumerate(nf) if "投信" in f and "買賣超" in f]
        d_all = [i for i, f in enumerate(nf) if f == "自營商買賣超股數"]
        d_idx = d_all or [i for i, f in enumerate(nf) if "自營商買賣超" in f and "外" not in f]
        tot = [i for i, f in enumerate(nf) if "三大法人" in f]
        for r in data:
            code = clean(r[ic])
            if not KEEP_RE.match(code):
                continue
            foreign = sum(num(r[i]) for i in f_idx)
            trust = sum(num(r[i]) for i in t_idx)
            dealer = sum(num(r[i]) for i in d_idx)
            total = num(r[tot[0]]) if tot else foreign + trust + dealer
            rows.append({"date": pd.Timestamp(d), "ticker": code, "foreign": foreign, "trust": trust,
                         "dealer": dealer, "total": total})
    return pd.DataFrame(rows)


def parse_twse_margin(payload: dict, d) -> pd.DataFrame:
    if _twse_empty(payload):
        return pd.DataFrame()
    rows = []
    for fields, data in _twse_tables(payload):
        if norm(fields[0]) != "代號" or len(fields) < 14:
            continue
        # positional (field names repeat for 融資/融券): 6=融資今日餘額 7=融資限額 12=融券今日餘額
        for r in data:
            code = clean(r[0])
            if KEEP_RE.match(code):
                rows.append({"date": pd.Timestamp(d), "ticker": code, "mbal": num(r[6]), "mlimit": num(r[7]),
                             "sbal": num(r[12])})
    return pd.DataFrame(rows)


def parse_twse_valuation(payload: dict, d) -> pd.DataFrame:
    if _twse_empty(payload):
        return pd.DataFrame()
    rows = []
    for fields, data in _twse_tables(payload):
        ic = find_col(fields, "證券代號")
        ipe, ipb, idy = find_col(fields, "本益比"), find_col(fields, "股價淨值比"), find_col(fields, "殖利率(%)", "殖利率")
        if ic is None or ipe is None:
            continue
        for r in data:
            code = clean(r[ic])
            if KEEP_RE.match(code):
                rows.append({"date": pd.Timestamp(d), "ticker": code, "pe": num(r[ipe]),
                             "pb": num(r[ipb]) if ipb is not None else np.nan,
                             "dy": num(r[idy]) if idy is not None else np.nan})
    return pd.DataFrame(rows)


def parse_twse_shares(payload: dict, d) -> pd.DataFrame:
    if _twse_empty(payload):
        return pd.DataFrame()
    rows = []
    for fields, data in _twse_tables(payload):
        ic, isz = find_col(fields, "證券代號"), find_col(fields, "發行股數")
        ifr = find_col(fields, "全體外資及陸資持股比率")
        if ic is None or isz is None:
            continue
        for r in data:
            code = clean(r[ic])
            if KEEP_RE.match(code):
                rows.append({"date": pd.Timestamp(d), "ticker": code, "shares": num(r[isz]),
                             "foreign_ratio": num(r[ifr]) if ifr is not None else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- TPEx parsers
def _tpex_tables(payload: dict):
    for t in payload.get("tables") or []:
        if t.get("fields") and t.get("data"):
            yield t["fields"], t["data"]
    if payload.get("aaData"):
        yield None, payload["aaData"]


def parse_tpex_price(payload: dict, d) -> pd.DataFrame:
    rows = []
    for fields, data in _tpex_tables(payload):
        if fields is None or find_col(fields, "代號") is None:
            continue
        c = {k: find_col(fields, *v) for k, v in {
            "code": ("代號",), "name": ("名稱",), "close": ("收盤",), "chg": ("漲跌",), "open": ("開盤",),
            "high": ("最高",), "low": ("最低",), "volume": ("成交股數",), "amount": ("成交金額(元)", "成交金額"),
            "shares": ("發行股數",)}.items()}
        for r in data:
            code = clean(r[c["code"]])
            if not KEEP_RE.match(code):
                continue
            raw = clean(r[c["chg"]]).replace(" ", "")
            event, ch = "", num(raw)
            if "除" in raw:
                event, ch = raw, np.nan
            rows.append({"date": pd.Timestamp(d), "ticker": code, "board": "TPEX", "name": clean(r[c["name"]]),
                         "open": num(r[c["open"]]), "high": num(r[c["high"]]), "low": num(r[c["low"]]),
                         "close": num(r[c["close"]]), "volume": num(r[c["volume"]]), "amount": num(r[c["amount"]]),
                         "change": ch, "event": event,
                         "shares": num(r[c["shares"]]) if c["shares"] is not None else np.nan})
    return pd.DataFrame(rows)


def parse_tpex_exright(payload: dict) -> pd.DataFrame:
    rows = []
    for fields, data in _tpex_tables(payload):
        if fields is None:
            continue
        idt, ic = find_col(fields, "除權息日期"), find_col(fields, "代號")
        ib, ir = find_col(fields, "除權息前收盤價"), find_col(fields, "除權息參考價")
        if None in (idt, ic, ib, ir):
            continue
        for r in data:
            code = clean(r[ic])
            if KEEP_RE.match(code):
                rows.append({"date": roc_to_date(r[idt]), "ticker": code, "board": "TPEX",
                             "before": num(r[ib]), "ref": num(r[ir])})
    f = pd.DataFrame(rows)
    return f.dropna(subset=["date"]) if not f.empty else f


def parse_tpex_inst(payload: dict, d) -> pd.DataFrame:
    rows = []
    for fields, data in _tpex_tables(payload):
        if fields is None or norm(fields[0]) != "代號":
            continue
        nf = [norm(f) for f in fields]
        if len(fields) >= 24:      # 2018+ positional layout (names repeat)
            pos = {"foreign": 10, "trust": 13, "dealer": 22, "total": 23}
        else:                      # 2017 named layout
            pos = {"foreign": next((i for i, f in enumerate(nf) if "外資" in f and ("淨買" in f or "買賣超" in f)), None),
                   "trust": next((i for i, f in enumerate(nf) if "投信" in f and ("淨買" in f or "買賣超" in f)), None),
                   "dealer": next((i for i, f in enumerate(nf) if f.startswith("自營") and "淨買" in f and "自行" not in f
                                   and "避險" not in f), None),
                   "total": next((i for i, f in enumerate(nf) if "三大法人" in f), None)}
        for r in data:
            code = clean(r[0])
            if not KEEP_RE.match(code):
                continue
            vals = {k: (num(r[i]) if i is not None else np.nan) for k, i in pos.items()}
            rows.append({"date": pd.Timestamp(d), "ticker": code, **vals})
    return pd.DataFrame(rows)


def parse_tpex_margin(payload: dict, d) -> pd.DataFrame:
    rows = []
    for fields, data in _tpex_tables(payload):
        if fields is None or norm(fields[0]) != "代號":
            continue
        ib, il, isb = find_col(fields, "資餘額"), find_col(fields, "資限額"), find_col(fields, "券餘額")
        for r in data:
            code = clean(r[0])
            if KEEP_RE.match(code):
                rows.append({"date": pd.Timestamp(d), "ticker": code, "mbal": num(r[ib]), "mlimit": num(r[il]),
                             "sbal": num(r[isb])})
    return pd.DataFrame(rows)


def parse_tpex_valuation(payload: dict, d) -> pd.DataFrame:
    rows = []
    for fields, data in _tpex_tables(payload):
        if fields is None:
            continue
        ic, ipe = find_col(fields, "股票代號", "代號"), find_col(fields, "本益比")
        ipb, idy = find_col(fields, "股價淨值比"), find_col(fields, "殖利率(%)", "殖利率")
        if ic is None or ipe is None:
            continue
        for r in data:
            code = clean(r[ic])
            if KEEP_RE.match(code):
                rows.append({"date": pd.Timestamp(d), "ticker": code, "pe": num(r[ipe]),
                             "pb": num(r[ipb]) if ipb is not None else np.nan,
                             "dy": num(r[idy]) if idy is not None else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- MOPS revenue
_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TD = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)


def parse_mops_revenue(html: str, year: int, month: int) -> pd.DataFrame:
    """t21sc03 static page: 公司代號, 公司名稱, 當月營收(千元), 上月營收, 去年當月營收, ..."""
    rows = []
    for tr in _TR.findall(html):
        tds = [clean(x) for x in _TD.findall(tr)]
        if len(tds) >= 5 and STOCK_RE.match(tds[0]):
            rev = num(tds[2])
            if np.isfinite(rev):
                rows.append({"ticker": tds[0], "ym": year * 100 + month, "revenue": rev * 1000.0})
    return pd.DataFrame(rows)


def parse_tdcc_holders(text: str) -> pd.DataFrame:
    """集保戶股權分散表（每週）→ 每檔一列：大戶(≥400張)、千張大戶、散戶(<50張) 持股比例與股東人數。

    分級 1~15 = 1-999股、1-5張、5-10、10-15、15-20、20-30、30-40、40-50、50-100、100-200、200-400、400-600、
    600-800、800-1000、>1000 張；17 = 合計。"""
    rows = {}
    for line in text.lstrip("\ufeff").splitlines()[1:]:
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6 or not p[0].isdigit():
            continue
        code = p[1]
        if not KEEP_RE.match(code):
            continue
        try:
            lv, n, pct = int(p[2]), float(p[3]), float(p[5])
        except ValueError:
            continue
        r = rows.setdefault(code, {"date": pd.Timestamp(p[0]), "ticker": code, "big400": 0.0, "big1000": 0.0,
                                   "retail50": 0.0, "holders": np.nan})
        if 12 <= lv <= 15:
            r["big400"] += pct
        if lv == 15:
            r["big1000"] += pct
        if 1 <= lv <= 8:
            r["retail50"] += pct
        if lv == 17:
            r["holders"] = n
    return pd.DataFrame(list(rows.values()))


def revenue_available_date(ym: int) -> pd.Timestamp:
    """Monthly revenue must be filed by the 10th of the following month (point-in-time rule)."""
    y, m = divmod(int(ym), 100)
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return pd.Timestamp(y, m, 10)


# --------------------------------------------------------------------------- fetchers
class Official:
    """High-level fetchers. Each returns a parsed DataFrame (possibly empty = holiday)."""

    def __init__(self, http: Http):
        self.http = http

    # TWSE
    def twse_price(self, d):
        return parse_twse_price(self.http.get(f"{TWSE}/rwd/zh/afterTrading/MI_INDEX",
                                              {"date": _ymd(d), "type": "ALLBUT0999", "response": "json"}), d)

    def twse_inst(self, d):
        return parse_twse_inst(self.http.get(f"{TWSE}/rwd/zh/fund/T86",
                                             {"date": _ymd(d), "selectType": "ALLBUT0999", "response": "json"}), d)

    def twse_margin(self, d):
        return parse_twse_margin(self.http.get(f"{TWSE}/rwd/zh/marginTrading/MI_MARGN",
                                               {"date": _ymd(d), "selectType": "ALL", "response": "json"}), d)

    def twse_valuation(self, d):
        return parse_twse_valuation(self.http.get(f"{TWSE}/rwd/zh/afterTrading/BWIBBU_d",
                                                  {"date": _ymd(d), "selectType": "ALL", "response": "json"}), d)

    def twse_shares(self, d):
        return parse_twse_shares(self.http.get(f"{TWSE}/rwd/zh/fund/MI_QFIIS",
                                               {"date": _ymd(d), "selectType": "ALLBUT0999", "response": "json"}), d)

    def twse_exright(self, start, end):
        return parse_twse_exright(self.http.get(f"{TWSE}/rwd/zh/exRight/TWT49U",
                                                {"startDate": _ymd(start), "endDate": _ymd(end), "response": "json"}))

    # TPEx
    def tpex_price(self, d):
        return parse_tpex_price(self.http.get(f"{TPEX}/www/zh-tw/afterTrading/otc",
                                              {"date": _slash(d), "type": "EW", "response": "json"}), d)

    def tpex_inst(self, d):
        return parse_tpex_inst(self.http.get(f"{TPEX}/www/zh-tw/insti/dailyTrade",
                                             {"type": "Daily", "sect": "EW", "date": _slash(d), "response": "json"}), d)

    def tpex_margin(self, d):
        return parse_tpex_margin(self.http.get(f"{TPEX}/www/zh-tw/margin/balance",
                                               {"date": _slash(d), "response": "json"}), d)

    def tpex_valuation(self, d):
        return parse_tpex_valuation(self.http.get(f"{TPEX}/www/zh-tw/afterTrading/peQryDate",
                                                  {"date": _slash(d), "response": "json"}), d)

    def tpex_exright(self, start, end):
        return parse_tpex_exright(self.http.get(f"{TPEX}/www/zh-tw/bulletin/exDailyQ",
                                                {"startDate": _slash(start), "endDate": _slash(end), "response": "json"}))

    # TDCC 集保戶股權分散表（最新一週，全市場）
    def tdcc_holders(self) -> pd.DataFrame:
        return parse_tdcc_holders(self.http.get(f"{TDCC}/getOD.ashx", {"id": "1-5"}, as_json=False, encoding="utf-8"))

    # MOPS
    def revenue(self, year: int, month: int) -> pd.DataFrame:
        roc = year - 1911
        parts = []
        for market in ("sii", "otc"):
            for kind in (0, 1):  # 0 = 國內公司, 1 = KY 外國公司
                url = f"{MOPS}/nas/t21/{market}/t21sc03_{roc}_{month}_{kind}.html"
                try:
                    html = self.http.get(url, as_json=False, encoding="big5")
                except SourceError:
                    if kind == 1:
                        continue
                    raise
                parts.append(parse_mops_revenue(html, year, month))
        f = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not f.empty:
            f = f.drop_duplicates(["ticker", "ym"], keep="last")
            f["avail_date"] = revenue_available_date(year * 100 + month)
        return f


DAILY_DATASETS = {
    # dataset: (twse method, tpex method)
    "price": ("twse_price", "tpex_price"),
    "institutional": ("twse_inst", "tpex_inst"),
    "margin": ("twse_margin", "tpex_margin"),
    "valuation": ("twse_valuation", "tpex_valuation"),
}


def weekdays(start, end) -> list[pd.Timestamp]:
    return list(pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end)))


def today_tw() -> date:
    return (datetime.utcnow() + pd.Timedelta(hours=8)).date()
