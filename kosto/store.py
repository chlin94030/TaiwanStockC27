"""Long-format Parquet store, one file per dataset per year + a JSON progress ledger.

Layout:  <dir>/raw/<dataset>/<year>.parquet     (columns: date, ticker, ...)
         <dir>/raw/revenue/all.parquet           (monthly; small)
         <dir>/state.json                         (dates done / empty, per-ticker FinMind progress)
Writes are atomic (tmp file + replace) so an interrupted GitHub Actions job never corrupts data.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pandas as pd

KEYS = {
    "price": ["date", "ticker"],
    "exright": ["date", "ticker"],
    "institutional": ["date", "ticker"],
    "margin": ["date", "ticker"],
    "valuation": ["date", "ticker"],
    "shares": ["date", "ticker"],
    "revenue": ["ticker", "ym"],
    "info": ["ticker"],
    "holders": ["date", "ticker"],
}
_LOCK = threading.RLock()  # two download threads (TWSE / TPEx) may write the same dataset files
YEARLY = {"price", "exright", "institutional", "margin", "valuation", "shares", "holders"}
# Download groups = one GitHub Actions matrix job + one release asset each (parallel, no write conflicts)
GROUPS = {
    "price": ["price"],
    "institutional": ["institutional"],
    "margin": ["margin"],
    "valuation": ["valuation"],
    "misc": ["exright", "shares", "revenue", "info", "holders"],
}


def _merge(a, b):
    """Union of two state dicts: lists of done keys are unioned, dicts merged recursively, b wins otherwise."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge(a[k], v) if k in a else v
        return out
    if isinstance(a, list) and isinstance(b, list):
        return sorted(set(a) | set(b))
    return b


class Store:
    def __init__(self, root: str | Path, state_name: str = "main"):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / f"state_{state_name}.json"
        self._state = None

    # ---------------- state ----------------
    @property
    def state(self) -> dict:
        if self._state is None:
            if self.state_path.is_file():
                self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            else:
                self._state = {}
        return self._state

    def save_state(self) -> None:
        """Merge with what is on disk (another thread/process may have saved) and write atomically."""
        with _LOCK:
            disk = {}
            if self.state_path.is_file():
                try:
                    disk = json.loads(self.state_path.read_text(encoding="utf-8"))
                except ValueError:
                    disk = {}
            merged = _merge(disk, self.state)
            self._state = merged
            tmp = self.state_path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.state_path)

    def done_set(self, key: str) -> set:
        return set(self.state.get("done", {}).get(key, []))

    def mark_done(self, key: str, items) -> None:
        d = self.state.setdefault("done", {})
        cur = set(d.get(key, []))
        cur.update(items)
        d[key] = sorted(cur)

    # ---------------- tables ----------------
    def _path(self, dataset: str, year: int | None = None) -> Path:
        folder = self.raw / dataset
        folder.mkdir(parents=True, exist_ok=True)
        return folder / (f"{year}.parquet" if dataset in YEARLY else "all.parquet")

    def append(self, dataset: str, frame: pd.DataFrame) -> int:
        """Upsert rows; later rows win on duplicate keys. Returns rows written."""
        if frame is None or frame.empty:
            return 0
        with _LOCK:
            return self._append(dataset, frame)

    def _append(self, dataset: str, frame: pd.DataFrame) -> int:
        frame = frame.copy()
        keys = KEYS[dataset]
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["ticker"] = frame["ticker"].astype(str)
        groups = frame.groupby(frame["date"].dt.year) if dataset in YEARLY else [(None, frame)]
        n = 0
        for year, part in groups:
            path = self._path(dataset, year)
            if path.is_file():
                old = pd.read_parquet(path)
                part = pd.concat([old, part], ignore_index=True)
            part = part.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
            tmp = path.with_suffix(".tmp")
            part.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            n += len(part)
        return n

    def read(self, dataset: str, start=None, end=None, columns=None) -> pd.DataFrame:
        folder = self.raw / dataset
        if not folder.is_dir():
            return pd.DataFrame()
        files = sorted(folder.glob("*.parquet"))
        if dataset in YEARLY and (start or end):
            y0 = pd.Timestamp(start).year if start else 0
            y1 = pd.Timestamp(end).year if end else 9999
            files = [f for f in files if f.stem.isdigit() and y0 <= int(f.stem) <= y1]
        if not files:
            return pd.DataFrame()
        out = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
        if "date" in out and (start or end):
            m = pd.Series(True, index=out.index)
            if start:
                m &= out["date"] >= pd.Timestamp(start)
            if end:
                m &= out["date"] <= pd.Timestamp(end)
            out = out[m]
        return out.reset_index(drop=True)

    def dates(self, dataset: str) -> set:
        f = self.read(dataset, columns=["date"])
        return set() if f.empty else set(pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d"))

    def summary(self) -> pd.DataFrame:
        rows = []
        for ds in KEYS:
            f = self.read(ds)
            if f.empty:
                rows.append({"dataset": ds, "rows": 0, "tickers": 0, "first": None, "last": None})
                continue
            col = "date" if "date" in f else ("avail_date" if "avail_date" in f else None)
            rows.append({"dataset": ds, "rows": len(f), "tickers": f["ticker"].nunique(),
                         "first": str(pd.to_datetime(f[col]).min().date()) if col else None,
                         "last": str(pd.to_datetime(f[col]).max().date()) if col else None})
        return pd.DataFrame(rows)
