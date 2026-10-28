"""GitHub Releases as the data/results store (no git-history bloat), + workflow dispatch."""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import requests

from .store import GROUPS

DATA_TAG = "kosto-data"
RESEARCH_TAG = "kosto-research"
API = "https://api.github.com"


def _headers(token: str | None, raw=False) -> dict:
    h = {"Accept": "application/octet-stream" if raw else "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_asset(repo: str, tag: str, name: str, token: str | None = None, timeout: int = 120) -> bytes:
    """Download a release asset. Public repos work without a token; private repos need one."""
    if token:
        r = requests.get(f"{API}/repos/{repo}/releases/tags/{tag}", headers=_headers(token), timeout=30)
        r.raise_for_status()
        asset = next((a for a in r.json().get("assets", []) if a["name"] == name), None)
        if asset is None:
            raise FileNotFoundError(f"{tag}/{name} 不存在")
        r = requests.get(asset["url"], headers=_headers(token, raw=True), timeout=timeout)
        r.raise_for_status()
        return r.content
    r = requests.get(f"https://github.com/{repo}/releases/download/{tag}/{name}", timeout=timeout)
    if r.status_code == 404:
        raise FileNotFoundError(f"{tag}/{name} 不存在（私人 repo 需要 GITHUB_TOKEN）")
    r.raise_for_status()
    return r.content


def release_info(repo: str, tag: str, token: str | None = None) -> dict:
    r = requests.get(f"{API}/repos/{repo}/releases/tags/{tag}", headers=_headers(token), timeout=30)
    if r.status_code != 200:
        return {}
    j = r.json()
    return {"published_at": j.get("published_at"),
            "assets": {a["name"]: {"size": a["size"], "updated_at": a["updated_at"]} for a in j.get("assets", [])}}


def upload_asset(repo: str, tag: str, path: str, token: str) -> None:
    """Upload (replace) a release asset with plain HTTPS — no gh CLI needed (local fallback runs)."""
    r = requests.get(f"{API}/repos/{repo}/releases/tags/{tag}", headers=_headers(token), timeout=30)
    if r.status_code == 404:
        r = requests.post(f"{API}/repos/{repo}/releases", headers=_headers(token), timeout=30,
                          json={"tag_name": tag, "name": tag, "body": "Kosto Lab store", "make_latest": "false"})
    r.raise_for_status()
    rel = r.json()
    name = Path(path).name
    for a in rel.get("assets", []):
        if a["name"] == name:
            requests.delete(a["url"], headers=_headers(token), timeout=30).raise_for_status()
    up = rel["upload_url"].split("{")[0]
    with open(path, "rb") as f:
        h = _headers(token)
        h["Content-Type"] = "application/octet-stream"
        requests.post(up, headers=h, params={"name": name}, data=f, timeout=600).raise_for_status()


def dispatch(repo: str, workflow_file: str, token: str, ref: str = "main", inputs: dict | None = None) -> None:
    r = requests.post(f"{API}/repos/{repo}/actions/workflows/{workflow_file}/dispatches",
                      headers=_headers(token), json={"ref": ref, "inputs": inputs or {}}, timeout=30)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"啟動失敗 HTTP {r.status_code}: {r.text[:200]}")


def latest_runs(repo: str, token: str | None, n: int = 5) -> list:
    r = requests.get(f"{API}/repos/{repo}/actions/runs", headers=_headers(token), params={"per_page": n}, timeout=30)
    if r.status_code != 200:
        return []
    return [{"name": x["name"], "status": x["status"], "conclusion": x["conclusion"], "created": x["created_at"],
             "url": x["html_url"]} for x in r.json().get("workflow_runs", [])]


# ------------------------------------------------------------------ group packs (used by Actions)
def pack(data_dir: str, group: str, out: str) -> str:
    data = Path(data_dir)
    with tarfile.open(out, "w:gz") as t:
        for ds in GROUPS[group]:
            d = data / "raw" / ds
            if d.is_dir():
                t.add(d, arcname=f"raw/{ds}")
        st = data / f"state_{group}.json"
        if st.is_file():
            t.add(st, arcname=st.name)
        for extra in ("SYNTHETIC_DEMO_DATA",):
            if (data / extra).is_file():
                t.add(data / extra, arcname=extra)
    return out


def unpack(data_dir: str, archive: str) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as t:
        members = [m for m in t.getmembers() if not (m.name.startswith("/") or ".." in Path(m.name).parts)]
        t.extractall(data_dir, members=members)


def token_from_env() -> str | None:
    return os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def bytes_io(b: bytes) -> io.BytesIO:
    return io.BytesIO(b)
