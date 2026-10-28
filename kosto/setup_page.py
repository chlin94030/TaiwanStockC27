"""Zero-configuration helpers for the Streamlit app.

* finds the GitHub repo by itself (no Secrets needed for public repos)
* reads the build progress written by the workflow (status/status.json)
* on a personal computer, starts/continues the local one-click build in the background
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / "status" / "status.json"
LOCAL_LOG = ROOT / "status" / "auto.log"
GROUP_ZH = {"price": "股價", "institutional": "三大法人", "margin": "融資融券", "valuation": "本益比/殖利率",
            "misc": "除權息/股本/月營收"}


def on_streamlit_cloud() -> bool:
    if os.getenv("KOSTO_FORCE_CLOUD"):
        return os.getenv("KOSTO_FORCE_CLOUD") == "1"
    return str(ROOT).startswith("/mount/src") or Path("/home/adminuser").exists()


def repo_from_git() -> str:
    cfg = ROOT / ".git" / "config"
    if not cfg.is_file():
        return ""
    m = re.search(r"url\s*=\s*\S*github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?\s*$", cfg.read_text(errors="ignore"), re.M)
    return m.group(1) if m else ""


def detect_repo(secret_value: str = "", query_value: str = "") -> tuple[str, str]:
    """(repo, how_found)."""
    for v, how in ((secret_value, "secrets"), (os.getenv("GITHUB_REPO", ""), "env"), (query_value, "網址參數"),
                   (repo_from_git(), "自動偵測")):
        v = (v or "").strip().strip("/")
        if re.fullmatch(r"[\w.-]+/[\w.-]+", v):
            return v, how
    return "", ""


WORKFLOW_COPY = ROOT / "pipeline_workflow_COPY.txt"   # visible copy (Windows/Mac may hide or skip .github)
WORKFLOW_PATH = ".github/workflows/pipeline.yml"


def workflow_text() -> str:
    for p in (ROOT / WORKFLOW_PATH, WORKFLOW_COPY):
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return ""


def workflow_present(repo: str, token: str | None = None):
    """True / False / None(unknown) — is the automation file on GitHub?"""
    h = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        for branch in ("HEAD", "main", "master"):
            r = requests.get(f"https://raw.githubusercontent.com/{repo}/{branch}/{WORKFLOW_PATH}", headers=h, timeout=10)
            if r.status_code == 200:
                return True
        return False
    except requests.RequestException:
        return None


def render_add_workflow(st, repo: str) -> None:
    """Three clicks: open GitHub's new-file page (path pre-filled) → copy → paste & commit."""
    st.error("GitHub 上缺少自動流程檔（`.github` 資料夾沒有上傳成功，Windows/Mac 常會隱藏或略過它）。"
             "請照下面 3 步補上，約 1 分鐘：")
    st.markdown(f"**① 按這個連結**（會打開 GitHub「新增檔案」頁，檔名已自動填好）："
                f"[➜ 在 GitHub 新增流程檔](https://github.com/{repo}/new/main?filename={WORKFLOW_PATH})")
    st.markdown("**② 按下方框框右上角的「複製」圖示**，回到 GitHub 頁面，在大的編輯框按 **Ctrl+V（Mac：⌘+V）** 貼上")
    st.code(workflow_text() or "（找不到流程檔內容：請改用 kosto-lab 資料夾中的 pipeline_workflow_COPY.txt）", language="yaml")
    st.markdown("**③ 按 GitHub 右上角綠色「Commit changes…」→ 再按一次「Commit changes」**。"
                "完成後約 1~5 分鐘會自動開始，本頁按「重新整理進度」即可看到進度條。")
    st.caption(f"若連結打開後檔名欄是空的，請手動輸入：{WORKFLOW_PATH}（打「/」會自動變成資料夾）")


def read_status(repo: str = "", token: str | None = None) -> dict:
    """Newest of: GitHub raw file (fresh) or the local checkout copy."""
    best = {}
    if repo:
        h = {"Authorization": f"Bearer {token}"} if token else {}
        for branch in ("HEAD", "main", "master"):
            try:
                r = requests.get(f"https://raw.githubusercontent.com/{repo}/{branch}/status/status.json",
                                 headers=h, timeout=10)
                if r.status_code == 200:
                    best = r.json()
                    break
            except (requests.RequestException, ValueError):
                break
    if STATUS.is_file():
        try:
            local = json.loads(STATUS.read_text(encoding="utf-8"))
            if local.get("checked_at", "") > best.get("checked_at", ""):
                best = local
        except ValueError:
            pass
    return best


def eta_hours(status: dict, local: bool) -> float:
    groups = status.get("groups", {})
    if not groups:
        return 5.0 if not local else 8.0
    slowest = min(groups.get(g, 0) for g in ("price", "institutional", "margin", "valuation"))
    if local:
        return (1 - status.get("percent", 0)) * 8.0
    return (1 - slowest) * 4.5


# ------------------------------------------------------------------ local background build
def local_running() -> bool:
    return LOCAL_LOG.is_file() and time.time() - LOCAL_LOG.stat().st_mtime < 300


def start_local_build() -> None:
    LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOCAL_LOG, "a", encoding="utf-8")
    kw = {"cwd": str(ROOT), "stdout": log, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    subprocess.Popen([sys.executable, "-u", "cli.py", "auto"], **kw)


def tail_log(n: int = 12) -> str:
    if not LOCAL_LOG.is_file():
        return ""
    lines = LOCAL_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def local_progress() -> dict:
    sys.path.insert(0, str(ROOT))
    from kosto.build import progress
    from kosto.config import load_config
    return progress(load_config())


def render_progress(st, status: dict, repo: str, local: bool) -> None:
    """Shared progress view (cloud or local)."""
    stage = status.get("stage", "")
    pct = float(status.get("percent", 0) or 0)
    if stage == "done" or status.get("research"):
        st.success("✅ 資料與研究已完成。若畫面沒更新，請按右上角 ⋮ → Rerun，或稍候 1~2 分鐘。")
    st.progress(min(pct, 1.0), text=f"資料庫完成度 {pct:.0%}" + (f"（約剩 {eta_hours(status, local):.1f} 小時）" if pct < 1 else ""))
    cols = st.columns(5)
    for k, (g, zh) in enumerate(GROUP_ZH.items()):
        cols[k].metric(zh, f"{float(status.get('groups', {}).get(g, 0)):.0%}")
    blocked = {k: v for k, v in status.get("blocked", {}).items() if v}
    if blocked:
        st.warning("部分官方網站拒絕連線（可能暫時封鎖了伺服器 IP）：\n\n" +
                   "\n".join(f"- {k}：{v}" for k, v in list(blocked.items())[:6]) +
                   "\n\n系統會自動重試；若一天後仍未改善，請改用「本機一鍵啟動」（見說明第 3 部分），"
                   "或在 GitHub 加入 FINMIND_TOKEN 讓系統改走 FinMind。")
    if stage == "stuck":
        st.error("已自動續跑 8 輪仍未完成。請依說明「常見問題」處理，或改用本機一鍵啟動。")
    if status.get("checked_at"):
        st.caption(f"最後更新：{status['checked_at']}")
    if repo and not local:
        st.markdown(f"執行紀錄：[github.com/{repo}/actions](https://github.com/{repo}/actions)")
