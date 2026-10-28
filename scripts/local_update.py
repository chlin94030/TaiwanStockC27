"""備援：若 GitHub Actions 的 IP 被證交所/公開資訊觀測站擋住，改在自己的電腦（台灣 IP）更新資料並上傳。

用法（Windows / Mac 皆可）：
    pip install -r requirements-pipeline.txt
    set GITHUB_REPO=你的帳號/kosto-lab        (Mac: export GITHUB_REPO=...)
    set GITHUB_TOKEN=github_pat_xxx           (Fine-grained token：Contents read/write)
    set FINMIND_TOKEN=xxx                     (可選)
    python scripts/local_update.py            (預設每組最多 330 分鐘，可中斷，下次會續傳)
之後到 GitHub Actions 手動執行 pipeline（mode=full），研究會讀取你上傳的資料。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kosto.build import build  # noqa: E402
from kosto.config import load_config  # noqa: E402
from kosto.remote import DATA_TAG, fetch_asset, pack, unpack, upload_asset  # noqa: E402
from kosto.store import GROUPS  # noqa: E402

repo, token = os.environ["GITHUB_REPO"], os.environ["GITHUB_TOKEN"]
budget = float(os.getenv("BUDGET_MINUTES", "330"))
cfg = load_config()
for g in GROUPS:
    name = f"data-{g}.tar.gz"
    try:
        Path(name).write_bytes(fetch_asset(repo, DATA_TAG, name, token))
        unpack(cfg["data"]["dir"], name)
        print("已下載既有", name)
    except FileNotFoundError:
        print("首次建立", name)
    print(build(cfg, g, os.getenv("FINMIND_TOKEN", ""), budget))
    pack(cfg["data"]["dir"], g, name)
    upload_asset(repo, DATA_TAG, name, token)
    print("已上傳", name)
