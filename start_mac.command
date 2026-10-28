#!/bin/bash
# Alpha Radar PRO — Mac 一鍵啟動（第一次請在檔案上按右鍵 → 打開）
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "找不到 Python。請到 https://www.python.org/downloads/ 下載 Python 3.12 以上安裝後，再雙擊本檔一次。"
  open "https://www.python.org/downloads/"
  read -p "按 Enter 結束"
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "第一次執行：建立獨立環境..."
  python3 -m venv .venv
fi
if [ ! -f .venv/kosto_ok ]; then
  echo "第一次執行：安裝套件，約 3~5 分鐘，請稍候..."
  .venv/bin/python -m pip install -q --upgrade pip && .venv/bin/python -m pip install -q -r requirements.txt || {
    echo "套件安裝失敗，請確認網路後再試一次"; read -p "按 Enter 結束"; exit 1; }
  touch .venv/kosto_ok
fi
echo "=========================================================="
echo "  Alpha Radar PRO 啟動中，瀏覽器會自動開啟 http://localhost:8501"
echo "  使用期間請不要關閉這個視窗，資料更新期間請保持電腦開機"
echo "=========================================================="
(sleep 8; open "http://localhost:8501") &
.venv/bin/python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
