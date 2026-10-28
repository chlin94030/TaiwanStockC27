@echo off
chcp 65001 >nul
title Alpha Radar PRO
cd /d "%~dp0"

set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
  python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo [!] 找不到 Python。即將開啟下載頁面：
  echo     請下載 Python 3.12 以上，安裝第一個畫面務必勾選「Add python.exe to PATH」，
  echo     安裝完成後再雙擊本檔一次。
  start "" https://www.python.org/downloads/
  pause
  exit /b
)

if not exist ".venv\Scripts\python.exe" (
  echo 第一次執行：建立獨立環境...
  %PY% -m venv .venv
)
if not exist ".venv\kosto_ok" (
  echo 第一次執行：安裝套件，約 3~5 分鐘，請稍候...
  ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
  if errorlevel 1 (
    echo [!] 套件安裝失敗，請確認網路連線後再雙擊本檔一次。
    pause
    exit /b
  )
  echo ok> ".venv\kosto_ok"
)

echo.
echo ==========================================================
echo   Alpha Radar PRO 啟動中，瀏覽器會自動開啟 http://localhost:8501
echo   使用期間請不要關閉這個黑色視窗（關閉 = 關掉 App）
echo   資料更新期間請保持電腦開機，完成後再關閉
echo ==========================================================
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause
