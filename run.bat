@echo off
chcp 65001 >nul
title AutoDock Studio - Molecular Docking Platform
cd /d "%~dp0"

echo ===================================================================
echo   AutoDock Studio: Autonomous Molecular Docking and Publication
echo ===================================================================
echo.
echo [1/3] Configuring UTF-8 Python environment...

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

set "PY_CMD=python"
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
)

echo [2/3] Opening AutoDock Studio in browser (http://localhost:8501)...
start /b "" cmd /c "ping 127.0.0.1 -n 3 >nul & start http://localhost:8501"

echo [3/3] Launching AutoDock Studio Server...
echo.
echo -------------------------------------------------------------------
echo  AutoDock Studio is running!
echo  URL: http://localhost:8501
echo  Press Ctrl+C in this terminal window to stop the server.
echo -------------------------------------------------------------------
echo.

"%PY_CMD%" -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

if %errorlevel% neq 0 (
    echo.
    echo ===================================================================
    echo [ERROR] AutoDock Studio encountered an error.
    echo ===================================================================
    pause
)
