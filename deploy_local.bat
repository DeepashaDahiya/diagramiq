@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo           🚀 DIAGRAMIQ - PRODUCTION LAUNCHER (WINDOWS)
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python dependencies...
python -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [!] Warning: Pip install returned a non-zero code. Proceeding with installed packages...
)

echo.
echo [2/3] Checking Frontend Production Build...
if not exist "frontend\build\index.html" (
    echo [*] Frontend build not detected. Compiling React frontend...
    cd frontend
    call npm run build
    cd ..
) else (
    echo [+] Production build verified at frontend\build
)

echo.
echo [3/3] Starting DiagramIQ Production WSGI Server on http://localhost:5000...
echo ======================================================================
echo [*] Open your browser at: http://localhost:5000
echo [*] Press CTRL+C to stop the server
echo ======================================================================
echo.

python backend\wsgi.py
pause
