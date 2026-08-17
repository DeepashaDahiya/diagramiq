#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "          🚀 DIAGRAMIQ - PRODUCTION LAUNCHER (LINUX/MAC)"
echo "======================================================================"
echo ""

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "[2/3] Checking Frontend Production Build..."
if [ ! -f "frontend/build/index.html" ]; then
    echo "[*] Frontend build not detected. Compiling React frontend..."
    cd frontend
    npm install --prefer-offline --no-audit
    npm run build
    cd ..
else
    echo "[+] Production build verified at frontend/build"
fi

echo ""
echo "[3/3] Starting DiagramIQ WSGI Server on http://localhost:5000..."
echo "======================================================================"
echo "[*] Open your browser at: http://localhost:5000"
echo "[*] Press CTRL+C to stop the server"
echo "======================================================================"
echo ""

if command -v gunicorn >/dev/null 2>&1; then
    exec gunicorn --workers 2 --threads 4 --bind 0.0.0.0:5000 --timeout 120 backend.wsgi:app
else
    exec python backend/wsgi.py
fi
