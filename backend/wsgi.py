"""
DiagramIQ — Production WSGI Entry Point
Supports Waitress (Windows/Cross-Platform) and Gunicorn (Linux/Containers).
Usage:
  Windows:  python backend/wsgi.py
  Linux:    gunicorn -w 4 -b 0.0.0.0:5000 backend.wsgi:app
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        from waitress import serve
        print(f"[*] DiagramIQ Production WSGI Server (Waitress) running on http://0.0.0.0:{port}")
        serve(app, host="0.0.0.0", port=port, threads=6)
    except ImportError:
        print(f"[*] Waitress not found. Running with standard Flask server on port {port}...")
        app.run(host="0.0.0.0", port=port, debug=False)
