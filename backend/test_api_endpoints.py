"""
DiagramIQ — REST API Verification Test Suite
Tests all endpoints on http://127.0.0.1:5000:
  1. GET  /health
  2. GET  /samples
  3. POST /analyze
  4. GET  /export/graphml
  5. GET  /export/json
"""

import json
import os
import sys
import time
import requests

BASE_URL = "http://127.0.0.1:5000"

def test_api():
    print("=" * 60)
    print("[*] TESTING DIAGRAMIQ REST API ENDPOINTS")
    print("=" * 60)

    # 1. Health Check
    print("\n[TEST 1] GET /health")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200, f"Health check returned status {resp.status_code}"
        data = resp.json()
        print(f"  [+] Status: {data.get('status')}, Ollama available: {data.get('ollama_connected')}")
    except Exception as e:
        print(f"  [-] Health check failed: {e}")
        return False

    # 2. Sample Diagrams
    print("\n[TEST 2] GET /samples")
    try:
        resp = requests.get(f"{BASE_URL}/samples", timeout=5)
        assert resp.status_code == 200, f"Samples returned status {resp.status_code}"
        samples = resp.json().get("samples", [])
        print(f"  [+] Retrieved {len(samples)} sample diagrams:")
        for s in samples[:3]:
            print(f"    - {s.get('name') or s.get('title')} ({s.get('type')}) -> Query: {s.get('default_query') or s.get('preset_queries', [''])[0]}")
        assert len(samples) >= 3, "Expected at least 3 sample diagrams"
    except Exception as e:
        print(f"  [-] Samples endpoint failed: {e}")
        return False

    # 3. Analyze Endpoint with sample image
    print("\n[TEST 3] POST /analyze (Multi-stage Neuro-Symbolic Analysis)")
    sample_img_path = os.path.join(os.path.dirname(__file__), "..", "samples", "sample_architecture.png")
    if not os.path.exists(sample_img_path):
        # Create dummy sample if missing
        from PIL import Image, ImageDraw
        os.makedirs(os.path.join(os.path.dirname(__file__), "..", "samples"), exist_ok=True)
        img = Image.new("RGB", (600, 300), color="#1e1e2e")
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "API Gateway", fill="#ffffff")
        draw.text((250, 50), "Auth Service", fill="#ffffff")
        draw.text((450, 50), "PostgreSQL DB", fill="#ffffff")
        img.save(sample_img_path)

    try:
        with open(sample_img_path, "rb") as f:
            files = {"image": ("diagram.png", f, "image/png")}
            payload = {"query": "What happens if the Auth Service fails?"}
            print("  [*] Sending request to /analyze (processing 6 stages)...")
            start = time.time()
            resp = requests.post(f"{BASE_URL}/analyze", files=files, data=payload, timeout=120)
            elapsed = time.time() - start
            
            assert resp.status_code == 200, f"Analyze returned status {resp.status_code}: {resp.text}"
            res_data = resp.json()
            
            print(f"  [+] /analyze succeeded in {elapsed:.2f}s!")
            print(f"    - Direct Answer: {res_data.get('direct_answer')[:80]}...")
            print(f"    - Grounded Claims: {len(res_data.get('grounded_claims', []))}")
            print(f"    - Overall Confidence: {res_data.get('overall_confidence')}")
            print(f"    - Follow-up Questions: {len(res_data.get('follow_up_questions', []))}")
            print(f"    - Trace Components: {res_data.get('reasoning_trace', {}).get('stage1_components')}")
    except Exception as e:
        print(f"  [-] Analyze endpoint failed: {e}")
        return False

    # 4. GraphML Export
    print("\n[TEST 4] GET /export/graphml")
    try:
        resp = requests.get(f"{BASE_URL}/export/graphml", timeout=5)
        assert resp.status_code == 200, f"GraphML export returned {resp.status_code}"
        assert "<graphml" in resp.text, "Response must contain GraphML XML tag"
        print(f"  [+] GraphML export returned valid XML ({len(resp.text)} bytes)")
    except Exception as e:
        print(f"  [-] GraphML export failed: {e}")
        return False

    # 5. JSON Export
    print("\n[TEST 5] GET /export/json")
    try:
        resp = requests.get(f"{BASE_URL}/export/json", timeout=5)
        assert resp.status_code == 200, f"JSON export returned {resp.status_code}"
        json_data = resp.json()
        assert "direct_answer" in json_data or "answer" in json_data, "Export JSON missing direct_answer"
        print(f"  [+] JSON export returned valid canonical structure")
    except Exception as e:
        print(f"  [-] JSON export failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL REST API ENDPOINTS VERIFIED AND PASSING!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    ok = test_api()
    sys.exit(0 if ok else 1)
