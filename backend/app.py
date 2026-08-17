"""
DiagramIQ — REST API Server
Flask backend serving the 6-stage neuro-symbolic multimodal reasoning pipeline.
Endpoints:
  - GET  /health          -> Service health, Ollama connectivity, OCR engine status
  - POST /analyze         -> Complete 6-stage CoT multimodal diagram analysis
  - GET  /samples         -> Preloaded sample diagrams and recommended queries
  - GET  /export/graphml  -> Download latest extracted graph in GraphML format
  - GET  /export/json     -> Download latest analysis result in JSON format
  - POST /preprocess      -> Image preprocessing (resize, denoise, contrast enhance)
"""

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure backend directory is on sys.path for direct or module execution
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import requests

from pipeline import (
    stage1_identify_components,
    stage2_extract_relationships,
    stage2_build_graph,
    stage3_classify_diagram,
    query_graph_comprehensive,
    run_stages_1_to_4
)
from grounding import (
    stage5_ground_answer,
    stage6_package_output
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DiagramIQ.App")

FRONTEND_BUILD_DIR = ROOT_DIR / "frontend" / "build"
if FRONTEND_BUILD_DIR.exists():
    app = Flask(__name__, static_folder=str(FRONTEND_BUILD_DIR), static_url_path="")
else:
    app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = ROOT_DIR / "data" / "diagrams"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Cache latest analysis result for quick GraphML & JSON export
LATEST_GRAPHML = "<graphml></graphml>"
LATEST_RESULT = {}


# ── UTILITIES: MULTI-FORMAT SUPPORT & PREPROCESSING ─────────────────────────

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".svg", ".pdf"}

def preprocess_and_save_image(file_obj, filename: str) -> Path:
    """
    Validates, optionally converts (PDF/SVG to PNG), resizes if > 4096px,
    and saves diagram image to UPLOAD_FOLDER.
    """
    ext = Path(filename).suffix.lower()
    target_path = UPLOAD_FOLDER / f"upload_{filename}"
    
    # Handle PDF conversion if PyMuPDF/fitz is available
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_obj.read(), filetype="pdf")
            if len(doc) > 0:
                page = doc[0]
                pix = page.get_pixmap(dpi=150)
                png_path = target_path.with_suffix(".png")
                pix.save(str(png_path))
                return png_path
        except Exception as pe:
            logger.warning(f"PDF conversion fallback: {pe}")
            file_obj.seek(0)

    # Standard image loading & dimension normalization via PIL
    try:
        image = Image.open(file_obj)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Max resolution constraint (4096 x 4096)
        max_dim = 4096
        if image.width > max_dim or image.height > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        png_path = target_path.with_suffix(".png")
        image.save(png_path, "PNG", quality=95)
        return png_path

    except Exception as e:
        logger.error(f"Image save error: {e}")
        # Save raw as fallback
        file_obj.seek(0)
        with open(target_path, "wb") as f:
            f.write(file_obj.read())
        return target_path


# ── API ENDPOINTS ────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """System health check and Ollama / model connectivity test."""
    ollama_url = os.environ.get("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
    ollama_connected = False
    available_models = []

    try:
        resp = requests.get(ollama_url, timeout=3)
        if resp.status_code == 200:
            ollama_connected = True
            available_models = [m.get("name") for m in resp.json().get("models", [])]
    except Exception:
        ollama_connected = False

    return jsonify({
        "status": "healthy",
        "service": "DiagramIQ Multimodal Reasoning System",
        "version": "1.0",
        "ollama_connected": ollama_connected,
        "ollama": {
            "connected": ollama_connected,
            "models": available_models,
            "recommended_model": "llava:latest"
        },
        "supported_diagrams": [
            "UML_CLASS", "UML_SEQUENCE", "SW_ARCHITECTURE", "FLOWCHART", "ER_DIAGRAM", "NETWORK_TOPOLOGY"
        ]
    })


@app.route('/samples', methods=['GET'])
def get_samples():
    """Return preloaded sample diagrams and recommended queries for quick testing."""
    sample_diagram_path = ROOT_DIR / "samples" / "sample_architecture.png"
    has_test_img = sample_diagram_path.exists()

    samples = [
        {
            "id": "microservices_arch",
            "name": "Microservices Cloud Architecture",
            "title": "Microservices Cloud Architecture",
            "type": "SW_ARCHITECTURE",
            "filename": "sample_architecture.png" if has_test_img else None,
            "description": "API Gateway, Auth Service, Database, and Cloud Storage topology with failure points.",
            "default_query": "What happens if the Auth Service fails?",
            "preset_queries": [
                "What happens if the Auth Service fails?",
                "Which components are single points of failure?",
                "Trace data flow from entry point to database",
                "What are the ingress entry points to this system?"
            ]
        },
        {
            "id": "ecommerce_uml_class",
            "name": "E-Commerce UML Class Diagram",
            "title": "E-Commerce UML Class Diagram",
            "type": "UML_CLASS",
            "description": "Customer, Order, PaymentProcessor, and OrderItem class inheritance and associations.",
            "default_query": "Explain the class relationships and inheritance hierarchy",
            "preset_queries": [
                "Explain the class relationships and inheritance hierarchy",
                "What dependencies exist between Order and PaymentProcessor?",
                "Which class acts as the central coordinator in this design?"
            ]
        },
        {
            "id": "auth_sequence",
            "name": "OAuth2 Authentication Sequence",
            "title": "OAuth2 Authentication Sequence",
            "type": "UML_SEQUENCE",
            "description": "User, Client App, Authorization Server, and Resource Server token exchange.",
            "default_query": "Trace the sequence of messages required to issue an access token",
            "preset_queries": [
                "Trace the sequence of messages required to issue an access token",
                "What happens if the Authorization Server token validation fails?",
                "Identify all synchronous vs asynchronous lifelines in this flow"
            ]
        }
    ]

    return jsonify({"samples": samples})


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint. Executes the full 6-Stage Neuro-Symbolic Pipeline.
    """
    global LATEST_GRAPHML, LATEST_RESULT

    # 1. Validate Input
    if 'image' not in request.files:
        return jsonify({"error": "No diagram image provided. Please upload a PNG, JPG, SVG, or PDF file."}), 400
    
    query = request.form.get('query', '').strip()
    if not query:
        query = "Explain this software diagram and analyze its core components, relationships, and single points of failure."

    image_file = request.files['image']
    if not image_file.filename:
        return jsonify({"error": "Uploaded file has no filename."}), 400

    try:
        # 2. Preprocess & Save Image
        saved_image_path = preprocess_and_save_image(image_file, image_file.filename)
        str_path = str(saved_image_path)
        logger.info(f"Processing diagram analysis request: image='{image_file.filename}', query='{query}'")

        # 3. Stage 1: Neural Visual Perception & Entity Extraction
        components = stage1_identify_components(str_path)

        # 4. Stage 2: Symbolic Graph Construction & Semantic Alignment
        relationships = stage2_extract_relationships(str_path, components)
        G, mgf_score, graphml_str = stage2_build_graph(components, relationships)
        LATEST_GRAPHML = graphml_str

        # 5. Stage 3: Semantic Diagram Classification & Domain Heuristics
        classification = stage3_classify_diagram(str_path, components)

        # 6. Stage 4: Neuro-Symbolic Graph Traversal Reasoning
        graph_reasoning = query_graph_comprehensive(G, query)

        # 7. Stage 5: Grounded Answer Synthesis
        synthesis = stage5_ground_answer(
            str_path,
            query,
            components,
            relationships,
            graph_reasoning,
            classification
        )

        # 8. Stage 6: Canonical Structured Output Packaging
        final_output = stage6_package_output(
            str_path,
            query,
            synthesis,
            classification,
            components,
            relationships,
            graph_reasoning,
            mgf_score,
            graphml_str
        )

        # Cache latest output
        LATEST_RESULT = final_output

        # Save record to disk
        out_file = ROOT_DIR / "data" / "grounding_output.json"
        with open(out_file, "w") as f:
            json.dump(final_output, f, indent=2)

        return jsonify(final_output), 200

    except Exception as e:
        logger.exception(f"Pipeline execution error: {e}")
        return jsonify({
            "error": f"Diagram reasoning pipeline failed: {str(e)}",
            "hint": "Check if diagram image is clear and valid."
        }), 500


@app.route('/export/graphml', methods=['GET'])
def export_graphml():
    """Export the latest extracted symbolic graph as a GraphML file (FR-18)."""
    global LATEST_GRAPHML
    if not LATEST_GRAPHML or LATEST_GRAPHML == "<graphml></graphml>":
        return jsonify({"error": "No graph has been analyzed yet."}), 404
    
    return Response(
        LATEST_GRAPHML,
        mimetype="application/xml",
        headers={"Content-Disposition": "attachment;filename=diagramiq_graph.graphml"}
    )


@app.route('/export/json', methods=['GET'])
def export_json():
    """Export the latest analysis JSON."""
    global LATEST_RESULT
    if not LATEST_RESULT:
        return jsonify({"error": "No analysis result available."}), 404
    return jsonify(LATEST_RESULT)


@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Image enhancement endpoint (denoise, contrast enhancement)."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Apply adaptive histogram equalization & light bilateral filter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    _, buffer = cv2.imencode('.png', filtered)
    return Response(buffer.tobytes(), mimetype='image/png')


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve production React frontend assets or fallback to index.html."""
    if FRONTEND_BUILD_DIR.exists():
        target_file = FRONTEND_BUILD_DIR / path
        if path != "" and target_file.exists():
            return send_file(str(target_file))
        index_file = FRONTEND_BUILD_DIR / "index.html"
        if index_file.exists():
            return send_file(str(index_file))
    return jsonify({
        "service": "DiagramIQ Multimodal Reasoning System",
        "status": "online",
        "message": "Frontend build not detected. Running in API-only mode. Use /health or /samples."
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"DiagramIQ starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)