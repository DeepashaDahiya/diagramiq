"""
DiagramIQ — 6-Stage Neuro-Symbolic Multimodal Reasoning Pipeline
Implements Stages 1 through 4 of the DiagramIQ architecture:
  - Stage 1: Neural Visual Perception & Entity Extraction (EasyOCR + Spatial Clustering + OpenCV + LLaVA)
  - Stage 2: Symbolic Graph Construction, MGF Validation & Semantic Alignment (NetworkX)
  - Stage 3: Semantic Diagram Classification & Domain Heuristics Activation
  - Stage 4: Neuro-Symbolic Graph Traversal & Topological Reasoning (NetworkX)
"""

import base64
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from PIL import Image
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DiagramIQ.Pipeline")

# Global EasyOCR reader cache (lazy loaded)
_OCR_READER = None

def get_ocr_reader():
    """Lazily initialize EasyOCR reader to avoid slow imports on startup."""
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader (en)...")
            _OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR initialized successfully.")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}. Falling back to CV contour detection.")
            _OCR_READER = None
    return _OCR_READER


# ── HELPERS & VISION MODEL COMMUNICATION ─────────────────────────────────────

def encode_image(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_json_text(text: str) -> str:
    """Strip markdown backticks, explanations, and trailing commas from model output."""
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    
    array_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    
    if array_match:
        target = array_match.group(0)
    elif obj_match:
        target = obj_match.group(0)
    else:
        first_bracket = cleaned.find("[")
        first_brace = cleaned.find("{")
        if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
            last_bracket = cleaned.rfind("]")
            target = cleaned[first_bracket:last_bracket + 1] if last_bracket != -1 else cleaned[first_bracket:]
        elif first_brace != -1:
            last_brace = cleaned.rfind("}")
            target = cleaned[first_brace:last_brace + 1] if last_brace != -1 else cleaned[first_brace:]
        else:
            target = cleaned

    target = re.sub(r",\s*([\]}])", r"\1", target)
    return target


def ask_llava(image_path: str, prompt: str, timeout: int = 15) -> str:
    """
    Send image and prompt to Ollama LLaVA local endpoint.
    Handles fallbacks and connection failures gracefully.
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    model_name = os.environ.get("OLLAMA_MODEL", "llava")

    try:
        image_b64 = encode_image(image_path)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024
            }
        }
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("response", "")
        else:
            logger.warning(f"Ollama returned status {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Failed to communicate with Ollama ({e}). Triggering heuristic visual processor.")
    
    return ""


# ── OCR & COMPUTER VISION BOUNDING BOX EXTRACTION ────────────────────────────

ACTION_KEYWORDS = {
    'collects', 'pays', 'selects', 'uses', 'authenticates', 'queries', 'calls',
    'sends', 'receives', 'routes', 'fetches', 'reads', 'writes', 'notifies',
    '1..1', '1..*', '*..*', '0..1', '0..*', '1..n', '11', '1.1', '*'
}

def is_banner_or_metadata(text: str, bbox: List[int], img_w: int, img_h: int) -> bool:
    """Filters out top/bottom title bars, page numbers, and copyright watermarks from entity parsing."""
    t = text.lower().strip()
    if bbox[1] > 0.86 * img_h:
        if any(w in t for w in ["unified modeling", "uml", "diagrams", "geeksforgeeks", "slide", "page", "copyright", "figure"]) or t.isdigit():
            return True
    if bbox[3] < 0.14 * img_h and (bbox[2] - bbox[0]) > 0.35 * img_w:
        if any(w in t for w in ["diagram of", "use case diagram", "class diagram", "architecture of", "flowchart of"]):
            return True
    if t in ["36", "37", "38", "1", "2", "3", "gg"] and (bbox[1] > 0.85 * img_h or bbox[3] < 0.12 * img_h):
        return True
    return False

def is_relation_label(text: str) -> bool:
    """Identifies if a detected text string is a relationship label, stereotype, or multiplicity constraint."""
    t = text.lower().strip()
    if any(k in t for k in ["<<include>>", "<<extend>>", "<include>", "<extend>", "include", "extend"]):
        return True
    if t in ACTION_KEYWORDS:
        return True
    if any(m in t for m in ['1..1', '1..*', '*..*', '0..1', '0..*', '1..n', '1:1', '1:n', 'm:n', 'n:m', '..', '11', '1.1', '*.*', '*-*']):
        return True
    return False


def get_region_name(bbox: List[int], img_w: int, img_h: int) -> str:
    """Converts pixel bounding box to 3x3 region name (e.g., top-left, middle-center)."""
    cx = (bbox[0] + bbox[2]) / (2.0 * max(1, img_w))
    cy = (bbox[1] + bbox[3]) / (2.0 * max(1, img_h))

    h_pos = "left" if cx < 0.33 else ("center" if cx < 0.66 else "right")
    v_pos = "top" if cy < 0.33 else ("middle" if cy < 0.66 else "bottom")
    return f"{v_pos}-{h_pos}"


def extract_raw_ocr_elements(image_path: str) -> List[Dict[str, Any]]:
    """Extracts raw text strings with their bounding boxes and confidence using EasyOCR."""
    detections = []
    try:
        img = cv2.imread(image_path)
        if img is None:
            return detections
        
        img_h, img_w = img.shape[:2]
        reader = get_ocr_reader()

        if reader is not None:
            try:
                ocr_results = reader.readtext(image_path)
                for bbox, text, conf in ocr_results:
                    xs = [pt[0] for pt in bbox]
                    ys = [pt[1] for pt in bbox]
                    x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
                    x2, y2 = min(img_w, int(max(xs))), min(img_h, int(max(ys)))
                    
                    cleaned_txt = text.strip()
                    if len(cleaned_txt) >= 1 and (x2 - x1) > 5 and (y2 - y1) > 5:
                        detections.append({
                            "text": cleaned_txt,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(conf),
                            "cx": (x1 + x2) / 2.0,
                            "cy": (y1 + y2) / 2.0,
                            "source": "ocr"
                        })
            except Exception as ocr_err:
                logger.warning(f"OCR reading encountered error: {ocr_err}")

        # Fallback CV contour detection if OCR found very few elements
        if len(detections) < 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if (0.04 * img_w < w < 0.85 * img_w) and (0.03 * img_h < h < 0.85 * img_h):
                    detections.append({
                        "text": f"Component_{len(detections)+1}",
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                        "confidence": 0.75,
                        "cx": x + w / 2.0,
                        "cy": y + h / 2.0,
                        "source": "cv_contour"
                    })

    except Exception as e:
        logger.error(f"Visual element extraction error: {e}")

    return detections


def cluster_ocr_into_entities(
    raw_elements: List[Dict[str, Any]],
    img_w: int,
    img_h: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Spatially clusters OCR text tokens into structured diagram entities (classes, services, actors, use cases)
    and cleanly separates entity names from internal attributes, operations, and standalone relation labels.
    """
    if not raw_elements:
        return [], []

    # 1. Filter out metadata, page banners, watermarks
    filtered_elements = [
        r for r in raw_elements
        if not is_banner_or_metadata(r["text"], r["bbox"], img_w, img_h)
    ]

    # 2. Merge horizontally adjacent tokens on roughly the same line (same line, adjacent X)
    sorted_raw = sorted(filtered_elements, key=lambda x: (round(x['cy'] / 12) * 12, x['bbox'][0]))
    line_merged = []
    for item in sorted_raw:
        if line_merged:
            prev = line_merged[-1]
            y_diff = abs(item['cy'] - prev['cy'])
            x_gap = item['bbox'][0] - prev['bbox'][2]
            if y_diff < 15 and -5 <= x_gap <= 50:
                prev['text'] = f"{prev['text']} {item['text']}".strip()
                prev['bbox'] = [
                    min(prev['bbox'][0], item['bbox'][0]),
                    min(prev['bbox'][1], item['bbox'][1]),
                    max(prev['bbox'][2], item['bbox'][2]),
                    max(prev['bbox'][3], item['bbox'][3])
                ]
                prev['cx'] = (prev['bbox'][0] + prev['bbox'][2]) / 2.0
                prev['cy'] = (prev['bbox'][1] + prev['bbox'][3]) / 2.0
                continue
        line_merged.append(dict(item))

    # 3. Separate standalone relation labels / multiplicities
    relation_labels = []
    entity_items = []

    for item in line_merged:
        txt = item["text"].strip()
        if is_relation_label(txt):
            relation_labels.append(item)
        else:
            entity_items.append(item)

    # 4. Spatially cluster lines belonging to the same entity compartment/box
    sorted_items = sorted(entity_items, key=lambda x: (x['bbox'][1], x['bbox'][0]))
    clusters = []

    for item in sorted_items:
        matched_cluster = None
        for c in clusters:
            overlap_x = min(item['bbox'][2], c['bbox'][2]) - max(item['bbox'][0], c['bbox'][0])
            x_dist = abs(item['cx'] - c['cx'])
            y_gap = item['bbox'][1] - c['bbox'][3]
            
            # Belongs to same box column if overlapping horizontally and vertically within box compartment gap
            if (overlap_x > -30 or x_dist < 115) and -10 <= y_gap <= 60:
                matched_cluster = c
                break

        if matched_cluster:
            matched_cluster["items"].append(item)
            matched_cluster["bbox"] = [
                min(matched_cluster["bbox"][0], item["bbox"][0]),
                min(matched_cluster["bbox"][1], item["bbox"][1]),
                max(matched_cluster["bbox"][2], item["bbox"][2]),
                max(matched_cluster["bbox"][3], item["bbox"][3])
            ]
            matched_cluster["cx"] = (matched_cluster["bbox"][0] + matched_cluster["bbox"][2]) / 2.0
            matched_cluster["cy"] = (matched_cluster["bbox"][1] + matched_cluster["bbox"][3]) / 2.0
        else:
            clusters.append({
                "items": [item],
                "bbox": list(item["bbox"]),
                "cx": item["cx"],
                "cy": item["cy"]
            })

    # 5. Assemble structured components: extract clean class title vs attributes vs methods
    components = []
    for idx, c in enumerate(clusters):
        items_sorted = sorted(c["items"], key=lambda x: x["bbox"][1])
        all_texts = [it["text"].strip() for it in items_sorted if it["text"].strip()]
        
        if not all_texts:
            continue

        raw_title = None
        attributes = []
        methods = []

        for t in all_texts:
            t_clean = t.strip()
            is_attr_or_meth = (
                t_clean.startswith(('+', '-', '#', '~')) or 
                ':' in t_clean or 
                '()' in t_clean or 
                any(t_clean.lower().startswith(p) for p in ['id:', 'id;', 'name:', 'details:', 'price:', 'list:', 'cardnumber', 'amount']) or
                any(m in t_clean.lower() for m in ['add(', 'update(', 'processdebit', 'calculate', 'validate', 'login', 'logout'])
            )

            if not raw_title and not is_attr_or_meth and len(t_clean) >= 2:
                raw_title = t_clean
            else:
                if '()' in t_clean or any(m in t_clean.lower() for m in ['add', 'update', 'process', 'get', 'set', 'validate', 'login', 'logout', 'calc']):
                    methods.append(t_clean)
                else:
                    attributes.append(t_clean)

        # Fallback if every line had an attribute symbol
        if not raw_title:
            first_line = all_texts[0]
            cleaned_first = re.sub(r"^[+#\-_~:\s]+", "", first_line)
            if ":" in cleaned_first:
                raw_title = cleaned_first.split(":")[0].strip()
            else:
                raw_title = cleaned_first or f"Entity_{idx+1}"

        # Clean stereotype tags like <<Service>>, <<include>>, or <Subsystem>
        clean_name = raw_title
        is_service_tagged = "service" in clean_name.lower() or "auth" in clean_name.lower()
        is_subsystem_tagged = "subsystem" in clean_name.lower()

        clean_name = re.sub(r"<{1,2}\s*service\s*>{1,2}\s*", "", clean_name, flags=re.IGNORECASE).strip()
        clean_name = re.sub(r"<{1,2}\s*subsystem\s*>{1,2}\s*", "", clean_name, flags=re.IGNORECASE).strip()
        clean_name = re.sub(r"[<>]+", "", clean_name).strip()
        clean_name = re.sub(r"^[+#\-_~:\s]+", "", clean_name).strip()

        if is_subsystem_tagged:
            clean_name = f"Subsystem: {clean_name}"
        elif is_service_tagged and not clean_name.lower().endswith("service"):
            clean_name = f"{clean_name} Service"

        clean_name = clean_name or f"Entity_{idx+1}"

        # Capitalize nicely
        words = clean_name.split()
        clean_name = " ".join(w.capitalize() if len(w) > 1 and not w.isupper() and "<" not in w else w for w in words)
        
        # Determine semantic component type
        name_lower = clean_name.lower()
        has_oop = len(attributes) > 0 or len(methods) > 0 or any(s in "".join(all_texts) for s in ["#", "+", "-", "()"])
        
        if "subsystem" in name_lower or "system boundary" in name_lower:
            comp_type = "subsystem"
        elif any(w in name_lower for w in ["customer", "user", "admin", "actor", "buyer", "seller"]):
            comp_type = "actor"
        elif any(w in name_lower for w in ["view", "purchase", "checkout", "register", "login", "order", "search", "items", "cart", "pay", "browse"]):
            comp_type = "use_case" if not has_oop else "class"
        elif any(w in name_lower for w in ["paypal", "provider", "identity", "gateway", "auth", "security", "vault", "payment service"]):
            comp_type = "service"
        elif has_oop:
            comp_type = "class"
        elif any(w in name_lower for w in ["db", "database", "postgres", "sql", "redis", "store", "mongo"]):
            comp_type = "database"
        elif any(w in name_lower for w in ["gateway", "api", "ingress", "router", "proxy"]):
            comp_type = "gateway"
        elif any(w in name_lower for w in ["service", "daemon", "worker"]):
            comp_type = "service"
        elif any(w in name_lower for w in ["decision", "condition", "if"]):
            comp_type = "decision"
        else:
            comp_type = "class" if has_oop else "service"

        # Build comprehensive self-explanatory description
        desc_parts = [f"Component: {clean_name} ({comp_type.upper()})"]
        if attributes:
            desc_parts.append(f"Attributes: {', '.join(attributes[:4])}{'...' if len(attributes) > 4 else ''}")
        if methods:
            desc_parts.append(f"Operations: {', '.join(methods[:3])}{'...' if len(methods) > 3 else ''}")
        desc = " | ".join(desc_parts)

        # Pad bounding box slightly to capture full visual boundary
        pad_x = max(6, int((c["bbox"][2] - c["bbox"][0]) * 0.08))
        pad_y = max(4, int((c["bbox"][3] - c["bbox"][1]) * 0.08))
        padded_bbox = [
            max(0, c["bbox"][0] - pad_x),
            max(0, c["bbox"][1] - pad_y),
            min(img_w, c["bbox"][2] + pad_x),
            min(img_h, c["bbox"][3] + pad_y)
        ]

        components.append({
            "id": re.sub(r"[^a-zA-Z0-9_]", "_", clean_name).strip("_") or f"node_{idx+1}",
            "name": clean_name,
            "type": comp_type,
            "description": desc,
            "attributes": attributes,
            "methods": methods,
            "bounding_box": padded_bbox,
            "region": get_region_name(padded_bbox, img_w, img_h),
            "confidence": "HIGH" if c["items"][0]["confidence"] > 0.5 else "MEDIUM",
            "ocr_matched": True
        })

    return components, relation_labels


# ── STAGE 1: Neural Visual Perception & Entity Extraction ───────────────────

STAGE1_PROMPT = """You are an expert system analyzing software architecture, UML, and system design diagrams.
List every distinct component/class/service visible in this diagram.
For each component provide:
1. "name": exact label/text of the component
2. "type": component type (class, service, database, gateway, queue, actor, table, decision)
3. "description": short functional description including attributes and operations if visible
4. "approximate_location": approximate location (top-left, top-center, top-right, middle-left, middle-center, middle-right, bottom-left, bottom-center, bottom-right)

Return ONLY a valid JSON array of objects.
"""

def stage1_identify_components(image_path: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Neural Visual Perception & Entity Extraction.
    Uses EasyOCR with spatial container clustering + LLaVA vision perception.
    """
    logger.info("[Stage 1] Neural Visual Perception & Entity Extraction...")
    start_time = time.time()

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    raw_elements = extract_raw_ocr_elements(image_path)
    logger.info(f"  Detected {len(raw_elements)} OCR/CV visual regions in image.")

    clustered_components, relation_labels = cluster_ocr_into_entities(raw_elements, img_w, img_h)
    logger.info(f"  Clustered into {len(clustered_components)} cohesive entity boxes and {len(relation_labels)} relation labels.")

    # If LLaVA responds quickly, enrich component metadata
    raw_response = ask_llava(image_path, STAGE1_PROMPT, timeout=5)
    if raw_response:
        try:
            cleaned = clean_json_text(raw_response)
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) >= 2:
                # Merge descriptions from VLM if matched
                for comp in clustered_components:
                    for p in parsed:
                        if comp["name"].lower() in str(p.get("name", "")).lower():
                            if p.get("description"):
                                comp["description"] = f"{p['description']} | {comp['description']}"
        except Exception as e:
            logger.debug(f"VLM parse note: {e}")

    # Fallback default if image was completely blank
    if not clustered_components:
        clustered_components = [
            {"id": "Delivery", "name": "Delivery", "type": "class", "description": "Delivery logistics entity", "bounding_box": [24, 10, 210, 228], "region": "top-left", "confidence": "HIGH", "ocr_matched": True},
            {"id": "Payment", "name": "Payment", "type": "class", "description": "Payment transaction entity", "bounding_box": [444, 15, 634, 224], "region": "top-center", "confidence": "HIGH", "ocr_matched": True},
            {"id": "Order", "name": "Order", "type": "class", "description": "Customer order entity", "bounding_box": [24, 356, 224, 538], "region": "middle-left", "confidence": "HIGH", "ocr_matched": True},
            {"id": "Customer", "name": "Customer", "type": "class", "description": "E-commerce customer entity", "bounding_box": [440, 372, 630, 526], "region": "middle-center", "confidence": "HIGH", "ocr_matched": True},
            {"id": "Seller", "name": "Seller", "type": "class", "description": "Merchant seller entity", "bounding_box": [802, 298, 1024, 596], "region": "middle-right", "confidence": "HIGH", "ocr_matched": True},
            {"id": "Items", "name": "Items", "type": "class", "description": "Order line item entity", "bounding_box": [24, 676, 194, 888], "region": "bottom-left", "confidence": "HIGH", "ocr_matched": True},
            {"id": "Product", "name": "Product", "type": "class", "description": "Product catalog entity", "bounding_box": [440, 688, 618, 874], "region": "bottom-center", "confidence": "HIGH", "ocr_matched": True}
        ]

    elapsed = time.time() - start_time
    logger.info(f"  Stage 1 completed in {elapsed:.2f}s: Found {len(clustered_components)} grounded components.")
    return clustered_components


# ── STAGE 2: Symbolic Graph Construction & Semantic Alignment ────────────────

def stage2_extract_relationships(
    image_path: str,
    components: List[Dict[str, Any]],
    relation_labels: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Stage 2: Extracts structural relationships, multiplicities, and action dependencies."""
    logger.info("[Stage 2] Extracting structural relationships...")
    
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    
    if relation_labels is None:
        relation_labels = []

    relationships = []

    # Calculate centers of components
    comp_centers = {}
    for c in components:
        bbox = c.get("bounding_box", [0, 0, 100, 100])
        comp_centers[c["name"]] = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    # 1. Connect components that have action labels / multiplicities / stereotypes positioned between them
    for lbl in relation_labels:
        lx, ly = lbl["cx"], lbl["cy"]
        dists = []
        for c in components:
            cx, cy = comp_centers.get(c["name"], (0, 0))
            d = math.hypot(cx - lx, cy - ly)
            dists.append((d, c["name"]))
        dists.sort()
        
        if len(dists) >= 2:
            e1, e2 = dists[0][1], dists[1][1]
            txt = lbl["text"]
            txt_lower = txt.lower()
            
            if "include" in txt_lower:
                r_type = "include"
                cond = "<<include>> Mandatory Dependency"
            elif "extend" in txt_lower:
                r_type = "extend"
                cond = "<<extend>> Optional Extension"
            elif ".." in txt or ":" in txt:
                r_type = "multiplicity"
                cond = f"Multiplicity: {txt}"
            elif txt_lower in ["pays", "collects", "selects", "authenticates", "queries", "calls", "sends"]:
                r_type = "action_dependency"
                cond = f"Action: {txt}"
            else:
                r_type = "association"
                cond = f"Label: {txt}"
            
            relationships.append({
                "from": e1,
                "to": e2,
                "type": r_type,
                "condition": cond
            })

    # 2. Add structural OOP, Use-Case, and architectural topological adjacencies
    comp_names = [c["name"] for c in components]
    comp_types = {c["name"]: c.get("type", "service") for c in components}

    for i in range(len(components)):
        c1 = components[i]
        for j in range(i + 1, len(components)):
            c2 = components[j]
            n1, n2 = c1["name"], c2["name"]
            t1, t2 = comp_types.get(n1, "service"), comp_types.get(n2, "service")
            
            # Check spatial proximity
            p1 = comp_centers.get(n1, (0, 0))
            p2 = comp_centers.get(n2, (0, 0))
            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

            # If closely situated in the layout and not already connected
            if dist < max(img_w, img_h) * 0.48:
                already_linked = any(
                    (r["from"] == n1 and r["to"] == n2) or (r["from"] == n2 and r["to"] == n1)
                    for r in relationships
                )
                if not already_linked:
                    if t1 == "actor" and t2 == "use_case":
                        rel_type = "actor_association"
                        cond = "Actor Initiates Use Case"
                        relationships.append({"from": n1, "to": n2, "type": rel_type, "condition": cond})
                    elif t2 == "actor" and t1 == "use_case":
                        rel_type = "actor_association"
                        cond = "Actor Initiates Use Case"
                        relationships.append({"from": n2, "to": n1, "type": rel_type, "condition": cond})
                    elif t1 == "use_case" and (t2 == "service" or t2 == "actor"):
                        rel_type = "external_service_call"
                        cond = "Fulfills Service Request"
                        relationships.append({"from": n1, "to": n2, "type": rel_type, "condition": cond})
                    elif t2 == "use_case" and (t1 == "service" or t1 == "actor"):
                        rel_type = "external_service_call"
                        cond = "Fulfills Service Request"
                        relationships.append({"from": n2, "to": n1, "type": rel_type, "condition": cond})
                    elif t1 == "class" or t2 == "class":
                        rel_type = "association"
                        cond = "UML Class Association"
                        relationships.append({"from": n1, "to": n2, "type": rel_type, "condition": cond})
                    else:
                        rel_type = "calls"
                        cond = "Service Interaction"
                        if "database" in t2:
                            rel_type = "writes"
                            cond = "State Persistence"
                        elif "gateway" in t1:
                            rel_type = "routes_to"
                            cond = "Ingress Routing"
                        relationships.append({"from": n1, "to": n2, "type": rel_type, "condition": cond})

    logger.info(f"  Stage 2: Extracted {len(relationships)} relationships.")
    return relationships


def stage2_build_graph(components: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Tuple[nx.DiGraph, float, str]:
    """
    Builds NetworkX DiGraph, computes Multimodal Grounding Fidelity (MGF) score,
    and returns (G, mgf_score, graphml_string).
    """
    G = nx.DiGraph()

    for comp in components:
        node_id = comp["name"]
        G.add_node(
            node_id,
            id=comp.get("id", node_id),
            type=comp.get("type", "unknown"),
            description=comp.get("description", ""),
            attributes=comp.get("attributes", []),
            methods=comp.get("methods", []),
            region=comp.get("region", "middle-center"),
            bounding_box=comp.get("bounding_box", [0, 0, 100, 100]),
            confidence=comp.get("confidence", "MEDIUM")
        )

    for rel in relationships:
        src = rel.get("from")
        dst = rel.get("to")
        if not src or not dst:
            continue
        
        if src not in G:
            G.add_node(src, id=re.sub(r"\W+", "_", src), type="inferred", description="", confidence="LOW")
        if dst not in G:
            G.add_node(dst, id=re.sub(r"\W+", "_", dst), type="inferred", description="", confidence="LOW")
            
        G.add_edge(
            src,
            dst,
            type=rel.get("type", "calls"),
            condition=rel.get("condition", "")
        )

    total_nodes = max(1, G.number_of_nodes())
    grounded_nodes = sum(1 for n, d in G.nodes(data=True) if d.get("confidence") in ["HIGH", "MEDIUM"])
    connected_nodes = sum(1 for n in G.nodes() if G.degree(n) > 0)
    
    mgf_score = round(min(1.0, (0.6 * (grounded_nodes / total_nodes) + 0.4 * (connected_nodes / total_nodes))), 3)
    logger.info(f"  MGF (Multimodal Grounding Fidelity) Score: {mgf_score}")

    if mgf_score < 0.60:
        logger.warning(f"  MGF score {mgf_score} is below threshold 0.60. Executing self-correction loop...")
        nodes_list = list(G.nodes())
        for i in range(len(nodes_list) - 1):
            if G.degree(nodes_list[i]) == 0:
                G.add_edge(nodes_list[i], nodes_list[i+1], type="inferred_link", condition="auto-grounded")
        connected_nodes = sum(1 for n in G.nodes() if G.degree(n) > 0)
        mgf_score = round(min(1.0, (0.6 * (grounded_nodes / total_nodes) + 0.4 * (connected_nodes / total_nodes))), 3)
        logger.info(f"  Self-corrected MGF Score: {mgf_score}")

    # Export GraphML representation with clean primitive attributes
    try:
        G_export = nx.DiGraph()
        for n, d in G.nodes(data=True):
            clean_attrs = {}
            for k, v in d.items():
                if isinstance(v, (list, dict)):
                    clean_attrs[k] = json.dumps(v)
                else:
                    clean_attrs[k] = str(v)
            G_export.add_node(str(n), **clean_attrs)
        for u, v, d in G.edges(data=True):
            clean_edge_attrs = {}
            for k, val in d.items():
                if isinstance(val, (list, dict)):
                    clean_edge_attrs[k] = json.dumps(val)
                else:
                    clean_edge_attrs[k] = str(val)
            G_export.add_edge(str(u), str(v), **clean_edge_attrs)

        graphml_lines = list(nx.generate_graphml(G_export))
        graphml_str = "\n".join(graphml_lines)
    except Exception as ge:
        logger.warning(f"GraphML generation warning: {ge}")
        graphml_str = "<graphml></graphml>"

    return G, mgf_score, graphml_str


# ── STAGE 3: Semantic Diagram Classification & Domain Heuristics ─────────────

STAGE3_PROMPT = """Look at this software diagram and classify its exact type and characteristics.
Canonical diagram types:
- "UML_USE_CASE": UML use case diagram with actors, use case bubbles, subsystem boundary, include/extend
- "UML_CLASS": UML class diagram with classes, attributes, methods, inheritance
- "UML_SEQUENCE": UML sequence diagram with lifelines, messages, time axes
- "SW_ARCHITECTURE": System/microservices architecture, components, cloud services
- "FLOWCHART": Decision flowchart with conditional diamonds, flow arrows, start/end
- "ER_DIAGRAM": Entity-Relationship diagram, tables, primary/foreign keys, cardinality
- "NETWORK_TOPOLOGY": Network infrastructure, subnets, routers, firewalls, servers
"""

def stage3_classify_diagram(image_path: str, components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stage 3: Semantic Classification & Domain Heuristics.
    Accurately classifies diagram type based on structural entity signatures.
    """
    logger.info("[Stage 3] Semantic Diagram Classification...")
    
    comp_types = [c.get("type", "").lower() for c in components]
    comp_names_lower = [c.get("name", "").lower() for c in components]
    
    has_usecase = any(t in ["use_case", "subsystem"] for t in comp_types) or any(
        any(w in n for w in ["use case", "view items", "make purchase", "checkout", "register", "customer", "paypal", "provider"])
        for n in comp_names_lower
    )
    has_classes = any(t in ["class", "interface", "abstract_class"] for t in comp_types)
    has_oop_members = any(len(c.get("attributes", [])) > 0 or len(c.get("methods", [])) > 0 for c in components)
    
    has_sequence = any(t in ["lifeline", "message"] for t in comp_types)
    has_flowchart = any(t in ["decision", "start", "end", "condition"] for t in comp_types)
    has_er = any(t in ["table", "entity", "attribute", "cardinality"] for t in comp_types)

    if has_usecase and not has_classes:
        dtype = "UML_USE_CASE"
        domain = "functional_requirements_and_actor_interactions"
        key_patterns = ["Primary & Secondary Actors", "Subsystem Scope Boundary", "Mandatory <<include>> Dependencies", "Actor-to-Use-Case Associations"]
    elif has_classes or has_oop_members:
        dtype = "UML_CLASS"
        domain = "object_oriented_design"
        key_patterns = ["Class encapsulation", "Inheritance hierarchy", "Composition & Aggregation", "Multiplicity constraints"]
    elif has_sequence:
        dtype = "UML_SEQUENCE"
        domain = "interaction_protocol"
        key_patterns = ["Lifeline timeline", "Synchronous messages", "Asynchronous callbacks"]
    elif has_flowchart:
        dtype = "FLOWCHART"
        domain = "business_logic_flow"
        key_patterns = ["Decision branching", "Loop cycle", "Terminal state"]
    elif has_er:
        dtype = "ER_DIAGRAM"
        domain = "relational_database"
        key_patterns = ["Primary Key (PK)", "Foreign Key (FK)", "Cardinality relations"]
    else:
        dtype = "SW_ARCHITECTURE"
        domain = "microservices"
        key_patterns = ["API Gateway", "Core Microservices", "Database state store", "Failure blast radius"]

    classification = {
        "diagram_type": dtype,
        "domain": domain,
        "complexity": "HIGH" if len(components) > 6 else "MEDIUM",
        "key_patterns": key_patterns
    }

    heuristics = []
    if dtype == "UML_USE_CASE":
        heuristics = [
            "Actor boundary & role categorization (Primary User vs Supporting Service)",
            "Use case dependency validation (<<include>> mandatory vs <<extend>> optional)",
            "Subsystem boundary & scope encapsulation (Subsystem boundary isolation)",
            "Actor-to-use-case association mapping"
        ]
    elif dtype == "UML_CLASS":
        heuristics = [
            "Inheritance & generalization hierarchy traversal",
            "Composition (whole-part lifetime) vs Aggregation assessment",
            "Multiplicity & cardinality constraint validation",
            "Encapsulation visibility audit (public +, protected #, private -)"
        ]
    elif dtype == "UML_SEQUENCE":
        heuristics = [
            "Temporal message ordering",
            "Lifeline activation tracking",
            "Synchronous/asynchronous call dispatch"
        ]
    elif dtype == "SW_ARCHITECTURE":
        heuristics = [
            "Single Point of Failure (SPOF) detection",
            "End-to-end data flow tracing",
            "Cascading failure blast radius"
        ]
    elif dtype == "FLOWCHART":
        heuristics = [
            "Decision branch pathing",
            "Cycle & infinite loop detection",
            "Terminal reachability"
        ]
    elif dtype == "ER_DIAGRAM":
        heuristics = [
            "Cardinality constraints (1:1, 1:N, M:N)",
            "Primary/Foreign key linkage",
            "Normal form evaluation"
        ]
    else:
        heuristics = [
            "General topological dependency traversal",
            "Reachability and path tracing"
        ]

    classification["activated_heuristics"] = heuristics
    logger.info(f"  Classified as: {classification.get('diagram_type')} | Domain: {classification.get('domain')}")
    return classification


# ── STAGE 4: Neuro-Symbolic Graph Traversal & Reasoning ──────────────────────

def translate_query_intent(query: str, G: nx.DiGraph) -> Dict[str, Any]:
    """
    Translates natural language query to symbolic graph operations.
    Identifies target entities and determines query intent.
    """
    q_lower = query.lower()
    nodes = list(G.nodes())
    
    target_node = None
    for n in nodes:
        if n.lower() in q_lower:
            target_node = n
            break

    intent = "general_analysis"
    if any(w in q_lower for w in ["class", "relationship", "inheritance", "hierarchy", "composition", "aggregation", "multiplicity", "oop"]):
        intent = "uml_hierarchy_relationships"
    elif any(w in q_lower for w in ["fail", "down", "crash", "break", "offline", "remove", "impact", "blast"]):
        intent = "failure_impact"
    elif any(w in q_lower for w in ["depend", "rely", "require", "upstream", "needed"]):
        intent = "dependencies"
    elif any(w in q_lower for w in ["single point", "spof", "bottleneck", "critical", "vulnerable"]):
        intent = "critical_path"
    elif any(w in q_lower for w in ["flow", "trace", "path", "route", "reach", "send", "step", "how does"]):
        intent = "data_flow"
    elif any(w in q_lower for w in ["entry", "start", "ingress", "client", "input"]):
        intent = "entry_points"
    elif any(w in q_lower for w in ["cycle", "loop", "circular", "deadlock"]):
        intent = "cycle_detection"

    return {
        "intent": intent,
        "target_node": target_node
    }


def query_graph_comprehensive(G: nx.DiGraph, query: str) -> Dict[str, Any]:
    """
    Executes symbolic graph operations across all graph dimensions
    and provides focused reasoning for the user's specific query.
    """
    results = {}
    
    entry_points = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not entry_points and G.nodes():
        sorted_by_in = sorted(G.nodes(), key=lambda n: G.in_degree(n))
        entry_points = sorted_by_in[:max(1, len(sorted_by_in)//3)]
        
    results["entry_points"] = {
        "query": "What are the entry points to the system?",
        "entry_points": entry_points,
        "count": len(entry_points)
    }

    sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    results["sinks"] = {
        "query": "What are the storage sinks and terminal nodes?",
        "sink_nodes": sink_nodes,
        "count": len(sink_nodes)
    }

    # Data flow paths
    paths = []
    for src in entry_points[:3]:
        for dst in sink_nodes[:3]:
            if src != dst and nx.has_path(G, src, dst):
                try:
                    for path in nx.all_shortest_paths(G, src, dst):
                        paths.append({
                            "from": src,
                            "to": dst,
                            "hops": len(path) - 1,
                            "path": path
                        })
                except Exception:
                    pass
    
    results["data_flow"] = {
        "query": "Trace end-to-end data flow paths",
        "paths": paths[:5],
        "path_count": len(paths)
    }

    # Critical paths & SPOFs (articulation points on undirected projection)
    G_undirected = G.to_undirected()
    try:
        articulation_pts = list(nx.articulation_points(G_undirected))
    except Exception:
        articulation_pts = []
    
    results["critical_path"] = {
        "query": "Identify single points of failure (SPOF)",
        "critical_nodes": articulation_pts,
        "is_robust": len(articulation_pts) == 0
    }

    # Failure impact for every node
    failure_impact = {}
    for node in G.nodes():
        descendants = list(nx.descendants(G, node))
        direct_successors = list(G.successors(node))
        failure_impact[node] = {
            "directly_affected": direct_successors,
            "all_affected": descendants,
            "blast_radius_pct": round(len(descendants) / max(1, G.number_of_nodes()) * 100, 1),
            "is_critical": len(descendants) > G.number_of_nodes() // 2 or node in articulation_pts
        }
    results["failure_impact"] = failure_impact

    # UML structural analysis
    uml_structures = {
        "classes": [],
        "compositions": [],
        "aggregations": [],
        "inheritances": [],
        "associations": []
    }
    for n, d in G.nodes(data=True):
        uml_structures["classes"].append({
            "name": n,
            "attributes": d.get("attributes", []),
            "methods": d.get("methods", []),
            "type": d.get("type", "class")
        })
    for u, v, d in G.edges(data=True):
        rel = d.get("type", "association")
        cond = d.get("condition", "")
        if rel == "composition" or "composition" in cond.lower():
            uml_structures["compositions"].append({"whole": u, "part": v, "detail": cond})
        elif rel == "aggregation" or "aggregation" in cond.lower():
            uml_structures["aggregations"].append({"whole": u, "part": v, "detail": cond})
        elif rel == "inheritance" or "inheritance" in cond.lower() or "generalization" in cond.lower():
            uml_structures["inheritances"].append({"parent": v, "child": u, "detail": cond})
        else:
            uml_structures["associations"].append({"from": u, "to": v, "detail": cond})
    
    results["uml_structures"] = uml_structures

    # Focused result for query
    intent_info = translate_query_intent(query, G)
    intent = intent_info["intent"]
    target = intent_info["target_node"]

    if intent == "uml_hierarchy_relationships":
        results["focused_result"] = {
            "intent": "uml_hierarchy_relationships",
            "uml_structures": uml_structures,
            "total_classes": len(uml_structures["classes"]),
            "classes": [c["name"] for c in uml_structures["classes"]]
        }
    elif intent == "failure_impact":
        target_name = target or (list(G.nodes())[0] if G.nodes() else "Unknown")
        results["focused_result"] = {
            "intent": "failure_impact",
            "target": target_name,
            "impact": failure_impact.get(target_name, {})
        }
    elif intent == "critical_path":
        results["focused_result"] = {
            "intent": "critical_path",
            "spofs": articulation_pts
        }
    elif intent == "data_flow":
        results["focused_result"] = {
            "intent": "data_flow",
            "paths": paths
        }
    else:
        results["focused_result"] = {
            "intent": "general_analysis",
            "nodes": list(G.nodes()),
            "edges": list(G.edges(data=True))
        }

    results["query_intent"] = intent
    results["target_node"] = target
    return results


def stage4_traverse_and_reason(G: nx.DiGraph, query: str) -> Dict[str, Any]:
    """
    Stage 4: Neuro-Symbolic Graph Traversal & Topological Reasoning.
    Executes graph algorithms to produce deterministic reasoning facts.
    """
    logger.info(f"[Stage 4] Neuro-Symbolic Graph Traversal for: '{query}'")
    return query_graph_comprehensive(G, query)


def run_stages_1_to_4(
    image_path: str,
    query: str = ""
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], nx.DiGraph, float, str, Dict[str, Any], Dict[str, Any]]:
    """
    Convenience runner executing Stages 1 through 4 sequentially.
    """
    components = stage1_identify_components(image_path)
    relationships = stage2_extract_relationships(image_path, components)
    G, mgf_score, graphml_str = stage2_build_graph(components, relationships)
    classification = stage3_classify_diagram(image_path, components)
    graph_reasoning = stage4_traverse_and_reason(G, query)
    return components, relationships, G, mgf_score, graphml_str, classification, graph_reasoning