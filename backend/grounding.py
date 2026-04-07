import requests
import base64
import json
from pathlib import Path
from PIL import Image


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_llava(image_path, prompt):
    image_b64 = encode_image(image_path)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llava",
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
    )
    return response.json().get("response", "")


# ── STAGE 5: Grounded Answer Synthesis ──────────────────────────────────────

def build_grounding_prompt(query, components, relationships, graph_result):
    """
    Build a prompt that forces LLaVA to answer AND cite which
    visual region supports each part of the answer.
    """
    component_names = [c["name"] for c in components]
    
    return f"""
You are analyzing a software architecture diagram.

Known components in this diagram: {', '.join(component_names)}

Graph analysis has already determined:
{json.dumps(graph_result, indent=2)}

User question: {query}

Answer the question using the graph analysis above.
For each claim in your answer, specify which component in the image supports it
and estimate its location as a region: top-left, top-center, top-right,
middle-left, middle-center, middle-right, bottom-left, bottom-center, bottom-right.

Return ONLY a valid JSON object in exactly this format:
{{
  "answer": "Direct answer to the question in 2-3 sentences",
  "claims": [
    {{
      "claim": "Specific fact stated in the answer",
      "supported_by": "Component name that visually supports this",
      "region": "middle-center",
      "confidence": "high"
    }}
  ],
  "follow_up_questions": [
    "A relevant follow-up question the user might want to ask"
  ]
}}

Confidence levels: high (visually confirmed), medium (inferred), low (assumed)
"""


def stage5_ground_answer(image_path, query, components, relationships, graph_result):
    print(f"[Stage 5] Grounding answer for query: '{query}'")
    
    prompt = build_grounding_prompt(query, components, relationships, graph_result)
    raw = ask_llava(image_path, prompt)

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        json_str = raw[start:end]
        grounded = json.loads(json_str)
        print(f"  Answer grounded with {len(grounded.get('claims', []))} claims")
        return grounded
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not parse grounded answer. Error: {e}")
        # Return a safe fallback
        return {
            "answer": raw,
            "claims": [],
            "follow_up_questions": []
        }


# ── STAGE 6: Structured Final Output ────────────────────────────────────────

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height


def region_to_bbox(region, img_width, img_height):
    """
    Convert a region name (e.g. 'middle-center') to
    approximate pixel bounding box [x1, y1, x2, y2].
    """
    region_map = {
        "top-left":       (0.0, 0.0, 0.33, 0.33),
        "top-center":     (0.33, 0.0, 0.66, 0.33),
        "top-right":      (0.66, 0.0, 1.0, 0.33),
        "middle-left":    (0.0, 0.33, 0.33, 0.66),
        "middle-center":  (0.33, 0.33, 0.66, 0.66),
        "middle-right":   (0.66, 0.33, 1.0, 0.66),
        "bottom-left":    (0.0, 0.66, 0.33, 1.0),
        "bottom-center":  (0.33, 0.66, 0.66, 1.0),
        "bottom-right":   (0.66, 0.66, 1.0, 1.0),
    }
    
    rx1, ry1, rx2, ry2 = region_map.get(region, (0.33, 0.33, 0.66, 0.66))
    return {
        "x1": int(rx1 * img_width),
        "y1": int(ry1 * img_height),
        "x2": int(rx2 * img_width),
        "y2": int(ry2 * img_height),
        "region": region
    }


def stage6_structured_output(image_path, query, grounded_answer,
                              classification, components,
                              relationships, graph_queries):
    print("[Stage 6] Building structured output...")

    img_width, img_height = get_image_dimensions(image_path)

    # Attach pixel bounding boxes to every claim
    claims_with_boxes = []
    for claim in grounded_answer.get("claims", []):
        bbox = region_to_bbox(claim.get("region", "middle-center"),
                              img_width, img_height)
        claims_with_boxes.append({**claim, "bbox": bbox})

    output = {
        "query": query,
        "answer": grounded_answer.get("answer", ""),
        "claims": claims_with_boxes,
        "follow_up_questions": grounded_answer.get("follow_up_questions", []),
        "reasoning_trace": {
            "stage1_components":     len(components),
            "stage2_relationships":  len(relationships),
            "stage3_classification": classification,
            "stage4_graph_queries":  graph_queries,
        },
        "image_dimensions": {
            "width": img_width,
            "height": img_height
        }
    }

    print(f"  Output ready — {len(claims_with_boxes)} grounded claims with bounding boxes")
    return output


# ── TEST GROUNDING STANDALONE ────────────────────────────────────────────────

if __name__ == "__main__":
    # Load the pipeline output we already generated
    with open("data/pipeline_output.json") as f:
        pipeline_data = json.load(f)

    image_path = "data/diagrams/test_diagram.png"
    query = "What happens if the Auth Service fails?"

    # Use the failure impact result from Stage 4 as graph context
    graph_result = pipeline_data["graph_queries"]["failure_impact"].get(
        "Auth Service", {}
    )

    grounded = stage5_ground_answer(
        image_path,
        query,
        pipeline_data["components"],
        pipeline_data["relationships"],
        graph_result
    )

    final_output = stage6_structured_output(
        image_path,
        query,
        grounded,
        pipeline_data["classification"],
        pipeline_data["components"],
        pipeline_data["relationships"],
        pipeline_data["graph_queries"]
    )

    # Save it
    with open("data/grounding_output.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("\n=== Stage 5+6 Output ===")
    print(f"Answer: {final_output['answer']}")
    print(f"\nClaims with bounding boxes:")
    for claim in final_output["claims"]:
        print(f"  - {claim['claim']}")
        print(f"    Supported by: {claim['supported_by']} | Region: {claim['region']} | BBox: {claim['bbox']} | Confidence: {claim['confidence']}")
    print(f"\nFollow-up questions:")
    for q in final_output["follow_up_questions"]:
        print(f"  - {q}")