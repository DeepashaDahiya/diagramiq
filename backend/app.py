import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from pipeline import (
    stage1_identify_components,
    stage2_extract_relationships,
    stage3_classify_diagram,
    stage4_build_graph,
    query_graph
)
from grounding import (
    stage5_ground_answer,
    stage6_structured_output
)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("data/diagrams")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "DiagramIQ backend running"})


@app.route('/analyze', methods=['POST'])
def analyze():
    # ── Validate input ───────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    if 'query' not in request.form:
        return jsonify({"error": "No query provided"}), 400

    image_file = request.files['image']
    query      = request.form['query']

    # ── Save uploaded image ──────────────────────────────────────────
    image_path = UPLOAD_FOLDER / image_file.filename
    image_file.save(image_path)

    try:
        # ── Run all 6 stages ─────────────────────────────────────────
        components     = stage1_identify_components(str(image_path))
        relationships  = stage2_extract_relationships(str(image_path))
        classification = stage3_classify_diagram(str(image_path))
        G              = stage4_build_graph(components, relationships)

        # Run all graph queries
        graph_queries = {
            "entry_points":  query_graph(G, "entry_points"),
            "critical_path": query_graph(G, "critical_path"),
            "data_flow":     query_graph(G, "data_flow"),
        }
        failure_impacts = {}
        for node in G.nodes():
            failure_impacts[node] = query_graph(G, "failure_impact", node)
        graph_queries["failure_impact"] = failure_impacts

        # Find the most relevant graph result for this query
        query_lower = query.lower()
        if "fail" in query_lower or "down" in query_lower:
            # Extract node name from query if possible
            graph_result = graph_queries.get("critical_path", {})
            for node in G.nodes():
                if node.lower() in query_lower:
                    graph_result = failure_impacts.get(node, {})
                    break
        elif "entry" in query_lower or "start" in query_lower:
            graph_result = graph_queries.get("entry_points", {})
        elif "flow" in query_lower or "path" in query_lower:
            graph_result = graph_queries.get("data_flow", {})
        else:
            graph_result = graph_queries.get("critical_path", {})

        # Stages 5 + 6
        grounded = stage5_ground_answer(
            str(image_path),
            query,
            components,
            relationships,
            graph_result
        )

        final_output = stage6_structured_output(
            str(image_path),
            query,
            grounded,
            classification,
            components,
            relationships,
            graph_queries
        )

        return jsonify(final_output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)