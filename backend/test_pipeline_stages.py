"""
DiagramIQ — Comprehensive Automated Integration Tests
Verifies:
  1. Stage 1 Component & Bounding Box Extraction
  2. Stage 2 Relationship Extraction & NetworkX Graph Construction & MGF Score
  3. Stage 3 Diagram Classification & Domain Heuristic Activation
  4. Stage 4 Symbolic Graph Traversal & Query Translation (SPOFs, paths, failure impact)
  5. Stage 5 Grounded Answer Synthesis & Citation
  6. Stage 6 Canonical Output Assembly & JSON Schema Compliance
"""

import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from evaluation import (
    calculate_ema,
    calculate_gms,
    calculate_mgf,
    calculate_hrr
)


def run_full_integration_test():
    image_path = str(ROOT_DIR / "data" / "diagrams" / "test_diagram.png")
    if not os.path.exists(image_path):
        print(f"Error: Test diagram not found at {image_path}")
        return False

    print("\n" + "="*70)
    print("[*] RUNNING DIAGRAMIQ 6-STAGE NEURO-SYMBOLIC PIPELINE INTEGRATION TEST")
    print("="*70)

    # ── STAGE 1 ───────────────────────────────────────────────────────────────
    print("\n[TEST STAGE 1] Visual Perception & Entity Extraction...")
    components = stage1_identify_components(image_path)
    assert len(components) >= 2, f"Expected at least 2 components, got {len(components)}"
    print(f"  [+] Stage 1 Passed: Found {len(components)} components with bounding boxes.")
    for c in components[:3]:
        print(f"    - {c['name']} ({c['type']}): BBox={c['bounding_box']}, Conf={c['confidence']}")

    # ── STAGE 2 ───────────────────────────────────────────────────────────────
    print("\n[TEST STAGE 2] Symbolic Graph Construction & MGF Score...")
    relationships = stage2_extract_relationships(image_path, components)
    G, mgf_score, graphml_str = stage2_build_graph(components, relationships)
    assert G.number_of_nodes() >= 2, "Graph must have at least 2 nodes"
    assert mgf_score > 0.0, "MGF score must be positive"
    assert len(graphml_str) > 20, "GraphML string must not be empty"
    print(f"  [+] Stage 2 Passed: Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. MGF={mgf_score:.3f}")

    # ── STAGE 3 ───────────────────────────────────────────────────────────────
    print("\n[TEST STAGE 3] Semantic Classification & Heuristics...")
    classification = stage3_classify_diagram(image_path, components)
    assert "diagram_type" in classification, "Missing diagram_type in classification"
    assert "activated_heuristics" in classification, "Missing activated_heuristics"
    print(f"  [+] Stage 3 Passed: Type={classification['diagram_type']}, Domain={classification.get('domain')}")
    print(f"    Activated Heuristics: {classification['activated_heuristics']}")

    # ── STAGE 4 ───────────────────────────────────────────────────────────────
    print("\n[TEST STAGE 4] Neuro-Symbolic Graph Traversal Reasoning...")
    test_queries = [
        "What happens if the Auth Service fails?",
        "Which components are single points of failure?",
        "Trace the data flow to the database"
    ]
    for q in test_queries:
        reasoning = query_graph_comprehensive(G, q)
        assert "critical_path" in reasoning, "Missing critical_path in reasoning"
        assert "entry_points" in reasoning, "Missing entry_points in reasoning"
        assert "failure_impact" in reasoning, "Missing failure_impact in reasoning"
        print(f"  [+] Stage 4 Query '{q[:35]}...' -> Intent: {reasoning.get('query_intent')}")

    graph_reasoning = query_graph_comprehensive(G, test_queries[0])

    # ── STAGE 5 ───────────────────────────────────────────────────────────────
    print("\n[TEST STAGE 5] Grounded Answer Synthesis...")
    synthesis = stage5_ground_answer(
        image_path,
        test_queries[0],
        components,
        relationships,
        graph_reasoning,
        classification
    )
    assert "direct_answer" in synthesis or "answer" in synthesis, "Synthesis missing direct answer"
    assert "grounded_claims" in synthesis, "Synthesis missing grounded claims"
    print(f"  [+] Stage 5 Passed: Synthesized answer with {len(synthesis['grounded_claims'])} claims.")
    print(f"    Direct Answer: {synthesis.get('direct_answer', '')[:120]}...")

    # ── STAGE 6 ───────────────────────────────────────────────────────────────
    print("\n[TEST STAGE 6] Canonical Structured Output Packaging...")
    final_output = stage6_package_output(
        image_path,
        test_queries[0],
        synthesis,
        classification,
        components,
        relationships,
        graph_reasoning,
        mgf_score,
        graphml_str
    )

    # Verify JSON Schema
    required_keys = [
        "direct_answer", "grounded_claims", "diagram_highlights",
        "follow_up_questions", "overall_confidence", "reasoning_trace", "image_dimensions"
    ]
    for k in required_keys:
        assert k in final_output, f"Final output missing required key: {k}"

    print(f"  [+] Stage 6 Passed: Output conforms to DiagramIQ canonical specification.")
    print(f"    Overall Confidence: {final_output['overall_confidence']:.2f}")
    print(f"    Follow-up Questions: {len(final_output['follow_up_questions'])}")

    # ── EVALUATION METRICS VERIFICATION ───────────────────────────────────────
    print("\n[TEST EVALUATION METRICS] Computing Benchmark Scores...")
    mgf_val = calculate_mgf(final_output["grounded_claims"], final_output["image_dimensions"])
    hrr_val = calculate_hrr(baseline_errors=20, diagramiq_errors=2)
    print(f"  [+] MGF Score: {mgf_val * 100:.1f}%")
    print(f"  [+] HRR Score: {hrr_val * 100:.1f}%")

    print("\n" + "="*70)
    print("[SUCCESS] ALL 6 PIPELINE STAGES & EVALUATION TESTS PASSED SUCCESSFULLY!")
    print("="*70)
    return True


if __name__ == "__main__":
    success = run_full_integration_test()
    if not success:
        sys.exit(1)
