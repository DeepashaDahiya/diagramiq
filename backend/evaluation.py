"""
DiagramIQ — Quantitative Evaluation & Benchmark Metrics Module
Implements metrics specified in requirements.md Section 7.1:
  - EMA: Exact Match Accuracy (Graph edge/node match precision)
  - GMS: Graph Matching Score (Normalized Graph Edit Distance)
  - MGF: Multimodal Grounding Fidelity (Visual bounding box attribution ratio)
  - HRR: Hallucination Reduction Rate (Reduction in false claims vs baseline)
  - PEQ: Pedagogical Explanation Quality (User study Likert aggregation)
"""

import math
from typing import Any, Dict, List, Set, Tuple
import networkx as nx


def calculate_ema(predicted_edges: List[Dict[str, Any]], ground_truth_edges: List[Dict[str, Any]]) -> float:
    """
    Exact Match Accuracy (EMA).
    Measures the precision of extracted graph edges matching ground-truth relationships.
    EMA = |Predicted Edges ∩ Ground Truth Edges| / |Ground Truth Edges|
    """
    if not ground_truth_edges:
        return 1.0 if not predicted_edges else 0.0

    def normalize_edge(e: Dict[str, Any]) -> Tuple[str, str]:
        src = str(e.get("from", "")).strip().lower()
        dst = str(e.get("to", "")).strip().lower()
        return (src, dst)

    gt_set = set(normalize_edge(e) for e in ground_truth_edges)
    pred_set = set(normalize_edge(e) for e in predicted_edges)

    matched = gt_set.intersection(pred_set)
    ema = len(matched) / len(gt_set)
    return round(ema, 4)


def calculate_gms(predicted_graph: nx.DiGraph, ground_truth_graph: nx.DiGraph) -> float:
    """
    Graph Matching Score (GMS).
    Computes structural similarity between extracted graph and ground truth graph
    using normalized graph similarity based on node/edge Jaccard and structural overlap.
    GMS in [0.0, 1.0].
    """
    pred_nodes = set(n.lower() for n in predicted_graph.nodes())
    gt_nodes = set(n.lower() for n in ground_truth_graph.nodes())

    pred_edges = set((u.lower(), v.lower()) for u, v in predicted_graph.edges())
    gt_edges = set((u.lower(), v.lower()) for u, v in ground_truth_graph.edges())

    # Node Jaccard
    node_union = pred_nodes.union(gt_nodes)
    node_sim = len(pred_nodes.intersection(gt_nodes)) / max(1, len(node_union))

    # Edge Jaccard
    edge_union = pred_edges.union(gt_edges)
    edge_sim = len(pred_edges.intersection(gt_edges)) / max(1, len(edge_union)) if edge_union else 1.0

    # Combined GMS (60% edge structure, 40% node alignment)
    gms = 0.40 * node_sim + 0.60 * edge_sim
    return round(gms, 4)


def calculate_mgf(grounded_claims: List[Dict[str, Any]], image_dimensions: Dict[str, int]) -> float:
    """
    Multimodal Grounding Fidelity (MGF).
    Computes percentage of claims that cite a valid, non-zero bounding box region
    grounded within the image dimensions.
    """
    if not grounded_claims:
        return 0.0

    img_w = image_dimensions.get("width", 1000)
    img_h = image_dimensions.get("height", 1000)

    valid_grounded_count = 0
    for claim in grounded_claims:
        bbox = claim.get("bounding_box") or (
            [claim["bbox"]["x1"], claim["bbox"]["y1"], claim["bbox"]["x2"], claim["bbox"]["y2"]]
            if "bbox" in claim else None
        )
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Validate coordinates are within image boundaries and form positive area
            if (0 <= x1 < x2 <= img_w) and (0 <= y1 < y2 <= img_h) and ((x2 - x1) * (y2 - y1) > 50):
                valid_grounded_count += 1

    mgf = valid_grounded_count / len(grounded_claims)
    return round(mgf, 4)


def calculate_hrr(baseline_errors: int, diagramiq_errors: int) -> float:
    """
    Hallucination Reduction Rate (HRR).
    Measures percentage reduction in structural hallucination errors
    compared to vanilla baseline VLM (e.g. LLaVA vanilla without graph).
    HRR = (baseline_errors - diagramiq_errors) / baseline_errors
    """
    if baseline_errors <= 0:
        return 0.0
    hrr = (baseline_errors - diagramiq_errors) / baseline_errors
    return round(max(0.0, min(1.0, hrr)), 4)


def calculate_peq(ratings: List[Dict[str, int]]) -> Dict[str, float]:
    """
    Pedagogical Explanation Quality (PEQ).
    Aggregates Likert scale (1-5) ratings across 5 criteria from 15-student user study:
    1. Clarity, 2. Completeness, 3. Accuracy, 4. Usefulness, 5. Trust.
    """
    if not ratings:
        return {"overall_peq": 0.0}

    criteria = ["clarity", "completeness", "accuracy", "usefulness", "trust"]
    results = {}
    
    for c in criteria:
        vals = [r[c] for r in ratings if c in r and 1 <= r[c] <= 5]
        results[c] = round(sum(vals) / len(vals), 2) if vals else 0.0

    results["overall_peq"] = round(sum(results.values()) / len(criteria), 2)
    return results


# ── SELF-TEST BENCHMARK ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing DiagramIQ Evaluation Metric Calculations ===")
    
    # 1. EMA test
    gt_edges = [
        {"from": "API Gateway", "to": "Auth Service"},
        {"from": "Auth Service", "to": "Database"},
        {"from": "API Gateway", "to": "Order Service"}
    ]
    pred_edges = [
        {"from": "API Gateway", "to": "Auth Service"},
        {"from": "Auth Service", "to": "Database"},
        {"from": "Order Service", "to": "Database"}  # Extra edge
    ]
    ema = calculate_ema(pred_edges, gt_edges)
    print(f"EMA (Exact Match Accuracy): {ema * 100:.1f}%")

    # 2. GMS test
    g_gt = nx.DiGraph()
    g_gt.add_edges_from([("A", "B"), ("B", "C"), ("A", "D")])
    g_pred = nx.DiGraph()
    g_pred.add_edges_from([("A", "B"), ("B", "C"), ("D", "C")])
    gms = calculate_gms(g_pred, g_gt)
    print(f"GMS (Graph Matching Score): {gms:.4f}")

    # 3. MGF test
    claims = [
        {"claim": "Fact 1", "bounding_box": [10, 10, 200, 200]},
        {"claim": "Fact 2", "bounding_box": [50, 50, 300, 300]},
        {"claim": "Fact 3", "bounding_box": [0, 0, 0, 0]}  # invalid
    ]
    mgf = calculate_mgf(claims, {"width": 800, "height": 600})
    print(f"MGF (Multimodal Grounding Fidelity): {mgf * 100:.1f}%")

    # 4. HRR test
    hrr = calculate_hrr(baseline_errors=25, diagramiq_errors=4)
    print(f"HRR (Hallucination Reduction Rate): {hrr * 100:.1f}%")

    print("\nAll evaluation metric functions executed successfully!")
