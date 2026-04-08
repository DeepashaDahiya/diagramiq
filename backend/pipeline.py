import requests
import base64
import json
import networkx as nx
from pathlib import Path


# ── HELPERS ─────────────────────────────────────────────────────────────────

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


# ── STAGE 1: Component Identification ───────────────────────────────────────

STAGE1_PROMPT = """
Analyze this diagram and list every distinct component you can see.
Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
Format exactly like this:

[
  {"name": "API Gateway", "type": "gateway", "description": "Entry point for all API calls"},
  {"name": "Database", "type": "database", "description": "Stores application data"}
]

Component types to use: service, database, gateway, load_balancer, server,
cache, queue, user, cloud, analytics, monitoring, security, network, unknown
"""

def stage1_identify_components(image_path):
    print("[Stage 1] Identifying components...")
    raw = ask_llava(image_path, STAGE1_PROMPT)
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        components = json.loads(raw[start:end])
        print(f"  Found {len(components)} components")
        return components
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not parse JSON. Error: {e}")
        return []


# ── STAGE 2: Relationship Extraction ────────────────────────────────────────

STAGE2_PROMPT = """
Analyze this diagram and list every connection or relationship between components.
Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
Format exactly like this:

[
  {"from": "API Gateway", "to": "Auth Service", "type": "calls", "condition": "on every request"},
  {"from": "App Server", "to": "Database", "type": "reads", "condition": "on data query"}
]

Relationship types to use: calls, reads, writes, sends, receives,
depends_on, routes_to, monitors, authenticates, caches
"""

def stage2_extract_relationships(image_path):
    print("[Stage 2] Extracting relationships...")
    raw = ask_llava(image_path, STAGE2_PROMPT)
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        relationships = json.loads(raw[start:end])
        print(f"  Found {len(relationships)} relationships")
        return relationships
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not parse JSON. Error: {e}")
        return []


# ── STAGE 3: Diagram Classification ─────────────────────────────────────────

STAGE3_PROMPT = """
Look at this diagram and classify it.
Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
Format exactly like this:

{
  "diagram_type": "software_architecture",
  "domain": "microservices",
  "complexity": "medium",
  "cloud_provider": "AWS"
}

diagram_type options: software_architecture, uml_class, uml_sequence,
flowchart, network_topology, er_diagram, unknown

domain options: microservices, monolith, cloud_native, on_premise,
hybrid, data_pipeline, unknown

complexity options: low, medium, high

cloud_provider: AWS, GCP, Azure, generic, none
"""

def stage3_classify_diagram(image_path):
    print("[Stage 3] Classifying diagram...")
    raw = ask_llava(image_path, STAGE3_PROMPT)
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        json_str = raw[start:end].replace("\\", "/")
        classification = json.loads(json_str)
        print(f"  Type: {classification.get('diagram_type')} | Domain: {classification.get('domain')}")
        return classification
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not parse JSON. Error: {e}")
        return {
            "diagram_type": "software_architecture",
            "domain": "unknown",
            "complexity": "medium",
            "cloud_provider": "unknown"
        }


# ── STAGE 4: Graph Construction + Traversal ──────────────────────────────────

def stage4_build_graph(components, relationships):
    print("[Stage 4] Building graph...")
    G = nx.DiGraph()

    for component in components:
        G.add_node(
            component["name"],
            type=component.get("type", "unknown"),
            description=component.get("description", "")
        )

    for rel in relationships:
        if rel["from"] not in G:
            G.add_node(rel["from"], type="unknown", description="")
        if rel["to"] not in G:
            G.add_node(rel["to"], type="unknown", description="")
        G.add_edge(
            rel["from"],
            rel["to"],
            type=rel.get("type", "unknown"),
            condition=rel.get("condition", "")
        )

    print(f"  Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def query_graph(G, query_type, target_node=None):
    results = {}

    if query_type == "failure_impact":
        if target_node not in G:
            return {"error": f"Node '{target_node}' not found in graph"}
        descendants = list(nx.descendants(G, target_node))
        results = {
            "query": f"What fails if '{target_node}' goes down?",
            "directly_affected": list(G.successors(target_node)),
            "all_affected": descendants,
            "affected_count": len(descendants)
        }

    elif query_type == "dependencies":
        if target_node not in G:
            return {"error": f"Node '{target_node}' not found in graph"}
        results = {
            "query": f"What does '{target_node}' depend on?",
            "direct_dependencies": list(G.predecessors(target_node)),
            "all_dependencies": list(nx.ancestors(G, target_node))
        }

    elif query_type == "critical_path":
        undirected = G.to_undirected()
        cut_nodes = []
        for component in nx.connected_components(undirected):
            subgraph = undirected.subgraph(component)
            cut_nodes.extend(nx.articulation_points(subgraph))
        results = {
            "query": "Which components are single points of failure?",
            "critical_nodes": cut_nodes,
            "count": len(cut_nodes)
        }

    elif query_type == "entry_points":
        entry_points = [n for n in G.nodes() if G.in_degree(n) == 0]
        results = {
            "query": "What are the entry points to the system?",
            "entry_points": entry_points
        }

    elif query_type == "data_flow":
        entry_points = [n for n in G.nodes() if G.in_degree(n) == 0]
        databases = [n for n, d in G.nodes(data=True) if d.get("type") == "database"]
        paths = []
        for entry in entry_points:
            for db in databases:
                try:
                    path = nx.shortest_path(G, entry, db)
                    paths.append({
                        "from": entry,
                        "to": db,
                        "path": path,
                        "steps": len(path) - 1
                    })
                except nx.NetworkXNoPath:
                    pass
        results = {
            "query": "Trace data flow from entry point to database",
            "paths": paths
        }

    return results


# ── FULL PIPELINE ─────────────────────────────────────────────────────────────

from concurrent.futures import ThreadPoolExecutor

def run_full_pipeline(image_path):
    print(f"\n=== DiagramIQ Full Pipeline ===")
    print(f"Image: {image_path}\n")

    # Run stages 1, 2, 3 in parallel — cuts time by ~60%
    with ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(stage1_identify_components, image_path)
        f2 = executor.submit(stage2_extract_relationships, image_path)
        f3 = executor.submit(stage3_classify_diagram, image_path)
        components     = f1.result()
        relationships  = f2.result()
        classification = f3.result()

    G = stage4_build_graph(components, relationships)

    print("\n[Stage 4] Running graph queries...")
    queries = {
        "entry_points":  query_graph(G, "entry_points"),
        "critical_path": query_graph(G, "critical_path"),
        "data_flow":     query_graph(G, "data_flow"),
    }

    failure_impacts = {}
    for node in G.nodes():
        failure_impacts[node] = query_graph(G, "failure_impact", node)
    queries["failure_impact"] = failure_impacts

    result = {
        "classification":     classification,
        "components":         components,
        "relationships":      relationships,
        "component_count":    len(components),
        "relationship_count": len(relationships),
        "graph_queries":      queries
    }

    output_path = Path("data/pipeline_output.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Pipeline Complete ===")
    print(f"Components:     {len(components)}")
    print(f"Relationships:  {len(relationships)}")
    print(f"Nodes in graph: {G.number_of_nodes()}")
    print(f"Edges in graph: {G.number_of_edges()}")

    return result, G

if __name__ == "__main__":
    run_full_pipeline("data/diagrams/test_diagram.png")