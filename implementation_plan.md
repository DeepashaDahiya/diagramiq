# Implementation Plan — DiagramIQ End-to-End System Refactoring & Debugging

Transform the current pre-development/broken prototype into a fully functioning, robust, locally-deployable neuro-symbolic multimodal reasoning system for software design diagram interpretation, strictly aligned with [requirements.md](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/requirements.md).

---

## 1. Problem & Context

DiagramIQ is designed to interpret complex software design diagrams using a 6-stage Chain-of-Thought (CoT) neuro-symbolic pipeline. It combines local Vision-Language Model inference (`LLaVA` via Ollama) with symbolic graph construction and traversal (`NetworkX`), producing grounded, confidence-scored, pedagogical explanations for computing students.

### Identified Bugs, Gaps, and Tweaks in Current Version:
1. **Broken Import & Directory Paths**: `from pipeline import ...` and `from grounding import ...` fail when executed from root or external runners due to unconfigured package resolution; relative file paths (`Path("data/diagrams")`) break depending on working directory.
2. **Missing OCR & Real Bounding Box Detection (Stage 1)**: Stage 1 does not use OCR (EasyOCR/CV) or extract real bounding boxes; Stage 6 uses a crude 3×3 grid lookup (`region_to_bbox`), resulting in fake component boundaries.
3. **Sequential Pipeline Violation (Stage 2)**: `run_full_pipeline` attempted to execute Stages 1, 2, and 3 in parallel via `ThreadPoolExecutor`, which overloaded local Ollama VLM workers and violated the prerequisite that Stage 2 requires Stage 1's extracted components.
4. **Missing MGF (Multimodal Grounding Fidelity) Scoring & Self-Correction (Stage 2)**: No MGF calculation or confidence threshold gating exists to verify graph fidelity against visual entities before reasoning.
5. **Incomplete Semantic Classification & Domain Heuristics (Stage 3)**: Classifier prompt used non-standard lowercase labels and did not activate diagram-specific heuristics (`UML_CLASS`, `UML_SEQUENCE`, `FLOWCHART`, `ER_DIAGRAM`, `SW_ARCHITECTURE`, `NETWORK_TOPOLOGY`).
6. **Limited Symbolic Graph Traversal & Query Translation (Stage 4)**: Basic substring search in `app.py` missed comprehensive query mappings, cycle detection, reachability, critical points of failure on directed graphs, and diagram-specific topological rules.
7. **JSON Schema Discrepancy (Stage 5 & 6)**: Stage 6 packaging and REST API did not conform to the canonical JSON response schema in Section 3.2 Stage 6 (`direct_answer`, `reasoning_trace`, `grounded_claims`, `diagram_highlights`, `follow_up_questions`, `overall_confidence`), causing potential frontend rendering mismatches.
8. **Missing GraphML & Graph Export Endpoints (FR-18)**: No endpoint or helper to export graph data as GraphML/JSON for academic reproducibility.
9. **Multi-Format Input & Preprocessing Incomplete (FR-01, FR-04, FR-05)**: No image sanitization, high-DPI scaling, or multi-format converters (PNG/JPG/SVG/PDF).
10. **Frontend UI Aesthetics & Missing Visualizer**: The frontend was a basic unstyled React prototype lacking graph visualization, interactive canvas bounding box overlays, 6-stage step visualizers, and sample diagram presets.

---

## User Review Required

> [!IMPORTANT]
> **Local Model Dependency**: The system connects to Ollama running locally at `http://localhost:11434` with model `llava:latest` (or `llava:13b`/`llava:7b`). A graceful fallback mock mode with realistic heuristics will also be built so the system can be demonstrated, tested, and evaluated even if Ollama is temporarily offline or processing heavy requests.

> [!NOTE]
> **Full Backward & Forward Schema Compatibility**: The backend API response will include both canonical specification fields (`direct_answer`, `grounded_claims`, `reasoning_trace`, `diagram_highlights`, `follow_up_questions`, `overall_confidence`) and alias fields (`answer`, `claims`) to guarantee zero frontend breakage.

---

## Open Questions

None at this stage. All requirements, metrics, and architecture stages are specified in [requirements.md](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/requirements.md).

---

## Proposed Changes

### Component 1: Python Backend Core & 6-Stage Pipeline

#### [MODIFY] [backend/pipeline.py](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/backend/pipeline.py)
- **Stage 1 (Neural Visual Perception & Entity Extraction)**:
  - Integrate `EasyOCR` text detection + OpenCV contour localization combined with LLaVA component identification.
  - Extract component names, types, descriptions, bounding boxes `[x1, y1, x2, y2]`, and OCR text labels.
  - Implement Stage 1 failure detection (if `< 2` components detected, flag low confidence / prompt re-upload).
- **Stage 2 (Symbolic Graph Construction & Semantic Alignment)**:
  - Construct typed directed graph (`nx.DiGraph`) with node attributes and edge relationships.
  - Compute Multimodal Grounding Fidelity (MGF) score based on component visual support ratio.
  - Implement self-correction loop if MGF `< 0.60`.
  - Generate GraphML export string and JSON adjacency list.
- **Stage 3 (Semantic Classification & Domain Heuristics)**:
  - Classify diagrams into canonical classes: `UML_CLASS`, `UML_SEQUENCE`, `FLOWCHART`, `ER_DIAGRAM`, `SW_ARCHITECTURE`, `NETWORK_TOPOLOGY`, `UNKNOWN`.
  - Activate domain heuristics (e.g., class inheritance & multiplicity for UML; message ordering for Sequence; SPOFs & data flows for Architecture; branches & loops for Flowchart; PK/FK cardinality for ER).
- **Stage 4 (Neuro-Symbolic Graph Traversal Reasoning)**:
  - Implement natural language query intent parser (identifies failure impacts, dependencies, single points of failure, shortest paths / data flow, cycles/loops, entry points, sinks, and structural counts).
  - Execute graph algorithms: `nx.descendants`, `nx.ancestors`, `nx.shortest_path`, `nx.articulation_points`, `nx.simple_cycles`, `nx.is_directed_acyclic_graph`.
  - Include diagram-type specific semantic reasoning facts.

#### [MODIFY] [backend/grounding.py](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/backend/grounding.py)
- **Stage 5 (Grounded Answer Synthesis)**:
  - Synthesize clear pedagogical answers linking graph traversal facts with diagram components.
  - Compute per-claim confidence scores (`HIGH`, `MEDIUM`, `LOW`) based on OCR visual confirmation, graph structure, and heuristic inference.
  - Attribute each claim to supporting component bounding box `[x1, y1, x2, y2]`.
- **Stage 6 (Structured Output Packaging)**:
  - Assemble full response conforming to Section 3.2 Stage 6 schema:
    `direct_answer`, `reasoning_trace` (with detailed CoT steps for Stages 1–4, MGF score, node/edge counts, GraphML), `grounded_claims`, `diagram_highlights`, `follow_up_questions`, `overall_confidence`.
  - Add image dimensions and bounding box normalizations.

#### [MODIFY] [backend/app.py](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/backend/app.py)
- Configure path resolution and CORS.
- Endpoints:
  - `GET /health` — Verifies backend status, Ollama connectivity, and loaded vision models.
  - `POST /analyze` — Runs full 6-stage pipeline on uploaded diagram (PNG, JPG, SVG, PDF) + query.
  - `GET /export/graphml` — Exports latest/specified graph as GraphML file for research papers.
  - `GET /samples` — Returns preloaded sample diagrams and recommended queries for quick testing.
  - `POST /preprocess` — Image preprocessing (resize, contrast enhancement, denoising).

#### [NEW] [backend/evaluation.py](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/backend/evaluation.py)
- Implement quantitative evaluation functions matching Section 7.1:
  - `calculate_ema(predicted_graph, ground_truth_graph)` (Exact Match Accuracy)
  - `calculate_gms(predicted_graph, ground_truth_graph)` (Graph Matching Score / Graph Edit Distance)
  - `calculate_mgf(grounded_claims, components)` (Multimodal Grounding Fidelity)
  - `calculate_hrr(baseline_errors, diagramiq_errors)` (Hallucination Reduction Rate)
- Provide benchmark test suite with sample test cases.

#### [NEW] [requirements.txt](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/requirements.txt)
- Pin required packages matching Section 5.5 of `requirements.md`.

---

### Component 2: Frontend UI & Interactive Visualizer

#### [MODIFY] [frontend/src/App.js](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/frontend/src/App.js) & [frontend/src/App.css](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/frontend/src/App.css) & [frontend/src/index.css](file:///c:/Users/DELL/OneDrive/Desktop/diagramiq/diagramiq/frontend/src/index.css)
- **Modern Glassmorphic Dark Design System**:
  - Deep slate dark theme (`#0b0f19`), neon cyan/indigo/emerald accent highlights, subtle glows, frosted glass cards, and clean typography.
- **Interactive Diagram Canvas with Real Bounding Box Highlighting**:
  - Accurate overlay of bounding boxes on rendered image.
  - Hovering/clicking any claim or highlighted component highlights its exact bounding box with glowing confidence badge and pulsing target effect.
  - Canvas zoom/reset controls.
- **Interactive Symbolic Graph Visualizer**:
  - Live SVG/Canvas rendering of extracted NetworkX graph nodes and directed edges.
  - Visual status badges for SPOFs (Single Points of Failure), entry points, databases, and services.
- **6-Stage CoT Pipeline Progress Visualizer**:
  - Live animated or collapsible stage-by-stage pipeline visualizer (Perception → Graph Construction → Classification → Traversal → Grounded Synthesis → Structured Output).
- **Pedagogical Explanation Tabs**:
  - **Overview & Direct Answer**: Clear pedagogical explanation with confidence meter.
  - **Grounded Claims & Evidence**: Per-claim card with component citations, confidence tags, and visual link.
  - **Graph Reasoning & Architecture Insights**: SPOFs, failure impact analysis, dependency paths.
  - **Reasoning Trace & GraphML Export**: Step-by-step logs, MGF score, one-click GraphML / JSON download.
- **Sample Diagram Selector**:
  - Quick-start chips for sample diagrams (Architecture, UML Class, Flowchart, ER Diagram) with 1-click load and query suggestions.

---

## Verification Plan

### Automated Tests
1. **Pipeline Unit & Integration Tests**:
   - `python backend/test_pipeline_stages.py`: Run all 6 stages sequentially on sample diagrams, verify node/edge extraction, MGF calculation, graph traversal queries, and JSON schema compliance.
2. **API Endpoint Verification**:
   - Test `GET /health`, `POST /analyze`, `GET /export/graphml`, `GET /samples`.
3. **Evaluation Metric Validation**:
   - Run `python backend/evaluation.py` to verify EMA, GMS, MGF, and HRR calculation algorithms.

### Manual / Browser Verification
1. Launch Flask backend (`python backend/app.py`) on port 5000.
2. Launch React frontend (`npm start`) on port 3000.
3. Test uploading a diagram image, selecting sample queries, verifying bounding box alignment on hover, inspecting the interactive symbolic graph, testing follow-up queries, and downloading the GraphML export.
