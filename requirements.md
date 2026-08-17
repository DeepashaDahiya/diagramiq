# DiagramIQ — Complete Requirements Document
### A Neuro-Symbolic Multimodal Reasoning System for Structured Software Design Diagram Interpretation in Computing Education

> **Version:** 1.0 | **Status:** Pre-development | **Authors:** [Your Name], [Co-author Name] | **Advisor:** [Professor Name]

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Paper Requirements](#2-research-paper-requirements)
3. [System Architecture Requirements](#3-system-architecture-requirements)
4. [Functional Requirements](#4-functional-requirements)
5. [Technical Stack & Dependencies](#5-technical-stack--dependencies)
6. [Dataset Requirements](#6-dataset-requirements)
7. [Evaluation & Metrics Requirements](#7-evaluation--metrics-requirements)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Development Milestones](#9-development-milestones)
10. [Team Responsibilities](#10-team-responsibilities)
11. [Paper Structure Requirements](#11-paper-structure-requirements)
12. [Citation & Reference Requirements](#12-citation--reference-requirements)
13. [Submission Requirements](#13-submission-requirements)

---

## 1. Project Overview

### 1.1 What We Are Building
DiagramIQ is a locally-deployable, open-source multimodal reasoning system that interprets software design diagrams using a 6-stage Chain-of-Thought (CoT) pipeline. It combines LLaVA 1.6 (13B) for visual understanding with NetworkX for symbolic graph traversal, producing structured, grounded, confidence-scored explanations — targeted at computing students learning software design.

### 1.2 Problem Being Solved
| Problem | Gap in Literature |
|---|---|
| Students cannot independently interpret complex software design diagrams | No existing system provides pedagogically structured diagram explanations |
| All capable multimodal tools require cloud APIs (GPT-4V, Gemini) | No privacy-safe, locally-deployable alternative exists |
| Pure vision-LLMs hallucinate structural relationships | No neuro-symbolic hybrid exists for software diagram interpretation |
| Error snowballing across multi-step visual reasoning chains | No intermediate confidence-gating mechanism in existing pipelines |

### 1.3 Key Differentiators
- **Local-first:** Runs entirely via Ollama — no API keys, no internet, no data leaves the device
- **Neuro-symbolic hybrid:** Neural perception (LLaVA) + symbolic graph traversal (NetworkX)
- **Grounded answers:** Every claim tied to a bounding box region in the source image
- **Confidence-scored output:** Per-claim confidence labels (High / Medium / Low)
- **6-stage forced decomposition:** Prevents error snowballing by separating perception from reasoning

---

## 2. Research Paper Requirements

### 2.1 Paper Identity
| Field | Value |
|---|---|
| **Full title** | DiagramIQ: A Neuro-Symbolic Multimodal Reasoning System for Structured Software Design Diagram Interpretation in Computing Education |
| **Short title** | DiagramIQ |
| **Target venue** | ICRITO 2025/2026 or equivalent Springer LNNS conference |
| **Paper type** | Original research / system paper |
| **Page limit** | 8–12 pages (Springer LNCS format) |
| **Format** | LaTeX via Overleaf — Springer LNCS template |

### 2.2 Author Order (Standard Convention)
1. **Author 1** — [Your name] — primary implementer, methodology + results
2. **Author 2** — [Co-author name] — literature review + experiments
3. **Author 3 (Senior/Last)** — [Professor name] — advisor, abstract + framing + final review

> ⚠️ Last-author position for the professor is the academic standard for advisor attribution. Confirm this with your professor early.

### 2.3 Research Contributions (Must Appear in Introduction)
The paper must explicitly claim these three contributions in a numbered list:

1. **DiagramIQ Pipeline** — A novel 6-stage neuro-symbolic CoT pipeline for multimodal software diagram interpretation, integrating LLaVA 1.6 with NetworkX-based graph traversal
2. **DiagramIQ-Eval Dataset** — A curated benchmark of 100 annotated software design diagrams (UML class, sequence, ER, flowchart) with ground-truth QA pairs for evaluation
3. **Empirical Evaluation** — Quantitative comparison of DiagramIQ against baseline VLMs (LLaVA vanilla, GPT-4o) on EMA, GMS, MGF, and HRR metrics, plus a 15-student pedagogical user study

---

## 3. System Architecture Requirements

### 3.1 High-Level Architecture

```
[User Input]
    │
    ├── Diagram Image (PNG/JPG/SVG)
    └── Natural Language Query
            │
            ▼
    ┌─────────────────────────────────────────────┐
    │           DiagramIQ 6-Stage Pipeline         │
    │                                             │
    │  Stage 1: Neural Visual Perception          │
    │  ├── CLIP visual encoder (patch embeddings) │
    │  ├── LLaVA 1.6 component identification     │
    │  └── EasyOCR text label extraction          │
    │                                             │
    │  Stage 2: Symbolic Graph Construction       │
    │  ├── Entity → node mapping                  │
    │  ├── Relationship → edge mapping            │
    │  └── Output: NetworkX DiGraph (JSON/GraphML)│
    │                                             │
    │  Stage 3: Semantic Classification           │
    │  ├── Diagram type classifier                │
    │  │   (UML / Architecture / Flowchart / ER)  │
    │  └── Domain heuristic activation            │
    │                                             │
    │  Stage 4: Graph Traversal Reasoning         │
    │  ├── NetworkX structural queries            │
    │  ├── Dependency chain traversal             │
    │  └── SPOF / critical path detection        │
    │                                             │
    │  Stage 5: Grounded Answer Synthesis         │
    │  ├── LLaVA final answer generation          │
    │  ├── Bounding box ↔ claim attribution       │
    │  └── Per-claim confidence scoring           │
    │                                             │
    │  Stage 6: Structured Output                 │
    │  ├── Direct answer                          │
    │  ├── Reasoning trace (collapsible)          │
    │  ├── Highlighted diagram regions            │
    │  ├── Confidence per claim                   │
    │  └── Proactive follow-up questions          │
    └─────────────────────────────────────────────┘
            │
            ▼
    [Structured JSON Response → Flask API → Frontend]
```

### 3.2 Stage-by-Stage Technical Requirements

#### Stage 1 — Neural Visual Perception & Entity Extraction
- **Input:** Raw diagram image (PNG/JPG, max 4096×4096px recommended)
- **Model:** LLaVA 1.6 (13B) via Ollama
- **OCR engine:** EasyOCR (preferred over pytesseract for varied diagram fonts)
- **Required outputs:**
  - List of detected components with type labels (node, service, database, actor, class, etc.)
  - Bounding box coordinates `[x1, y1, x2, y2]` for each detected entity
  - Text labels extracted from diagram
- **Prompt template for Stage 1:**
  ```
  "List every distinct component visible in this diagram. For each component,
  provide: (1) its name/label, (2) its type (service/database/actor/class/etc),
  (3) its approximate position in the image. Do not reason about relationships yet."
  ```
- **Failure handling:** If fewer than 2 components detected → flag as low-confidence, request image re-upload with higher resolution

#### Stage 2 — Symbolic Graph Construction & Semantic Alignment
- **Input:** Component list from Stage 1
- **Library:** NetworkX `DiGraph` (directed graph)
- **Required outputs:**
  - JSON adjacency structure:
    ```json
    {
      "nodes": [{"id": "AuthService", "type": "microservice"}],
      "edges": [{"from": "APIGateway", "to": "AuthService", "type": "calls", "condition": "on every request"}]
    }
    ```
  - GraphML export (for paper figures and reproducibility)
- **Multimodal Grounding Fidelity (MGF) check:** Compute alignment score between JSON graph and source image before proceeding. If MGF < threshold → trigger self-correction loop (re-query Stage 1)
- **Prompt template for Stage 2:**
  ```
  "Based on the components identified, now describe every connection, arrow,
  or relationship between them. For each relationship specify: source component,
  target component, relationship type, and any condition labels."
  ```

#### Stage 3 — Semantic Classification
- **Diagram type classes:** `UML_CLASS` | `UML_SEQUENCE` | `FLOWCHART` | `ER_DIAGRAM` | `SW_ARCHITECTURE` | `NETWORK_TOPOLOGY` | `UNKNOWN`
- **Classification method:** LLaVA zero-shot classification prompt + keyword matching on Stage 1 labels
- **Domain heuristic activation:**
  - `UML_CLASS` → activate inheritance chain reasoning, multiplicity parsing
  - `UML_SEQUENCE` → activate temporal ordering reasoning, lifeline tracking
  - `SW_ARCHITECTURE` → activate SPOF detection, data flow tracing
  - `FLOWCHART` → activate decision branch tracking, loop detection
  - `ER_DIAGRAM` → activate cardinality reasoning, primary/foreign key detection

#### Stage 4 — Neuro-Symbolic Graph Traversal Reasoning
- **Library:** NetworkX (Python)
- **Key operations to implement:**
  - `nx.descendants(G, node)` → find all downstream dependents
  - `nx.ancestors(G, node)` → find all upstream dependencies
  - `nx.shortest_path(G, source, target)` → data flow tracing
  - `nx.articulation_points(G)` → single point of failure detection
  - `nx.is_directed_acyclic_graph(G)` → cycle detection
- **Query translation:** User NL query → symbolic graph operation (rule-based mapping)
  - "What breaks if X fails?" → `nx.descendants(G, X)`
  - "What does X depend on?" → `nx.ancestors(G, X)`
  - "How does data flow from A to B?" → `nx.shortest_path(G, A, B)`
  - "What are the single points of failure?" → `nx.articulation_points(G)`

#### Stage 5 — Grounded Answer Synthesis
- **Input:** Graph traversal results (Stage 4) + original image + component bounding boxes (Stage 1)
- **Confidence levels:**
  - `HIGH` — Component visually confirmed in Stage 1 with high OCR confidence
  - `MEDIUM` — Component inferred from graph structure, partially visible
  - `LOW` — Component assumed from domain heuristics, not directly visible
- **Grounding format:** Each claim in the answer must cite its source component ID from Stage 1
- **Prompt template for Stage 5:**
  ```
  "Using the following graph traversal results: {graph_results}, synthesize a clear
  answer to the user's question: {query}. For each claim you make, cite the specific
  component from the diagram that supports it."
  ```

#### Stage 6 — Structured Output
- **Response schema (JSON):**
  ```json
  {
    "direct_answer": "string",
    "reasoning_trace": ["step1", "step2", "step3"],
    "grounded_claims": [
      {
        "claim": "string",
        "supporting_component": "component_id",
        "bounding_box": [x1, y1, x2, y2],
        "confidence": "HIGH|MEDIUM|LOW"
      }
    ],
    "diagram_highlights": ["component_id_1", "component_id_2"],
    "follow_up_questions": ["question1", "question2", "question3"],
    "overall_confidence": 0.0
  }
  ```

---

## 4. Functional Requirements

### 4.1 Input Requirements
| Requirement ID | Requirement | Priority |
|---|---|---|
| FR-01 | System SHALL accept PNG, JPG, JPEG, SVG, and PDF diagram images | MUST |
| FR-02 | System SHALL accept natural language queries in English | MUST |
| FR-03 | System SHALL handle images up to 10MB in size | MUST |
| FR-04 | System SHALL accept multi-page PDFs and process the first diagram page | SHOULD |
| FR-05 | System SHALL provide an image preprocessing option (resize, denoise) | SHOULD |

### 4.2 Processing Requirements
| Requirement ID | Requirement | Priority |
|---|---|---|
| FR-06 | System SHALL execute all 6 pipeline stages for every query | MUST |
| FR-07 | System SHALL construct a valid NetworkX graph from Stage 2 before Stage 4 | MUST |
| FR-08 | System SHALL compute MGF score after Stage 2 and halt if below threshold | MUST |
| FR-09 | System SHALL classify diagram type before activating domain heuristics | MUST |
| FR-10 | System SHALL assign confidence level to every claim in the output | MUST |
| FR-11 | System SHALL run entirely locally via Ollama (zero cloud dependency) | MUST |
| FR-12 | System SHALL generate proactive follow-up questions after every response | SHOULD |

### 4.3 Output Requirements
| Requirement ID | Requirement | Priority |
|---|---|---|
| FR-13 | System SHALL return a direct answer within the structured JSON schema | MUST |
| FR-14 | System SHALL return a collapsible reasoning trace | MUST |
| FR-15 | System SHALL return bounding box coordinates for every cited component | MUST |
| FR-16 | System SHALL highlight cited diagram regions on the original image | SHOULD |
| FR-17 | System SHALL return an overall confidence score (0.0–1.0) | MUST |
| FR-18 | System SHALL export the extracted graph as GraphML on request | SHOULD |

---

## 5. Technical Stack & Dependencies

### 5.1 Core Model Stack
| Component | Tool/Library | Version | Purpose |
|---|---|---|---|
| Vision-Language Model | LLaVA 1.6 | 13B | Visual understanding + NL generation |
| Local inference engine | Ollama | Latest | Run LLaVA locally, no API needed |
| Graph library | NetworkX | ≥ 3.0 | Symbolic graph construction + traversal |
| OCR engine | EasyOCR | ≥ 1.7 | Text label extraction from diagrams |
| Image processing | OpenCV + PIL | ≥ 4.8 / 10.0 | Preprocessing, bounding box drawing |

### 5.2 Backend Stack
| Component | Tool | Version | Purpose |
|---|---|---|---|
| Web framework | Flask | ≥ 3.0 | REST API serving the 6-stage pipeline |
| API format | JSON (REST) | — | Input/output contract |
| CORS | flask-cors | ≥ 4.0 | Allow frontend to call Flask |
| Environment | Python venv | 3.10+ | Dependency isolation |

### 5.3 Frontend Stack
| Component | Tool | Purpose |
|---|---|---|
| UI framework | React (or plain HTML/JS) | Upload interface + result display |
| Diagram overlay | Canvas API or fabric.js | Bounding box highlighting on image |
| Reasoning trace | Collapsible accordion component | Step-by-step CoT display |

### 5.4 Development & Research Tools
| Tool | Purpose |
|---|---|
| Overleaf (Springer LNCS template) | Paper writing in LaTeX |
| Zotero + BibTeX | Citation management → exports directly to Overleaf |
| PlantUML | Generate primary evaluation dataset (DiagramIQ-Eval) |
| pandas + matplotlib/seaborn | Generate evaluation charts for results section |
| Jupyter Notebook | Experimentation, metric computation, ablation analysis |
| GitHub (private repo) | Version control for code + paper source |
| Google Scholar + Semantic Scholar | Literature search |
| Connected Papers | Visual citation mapping for related work |

### 5.5 Full Python Requirements (`requirements.txt`)
```
flask>=3.0.0
flask-cors>=4.0.0
requests>=2.31.0
Pillow>=10.0.0
opencv-python>=4.8.0
easyocr>=1.7.0
networkx>=3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
datasets>=2.14.0
plantuml>=0.3.0
jupyter>=1.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### 5.6 Hardware Requirements
| Spec | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| VRAM (GPU) | 8 GB | 12–16 GB (for LLaVA 13B) |
| Storage | 20 GB free | 40 GB free |
| OS | Windows 10 / Ubuntu 20.04 | Ubuntu 22.04 LTS |
| Ollama | Must be installed and running | — |

> ℹ️ If VRAM < 8GB, use LLaVA 7B instead of 13B. Results may differ slightly — note this in the paper's limitations section.

---

## 6. Dataset Requirements

### 6.1 Dataset Overview

| Dataset | Role | Size | Source | Access |
|---|---|---|---|---|
| **DiagramIQ-Eval** | Primary evaluation | ~100 diagrams | Self-generated via PlantUML | You own it |
| **OpenRaiser/DiagramGen** | Supplementary benchmark | ~thousands | HuggingFace (CC BY 4.0) | Free login |
| **AI2D** | Out-of-domain ablation | ~5000 diagrams | Allen Institute | Free, no login |
| **UMLCode-ClassDiagram-Scored** | Ablation (quality-stratified) | 5000 samples | HuggingFace (DOI: 10.57967/hf/5932) | Free login |

### 6.2 DiagramIQ-Eval — Construction Requirements (PRIMARY)
This is the most important dataset. You build it yourself.

**Composition (100 diagrams total):**
| Diagram Type | Count | Rationale |
|---|---|---|
| UML Class Diagrams | 30 | Core software design skill |
| UML Sequence Diagrams | 25 | Temporal reasoning test |
| ER Diagrams | 25 | Structural relationship test |
| Flowcharts | 20 | Control flow reasoning test |

**Per-diagram annotation requirements:**
- 3 ground-truth QA pairs (question + reference answer)
- Diagram type label
- Complexity label: `SIMPLE` (< 5 components) / `MEDIUM` (5–10) / `COMPLEX` (> 10)
- PlantUML source code (preserves reproducibility)
- Rendered PNG export (800×600px minimum)

**QA pair types to include (distribute across the set):**
1. Component identification: *"What type of component is X?"*
2. Relationship reasoning: *"What is the relationship between A and B?"*
3. Structural analysis: *"What breaks if X is removed?"*
4. Flow tracing: *"Explain the data flow from A to B"*
5. Pedagogical explanation: *"Explain this diagram to a student learning software design"*

**Construction steps:**
1. Write PlantUML `.puml` scripts (25–30 scripts per diagram type)
2. Render to PNG using `plantuml -tpng *.puml`
3. Write 3 QA pairs per diagram in a CSV file: `diagram_id, question, reference_answer, question_type, complexity`
4. Peer-validate 20% of QA pairs with your co-author
5. Professor spot-checks 10 diagrams before finalising

### 6.3 External Dataset Access
```python
# OpenRaiser/DiagramGen
from datasets import load_dataset
ds = load_dataset("OpenRaiser/DiagramGen")

# UMLCode-ClassDiagram-Scored
ds = load_dataset("nguyenvanviet/UMLCode-ClassDiagram-DeepSeek-32B-Scored")

# AI2D — download directly from:
# https://prior.allenai.org/projects/diagram-understanding
# (no login required, 945MB zip)
```

### 6.4 Dataset Citation Requirements (for paper)
Every dataset used must be cited with:
- Dataset name + version/revision
- Author(s)
- DOI or URL
- Licence
- Year accessed

---

## 7. Evaluation & Metrics Requirements

### 7.1 Quantitative Metrics

| Metric | Full Name | What It Measures | How to Compute |
|---|---|---|---|
| **EMA** | Exact Match Accuracy | How perfectly the extracted graph matches ground truth | `correct_edges / total_edges` across DiagramIQ-Eval |
| **GMS** | Graph Matching Score | Structural similarity of extracted vs ground-truth graph | Graph edit distance normalised 0–1 |
| **MGF** | Multimodal Grounding Fidelity | How accurately final answer claims map to image regions | % of claims with valid bounding box attribution |
| **HRR** | Hallucination Reduction Rate | Reduction in false claims vs baseline LLaVA | `(baseline_errors - DiagramIQ_errors) / baseline_errors` |
| **PEQ** | Pedagogical Explanation Quality | Student-rated usefulness of explanations | Likert scale 1–5 from 15-student user study |

### 7.2 Baseline Models for Comparison
DiagramIQ must be compared against at least two baselines:

| Baseline | Why include it |
|---|---|
| **LLaVA 1.6 (vanilla, no pipeline)** | Isolates the contribution of the 6-stage pipeline itself |
| **GPT-4o (via API, few queries)** | State-of-the-art cloud model — shows trade-off: accuracy vs privacy/cost |

> ℹ️ You only need ~20–30 GPT-4o API calls for comparison (free tier or minimal paid use). Don't run the full 100 diagrams through GPT-4o.

### 7.3 Ablation Study Requirements
Run three ablation experiments to prove each component's contribution:

| Ablation | What you remove | Expected finding |
|---|---|---|
| **No Stage 4** (no graph traversal) | Remove NetworkX, use only LLaVA | Lower EMA and HRR — proves symbolic reasoning contribution |
| **No Stage 2** (no graph construction) | Skip JSON graph, ask LLaVA directly | Higher hallucination rate — proves structured extraction contribution |
| **No confidence scoring** | Remove MGF check and confidence labels | Higher error propagation — proves confidence gating contribution |

### 7.4 User Study Requirements (PEQ)
- **Participants:** 15 classmates from your computing programme
- **Task:** Given a diagram + DiagramIQ's explanation, rate on 5 criteria (1–5 Likert each):
  1. Clarity — "The explanation was easy to understand"
  2. Completeness — "All key components were covered"
  3. Accuracy — "The explanation matched what I see in the diagram"
  4. Usefulness — "This would help me learn from this diagram"
  5. Trust — "I trust the confidence scores provided"
- **Format:** Google Form (anonymous, 15 minutes per participant)
- **Ethics note:** Since this involves human participants, mention in the paper that participation was voluntary and anonymous. Confirm with your professor whether institutional ethics approval is required at your university.

---

## 8. Non-Functional Requirements

### 8.1 Performance Requirements
| Requirement | Target |
|---|---|
| End-to-end response time (all 6 stages) | < 45 seconds on recommended hardware |
| Stage 1 completion time | < 15 seconds |
| Stage 4 graph traversal time | < 1 second (NetworkX is fast) |
| Maximum image size handled | 10 MB |
| Concurrent users (for demo) | 1 (local system, single-user) |

### 8.2 Reliability Requirements
| Requirement | Detail |
|---|---|
| Graceful failure | If any stage fails, system returns partial output with error annotation — never a silent failure |
| MGF self-correction | If Stage 2 MGF < 0.6, automatically re-query Stage 1 once before flagging |
| Ollama connection check | On startup, system verifies LLaVA model is loaded before accepting requests |

### 8.3 Reproducibility Requirements (Critical for paper)
- All PlantUML source files for DiagramIQ-Eval committed to GitHub
- All evaluation scripts (`evaluate_ema.py`, `evaluate_hmr.py`, etc.) committed to GitHub
- `requirements.txt` pinned with exact versions
- Random seeds set for any stochastic operations (`random.seed(42)`)
- README with full setup instructions (Ollama install → model pull → Flask run)
- GitHub repo must be public before paper submission (reviewers may check)

### 8.4 Openness Requirements
- Code: MIT Licence
- Dataset (DiagramIQ-Eval): CC BY 4.0
- Paper: Submit to arXiv as preprint after conference acceptance

---

## 9. Development Milestones

| Phase | Milestone | Deliverable | Target Duration |
|---|---|---|---|
| **M1** | Environment setup | Ollama + LLaVA 1.6 running, Flask API skeleton | Week 1 |
| **M2** | Stage 1 complete | Component + OCR extraction working on 5 test diagrams | Week 2 |
| **M3** | Stage 2 complete | NetworkX graph construction from Stage 1 output | Week 3 |
| **M4** | Stage 3 complete | Diagram type classifier working on all 4 types | Week 3 |
| **M5** | Stage 4 complete | Graph traversal queries answering correctly | Week 4–5 |
| **M6** | Stage 5 + 6 complete | Full pipeline end-to-end on 10 test diagrams | Week 5–6 |
| **M7** | DiagramIQ-Eval built | 100 annotated diagrams with 300 QA pairs | Week 6–7 |
| **M8** | Evaluation complete | All 5 metrics computed, ablation done, user study done | Week 8–9 |
| **M9** | Paper first draft | All sections written in Overleaf | Week 10–11 |
| **M10** | Professor review | Full paper reviewed + revised | Week 12 |
| **M11** | Submission | Formatted + submitted to target conference | Week 13 |

---

## 10. Team Responsibilities

| Section / Task | Primary Owner | Reviewer |
|---|---|---|
| System implementation (Stages 1–4) | Author 1 (You) | Author 2 |
| System implementation (Stages 5–6) | Author 1 (You) | Professor |
| DiagramIQ-Eval dataset construction | Both authors | Professor |
| Evaluation scripts + metric computation | Author 1 | Author 2 |
| User study design + execution | Author 2 | Professor |
| Literature review + Related Work section | Author 2 | Author 1 |
| Methodology section | Author 1 | Professor |
| Results + Discussion section | Both authors | Professor |
| Introduction + Abstract | Professor (leads) | Both authors |
| Conclusion + Future Work | Author 2 | Author 1 |
| References + BibTeX management | Author 2 (Zotero) | Author 1 |
| Overleaf setup + formatting | Author 1 | — |
| GitHub repo setup + README | Author 1 | — |

---

## 11. Paper Structure Requirements

Each section has a minimum content requirement:

### Abstract (250 words max)
- Problem in 1 sentence
- Proposed system in 1 sentence
- Key technical approach in 1 sentence
- Main results (2–3 numbers: EMA%, HRR%, PEQ score)
- Conclusion in 1 sentence

### 1. Introduction (~1 page)
- Hook: the gap students face with software diagrams
- What exists and why it fails (3 gaps, 3 sentences)
- What DiagramIQ does
- Explicit contributions as a numbered list (3 contributions — see Section 2.3)
- Paper organisation paragraph

### 2. Related Work (~1.5 pages)
Must cover and cite:
- Multimodal Chain-of-Thought reasoning (LLaVA-CoT, LLaVA-1.5, InstructBLIP)
- UML diagram generation using LLMs (NOMAD, PLM-based approaches, the Nguyen et al. FDSE 2025 paper)
- MLLM evaluation for software diagrams (Ibáñez et al. 2025, Wiley CAE — doi:10.1002/cae.70080)
- Neuro-symbolic AI (NPS-MMR, GraphRAG, ChatP&ID)
- Educational AI tools for software engineering learners (Garaccione et al. 2025, MDPI)
- Conclude with a table: existing systems vs DiagramIQ across 5 axes

### 3. Methodology (~2.5 pages)
- System overview diagram (the architecture figure — required)
- Each of the 6 stages described with: inputs, process, outputs, prompt template
- Graph construction formalism (define G = (V, E, L) mathematically)
- Confidence scoring formula
- MGF formula

### 4. Experimental Setup (~1 page)
- DiagramIQ-Eval construction details
- External datasets used + justification
- Baseline models
- Hardware/software environment
- Evaluation metric definitions (formulae)

### 5. Results & Discussion (~1.5 pages)
- Main results table (DiagramIQ vs baselines on all 5 metrics)
- Ablation study results table
- User study PEQ results + bar chart
- Qualitative examples (2–3 diagrams with DiagramIQ output shown)
- Discussion: where does DiagramIQ fail? (be honest — reviewers respect this)

### 6. Conclusion (~0.5 pages)
- Restate contributions in past tense
- Key finding in one sentence
- 2–3 specific future work directions:
  1. Multi-agent debate for deeper architectural critique
  2. Fine-tuning LLaVA on DiagramIQ-Eval for domain specialisation
  3. Expanding to dynamic diagrams (animated sequence diagrams)

### References
- Minimum 20 references
- All cited in-text
- Formatted in Springer LNCS BibTeX style
- No reference older than 2019 unless it's a foundational paper (e.g., original LLaVA, UML spec)

---

## 12. Citation & Reference Requirements

### 12.1 Must-Cite Papers
| Paper | Why | Citation type |
|---|---|---|
| LLaVA-CoT (Xu et al., arXiv:2411.10440) | Direct technical ancestor | Related Work + Methodology |
| Ibáñez et al. 2025 (Wiley CAE, doi:10.1002/cae.70080) | Establishes limitation DiagramIQ solves | Related Work + Introduction |
| LLaVA 1.5 / LLaVA 1.6 (Liu et al.) | Base model being used | Methodology |
| NetworkX (Hagberg et al.) | Graph library used | Methodology |
| Nguyen et al. FDSE 2025 (Springer CCIS 2709) | Dataset being used + related pipeline | Related Work + Experiments |
| Garaccione et al. 2025 (MDPI) | Education use case evidence | Introduction + Related Work |
| ScienceQA (Lu et al., NeurIPS 2022) | Benchmark context for LLaVA evaluation | Experiments |
| AI2D (Kembhavi et al.) | Ablation dataset | Experiments |

### 12.2 BibTeX Management
- Use Zotero browser plugin to import directly from Google Scholar / Springer / arXiv
- Export as BibTeX and sync to Overleaf
- Keep a `references.bib` file in the shared Overleaf project
- Check for duplicate keys before submission

---

## 13. Submission Requirements

### 13.1 Pre-Submission Checklist
- [ ] All 6 pipeline stages implemented and tested
- [ ] DiagramIQ-Eval dataset complete (100 diagrams, 300 QA pairs)
- [ ] All 5 metrics computed and results tables filled
- [ ] Ablation study (3 variants) complete
- [ ] User study (15 participants) complete
- [ ] GitHub repo public with README, code, and dataset
- [ ] Paper formatted in Springer LNCS LaTeX template
- [ ] Page limit respected (check with target venue — typically 8–12 pages)
- [ ] All figures are high-resolution (300 DPI minimum for camera-ready)
- [ ] All references formatted correctly in BibTeX
- [ ] Professor has reviewed and approved the final version
- [ ] Abstract is within word limit
- [ ] Author affiliations and emails correct
- [ ] Plagiarism check run (use iThenticate or Turnitin — ask your professor)
- [ ] Paper proofread by at least 2 people
- [ ] arXiv preprint uploaded (after acceptance)

### 13.2 Target Conference Details to Confirm
Research and fill in before submission:
- [ ] Conference name + year + edition
- [ ] Submission portal URL (usually EasyChair or CMT)
- [ ] Paper submission deadline
- [ ] Notification date
- [ ] Camera-ready deadline
- [ ] Conference date and location
- [ ] Registration fee and student discount availability
- [ ] Whether Scopus / SCIE indexing is confirmed for that year's proceedings

---

*Document maintained by: Deepasha | Last updated: August 2026 | Version: 1.0*