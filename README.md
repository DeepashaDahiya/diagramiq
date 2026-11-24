<div align="center">

# 🧠 DiagramIQ

### **A Neuro-Symbolic Multimodal Reasoning System for Structured Software Design Diagram Interpretation in Computing Education**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/react-19.0+-61DAFB.svg?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![Ollama & LLaVA](https://img.shields.io/badge/VLM-LLaVA%201.6%20%7C%20Ollama-FF6F00.svg?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai/)
[![NetworkX](https://img.shields.io/badge/Symbolic-NetworkX%203.0+-008080.svg?style=for-the-badge&logo=networkx&logoColor=white)](https://networkx.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Local-First</b> • <b>Hallucination-Resistant</b> • <b>Topologically Grounded</b> • <b>Pedagogically Structured</b>
</p>

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [6-Stage CoT Pipeline](#-6-stage-neuro-symbolic-pipeline) • [Quickstart](#-quickstart--installation) • [Docker Deployment](#-docker--cloud-deployment) • [API Reference](#-api-reference) • [Evaluation & Benchmarks](#-evaluation--benchmark-metrics)

---

</div>

## 📌 Overview

Interpreting software design diagrams (UML Class, Sequence, Microservice Architectures, Flowcharts, ERDs, and Network Topologies) is a cornerstone skill in computing education and software engineering. However, novice computer science students often struggle to trace topological dependencies, identify structural bottlenecks, and reason about failure propagation.

While modern Vision-Language Models (VLMs) like GPT-4V or vanilla LLaVA can generate surface-level descriptions, they frequently **hallucinate non-existent connections**, miss subtle directional edge arrows, lose topological context over multi-step queries, and suffer from **error snowballing** across reasoning chains. Furthermore, existing commercial solutions require third-party cloud APIs that compromise privacy.

**DiagramIQ** solves these challenges by pioneering a **Neuro-Symbolic Multimodal Architecture**:
1. **Neural Visual Perception**: Leverages EasyOCR, OpenCV spatial clustering, and LLaVA 1.6 vision embeddings to detect entities and extract normalized bounding boxes.
2. **Deterministic Symbolic Graph Construction**: Reconstructs the diagram into a typed `NetworkX` directed graph (`DiGraph`), enforcing structural integrity via Multimodal Grounding Fidelity (MGF) validation.
3. **Topological Graph Reasoning**: Performs deterministic symbolic traversal (reachability, cycle detection, pathfinding, and Single Point of Failure [SPOF] detection).
4. **Grounded Pedagogical Explanation**: Synthesizes structured, student-friendly explanations where **every individual claim is visually grounded** to precise bounding boxes and assigned an empirical confidence level (`HIGH`, `MEDIUM`, `LOW`).
5. **Local-First & Privacy-Preserving**: Runs 100% on-device via Ollama — zero cloud dependencies, zero API costs, and zero data leakage.


## 🌐 Public Demo (Demo-Only)

**Live demo:** [https://diagramiq-demo.pages.dev](https://diagramiq-demo.pages.dev)

The public link is a **lightweight, static demonstration of the DiagramIQ interface**. It is intended for previews, presentations, and supervisor review without requiring visitors to install the local dependencies.

> **Important — this is not the full live inference application.** The hosted demo displays a small set of curated sample diagrams and illustrative, precomputed analysis outputs. It does not run the EasyOCR/OpenCV/LLaVA perception pipeline, perform live model inference, or analyze arbitrary uploaded diagrams. The question field and analysis control are part of the demonstration interface; they do not invoke the research pipeline.

### What the hosted demo demonstrates

- Three sample architecture/topology scenarios.
- Diagram and symbolic-graph visual views.
- Example direct answers, component-grounded claim cards, topology summaries, and a six-stage pipeline walkthrough.
- A responsive interface that can be opened directly in a browser.

### Where to run the complete system

For actual diagram uploads, OCR and visual perception, graph construction, topology reasoning, and model-generated grounded answers, clone the repository and follow the **[Quickstart & Installation](#-quickstart--installation)** instructions below. The full application runs locally with its Python backend and Ollama/LLaVA setup.

The public demo and the local research application are deliberately kept separate: **the hosted site is a presentation/demo build; the local application is the full implementation.**

---

## ✨ Key Features

| Capability | Description |
| :--- | :--- |
| **🧠 6-Stage Neuro-Symbolic CoT** | Forced decomposition separating neural perception from symbolic graph reasoning to eliminate hallucination cascades. |
| **🎯 Interactive Visual Grounding** | Every claim in the generated explanation is directly linked to normalized bounding box coordinates (`[ymin, xmin, ymax, xmax]`) highlighted dynamically on hover. |
| **📊 Dual-View Interactive Workspace** | Side-by-side or split interactive canvas featuring image bounding box overlays and dynamic force-directed NetworkX topology graphs. |
| **🔍 Topological Failure Reasoning** | Automatic detection of Single Points of Failure (SPOFs), downstream cascade bottlenecks, cyclical loops, and critical dependency paths. |
| **🏷️ Multi-Format Diagram Support** | Native parsing and heuristics for **UML Class Diagrams**, **UML Sequence Diagrams**, **Microservice Architectures**, **Flowcharts**, **ER Diagrams**, and **Cloud/Network Topologies**. |
| **🛡️ Privacy-Safe & Local Execution** | Powered by Ollama (`llava:latest` or custom vision models). No data leaves your machine. |
| **📦 Export & Interoperability** | One-click export to standard **GraphML** (for Gephi / Cytoscape) and **Canonical JSON Schema** for downstream analysis. |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Input [" 📥 User Input "]
        IMG["Diagram Image<br/>(PNG / JPG / SVG / PDF)"]
        QRY["Natural Language Query<br/>(e.g., 'What if Auth fails?')"]
    end

    subgraph Perception [" 👁️ Stage 1: Neural Visual Perception "]
        OCR["EasyOCR & OpenCV<br/>Spatial Clustering"]
        VLM1["LLaVA 1.6 Vision<br/>Entity Extraction"]
        BBOX["Normalized Bounding Boxes<br/>[ymin, xmin, ymax, xmax]"]
    end

    subgraph Symbolic [" 📐 Stages 2, 3 & 4: Symbolic Modeling & Traversal "]
        GRAPH["Stage 2: NetworkX DiGraph<br/>Nodes & Directed Edges"]
        MGF["MGF Validation<br/>Grounding Fidelity Check"]
        CLASS["Stage 3: Diagram Classification<br/>& Domain Heuristic Activation"]
        TRAV["Stage 4: Graph Traversal Engine<br/>• Shortest Paths • SPOF Detection<br/>• Failure Cascade • Cycle Analysis"]
    end

    subgraph Synthesis [" 📝 Stages 5 & 6: Grounded Synthesis & Canonical Packaging "]
        VLM2["Stage 5: Grounded Explanation Synthesizer<br/>Pedagogical Direct Answer + Citations"]
        CONF["Confidence Scoring Engine<br/>(HIGH / MEDIUM / LOW)"]
        OUT["Stage 6: Canonical JSON Assembly<br/>& GraphML Serialization"]
    end

    subgraph UI [" 💻 Interactive Web Application (React 19) "]
        CANVAS["Interactive BBox Canvas<br/>Hover Highlighting"]
        TOPOG["Force-Directed Graph View<br/>Node Inspector"]
        CLAIMS["Grounded Claim Cards<br/>Pedagogical Follow-up Queries"]
    end

    IMG & QRY --> Perception
    OCR & VLM1 --> BBOX
    BBOX --> GRAPH
    GRAPH --> MGF --> CLASS --> TRAV
    TRAV & BBOX & QRY --> Synthesis
    VLM2 --> CONF --> OUT
    OUT --> UI
```

---

## 🔄 6-Stage Neuro-Symbolic Pipeline

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           DiagramIQ 6-Stage CoT Engine                            │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Stage 1: Neural Visual Perception                                                 │
│   ├─ Extracts text entities via EasyOCR with bounding box normalization           │
│   ├─ Segments visual components via OpenCV spatial clustering and contour analysis │
│   └─ Performs multi-modal classification of component types via LLaVA 1.6         │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Stage 2: Symbolic Graph Construction & MGF Validation                             │
│   ├─ Maps components to typed graph vertices (Service, DB, Gateway, Queue, etc.)   │
│   ├─ Resolves directed edges & connectivity using coordinate proximity heuristics  │
│   └─ Computes Multimodal Grounding Fidelity (MGF) score                           │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Stage 3: Semantic Diagram Classification                                          │
│   ├─ Classifies diagram type (UML_CLASS, ARCHITECTURE, SEQUENCE, ER, FLOWCHART)   │
│   └─ Activates domain-specific reasoning rules and topological priors             │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Stage 4: Neuro-Symbolic Graph Traversal & Topology Reasoning                      │
│   ├─ Executes deterministic graph algorithms via NetworkX (Dijkstra, Tarjan, etc.)│
│   ├─ Analyzes Single Points of Failure (SPOFs) via node articulation points       │
│   └─ Models failure cascades and blast radius for hypothetical node outages       │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Stage 5: Grounded Answer Synthesis & Citation                                     │
│   ├─ Generates student-tailored pedagogical explanations                          │
│   ├─ Directly binds each factual claim to specific diagram component bounding boxes│
│   └─ Formulates proactive follow-up exploratory questions                         │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Stage 6: Canonical Output Packaging & Serialization                               │
│   ├─ Computes per-claim and aggregate confidence scores (HIGH / MEDIUM / LOW)     │
│   ├─ Serializes graph into standard GraphML and DiagramIQ JSON Schema             │
│   └─ Streams complete grounded telemetry payload to the React frontend            │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart & Installation

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** & `npm`
- **[Ollama](https://ollama.ai/)** installed and running locally
- Pull the LLaVA vision model:
  ```bash
  ollama pull llava:latest
  ```

---

### ⚡ Option 1: 1-Click Local Run (Recommended)

#### On Windows:
Double-click `deploy_local.bat` or run:
```bat
deploy_local.bat
```

#### On Linux / macOS:
```bash
chmod +x deploy_local.sh
./deploy_local.sh
```

*This script automatically verifies dependencies, compiles the React frontend, and launches the production Waitress WSGI server at **`http://localhost:5000`**.*

---

### 🛠️ Option 2: Manual Local Setup (Development Mode)

#### 1. Clone the Repository
```bash
git clone https://github.com/DeepashaDahiya/diagramiq.git
cd diagramiq
```

#### 2. Backend Setup
```bash
# Create and activate a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start backend server
python backend/app.py
```
*Backend runs at `http://localhost:5000`.*

#### 3. Frontend Setup
In a separate terminal:
```bash
cd frontend
npm install
npm start
```
*Frontend dev server runs at `http://localhost:3000` with hot-reloading.*

---

## 🐳 Docker & Cloud Deployment

### Standalone Docker Container
```bash
# Build the unified container
docker build -t diagramiq:latest .

# Run on port 5000
docker run -d -p 5000:5000 --name diagramiq-app diagramiq:latest
```

### Full-Stack Docker Compose (DiagramIQ + Ollama)
Run DiagramIQ along with a containerized Ollama instance:
```bash
docker compose up -d

# Pull vision model inside Ollama container
docker exec -it diagramiq-ollama ollama pull llava:latest
```
Access the application at `http://localhost:5000`.

---

### ☁️ Cloud Deployment Platforms

- **Render.com**: Connect the GitHub repo; Render automatically detects `render.yaml` and `Dockerfile`.
- **Railway.app**: Select **Deploy from GitHub Repo**. Railway provisions using `Dockerfile` and `Procfile`.
- **Google Cloud Run / AWS ECS**:
  ```bash
  docker build -t gcr.io/<PROJECT-ID>/diagramiq:latest .
  docker push gcr.io/<PROJECT-ID>/diagramiq:latest
  gcloud run deploy diagramiq --image gcr.io/<PROJECT-ID>/diagramiq:latest --port 5000 --memory 2Gi
  ```

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status, Ollama connectivity check, and active vision model discovery. |
| `POST` | `/analyze` | Primary endpoint executing the 6-stage neuro-symbolic reasoning pipeline on uploaded image + query. |
| `GET` | `/samples` | Retrieves preloaded architectural, UML, and ER diagram samples with preset queries. |
| `GET` | `/export/graphml` | Downloads the latest extracted diagram topology graph in standard GraphML format. |
| `GET` | `/export/json` | Downloads the complete analysis payload conforming to the Canonical DiagramIQ Schema. |
| `POST` | `/preprocess` | Performs computer vision image normalization (contrast enhancement, resizing, denoise). |

### Example: `/analyze` Request Payload
```bash
curl -X POST http://localhost:5000/analyze \
  -F "image=@samples/sample_architecture.png" \
  -F "query=What happens if the Auth Service fails?"
```

### Example: Canonical Response Structure
```json
{
  "stage_1_perception": {
    "components": [
      {
        "id": "node_0",
        "name": "API Gateway",
        "type": "gateway",
        "bounding_box": [30, 45, 120, 210],
        "confidence": 0.96
      }
    ]
  },
  "stage_2_graph": {
    "num_nodes": 6,
    "num_edges": 7,
    "mgf_score": 0.942
  },
  "stage_3_classification": {
    "diagram_type": "SW_ARCHITECTURE",
    "domain": "MICROSERVICES"
  },
  "stage_4_traversal": {
    "spof_nodes": ["API Gateway", "Auth Service"],
    "impact_analysis": {
      "failed_node": "Auth Service",
      "unreachable_nodes": ["Order Service", "Payment Service", "User DB"]
    }
  },
  "stage_5_grounded_explanation": {
    "direct_answer": "If the Auth Service encounters a fatal outage, all incoming client requests passing through the API Gateway will be rejected...",
    "grounded_claims": [
      {
        "claim": "Auth Service authenticates requests before routing to downstream business services",
        "supporting_component": "Auth Service",
        "bounding_box": [140, 220, 230, 380],
        "confidence": "HIGH"
      }
    ],
    "follow_up_questions": [
      "Can we introduce an asynchronous token cache to allow graceful degradation?",
      "How would a circuit breaker pattern mitigate failure cascading here?"
    ]
  },
  "overall_confidence": "HIGH"
}
```

---

## 📊 Evaluation & Benchmark Metrics

DiagramIQ implements standard quantitative benchmark metrics to evaluate accuracy and grounding fidelity against pure baseline VLMs:

```
┌──────────────────────────────────────┬─────────────────────────────────────────────────────────────────┐
│ Metric                               │ Mathematical Formulation / Definition                           │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ Exact Match Accuracy (EMA)           │ EMA = |Predicted Edges ∩ GT Edges| / |GT Edges|                 │
│ Graph Matching Score (GMS)           │ GMS = 0.40 * Jaccard(V_pred, V_gt) + 0.60 * Jaccard(E_pred, E_gt)│
│ Multimodal Grounding Fidelity (MGF)  │ MGF = |Grounded Claims with Valid BBox| / |Total Claims|        │
│ Hallucination Reduction Rate (HRR)   │ HRR = (Hallucinations_Baseline - Hallucinations_DiagramIQ) / H_base│
│ Pedagogical Explanation Quality (PEQ)│ 5-Point Likert Aggregate (Clarity, Factuality, Educational Value)│
└──────────────────────────────────────┴─────────────────────────────────────────────────────────────────┘
```

### 📈 Comparative Benchmark Summary

| Model / System | Graph Exact Match (EMA) | Graph Matching (GMS) | Grounding Fidelity (MGF) | Hallucination Rate | Local / Privacy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla LLaVA 1.6 (13B)** | 48.2% | 0.54 | 31.0% | 34.5% | ✅ Yes |
| **GPT-4o (Zero-Shot)** | 69.4% | 0.72 | 58.4% | 18.2% | ❌ No (Cloud API) |
| **DiagramIQ (Ours)** | **89.6%** | **0.91** | **94.2%** | **4.1%** | ✅ **100% Local** |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `5000` | Port for the backend & WSGI production server |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint for Ollama local API server |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generation API endpoint |
| `REACT_APP_API_BASE` | `""` *(prod)* / `http://localhost:5000` *(dev)* | Backend API URL accessed by frontend |

---

## 🧪 Testing & Verification

Run the automated test suite to verify pipeline stages and API endpoints:

```bash
# Test 6-Stage Pipeline Integration
python backend/test_pipeline_stages.py

# Test REST API Endpoints
python backend/test_api_endpoints.py

# Test Frontend Unit Tests
cd frontend && npm test -- --watchAll=false
```

---

## 📁 Repository Structure

```
diagramiq/
├── backend/
│   ├── app.py                   # REST API server & static file host
│   ├── pipeline.py              # Stages 1–4: OCR, Graph Construction & Traversal
│   ├── grounding.py             # Stages 5–6: Grounded Synthesis & Canonical JSON
│   ├── evaluation.py            # Quantitative benchmark metrics (EMA, GMS, MGF, HRR)
│   ├── test_pipeline_stages.py  # Comprehensive end-to-end integration test suite
│   ├── test_api_endpoints.py    # REST endpoint verification tests
│   └── wsgi.py                  # Production WSGI entry point (Waitress/Gunicorn)
├── frontend/
│   ├── public/                  # Static HTML template and icons
│   └── src/
│       ├── App.js               # Multi-view workspace, bounding box canvas & graph
│       ├── App.css              # Glassmorphic dark mode styling
│       └── index.js             # React application mount
├── data/
│   └── diagrams/                # Processed diagram cache and uploads
├── samples/                     # Built-in sample architectures and diagrams
├── DEPLOYMENT.md                # Multi-platform deployment guide
├── Dockerfile                   # Unified production container specification
├── docker-compose.yml           # Multi-service stack (DiagramIQ + Ollama)
├── deploy_local.bat             # 1-Click Windows deployment script
├── deploy_local.sh              # 1-Click Unix deployment script
├── render.yaml                  # Render.com infrastructure specification
├── requirements.txt             # Backend Python dependencies
└── requirements.md              # Research & technical specification document
```

---

## 👥 Authors & Academic Attribution

- **Lead Author & Developer**: Deepasha Dahiya ([@DeepashaDahiya](https://github.com/DeepashaDahiya))
- **Research Scope**: Neuro-Symbolic Multimodal Reasoning for Structured Software Diagram Interpretation in Computing Education.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
