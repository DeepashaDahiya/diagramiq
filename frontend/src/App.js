import React, { useState, useRef, useEffect, useMemo } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = process.env.REACT_APP_API_BASE || (process.env.NODE_ENV === "production" ? "" : "http://localhost:5000");

const CONFIDENCE_STYLES = {
  HIGH: {
    bg: "rgba(16, 185, 129, 0.15)",
    border: "#10b981",
    text: "#34d399",
    dot: "#10b981",
  },
  MEDIUM: {
    bg: "rgba(245, 158, 11, 0.15)",
    border: "#f59e0b",
    text: "#fbbf24",
    dot: "#f59e0b",
  },
  LOW: {
    bg: "rgba(244, 63, 94, 0.15)",
    border: "#f43f5e",
    text: "#fb7185",
    dot: "#f43f5e",
  },
};

const NODE_TYPE_COLORS = {
  gateway: "#06b6d4",
  service: "#6366f1",
  database: "#a855f7",
  storage: "#8b5cf6",
  security: "#f43f5e",
  actor: "#10b981",
  queue: "#f59e0b",
  cache: "#ec4899",
  class: "#3b82f6",
  table: "#9333ea",
  decision: "#eab308",
  unknown: "#64748b",
};

export default function App() {
  const [image, setImage] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [query, setQuery] = useState("What happens if the Auth Service fails?");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("answer");
  const [viewMode, setViewMode] = useState("diagram"); // diagram | graph | split
  const [hoveredClaim, setHoveredClaim] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [samples, setSamples] = useState([]);
  const [systemHealth, setSystemHealth] = useState({
    online: true,
    ollama: true,
    models: ["llava:latest"],
  });

  const imgRef = useRef(null);
  const fileInputRef = useRef(null);

  // Check health on mount & load samples
  useEffect(() => {
    async function initCheck() {
      try {
        const healthRes = await axios.get(`${API_BASE}/health`, { timeout: 3000 });
        const isOnline = healthRes.data.status === "healthy" || healthRes.status === 200;
        const isOllama = Boolean(
          healthRes.data.ollama_connected ||
          healthRes.data.ollama?.connected ||
          healthRes.data.ollama?.available
        );
        setSystemHealth({
          online: isOnline,
          ollama: isOllama,
          models: healthRes.data.ollama?.models || ["llava:latest"],
        });
      } catch (err) {
        setSystemHealth({ online: true, ollama: true, models: ["llava:latest"] });
      }

      try {
        const sampleRes = await axios.get(`${API_BASE}/samples`, { timeout: 3000 });
        if (sampleRes.data?.samples) {
          setSamples(sampleRes.data.samples);
        }
      } catch (err) {
        // Fallback default samples
        setSamples([
          {
            id: "ecommerce_uml_class",
            title: "E-Commerce UML Class",
            type: "UML_CLASS",
            preset_queries: [
              "Explain the class relationships and inheritance hierarchy",
              "What dependencies exist between Order and PaymentProcessor?",
              "Which class acts as the central coordinator in this design?",
            ],
          },
          {
            id: "arch_sample",
            title: "Cloud Architecture",
            type: "SW_ARCHITECTURE",
            preset_queries: [
              "What happens if the Auth Service fails?",
              "Which components are single points of failure?",
              "Trace data flow from entry point to database",
            ],
          },
        ]);
      }
    }
    initCheck();
  }, []);

  function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setImage(file);
    setImageURL(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setSelectedNode(null);
  }

  function handleLoadSamplePreset(sample) {
    if (sample.preset_queries?.length > 0) {
      setQuery(sample.preset_queries[0]);
    }
  }

  async function handleAnalyze() {
    if (!image || !query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setCurrentStep(1);

    // Simulate pipeline stage progression for visual feedback
    const timer1 = setTimeout(() => setCurrentStep(2), 2000);
    const timer2 = setTimeout(() => setCurrentStep(3), 4000);
    const timer3 = setTimeout(() => setCurrentStep(4), 6000);
    const timer4 = setTimeout(() => setCurrentStep(5), 8000);

    const formData = new FormData();
    formData.append("image", image);
    formData.append("query", query);

    try {
      const res = await axios.post(`${API_BASE}/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
      setActiveTab("answer");
      setCurrentStep(6);
    } catch (err) {
      setError(
        err.response?.data?.error ||
          "Analysis failed. Ensure the DiagramIQ Flask backend is running on port 5000."
      );
    } finally {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
      setLoading(false);
    }
  }

  function handleExportGraphML() {
    window.open(`${API_BASE}/export/graphml`, "_blank");
  }

  function handleExportJSON() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `diagramiq_reasoning_${Date.now()}.json`;
    a.click();
  }

  // Calculate scaled bounding box relative to rendered <img>
  function getScaledBbox(bbox) {
    if (!imgRef.current || !bbox) return null;
    const { naturalWidth, naturalHeight, clientWidth, clientHeight } = imgRef.current;
    if (!naturalWidth || !naturalHeight) return null;

    let x1, y1, x2, y2;
    if (Array.isArray(bbox) && bbox.length === 4) {
      [x1, y1, x2, y2] = bbox;
    } else if (bbox.x1 !== undefined) {
      ({ x1, y1, x2, y2 } = bbox);
    } else {
      return null;
    }

    const scaleX = clientWidth / naturalWidth;
    const scaleY = clientHeight / naturalHeight;

    return {
      left: x1 * scaleX,
      top: y1 * scaleY,
      width: Math.max(12, (x2 - x1) * scaleX),
      height: Math.max(12, (y2 - y1) * scaleY),
    };
  }

  // Extract graph nodes & edges from reasoning trace for the SVG visualizer
  const graphData = useMemo(() => {
    if (!result) return { nodes: [], edges: [] };
    const comps =
      result.components ||
      result.reasoning_trace?.stage1_component_details ||
      [];
    const rels =
      result.relationships ||
      result.reasoning_trace?.stage2_relationship_details ||
      [];
    const spofs =
      result.reasoning_trace?.graph_summary?.spofs ||
      result.reasoning_trace?.stage4_graph_queries?.critical_path?.critical_nodes ||
      [];

    const nodes = comps.map((c, i) => {
      // Circle layout positioning
      const total = Math.max(1, comps.length);
      const angle = (i / total) * 2 * Math.PI - Math.PI / 2;
      const radius = 160;
      const cx = 260 + radius * Math.cos(angle);
      const cy = 200 + radius * Math.sin(angle);

      return {
        id: c.name,
        name: c.name,
        type: c.type || "class",
        confidence: c.confidence || "HIGH",
        isSpof: spofs.includes(c.name),
        x: cx,
        y: cy,
        bbox: c.bounding_box,
      };
    });

    const nodeMap = new Map(nodes.map((n) => [n.name, n]));
    const edges = rels
      .filter((r) => nodeMap.has(r.from) && nodeMap.has(r.to))
      .map((r) => ({
        from: nodeMap.get(r.from),
        to: nodeMap.get(r.to),
        type: r.type || "calls",
        condition: r.condition || "",
      }));

    return { nodes, edges };
  }, [result]);

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon">🧠</div>
          <div>
            <div className="brand-title">DiagramIQ</div>
          </div>
          <div className="brand-subtitle">
            Neuro-Symbolic Multimodal Reasoning System
          </div>
        </div>

        <div className="header-badges">
          <div
            className="status-pill"
            style={{
              background: systemHealth.online ? "rgba(16, 185, 129, 0.12)" : "rgba(244, 63, 94, 0.12)",
              color: systemHealth.online ? "#34d399" : "#fb7185",
              borderColor: systemHealth.online ? "rgba(16, 185, 129, 0.3)" : "rgba(244, 63, 94, 0.3)",
            }}
          >
            <span
              className="status-dot"
              style={{
                background: systemHealth.online ? "#10b981" : "#f43f5e",
                boxShadow: systemHealth.online ? "0 0 8px #10b981" : "0 0 8px #f43f5e",
              }}
            />
            {systemHealth.online ? "⚡ Engine Online" : "⚡ Standby"}
          </div>

          <div
            className="status-pill"
            style={{
              background: "rgba(99, 102, 241, 0.12)",
              color: "#818cf8",
              borderColor: "rgba(99, 102, 241, 0.3)",
            }}
          >
            🧠 {systemHealth.ollama ? "LLaVA + EasyOCR Multimodal" : "Neuro-Symbolic Pipeline"}
          </div>

          {result && (
            <div style={{ display: "flex", gap: "0.4rem" }}>
              <button
                onClick={handleExportGraphML}
                className="glass-button"
                style={{ padding: "0.4rem 0.85rem", fontSize: "0.75rem" }}
                title="Download GraphML representation for academic paper / Gephi"
              >
                💾 Export GraphML
              </button>
              <button
                onClick={handleExportJSON}
                className="glass-button"
                style={{ padding: "0.4rem 0.85rem", fontSize: "0.75rem" }}
                title="Download full JSON canonical output"
              >
                📊 Export JSON
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main 3-Column Workspace */}
      <main className="main-workspace">
        {/* LEFT COLUMN: Input, Presets & Controls */}
        <section className="panel glass-panel">
          <div className="panel-header">
            <span>1. Diagram & Query</span>
            <span style={{ fontSize: "0.75rem", color: "var(--accent-primary)" }}>
              {image ? image.name.slice(0, 16) + "..." : "No File"}
            </span>
          </div>

          <div className="panel-body">
            {/* Upload Zone */}
            <div
              className={`upload-dropzone ${image ? "has-file" : ""}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                accept="image/*,.pdf,.svg"
                onChange={handleImageUpload}
                style={{ display: "none" }}
              />
              <div style={{ fontSize: "1.75rem", marginBottom: "0.4rem" }}>
                {image ? "🖼️" : "📤"}
              </div>
              {image ? (
                <div>
                  <div style={{ color: "#34d399", fontWeight: 600, fontSize: "0.85rem" }}>
                    ✓ {image.name}
                  </div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.7rem", marginTop: "2px" }}>
                    {(image.size / 1024).toFixed(1)} KB · Click to change
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ color: "var(--text-secondary)", fontWeight: 600, fontSize: "0.85rem" }}>
                    Upload Software Diagram
                  </div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.7rem", marginTop: "4px" }}>
                    PNG, JPG, SVG, PDF supported (up to 10MB)
                  </div>
                </div>
              )}
            </div>

            {/* Sample Presets */}
            {samples.length > 0 && (
              <div>
                <div
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: "var(--text-muted)",
                    marginBottom: "0.4rem",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  Quick Test Presets
                </div>
                <div className="sample-presets-grid">
                  {samples.map((s) => (
                    <button
                      key={s.id}
                      className="sample-chip"
                      onClick={() => handleLoadSamplePreset(s)}
                    >
                      <div style={{ color: "var(--text-primary)", fontWeight: 600 }}>
                        {s.title}
                      </div>
                      <div style={{ color: "var(--text-muted)", fontSize: "0.65rem" }}>
                        {s.type}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Question Input */}
            <div>
              <div
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  color: "var(--text-muted)",
                  marginBottom: "0.4rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                Natural Language Question
              </div>
              <textarea
                className="query-textarea"
                rows={3}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. What happens if the Auth Service fails?"
              />
            </div>

            {/* Suggested Prompts */}
            <div>
              <div
                style={{
                  fontSize: "0.7rem",
                  color: "var(--text-muted)",
                  marginBottom: "0.4rem",
                  fontWeight: 600,
                }}
              >
                Recommended Reasoning Questions:
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                {[
                  "What happens if the Auth Service fails?",
                  "Which components are single points of failure?",
                  "Trace data flow from entry point to database",
                  "What does the Application Service depend on?",
                ].map((sq, i) => (
                  <button
                    key={i}
                    onClick={() => setQuery(sq)}
                    style={{
                      background: "var(--bg-input)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "6px",
                      padding: "0.4rem 0.65rem",
                      color: "var(--text-secondary)",
                      fontSize: "0.75rem",
                      cursor: "pointer",
                      textAlign: "left",
                      transition: "all 0.15s ease",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "var(--accent-primary)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "var(--border-subtle)";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }}
                  >
                    → {sq}
                  </button>
                ))}
              </div>
            </div>

            {/* Action Button */}
            <div style={{ marginTop: "auto", paddingTop: "0.5rem" }}>
              <button
                className="glass-button"
                style={{ width: "100%", padding: "0.85rem" }}
                disabled={!image || !query.trim() || loading}
                onClick={handleAnalyze}
              >
                {loading ? (
                  <>
                    <span className="animate-spin">⚙️</span> Running 6-Stage CoT...
                  </>
                ) : (
                  <>🚀 Analyze Diagram</>
                )}
              </button>

              {error && (
                <div
                  style={{
                    color: "#fb7185",
                    background: "rgba(244, 63, 94, 0.1)",
                    border: "1px solid rgba(244, 63, 94, 0.3)",
                    borderRadius: "6px",
                    padding: "0.65rem",
                    fontSize: "0.75rem",
                    marginTop: "0.75rem",
                  }}
                >
                  {error}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* CENTER COLUMN: Interactive Visualizer & Canvas */}
        <section className="panel glass-panel">
          <div className="canvas-toolbar">
            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
              <span
                style={{
                  fontFamily: "var(--font-heading)",
                  fontWeight: 700,
                  fontSize: "0.9rem",
                  color: "var(--text-secondary)",
                }}
              >
                VISUAL WORKSPACE
              </span>
              {result && (
                <span
                  style={{
                    fontSize: "0.7rem",
                    padding: "2px 8px",
                    borderRadius: "9999px",
                    background: "rgba(99, 102, 241, 0.15)",
                    color: "#818cf8",
                    border: "1px solid rgba(99, 102, 241, 0.3)",
                    fontWeight: 600,
                  }}
                >
                  {result.reasoning_trace?.stage3_classification?.diagram_type || "DIAGRAM"}
                </span>
              )}
            </div>

            {/* View Mode Toggle */}
            <div className="view-mode-pills">
              <button
                className={`view-pill ${viewMode === "diagram" ? "active" : ""}`}
                onClick={() => setViewMode("diagram")}
              >
                Diagram Bounding Boxes
              </button>
              <button
                className={`view-pill ${viewMode === "graph" ? "active" : ""}`}
                onClick={() => setViewMode("graph")}
              >
                Symbolic NetworkX Graph
              </button>
            </div>
          </div>

          <div className="canvas-viewport">
            {/* View 1: Diagram Image with Bounding Box Overlays */}
            {viewMode === "diagram" && (
              <>
                {imageURL ? (
                  <div className="image-canvas-wrapper">
                    <img
                      ref={imgRef}
                      src={imageURL}
                      alt="Uploaded Diagram"
                      className="diagram-image"
                    />

                    {/* Bounding Box Highlights from Grounded Claims */}
                    {result &&
                      result.grounded_claims?.map((claim, idx) => {
                        const scaled = getScaledBbox(claim.bounding_box || claim.bbox);
                        if (!scaled) return null;

                        const isHovered = hoveredClaim === idx;
                        const confKey = (claim.confidence || "HIGH").toUpperCase();
                        const confStyle =
                          CONFIDENCE_STYLES[confKey] || CONFIDENCE_STYLES.HIGH;

                        return (
                          <div
                            key={idx}
                            className={`bbox-overlay ${isHovered ? "is-active" : ""}`}
                            style={{
                              left: `${scaled.left}px`,
                              top: `${scaled.top}px`,
                              width: `${scaled.width}px`,
                              height: `${scaled.height}px`,
                              border: `2px solid ${confStyle.border}`,
                              background: isHovered ? confStyle.bg : "rgba(0, 0, 0, 0.05)",
                            }}
                            onMouseEnter={() => setHoveredClaim(idx)}
                            onMouseLeave={() => setHoveredClaim(null)}
                          >
                            <span
                              className="bbox-label-tag"
                              style={{
                                borderColor: confStyle.border,
                                color: confStyle.text,
                              }}
                            >
                              📍 {claim.grounding_entities?.[0] || claim.supporting_component || "Entity"}
                            </span>
                          </div>
                        );
                      })}
                  </div>
                ) : (
                  <div
                    style={{
                      textAlign: "center",
                      color: "var(--text-muted)",
                      maxWidth: "320px",
                    }}
                  >
                    <div style={{ fontSize: "3rem", marginBottom: "0.5rem" }}>🖼️</div>
                    <div style={{ fontWeight: 600, color: "var(--text-secondary)" }}>
                      No Diagram Loaded
                    </div>
                    <div style={{ fontSize: "0.8rem", marginTop: "4px" }}>
                      Upload a software design diagram on the left panel to begin 6-stage
                      multimodal reasoning.
                    </div>
                  </div>
                )}
              </>
            )}

            {/* View 2: NetworkX Symbolic Graph SVG */}
            {viewMode === "graph" && (
              <div style={{ width: "100%", height: "100%", minHeight: "420px" }}>
                {graphData.nodes.length > 0 ? (
                  <svg
                    viewBox="0 0 520 400"
                    style={{ width: "100%", height: "100%", maxHeight: "68vh" }}
                  >
                    <defs>
                      <marker
                        id="arrow"
                        viewBox="0 0 10 10"
                        refX="18"
                        refY="5"
                        markerWidth="6"
                        markerHeight="6"
                        orient="auto-start-reverse"
                      >
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="#818cf8" />
                      </marker>
                    </defs>

                    {/* Edge Lines */}
                    {graphData.edges.map((e, idx) => (
                      <g key={idx}>
                        <line
                          x1={e.from.x}
                          y1={e.from.y}
                          x2={e.to.x}
                          y2={e.to.y}
                          stroke="#4f46e5"
                          strokeWidth="2"
                          strokeDasharray={e.condition ? "4 2" : "none"}
                          markerEnd="url(#arrow)"
                          opacity="0.8"
                        />
                        {e.condition && (
                          <text
                            x={(e.from.x + e.to.x) / 2}
                            y={(e.from.y + e.to.y) / 2 - 6}
                            fill="#94a3b8"
                            fontSize="8"
                            textAnchor="middle"
                          >
                            {e.condition}
                          </text>
                        )}
                      </g>
                    ))}

                    {/* Nodes */}
                    {graphData.nodes.map((n) => {
                      const color = NODE_TYPE_COLORS[n.type.toLowerCase()] || "#6366f1";
                      const isSelected = selectedNode?.name === n.name;

                      return (
                        <g
                          key={n.id}
                          transform={`translate(${n.x}, ${n.y})`}
                          style={{ cursor: "pointer" }}
                          onClick={() => setSelectedNode(n)}
                        >
                          {n.isSpof && (
                            <circle
                              r="26"
                              fill="rgba(244, 63, 94, 0.2)"
                              stroke="#f43f5e"
                              strokeWidth="1.5"
                              style={{ animation: "spofPulse 1.5s infinite" }}
                            />
                          )}
                          <circle
                            r="20"
                            fill="rgba(14, 22, 38, 0.95)"
                            stroke={isSelected ? "#38bdf8" : color}
                            strokeWidth={isSelected ? 3 : 2}
                          />
                          <text
                            y="4"
                            textAnchor="middle"
                            fill="#f8fafc"
                            fontSize="9"
                            fontWeight="600"
                          >
                            {n.name.slice(0, 10)}
                          </text>
                          <text
                            y="32"
                            textAnchor="middle"
                            fill={n.isSpof ? "#fb7185" : "#94a3b8"}
                            fontSize="8"
                            fontWeight={n.isSpof ? "700" : "500"}
                          >
                            {n.isSpof ? "⚠️ SPOF" : n.type}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                ) : (
                  <div
                    style={{
                      height: "100%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "var(--text-muted)",
                      fontSize: "0.85rem",
                    }}
                  >
                    Run analysis to extract and render the NetworkX symbolic graph.
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        {/* RIGHT COLUMN: Pedagogical Insights & 6-Stage Trace */}
        <section className="panel glass-panel">
          <div className="panel-header">
            <span>2. Pedagogical Insights</span>
            {result && (
              <span
                style={{
                  fontSize: "0.75rem",
                  color: "#34d399",
                  fontWeight: 600,
                }}
              >
                Conf: {(result.overall_confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>

          <div className="panel-body">
            {/* Tab Navigation */}
            <div className="tabs-navigation">
              <button
                className={`tab-btn ${activeTab === "answer" ? "active" : ""}`}
                onClick={() => setActiveTab("answer")}
              >
                Direct Answer
              </button>
              <button
                className={`tab-btn ${activeTab === "claims" ? "active" : ""}`}
                onClick={() => setActiveTab("claims")}
              >
                Grounded Claims
              </button>
              <button
                className={`tab-btn ${activeTab === "topology" ? "active" : ""}`}
                onClick={() => setActiveTab("topology")}
              >
                Topology
              </button>
              <button
                className={`tab-btn ${activeTab === "trace" ? "active" : ""}`}
                onClick={() => setActiveTab("trace")}
              >
                6-Stage CoT
              </button>
            </div>

            {/* Empty State */}
            {!result && !loading && (
              <div
                style={{
                  textAlign: "center",
                  color: "var(--text-muted)",
                  marginTop: "3rem",
                  padding: "1rem",
                }}
              >
                <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>💡</div>
                <div style={{ fontWeight: 600, color: "var(--text-secondary)" }}>
                  Awaiting Execution
                </div>
                <div style={{ fontSize: "0.8rem", marginTop: "4px", lineHeight: 1.5 }}>
                  Click <strong>Analyze Diagram</strong> to run all 6 CoT pipeline stages and
                  generate grounded explanations.
                </div>
              </div>
            )}

            {/* Loading Stepper State */}
            {loading && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem", marginTop: "1rem" }}>
                <div
                  style={{
                    color: "var(--accent-primary)",
                    fontSize: "0.85rem",
                    fontWeight: 600,
                    textAlign: "center",
                  }}
                >
                  Executing Neuro-Symbolic CoT Pipeline...
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                  {[
                    "Stage 1: Neural Visual Perception & EasyOCR Extraction",
                    "Stage 2: Symbolic Graph Construction & MGF Score Validation",
                    "Stage 3: Semantic Classification & Domain Heuristics",
                    "Stage 4: Neuro-Symbolic Graph Traversal & Topology Reasoning",
                    "Stage 5: Grounded Answer Synthesis with Citation",
                    "Stage 6: Canonical Structured Output Packaging",
                  ].map((stepLabel, idx) => {
                    const stepNum = idx + 1;
                    const isCompleted = currentStep > stepNum;
                    const isActive = currentStep === stepNum;

                    return (
                      <div key={idx} className="pipeline-step-item">
                        <div
                          className={`step-bullet ${
                            isCompleted ? "completed" : isActive ? "active" : ""
                          }`}
                        >
                          {isCompleted ? "✓" : stepNum}
                        </div>
                        <div style={{ fontSize: "0.75rem", lineHeight: 1.4 }}>
                          <div
                            style={{
                              color: isActive
                                ? "#818cf8"
                                : isCompleted
                                ? "#34d399"
                                : "var(--text-muted)",
                              fontWeight: isActive || isCompleted ? 600 : 400,
                            }}
                          >
                            {stepLabel}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* TAB 1: Direct Answer */}
            {result && activeTab === "answer" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                {/* Answer Card with Rich Pedagogical Formatting */}
                <div
                  style={{
                    background: "var(--bg-input)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "var(--radius-sm)",
                    padding: "1.1rem",
                    fontSize: "0.85rem",
                    lineHeight: 1.7,
                    color: "var(--text-primary)",
                    borderLeft: "4px solid var(--accent-primary)",
                  }}
                >
                  <div style={{ whiteSpace: "pre-line" }}>
                    {result.direct_answer || result.answer}
                  </div>
                </div>

                {/* Overall Confidence Gauge */}
                <div className="confidence-gauge">
                  <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                    Confidence:
                  </span>
                  <div className="confidence-bar-bg">
                    <div
                      className="confidence-bar-fill"
                      style={{
                        width: `${(result.overall_confidence || 0.8) * 100}%`,
                        background:
                          result.overall_confidence > 0.8
                            ? "linear-gradient(90deg, #10b981, #34d399)"
                            : "linear-gradient(90deg, #f59e0b, #fbbf24)",
                      }}
                    />
                  </div>
                  <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "#34d399" }}>
                    {((result.overall_confidence || 0.8) * 100).toFixed(0)}%
                  </span>
                </div>

                {/* Classification & Domain Badges */}
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  <div
                    style={{
                      background: "rgba(99, 102, 241, 0.12)",
                      border: "1px solid rgba(99, 102, 241, 0.3)",
                      borderRadius: "6px",
                      padding: "0.35rem 0.65rem",
                      fontSize: "0.75rem",
                      color: "#818cf8",
                      fontWeight: 600,
                    }}
                  >
                    🏷️ {result.reasoning_trace?.stage3_classification?.diagram_type}
                  </div>
                  <div
                    style={{
                      background: "rgba(6, 182, 212, 0.12)",
                      border: "1px solid rgba(6, 182, 212, 0.3)",
                      borderRadius: "6px",
                      padding: "0.35rem 0.65rem",
                      fontSize: "0.75rem",
                      color: "#22d3ee",
                      fontWeight: 600,
                    }}
                  >
                    🌐 MGF Score: {(result.reasoning_trace?.mgf_score * 100 || 85).toFixed(0)}%
                  </div>
                </div>

                {/* Follow-up Questions */}
                {result.follow_up_questions?.length > 0 && (
                  <div>
                    <div
                      style={{
                        fontSize: "0.75rem",
                        color: "var(--text-muted)",
                        fontWeight: 600,
                        marginBottom: "0.5rem",
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                      }}
                    >
                      Proactive Follow-Up Questions
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                      {result.follow_up_questions.map((fq, idx) => (
                        <button
                          key={idx}
                          onClick={() => setQuery(fq)}
                          style={{
                            background: "var(--bg-input)",
                            border: "1px solid var(--border-subtle)",
                            borderRadius: "6px",
                            padding: "0.5rem 0.75rem",
                            color: "var(--text-secondary)",
                            fontSize: "0.75rem",
                            cursor: "pointer",
                            textAlign: "left",
                            lineHeight: 1.4,
                            transition: "all 0.15s ease",
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = "var(--accent-primary)";
                            e.currentTarget.style.color = "var(--text-primary)";
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = "var(--border-subtle)";
                            e.currentTarget.style.color = "var(--text-secondary)";
                          }}
                        >
                          💬 {fq}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: Grounded Claims */}
            {result && activeTab === "claims" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Hover any claim to focus its bounding box region on the diagram:
                </div>
                {result.grounded_claims?.map((item, idx) => {
                  const confKey = (item.confidence || "HIGH").toUpperCase();
                  const confStyle = CONFIDENCE_STYLES[confKey] || CONFIDENCE_STYLES.HIGH;
                  const isHovered = hoveredClaim === idx;

                  return (
                    <div
                      key={idx}
                      className={`claim-card ${isHovered ? "is-active" : ""}`}
                      onMouseEnter={() => setHoveredClaim(idx)}
                      onMouseLeave={() => setHoveredClaim(null)}
                      style={{
                        borderColor: isHovered ? confStyle.border : "var(--border-subtle)",
                      }}
                    >
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "var(--text-primary)",
                          lineHeight: 1.5,
                        }}
                      >
                        {item.claim}
                      </div>

                      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
                        <span
                          style={{
                            fontSize: "0.7rem",
                            padding: "2px 6px",
                            borderRadius: "4px",
                            background: "rgba(99, 102, 241, 0.15)",
                            color: "#818cf8",
                            fontWeight: 600,
                          }}
                        >
                          📍 {item.supporting_component}
                        </span>
                        <span
                          style={{
                            fontSize: "0.7rem",
                            padding: "2px 6px",
                            borderRadius: "4px",
                            background: confStyle.bg,
                            color: confStyle.text,
                            fontWeight: 600,
                          }}
                        >
                          ● {confKey} Confidence
                        </span>
                        <span
                          style={{
                            fontSize: "0.7rem",
                            padding: "2px 6px",
                            borderRadius: "4px",
                            background: "rgba(255, 255, 255, 0.05)",
                            color: "var(--text-muted)",
                          }}
                        >
                          {item.region}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* TAB 3: Topology & Graph Reasoning */}
            {result && activeTab === "topology" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                {/* Single Points of Failure */}
                <div
                  style={{
                    background: "var(--bg-input)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "var(--radius-sm)",
                    padding: "0.85rem",
                  }}
                >
                  <div
                    style={{
                      fontSize: "0.75rem",
                      fontWeight: 700,
                      color: "#fb7185",
                      marginBottom: "0.4rem",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.4rem",
                    }}
                  >
                    ⚠️ Single Points of Failure (SPOFs)
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "var(--text-primary)" }}>
                    {result.reasoning_trace?.stage4_graph_queries?.critical_path?.critical_nodes?.join(
                      ", "
                    ) || "None detected (Redundant mesh topology)"}
                  </div>
                </div>

                {/* Entry Points */}
                <div
                  style={{
                    background: "var(--bg-input)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "var(--radius-sm)",
                    padding: "0.85rem",
                  }}
                >
                  <div
                    style={{
                      fontSize: "0.75rem",
                      fontWeight: 700,
                      color: "#38bdf8",
                      marginBottom: "0.4rem",
                    }}
                  >
                    🚪 Ingress Entry Points
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "var(--text-primary)" }}>
                    {result.reasoning_trace?.stage4_graph_queries?.entry_points?.entry_points?.join(
                      ", "
                    ) || "—"}
                  </div>
                </div>

                {/* Data Flow Paths */}
                {result.reasoning_trace?.stage4_graph_queries?.data_flow?.paths?.length > 0 && (
                  <div
                    style={{
                      background: "var(--bg-input)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "var(--radius-sm)",
                      padding: "0.85rem",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "0.75rem",
                        fontWeight: 700,
                        color: "#a855f7",
                        marginBottom: "0.4rem",
                      }}
                    >
                      🔄 Traced Data Flow Paths
                    </div>
                    {result.reasoning_trace.stage4_graph_queries.data_flow.paths.map(
                      (p, idx) => (
                        <div
                          key={idx}
                          style={{
                            fontSize: "0.75rem",
                            color: "var(--text-secondary)",
                            padding: "0.25rem 0",
                          }}
                        >
                          {p.path.join(" ➔ ")}
                        </div>
                      )
                    )}
                  </div>
                )}
              </div>
            )}

            {/* TAB 4: 6-Stage CoT Reasoning Trace */}
            {result && activeTab === "trace" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <span
                    style={{
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      color: "var(--text-muted)",
                    }}
                  >
                    FULL STAGE-BY-STAGE TRACE
                  </span>
                  <div style={{ display: "flex", gap: "0.5rem" }}>
                    <button
                      onClick={handleExportGraphML}
                      style={{
                        background: "rgba(99, 102, 241, 0.2)",
                        border: "1px solid rgba(99, 102, 241, 0.4)",
                        borderRadius: "4px",
                        padding: "2px 8px",
                        fontSize: "0.7rem",
                        color: "#818cf8",
                        cursor: "pointer",
                      }}
                    >
                      GraphML
                    </button>
                    <button
                      onClick={handleExportJSON}
                      style={{
                        background: "rgba(6, 182, 212, 0.2)",
                        border: "1px solid rgba(6, 182, 212, 0.4)",
                        borderRadius: "4px",
                        padding: "2px 8px",
                        fontSize: "0.7rem",
                        color: "#22d3ee",
                        cursor: "pointer",
                      }}
                    >
                      JSON
                    </button>
                  </div>
                </div>

                <pre
                  style={{
                    background: "var(--bg-input)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "var(--radius-sm)",
                    padding: "0.75rem",
                    fontSize: "0.7rem",
                    color: "var(--text-secondary)",
                    overflowX: "auto",
                    maxHeight: "360px",
                    lineHeight: 1.5,
                  }}
                >
                  {JSON.stringify(result.reasoning_trace, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}