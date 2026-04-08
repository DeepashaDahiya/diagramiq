import { useState, useRef } from "react";
import axios from "axios";

const CONFIDENCE_COLORS = {
  high:   "rgba(34, 197, 94, 0.35)",
  medium: "rgba(251, 191, 36, 0.35)",
  low:    "rgba(239, 68, 68, 0.25)",
};

const CONFIDENCE_BORDER = {
  high:   "#22c55e",
  medium: "#fbbf24",
  low:    "#ef4444",
};

export default function App() {
  const [image, setImage]               = useState(null);
  const [imageURL, setImageURL]         = useState(null);
  const [query, setQuery]               = useState("");
  const [result, setResult]             = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [activeTab, setActiveTab]       = useState("answer");
  const [hoveredClaim, setHoveredClaim] = useState(null);
  const imgRef = useRef(null);

  function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setImage(file);
    setImageURL(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }

  async function handleAnalyze() {
    if (!image || !query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", image);
    formData.append("query", query);

    try {
      const res = await axios.post("http://localhost:5000/analyze", formData);
      setResult(res.data);
      setActiveTab("answer");
    } catch (err) {
      setError("Analysis failed. Make sure the backend is running on port 5000.");
    } finally {
      setLoading(false);
    }
  }

  function getScaledBbox(bbox) {
    if (!imgRef.current || !result) return null;
    const { naturalWidth, naturalHeight, width, height } = imgRef.current;
    const scaleX = width  / naturalWidth;
    const scaleY = height / naturalHeight;
    return {
      left:   bbox.x1 * scaleX,
      top:    bbox.y1 * scaleY,
      width:  (bbox.x2 - bbox.x1) * scaleX,
      height: (bbox.y2 - bbox.y1) * scaleY,
    };
  }

  return (
    <div style={{ minHeight: "100vh", background: "#0f172a", color: "#f1f5f9",
                  fontFamily: "system-ui, sans-serif" }}>

      {/* Header */}
      <div style={{ background: "#1e293b", borderBottom: "1px solid #334155",
                    padding: "1rem 2rem", display: "flex",
                    alignItems: "center", gap: "0.75rem" }}>
        <span style={{ fontSize: "1.5rem" }}>🧠</span>
        <span style={{ fontSize: "1.25rem", fontWeight: 600 }}>DiagramIQ</span>
        <span style={{ fontSize: "0.8rem", color: "#64748b", marginLeft: "0.5rem" }}>
          Multimodal Diagram Reasoning
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr 380px",
                    gap: "1rem", padding: "1rem",
                    minHeight: "calc(100vh - 60px)" }}>

        {/* LEFT: Upload + Query */}
        <div style={{ background: "#1e293b", borderRadius: "12px",
                      padding: "1.25rem", display: "flex",
                      flexDirection: "column", gap: "1rem" }}>

          <div style={{ fontWeight: 600, fontSize: "0.9rem", color: "#94a3b8" }}>
            UPLOAD DIAGRAM
          </div>

          <label style={{ border: "2px dashed #334155", borderRadius: "8px",
                          padding: "1.5rem", textAlign: "center",
                          cursor: "pointer", color: "#64748b",
                          fontSize: "0.85rem" }}>
            <input type="file" accept="image/*" onChange={handleImageUpload}
                   style={{ display: "none" }} />
            {image ? (
              <span style={{ color: "#22c55e" }}>✓ {image.name}</span>
            ) : (
              <span>Click to upload<br />PNG, JPG supported</span>
            )}
          </label>

          <div style={{ fontWeight: 600, fontSize: "0.9rem", color: "#94a3b8" }}>
            YOUR QUESTION
          </div>

          <textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="e.g. What happens if the Auth Service fails?"
            rows={4}
            style={{ background: "#0f172a", border: "1px solid #334155",
                     borderRadius: "8px", padding: "0.75rem",
                     color: "#f1f5f9", fontSize: "0.875rem",
                     resize: "vertical", outline: "none" }}
          />

          <div style={{ fontSize: "0.75rem", color: "#64748b" }}>
            Suggested queries:
          </div>

          {[
            "What are the entry points to the system?",
            "Which components are single points of failure?",
            "Trace the data flow to the database",
            "What happens if the Auth Service fails?",
          ].map(q => (
            <button key={q} onClick={() => setQuery(q)}
              style={{ background: "#0f172a", border: "1px solid #334155",
                       borderRadius: "6px", padding: "0.4rem 0.75rem",
                       color: "#94a3b8", fontSize: "0.75rem",
                       cursor: "pointer", textAlign: "left" }}>
              {q}
            </button>
          ))}

          <button onClick={handleAnalyze}
            disabled={!image || !query.trim() || loading}
            style={{ background: loading ? "#334155" : "#6366f1",
                     border: "none", borderRadius: "8px",
                     padding: "0.75rem", color: "#fff",
                     fontWeight: 600, cursor: "pointer",
                     fontSize: "0.9rem", marginTop: "auto",
                     opacity: (!image || !query.trim()) ? 0.5 : 1 }}>
            {loading ? "Analyzing... (~30s)" : "Analyze Diagram"}
          </button>

          {error && (
            <div style={{ color: "#ef4444", fontSize: "0.8rem",
                          background: "#1e1010", borderRadius: "6px",
                          padding: "0.5rem" }}>
              {error}
            </div>
          )}
        </div>

        {/* CENTER: Diagram with bounding boxes */}
        <div style={{ background: "#1e293b", borderRadius: "12px",
                      padding: "1.25rem" }}>

          <div style={{ fontWeight: 600, fontSize: "0.9rem",
                        color: "#94a3b8", marginBottom: "1rem" }}>
            DIAGRAM
            {result && (
              <span style={{ color: "#6366f1", marginLeft: "0.5rem" }}>
                · {result.reasoning_trace?.stage3_classification?.diagram_type}
                &nbsp;·&nbsp;
                {result.reasoning_trace?.stage1_components} components
              </span>
            )}
          </div>

          {imageURL ? (
            <div style={{ position: "relative", display: "inline-block",
                          width: "100%" }}>
              <img ref={imgRef} src={imageURL} alt="diagram"
                   style={{ width: "100%", borderRadius: "8px",
                             display: "block" }} />

              {result && result.claims.map((claim, i) => {
                const scaled = getScaledBbox(claim.bbox);
                if (!scaled) return null;
                const isHovered = hoveredClaim === i;
                return (
                  <div key={i} style={{
                    position: "absolute",
                    left:     scaled.left,
                    top:      scaled.top,
                    width:    scaled.width,
                    height:   scaled.height,
                    background: isHovered
                      ? CONFIDENCE_COLORS[claim.confidence]
                      : "transparent",
                    border: `2px solid ${CONFIDENCE_BORDER[claim.confidence]}`,
                    borderRadius: "4px",
                    transition: "background 0.2s",
                    pointerEvents: "none",
                    boxSizing: "border-box",
                  }} />
                );
              })}
            </div>
          ) : (
            <div style={{ height: "400px", display: "flex",
                          alignItems: "center", justifyContent: "center",
                          color: "#334155", fontSize: "0.9rem",
                          border: "2px dashed #1e293b",
                          borderRadius: "8px" }}>
              Upload a diagram to get started
            </div>
          )}
        </div>

        {/* RIGHT: Answer + Reasoning Trace */}
        <div style={{ background: "#1e293b", borderRadius: "12px",
                      padding: "1.25rem", display: "flex",
                      flexDirection: "column", gap: "1rem",
                      overflowY: "auto", maxHeight: "calc(100vh - 80px)" }}>

          {/* Tabs */}
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {["answer", "claims", "trace"].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                style={{ flex: 1, padding: "0.4rem",
                         background: activeTab === tab ? "#6366f1" : "#0f172a",
                         border: "1px solid #334155", borderRadius: "6px",
                         color: activeTab === tab ? "#fff" : "#64748b",
                         fontSize: "0.75rem", cursor: "pointer",
                         textTransform: "capitalize" }}>
                {tab}
              </button>
            ))}
          </div>

          {!result && !loading && (
            <div style={{ color: "#334155", fontSize: "0.85rem",
                          textAlign: "center", marginTop: "3rem" }}>
              Results will appear here after analysis
            </div>
          )}

          {loading && (
            <div style={{ textAlign: "center", marginTop: "3rem" }}>
              <div style={{ color: "#6366f1", fontSize: "0.9rem",
                            marginBottom: "0.5rem" }}>
                Running 6-stage pipeline...
              </div>
              <div style={{ color: "#64748b", fontSize: "0.75rem",
                            lineHeight: 1.6 }}>
                Stage 1: Identifying components<br />
                Stage 2: Extracting relationships<br />
                Stage 3: Classifying diagram<br />
                Stage 4: Graph traversal reasoning<br />
                Stage 5: Grounding answer<br />
                Stage 6: Building structured output
              </div>
            </div>
          )}

          {/* Answer tab */}
          {result && activeTab === "answer" && (
            <div style={{ display: "flex", flexDirection: "column",
                          gap: "1rem" }}>

              <div style={{ background: "#0f172a", borderRadius: "8px",
                            padding: "1rem", fontSize: "0.875rem",
                            lineHeight: 1.7, color: "#e2e8f0" }}>
                {result.answer}
              </div>

              {/* Graph query summary */}
              <div style={{ fontSize: "0.75rem", color: "#64748b",
                            fontWeight: 600 }}>
                GRAPH ANALYSIS
              </div>

              <div style={{ background: "#0f172a", borderRadius: "8px",
                            padding: "0.75rem", fontSize: "0.75rem",
                            color: "#94a3b8", lineHeight: 1.6 }}>
                <div>Entry points: <span style={{ color: "#e2e8f0" }}>
                  {result.reasoning_trace?.stage4_graph_queries
                    ?.entry_points?.entry_points?.join(", ") || "—"}
                </span></div>
                <div style={{ marginTop: "0.4rem" }}>
                  Single points of failure: <span style={{ color: "#ef4444" }}>
                    {result.reasoning_trace?.stage4_graph_queries
                      ?.critical_path?.critical_nodes?.join(", ") || "None"}
                  </span>
                </div>
              </div>

              {result.follow_up_questions?.length > 0 && (
                <>
                  <div style={{ fontSize: "0.75rem", color: "#64748b",
                                fontWeight: 600 }}>
                    FOLLOW-UP QUESTIONS
                  </div>
                  {result.follow_up_questions.map((q, i) => (
                    <button key={i} onClick={() => setQuery(q)}
                      style={{ background: "#0f172a",
                               border: "1px solid #334155",
                               borderRadius: "6px",
                               padding: "0.5rem 0.75rem",
                               color: "#94a3b8", fontSize: "0.75rem",
                               cursor: "pointer", textAlign: "left" }}>
                      → {q}
                    </button>
                  ))}
                </>
              )}
            </div>
          )}

          {/* Claims tab */}
          {result && activeTab === "claims" && (
            <div style={{ display: "flex", flexDirection: "column",
                          gap: "0.75rem" }}>
              <div style={{ fontSize: "0.75rem", color: "#64748b" }}>
                Hover a claim to highlight its region on the diagram
              </div>
              {result.claims.map((claim, i) => (
                <div key={i}
                  onMouseEnter={() => setHoveredClaim(i)}
                  onMouseLeave={() => setHoveredClaim(null)}
                  style={{ background: "#0f172a", borderRadius: "8px",
                           padding: "0.75rem", cursor: "pointer",
                           border: `1px solid ${hoveredClaim === i
                             ? CONFIDENCE_BORDER[claim.confidence]
                             : "#334155"}`,
                           transition: "border-color 0.2s" }}>
                  <div style={{ fontSize: "0.8rem", color: "#e2e8f0",
                                marginBottom: "0.5rem", lineHeight: 1.5 }}>
                    {claim.claim}
                  </div>
                  <div style={{ display: "flex", gap: "0.4rem",
                                flexWrap: "wrap" }}>
                    <span style={{ fontSize: "0.7rem", padding: "2px 6px",
                                   borderRadius: "4px",
                                   background: "#1e293b", color: "#64748b" }}>
                      📍 {claim.supported_by}
                    </span>
                    <span style={{ fontSize: "0.7rem", padding: "2px 6px",
                                   borderRadius: "4px",
                                   background: CONFIDENCE_COLORS[claim.confidence],
                                   color: CONFIDENCE_BORDER[claim.confidence] }}>
                      {claim.confidence} confidence
                    </span>
                    <span style={{ fontSize: "0.7rem", padding: "2px 6px",
                                   borderRadius: "4px",
                                   background: "#1e293b", color: "#64748b" }}>
                      {claim.region}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Trace tab */}
          {result && activeTab === "trace" && (
            <div>
              <div style={{ fontSize: "0.75rem", color: "#94a3b8",
                            fontWeight: 600, marginBottom: "0.75rem" }}>
                FULL REASONING TRACE
              </div>
              <pre style={{ background: "#0f172a", borderRadius: "8px",
                            padding: "1rem", overflowX: "auto",
                            whiteSpace: "pre-wrap", lineHeight: 1.5,
                            fontSize: "0.7rem", color: "#64748b" }}>
                {JSON.stringify(result.reasoning_trace, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}