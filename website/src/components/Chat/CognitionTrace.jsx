import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

/**
 * Collapsible cognition-trace inspector.
 *
 * Renders only when `trace` is a non-null object with at least one field.
 * Collapsed by default; messages without a trace render exactly as before.
 *
 * Expected trace shape (all fields optional):
 *   memories:   [{ text: string, score: number }]
 *   affect:     number[]
 *   affect_adj: number
 *   sampling:   Record<string, number>
 *   tools:      [{ name, input, output }]
 */
export default function CognitionTrace({ trace }) {
  const [open, setOpen] = useState(false);

  if (!trace || typeof trace !== "object") return null;
  if (!Object.keys(trace).length) return null;

  // Check if trace has any meaningful content to display
  const hasContent = () => {
    // Non-empty memories array
    if (Array.isArray(trace.memories) && trace.memories.length > 0) return true;
    // Non-empty affect array
    if (Array.isArray(trace.affect) && trace.affect.length > 0) return true;
    // affect_adj with a value
    if (trace.affect_adj != null) return true;
    // Non-empty sampling object
    if (trace.sampling && typeof trace.sampling === "object" && Object.keys(trace.sampling).length > 0) return true;
    // Non-empty tools array
    if (Array.isArray(trace.tools) && trace.tools.length > 0) return true;
    return false;
  };

  if (!hasContent()) return null;

  return (
    <div className="cognition-trace">
      <button
        className="cognition-trace-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="cognition-trace-body"
      >
        {open ? (
          <ChevronDown size={13} aria-hidden="true" />
        ) : (
          <ChevronRight size={13} aria-hidden="true" />
        )}
        <span>Cognition trace</span>
      </button>

      {open && (
        <div id="cognition-trace-body" className="cognition-trace-body">
          {/* Recalled memories */}
          {Array.isArray(trace.memories) && trace.memories.length > 0 && (
            <section className="trace-section">
              <h4 className="trace-section-title">Recalled memories</h4>
              <ul className="trace-memory-list">
                {trace.memories.map((m, i) => (
                  <li key={i} className="trace-memory">
                    <span className="trace-memory-score">{m.score.toFixed(3)}</span>
                    <span className="trace-memory-text">{m.text}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Affective vector */}
          {Array.isArray(trace.affect) && trace.affect.length > 0 && (
            <section className="trace-section">
              <h4 className="trace-section-title">Affective vector</h4>
              <div className="trace-affect">
                {trace.affect.map((v, i) => (
                  <span key={i} className="trace-affect-val">
                    {v.toFixed(3)}
                  </span>
                ))}
                {trace.affect_adj != null && (
                  <span className="trace-affect-adj">
                    → sampling Δ {trace.affect_adj > 0 ? "+" : ""}
                    {trace.affect_adj.toFixed(3)}
                  </span>
                )}
              </div>
            </section>
          )}

          {/* Sampling adjustments */}
          {trace.sampling && Object.keys(trace.sampling).length > 0 && (
            <section className="trace-section">
              <h4 className="trace-section-title">Sampling</h4>
              <dl className="trace-kv">
                {Object.entries(trace.sampling).map(([k, v]) => (
                  <div key={k} className="trace-kv-row">
                    <dt className="trace-key">{k}</dt>
                    <dd className="trace-val">{Number(v).toFixed(4)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          {/* Tool steps */}
          {Array.isArray(trace.tools) && trace.tools.length > 0 && (
            <section className="trace-section">
              <h4 className="trace-section-title">Tool steps</h4>
              <ol className="trace-tool-list">
                {trace.tools.map((t, i) => (
                  <li key={i} className="trace-tool">
                    <span className="trace-tool-name">{t.name}</span>
                    {t.input != null && (
                      <pre className="trace-tool-io">{JSON.stringify(t.input, null, 2)}</pre>
                    )}
                    {t.output != null && (
                      <pre className="trace-tool-io trace-tool-output">
                        {JSON.stringify(t.output, null, 2)}
                      </pre>
                    )}
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
