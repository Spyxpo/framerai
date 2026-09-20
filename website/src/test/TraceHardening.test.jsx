/**
 * Regression tests for trace-rendering hardening (Issue #XXX).
 *
 * Covers three protections:
 *
 * A. sanitizeTrace() in storage.js
 *    — normalizes non-numeric scores, affect values, and affect_adj from
 *      stored metadata so CognitionTrace's .toFixed() calls cannot crash.
 *
 * B. sanitizeMessage() revalidates trace inside metadata
 *    — ensures the storage boundary applies sanitizeTrace() to any trace
 *      that survived earlier validation with malformed numeric fields.
 *
 * C. CognitionTraceErrorBoundary
 *    — catches descendant render errors so a malformed trace cannot crash
 *      the surrounding message tree.
 *
 * Discriminating tests (fail on old code):
 *   A: "sanitizeTrace normalizes null score to 0"
 *   B: "sanitizeMessage coerces null score in trace metadata"
 *   C: "CognitionTraceErrorBoundary catches a render error and renders null"
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render } from "@testing-library/react";
import { sanitizeTrace, sanitizeMessage } from "../utils/storage";
import { CognitionTraceErrorBoundary } from "../components/Chat/CognitionTrace";

// ---------------------------------------------------------------------------
// A. sanitizeTrace()
// ---------------------------------------------------------------------------

describe("sanitizeTrace — memories", () => {
  // DISCRIMINATING: old code = sanitizeTrace doesn't exist (undefined import).
  // New code: score: null → score: 0
  it("normalizes null score to 0", () => {
    const result = sanitizeTrace({
      memories: [{ text: "memory", score: null }],
    });
    expect(result).not.toBeNull();
    expect(result.memories[0].score).toBe(0);
    expect(typeof result.memories[0].score).toBe("number");
  });

  it("normalizes non-numeric string score to 0", () => {
    const result = sanitizeTrace({
      memories: [{ text: "memory", score: "notanumber" }],
    });
    expect(result.memories[0].score).toBe(0);
  });

  it("preserves valid numeric score", () => {
    const result = sanitizeTrace({
      memories: [{ text: "memory", score: 0.85 }],
    });
    expect(result.memories[0].score).toBe(0.85);
  });

  it("normalizes numeric string score to number", () => {
    const result = sanitizeTrace({
      memories: [{ text: "memory", score: "0.9" }],
    });
    expect(result.memories[0].score).toBe(0.9);
    expect(typeof result.memories[0].score).toBe("number");
  });
});

describe("sanitizeTrace — affect_adj", () => {
  it("preserves valid numeric affect_adj", () => {
    const result = sanitizeTrace({ affect: [0.1], affect_adj: 0.05 });
    expect(result.affect_adj).toBe(0.05);
  });

  it("normalizes numeric-string affect_adj", () => {
    const result = sanitizeTrace({ affect: [0.1], affect_adj: "0.05" });
    expect(result.affect_adj).toBe(0.05);
    expect(typeof result.affect_adj).toBe("number");
  });

  it("drops non-numeric affect_adj", () => {
    const result = sanitizeTrace({ affect: [0.1], affect_adj: "notanumber" });
    expect(result).not.toBeNull();
    expect(result.affect_adj).toBeUndefined();
  });
});

describe("sanitizeTrace — tools", () => {
  it("preserves valid tool entries", () => {
    const result = sanitizeTrace({
      tools: [
        { name: "tool_a", input: { x: 1 }, output: { y: 2 } },
        { name: "tool_b" },
      ],
    });
    expect(result.tools).toHaveLength(2);
    expect(result.tools[0].name).toBe("tool_a");
    expect(result.tools[0].input).toEqual({ x: 1 });
    expect(result.tools[1].name).toBe("tool_b");
  });

  it("drops tool entries with non-string name", () => {
    const result = sanitizeTrace({
      tools: [
        { name: 42 },
        { name: "valid" },
      ],
    });
    expect(result.tools).toHaveLength(1);
    expect(result.tools[0].name).toBe("valid");
  });

  it("drops non-JSON-safe tool input", () => {
    const circular = {};
    circular.self = circular;
    const result = sanitizeTrace({
      tools: [{ name: "t", input: circular }],
    });
    expect(result.tools[0].name).toBe("t");
    expect(result.tools[0].input).toBeUndefined();
  });
});

describe("sanitizeTrace — preserves originals", () => {
  it("does not mutate the original trace object", () => {
    const original = {
      memories: [{ text: "m", score: null }],
      tools: [{ name: "t", input: { x: 1 } }],
    };
    const toolRef = original.tools[0];
    sanitizeTrace(original);
    expect(original.memories[0].score).toBeNull();
    expect(original.tools[0]).toBe(toolRef);
  });
});

// ---------------------------------------------------------------------------
// B. sanitizeMessage() — trace in metadata
// ---------------------------------------------------------------------------

describe("sanitizeMessage — trace metadata revalidation", () => {
  // DISCRIMINATING: old code passes metadata through verbatim, so score stays null.
  // New code: sanitizeTrace() normalizes score: null → 0.
  it("coerces null score in trace metadata", () => {
    const msg = {
      id: "m1",
      role: "assistant",
      content: "hello",
      metadata: {
        model: "framerai-text",
        trace: {
          memories: [{ text: "recall", score: null }],
        },
      },
    };

    const result = sanitizeMessage(msg);

    expect(result).not.toBeNull();
    expect(result.metadata.trace.memories[0].score).toBe(0);
  });

  it("preserves other metadata fields alongside trace sanitization", () => {
    const msg = {
      id: "m1",
      role: "assistant",
      content: "hello",
      metadata: {
        model: "framerai-text",
        url: "/uploads/img.png",
        trace: {
          memories: [{ text: "recall", score: 0.7 }],
        },
      },
    };

    const result = sanitizeMessage(msg);

    expect(result.metadata.model).toBe("framerai-text");
    expect(result.metadata.url).toBe("/uploads/img.png");
    expect(result.metadata.trace.memories[0].score).toBe(0.7);
  });

  it("removes invalid trace from metadata", () => {
    const msg = {
      id: "m1",
      role: "assistant",
      content: "hello",
      metadata: {
        model: "framerai-text",
        trace: "not an object",
      },
    };

    const result = sanitizeMessage(msg);

    expect(result.metadata.model).toBe("framerai-text");
    expect(result.metadata.trace).toBeUndefined();
  });

  it("passes through message without trace unchanged", () => {
    const msg = {
      id: "m1",
      role: "assistant",
      content: "hello",
      metadata: { model: "framerai-text" },
    };

    const result = sanitizeMessage(msg);

    expect(result.metadata.model).toBe("framerai-text");
    expect(result.metadata.trace).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// C. CognitionTraceErrorBoundary
// ---------------------------------------------------------------------------

describe("CognitionTraceErrorBoundary", () => {
  // Suppress React's expected console.error output for error boundary tests.
  // React logs caught errors to console.error even when a boundary handles them.
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  // DISCRIMINATING: old code = class doesn't exist (undefined import) → render throws.
  // New code: boundary catches the error and renders null.
  it("catches a render error from a descendant and renders null", () => {
    const Thrower = () => {
      throw new Error("deliberate test render error");
    };

    const { container } = render(
      <CognitionTraceErrorBoundary>
        <Thrower />
      </CognitionTraceErrorBoundary>
    );

    expect(container.firstChild).toBeNull();
  });

  it("renders normal children when no error occurs", () => {
    const { container } = render(
      <CognitionTraceErrorBoundary>
        <div data-testid="safe-child">OK</div>
      </CognitionTraceErrorBoundary>
    );

    expect(container.querySelector("[data-testid='safe-child']")).not.toBeNull();
    expect(container.textContent).toBe("OK");
  });

});
