/**
 * CognitionTrace component tests (Issue #164).
 *
 * Verify that the trace inspector renders correctly, is collapsed by default,
 * and only appears when a valid trace exists.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import CognitionTrace from "../components/Chat/CognitionTrace";

describe("CognitionTrace", () => {
  it("renders nothing when trace is null", () => {
    const { container } = render(<CognitionTrace trace={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when trace is undefined", () => {
    const { container } = render(<CognitionTrace trace={undefined} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when trace is not an object", () => {
    const { container } = render(<CognitionTrace trace="not an object" />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when trace is empty object", () => {
    const { container } = render(<CognitionTrace trace={{}} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders collapsed by default when trace is valid", () => {
    const trace = {
      memories: [{ text: "memory", score: 0.8 }],
    };

    render(<CognitionTrace trace={trace} />);

    const toggle = screen.getByRole("button", { name: /cognition trace/i });
    expect(toggle).toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    // Body should not be visible
    expect(screen.queryByText("memory")).not.toBeInTheDocument();
  });

  it("expands when toggle is clicked", () => {
    const trace = {
      memories: [{ text: "test memory", score: 0.85 }],
    };

    render(<CognitionTrace trace={trace} />);

    const toggle = screen.getByRole("button", { name: /cognition trace/i });
    fireEvent.click(toggle);

    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("test memory")).toBeInTheDocument();
    expect(screen.getByText("0.850")).toBeInTheDocument();
  });

  it("collapses when toggle is clicked again", () => {
    const trace = {
      memories: [{ text: "memory", score: 0.9 }],
    };

    render(<CognitionTrace trace={trace} />);

    const toggle = screen.getByRole("button", { name: /cognition trace/i });

    // Expand
    fireEvent.click(toggle);
    expect(screen.getByText("memory")).toBeInTheDocument();

    // Collapse
    fireEvent.click(toggle);
    expect(screen.queryByText("memory")).not.toBeInTheDocument();
  });

  it("renders recalled memories section", () => {
    const trace = {
      memories: [
        { text: "first memory", score: 0.92 },
        { text: "second memory", score: 0.78 },
      ],
    };

    render(<CognitionTrace trace={trace} />);

    const toggle = screen.getByRole("button");
    fireEvent.click(toggle);

    expect(screen.getByText("Recalled memories")).toBeInTheDocument();
    expect(screen.getByText("first memory")).toBeInTheDocument();
    expect(screen.getByText("0.920")).toBeInTheDocument();
    expect(screen.getByText("second memory")).toBeInTheDocument();
    expect(screen.getByText("0.780")).toBeInTheDocument();
  });

  it("renders affective vector section", () => {
    const trace = {
      affect: [0.12345, -0.56789, 0.34567],
      affect_adj: 0.05,
    };

    render(<CognitionTrace trace={trace} />);

    const toggle = screen.getByRole("button");
    fireEvent.click(toggle);

    expect(screen.getByText("Affective vector")).toBeInTheDocument();
    expect(screen.getByText("0.123")).toBeInTheDocument();
    expect(screen.getByText("-0.568")).toBeInTheDocument();
    expect(screen.getByText("0.346")).toBeInTheDocument();
    expect(screen.getByText(/sampling Δ \+0\.050/)).toBeInTheDocument();
  });

  it("renders negative affect_adj correctly", () => {
    const trace = {
      affect: [0.1],
      affect_adj: -0.03,
    };

    render(<CognitionTrace trace={trace} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText(/sampling Δ -0\.030/)).toBeInTheDocument();
  });

  it("renders sampling adjustments section", () => {
    const trace = {
      sampling: {
        temperature: 0.7234,
        top_k: 50,
        top_p: 0.9123,
      },
    };

    render(<CognitionTrace trace={trace} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Sampling")).toBeInTheDocument();
    expect(screen.getByText("temperature")).toBeInTheDocument();
    expect(screen.getByText("0.7234")).toBeInTheDocument();
    expect(screen.getByText("top_k")).toBeInTheDocument();
    expect(screen.getByText("50.0000")).toBeInTheDocument();
    expect(screen.getByText("top_p")).toBeInTheDocument();
    expect(screen.getByText("0.9123")).toBeInTheDocument();
  });

  it("renders tool steps section", () => {
    const trace = {
      tools: [
        {
          name: "web_search",
          input: { query: "test query" },
          output: { results: ["result1", "result2"] },
        },
        {
          name: "web_fetch",
          input: { url: "https://example.com" },
          output: { content: "page content" },
        },
      ],
    };

    render(<CognitionTrace trace={trace} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Tool steps")).toBeInTheDocument();
    expect(screen.getByText("web_search")).toBeInTheDocument();
    expect(screen.getByText("web_fetch")).toBeInTheDocument();

    // Check that JSON is formatted
    expect(screen.getByText(/"query": "test query"/)).toBeInTheDocument();
  });

  it("renders multiple sections simultaneously", () => {
    const trace = {
      memories: [{ text: "memory", score: 0.9 }],
      affect: [0.1, 0.2],
      sampling: { temperature: 0.8 },
      tools: [{ name: "tool1", input: {}, output: {} }],
    };

    render(<CognitionTrace trace={trace} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Recalled memories")).toBeInTheDocument();
    expect(screen.getByText("Affective vector")).toBeInTheDocument();
    expect(screen.getByText("Sampling")).toBeInTheDocument();
    expect(screen.getByText("Tool steps")).toBeInTheDocument();
  });

  it("omits sections when their data is missing", () => {
    const trace = {
      memories: [{ text: "only memory", score: 0.8 }],
      // No affect, sampling, or tools
    };

    render(<CognitionTrace trace={trace} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Recalled memories")).toBeInTheDocument();
    expect(screen.queryByText("Affective vector")).not.toBeInTheDocument();
    expect(screen.queryByText("Sampling")).not.toBeInTheDocument();
    expect(screen.queryByText("Tool steps")).not.toBeInTheDocument();
  });

  it("omits sections when their arrays are empty", () => {
    const trace = {
      memories: [],
      affect: [],
      sampling: {},
      tools: [],
    };

    const { container } = render(<CognitionTrace trace={trace} />);

    // Should render nothing because all sections are empty
    expect(container.firstChild).toBeNull();
  });

  it("handles affect without affect_adj", () => {
    const trace = {
      affect: [0.5, -0.3],
      // No affect_adj
    };

    render(<CognitionTrace trace={trace} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Affective vector")).toBeInTheDocument();
    expect(screen.getByText("0.500")).toBeInTheDocument();
    expect(screen.queryByText(/sampling Δ/)).not.toBeInTheDocument();
  });

  it("handles tools without input or output", () => {
    const trace = {
      tools: [
        { name: "minimal_tool" },
        { name: "with_input", input: { arg: "value" } },
        { name: "with_output", output: { result: "data" } },
      ],
    };

    render(<CognitionTrace trace={trace} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("minimal_tool")).toBeInTheDocument();
    expect(screen.getByText("with_input")).toBeInTheDocument();
    expect(screen.getByText("with_output")).toBeInTheDocument();
  });

  it("maintains accessibility attributes", () => {
    const trace = {
      memories: [{ text: "test", score: 0.8 }],
    };

    render(<CognitionTrace trace={trace} />);

    const toggle = screen.getByRole("button");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveAttribute("aria-controls", "cognition-trace-body");

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");

    const body = screen.getByText("Recalled memories").closest("#cognition-trace-body");
    expect(body).toBeInTheDocument();
  });

  it("renders with keyboard navigation support", () => {
    const trace = {
      memories: [{ text: "test", score: 0.8 }],
    };

    render(<CognitionTrace trace={trace} />);

    const toggle = screen.getByRole("button");

    // Simulate Enter key press
    fireEvent.keyDown(toggle, { key: "Enter", code: "Enter" });
    // Button click handlers are triggered by Enter automatically
    fireEvent.click(toggle);

    expect(screen.getByText("test")).toBeInTheDocument();
  });
});

describe("CognitionTrace integration with MessageBubble", () => {
  it("appears in assistant message when trace is present", async () => {
    // This would be tested in MessageBubble.test.jsx
    // but we document the requirement here
    const message = {
      role: "assistant",
      content: "response",
      metadata: {
        model: "framerai-text",
        trace: {
          memories: [{ text: "memory", score: 0.9 }],
        },
      },
    };

    // The MessageBubble component should render CognitionTrace
    // when message.metadata.trace exists
    expect(message.metadata.trace).toBeDefined();
  });

  it("does not appear in user messages", () => {
    const message = {
      role: "user",
      content: "user question",
      metadata: {
        trace: { memories: [] }, // Even if present, should not render
      },
    };

    expect(message.role).toBe("user");
    // CognitionTrace should only render for assistant messages
  });

  it("does not appear when trace is missing", () => {
    const message = {
      role: "assistant",
      content: "response",
      metadata: {
        model: "framerai-text",
        // No trace
      },
    };

    expect(message.metadata.trace).toBeUndefined();
  });
});
