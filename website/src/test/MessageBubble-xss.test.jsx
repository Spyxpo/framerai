/**
 * Regression tests for Issue #240: Message content must not execute as HTML/JavaScript.
 *
 * THE BUG: renderContent() used dangerouslySetInnerHTML after regex-based
 * Markdown replacement without escaping the input first. Any HTML in a message
 * — including event-handler attributes — was injected directly into the DOM.
 *
 * Example payload:  <img src="broken" onerror="window.__xss=1">
 *
 * FIX: Replace dangerouslySetInnerHTML with <ReactMarkdown>, which converts
 * Markdown to React elements and escapes raw HTML by default. No unsafe HTML
 * ever reaches the DOM.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import MessageBubble from "../components/Chat/MessageBubble";

function makeMessage(overrides = {}) {
  return {
    id: "msg-xss",
    role: "assistant",
    type: "text",
    content: "hello",
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

describe("MessageBubble — XSS / HTML-injection safety (Issue #240)", () => {
  beforeEach(() => {
    // Clear any side-effects from a prior test
    delete window.__xss;
  });

  // ── 1. img onerror (the canonical example from the issue) ─────────────────
  it("REGRESSION #240: onerror handler in img tag must not execute", () => {
    const payload = '<img src="broken" onerror="window.__xss=1">';
    const { container } = render(
      <MessageBubble message={makeMessage({ content: payload })} />
    );

    // The global must not have been set (handler didn't run)
    expect(window.__xss).toBeUndefined();

    // No actual <img> element should exist in the message body
    const imgs = container.querySelectorAll(".text-content img");
    imgs.forEach((img) => {
      expect(img.getAttribute("onerror")).toBeNull();
    });
  });

  // ── 2. script tag must not execute ────────────────────────────────────────
  it("REGRESSION #240: inline <script> tag must not execute", () => {
    const payload = '<script>window.__xss="script"</script>';
    render(<MessageBubble message={makeMessage({ content: payload })} />);
    expect(window.__xss).toBeUndefined();
  });

  // ── 3. href javascript: URI must not be a live link ───────────────────────
  it("REGRESSION #240: javascript: href must not be rendered as a live link", () => {
    const payload = '<a href="javascript:window.__xss=\'href\'">click</a>';
    const { container } = render(
      <MessageBubble message={makeMessage({ content: payload })} />
    );
    const links = container.querySelectorAll("a[href^='javascript:']");
    expect(links.length).toBe(0);
  });

  // ── 4. event handler on a plain tag ───────────────────────────────────────
  it("REGRESSION #240: onclick attribute must not be present in the DOM", () => {
    const payload = '<span onclick="window.__xss=1">text</span>';
    const { container } = render(
      <MessageBubble message={makeMessage({ content: payload })} />
    );
    const nodes = container.querySelectorAll("[onclick]");
    expect(nodes.length).toBe(0);
    expect(window.__xss).toBeUndefined();
  });

  // ── 5. User message role is also safe ─────────────────────────────────────
  it("REGRESSION #240: malicious payload in user message is also safe", () => {
    const payload = '<img src="x" onerror="window.__xss=\'user\'">';
    render(
      <MessageBubble
        message={makeMessage({ role: "user", content: payload })}
      />
    );
    expect(window.__xss).toBeUndefined();
  });

  // ── 6. Normal Markdown still renders correctly ────────────────────────────
  it("bold Markdown text renders as visible bold text", () => {
    render(
      <MessageBubble message={makeMessage({ content: "Hello **world**!" })} />
    );
    // react-markdown renders **…** as a <strong> element
    const strong = document.querySelector("strong");
    expect(strong).not.toBeNull();
    expect(strong.textContent).toBe("world");
  });

  it("inline code renders with visible backtick syntax", () => {
    render(
      <MessageBubble message={makeMessage({ content: "Use `const` keyword." })} />
    );
    const code = document.querySelector("code");
    expect(code).not.toBeNull();
    expect(code.textContent).toContain("const");
  });

  it("fenced code block renders via CodeBlock component", () => {
    const content = "```python\nprint('hello')\n```";
    render(<MessageBubble message={makeMessage({ content })} />);
    // The CodeBlock component shows the language label
    expect(screen.getByText("python")).toBeInTheDocument();
    expect(screen.getByText("print('hello')")).toBeInTheDocument();
  });

  it("plain text without Markdown renders as-is", () => {
    render(
      <MessageBubble message={makeMessage({ content: "Just plain text here." })} />
    );
    expect(screen.getByText("Just plain text here.")).toBeInTheDocument();
  });
});
