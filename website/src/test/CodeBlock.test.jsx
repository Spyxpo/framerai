import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CodeBlock from "../components/CodeBlock/CodeBlock";

describe("CodeBlock — rendering", () => {
  it("renders the language label", () => {
    render(<CodeBlock language="python" code="print('hello')" />);
    expect(screen.getByText("python")).toBeInTheDocument();
  });

  it("renders the code content", () => {
    render(<CodeBlock language="js" code="const x = 1;" />);
    expect(screen.getByText("const x = 1;")).toBeInTheDocument();
  });

  it("renders Copy button by default", () => {
    render(<CodeBlock language="ts" code="let y = 2;" />);
    expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
  });

  it("renders multiline code correctly", () => {
    const code = "line one\nline two\nline three";
    render(<CodeBlock language="text" code={code} />);
    expect(screen.getByText(/line one/)).toBeInTheDocument();
  });

  it("renders with unknown language label as-is", () => {
    render(<CodeBlock language="brainfuck" code="+++." />);
    expect(screen.getByText("brainfuck")).toBeInTheDocument();
  });
});

describe("CodeBlock — copy behaviour", () => {
  let user;
  beforeEach(() => {
    user = userEvent.setup();
    vi.clearAllMocks();
  });

  it("calls clipboard.writeText with the code when Copy is clicked", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true, writable: true });
    render(<CodeBlock language="python" code="print('hi')" />);
    await user.click(screen.getByRole("button", { name: /copy/i }));
    expect(writeText).toHaveBeenCalledWith("print('hi')");
  });

  it("button label changes to Copied after clicking", async () => {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true, writable: true,
    });
    render(<CodeBlock language="python" code="x = 1" />);
    await user.click(screen.getByRole("button", { name: /copy/i }));
    expect(screen.getByText(/copied/i)).toBeInTheDocument();
  });

  it("button reverts to Copy after 2 seconds", async () => {
    vi.useFakeTimers();
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true, writable: true,
    });
    render(<CodeBlock language="python" code="x = 1" />);
    const btn = screen.getByRole("button", { name: /copy/i });
    await act(async () => { btn.click(); });
    expect(screen.getByText(/copied/i)).toBeInTheDocument();
    act(() => { vi.advanceTimersByTime(2100); });
    expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
    vi.useRealTimers();
  });

  it("can copy multiple times in sequence", async () => {
    vi.useFakeTimers();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true, writable: true });
    render(<CodeBlock language="js" code="foo()" />);
    const btn = screen.getByRole("button", { name: /copy/i });
    await act(async () => { btn.click(); });
    act(() => { vi.advanceTimersByTime(2100); });
    await act(async () => { btn.click(); });
    expect(writeText).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });
});
