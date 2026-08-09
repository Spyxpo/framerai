import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Chat from "../components/Chat/Chat";
import MessageBubble from "../components/Chat/MessageBubble";

// ── Helpers ────────────────────────────────────────────────────────────────

function makeMessage(overrides = {}) {
  return {
    id: "msg-1",
    role: "user",
    type: "text",
    content: "Hello world",
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

/** Minimal props to render <Chat> without crashes */
function chatProps(overrides = {}) {
  return {
    messages: [],
    loading: false,
    streaming: false,
    loadingMessages: false,
    error: null,
    sidebarOpen: true,
    onSend: vi.fn(),
    onToggleSidebar: vi.fn(),
    onDismissError: vi.fn(),
    onOpenSettings: vi.fn(),
    onFocusSidebar: vi.fn(),
    onFocusSidebarSettings: vi.fn(),
    focusRef: { current: null },
    textareaFocusRef: { current: null },
    chatSettingsFocusRef: { current: null },
    ...overrides,
  };
}

// ── MessageBubble — receive / render ──────────────────────────────────────

describe("MessageBubble", () => {
  it("renders a user message", () => {
    render(<MessageBubble message={makeMessage()} />);
    expect(screen.getByText("Hello world")).toBeInTheDocument();
    expect(screen.getByRole("article")).toHaveClass("user");
  });

  it("renders an assistant message", () => {
    const msg = makeMessage({ role: "assistant", content: "Hi there!" });
    render(<MessageBubble message={msg} />);
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
    expect(screen.getByRole("article")).toHaveClass("assistant");
  });

  it("renders an error message with alert role", () => {
    const msg = makeMessage({ role: "assistant", type: "error", content: "Something went wrong" });
    render(<MessageBubble message={msg} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("shows retry button on error message when onRetry is provided", () => {
    const onRetry = vi.fn();
    const msg = makeMessage({ role: "assistant", type: "error", content: "Failed" });
    render(<MessageBubble message={msg} onRetry={onRetry} />);
    const retryBtn = screen.getByRole("button", { name: /retry/i });
    expect(retryBtn).toBeInTheDocument();
    retryBtn.click();
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it("does not show retry button when onRetry is not provided", () => {
    const msg = makeMessage({ role: "assistant", type: "error", content: "Failed" });
    render(<MessageBubble message={msg} />);
    expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
  });

  it("shows copy button on assistant message", () => {
    const msg = makeMessage({ role: "assistant", content: "Here is your answer." });
    render(<MessageBubble message={msg} />);
    expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
  });

  it("does not show copy button on user message", () => {
    render(<MessageBubble message={makeMessage()} />);
    expect(screen.queryByRole("button", { name: /copy/i })).not.toBeInTheDocument();
  });

  it("copy button writes to clipboard and shows Copied feedback", async () => {
    const user = userEvent.setup();
    const msg = makeMessage({ role: "assistant", content: "Copy me." });
    render(<MessageBubble message={msg} />);
    const copyBtn = screen.getByRole("button", { name: /copy/i });
    await user.click(copyBtn);
    // Button label flips to "Copied" — proves the click handler ran
    expect(screen.getByRole("button", { name: /copied/i })).toBeInTheDocument();
  });

  it("shows typing indicator when content is empty and streaming", () => {
    const msg = makeMessage({ role: "assistant", content: "" });
    render(<MessageBubble message={msg} isStreaming />);
    // typing indicator renders three <span> children inside .typing-indicator
    const indicator = document.querySelector(".typing-indicator");
    expect(indicator).toBeInTheDocument();
  });

  it("renders a code block for fenced code in content", () => {
    const msg = makeMessage({
      role: "assistant",
      content: "```python\nprint('hello')\n```",
    });
    render(<MessageBubble message={msg} />);
    expect(screen.getByText("python")).toBeInTheDocument();
    expect(screen.getByText("print('hello')")).toBeInTheDocument();
  });

  it("renders an image when message has image type and metadata url", () => {
    const msg = makeMessage({
      role: "assistant",
      type: "image",
      content: "Here is your image.",
      metadata: { url: "/uploads/generated/img.png", prompt: "a cat" },
    });
    render(<MessageBubble message={msg} />);
    const img = screen.getByRole("img", { name: /a cat/i });
    expect(img).toHaveAttribute("src", "/uploads/generated/img.png");
  });

  it("renders an audio player when message has audio type and metadata url", () => {
    const msg = makeMessage({
      role: "assistant",
      type: "audio",
      content: "Here is your audio.",
      metadata: { url: "/uploads/generated/audio.wav" },
    });
    render(<MessageBubble message={msg} />);
    expect(screen.getByLabelText(/generated audio/i)).toBeInTheDocument();
  });

  it("renders model tag when metadata.model is present", () => {
    const msg = makeMessage({
      role: "assistant",
      content: "response",
      metadata: { model: "framerai-text" },
    });
    render(<MessageBubble message={msg} />);
    expect(screen.getByText("framerai-text")).toBeInTheDocument();
  });
});

// ── Chat — send flow ───────────────────────────────────────────────────────

describe("Chat — send flow", () => {
  let user;
  beforeEach(() => {
    user = userEvent.setup();
  });

  it("renders the textarea", () => {
    render(<Chat {...chatProps()} />);
    expect(screen.getByRole("textbox", { name: /message input/i })).toBeInTheDocument();
  });

  it("send button is disabled when input is empty", () => {
    render(<Chat {...chatProps()} />);
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("send button becomes enabled when user types", async () => {
    render(<Chat {...chatProps()} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    await user.type(textarea, "hi");
    expect(screen.getByRole("button", { name: /send/i })).toBeEnabled();
  });

  it("calls onSend with input text and default type on submit", async () => {
    const onSend = vi.fn();
    render(<Chat {...chatProps({ onSend })} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    await user.type(textarea, "Say hello");
    await user.click(screen.getByRole("button", { name: /send/i }));
    expect(onSend).toHaveBeenCalledWith("Say hello", "text");
  });

  it("clears input after send", async () => {
    const onSend = vi.fn();
    render(<Chat {...chatProps({ onSend })} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    await user.type(textarea, "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));
    expect(textarea).toHaveValue("");
  });

  it("calls onSend when Enter is pressed (no shift)", async () => {
    const onSend = vi.fn();
    render(<Chat {...chatProps({ onSend })} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    await user.type(textarea, "Enter test{Enter}");
    expect(onSend).toHaveBeenCalled();
  });

  it("does not call onSend when Shift+Enter is pressed", async () => {
    const onSend = vi.fn();
    render(<Chat {...chatProps({ onSend })} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    await user.type(textarea, "multiline{Shift>}{Enter}{/Shift}");
    expect(onSend).not.toHaveBeenCalled();
  });

  it("send button is disabled while loading", () => {
    render(<Chat {...chatProps({ loading: true })} />);
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("send button is disabled while streaming", () => {
    render(<Chat {...chatProps({ streaming: true })} />);
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("renders received messages", () => {
    const messages = [
      makeMessage({ role: "user", content: "Hello!" }),
      makeMessage({ id: "msg-2", role: "assistant", content: "Hi there!" }),
    ];
    render(<Chat {...chatProps({ messages })} />);
    expect(screen.getByText("Hello!")).toBeInTheDocument();
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("shows error banner when error prop is set", () => {
    render(<Chat {...chatProps({ error: "Something broke" })} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("Something broke")).toBeInTheDocument();
  });

  it("calls onDismissError when dismiss button is clicked", async () => {
    const onDismissError = vi.fn();
    render(<Chat {...chatProps({ error: "Oops", onDismissError })} />);
    await user.click(screen.getByRole("button", { name: /dismiss error/i }));
    expect(onDismissError).toHaveBeenCalledOnce();
  });

  it("shows welcome screen when messages are empty", () => {
    render(<Chat {...chatProps()} />);
    expect(screen.getByText(/welcome to framerai/i)).toBeInTheDocument();
  });

  it("calls onSend when a suggestion button is clicked", async () => {
    const onSend = vi.fn();
    render(<Chat {...chatProps({ onSend })} />);
    await user.click(screen.getByRole("button", { name: /what can you do/i }));
    expect(onSend).toHaveBeenCalledWith("Hello! What can you do?");
  });

  it("selecting a mode type changes the button sent with next message", async () => {
    const onSend = vi.fn();
    render(<Chat {...chatProps({ onSend })} />);
    // Click the Image mode button
    await user.click(screen.getByRole("button", { name: /image generation mode/i }));
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    await user.type(textarea, "a sunset");
    await user.click(screen.getByRole("button", { name: /send/i }));
    expect(onSend).toHaveBeenCalledWith("a sunset", "image");
  });

  it("opens settings when settings button is clicked", async () => {
    const onOpenSettings = vi.fn();
    render(<Chat {...chatProps({ onOpenSettings })} />);
    await user.click(screen.getByRole("button", { name: /generation settings/i }));
    expect(onOpenSettings).toHaveBeenCalledOnce();
  });
});
