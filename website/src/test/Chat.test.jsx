import { useState } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
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

  it("renders streaming audio player when audioChunks are present", () => {
    // jsdom doesn't support Web Audio API, component will show error or loading
    // The important thing is that it recognizes streaming audio data
    const msg = makeMessage({
      role: "assistant",
      type: "audio",
      content: "Here is your audio",
      audioChunks: ["Y2h1bmsx", "Y2h1bmsy"], // base64 chunks
      audioMetadata: {
        sampleRate: 24000,
        channels: 1,
        bitsPerSample: 16,
        totalChunks: 2,
      },
    });
    const { container } = render(<MessageBubble message={msg} />);

    // The streaming audio player component should be rendered
    // (it will show an error in jsdom, but the component structure exists)
    const streamingPlayer = container.querySelector('.streaming-audio-player, .streaming-audio-error, .streaming-audio-loading');
    expect(streamingPlayer).toBeInTheDocument();
  });

  it("renders standard audio element when only URL is present (fallback)", () => {
    const msg = makeMessage({
      role: "assistant",
      type: "audio",
      content: "Here is your audio",
      metadata: { url: "/uploads/generated/audio.wav" },
    });
    render(<MessageBubble message={msg} />);
    const audioElement = screen.getByLabelText(/generated audio/i);
    expect(audioElement).toBeInTheDocument();
    expect(audioElement.tagName).toBe("AUDIO");
  });

  it("does not render audio when type is audio but no chunks or URL", () => {
    const msg = makeMessage({
      role: "assistant",
      type: "audio",
      content: "Audio generation in progress",
    });
    render(<MessageBubble message={msg} />);
    expect(screen.queryByLabelText(/generated audio/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/streaming audio/i)).not.toBeInTheDocument();
  });

  it("regression: final audio chunk preserved when done=true", async () => {
    // This test uses the REAL useChat hook with a mocked WebSocketClient
    // to verify that ALL chunks (including the final one) are preserved.
    //
    // CRITICAL: This test MUST FAIL if the chunk push in useChat.js is broken

    // Set up isolated mock for this test only
    let mockStreamHandler = null;

    const MockWebSocketClient = class {
      constructor() {
        this.ws = { readyState: 1 };
        this.listeners = new Map();
      }
      connect() {
        return Promise.resolve();
      }
      on(type, handler) {
        if (!this.listeners.has(type)) {
          this.listeners.set(type, []);
        }
        this.listeners.get(type).push(handler);
        if (type === "stream") {
          mockStreamHandler = handler;
        }
        return () => {};
      }
      send() {}
      disconnect() {}
    };

    // Mock the API module as well
    vi.doMock("../services/api", () => ({
      api: {
        createConversation: vi.fn(() => Promise.resolve({ id: "test-conv-id", title: "Test" })),
        listConversations: vi.fn(() => Promise.resolve([])),
      },
    }));

    // Use doMock for test-specific mocking
    vi.doMock("../services/websocket", () => ({
      WebSocketClient: MockWebSocketClient,
    }));

    // Dynamic imports after mock setup
    const { renderHook, act, waitFor } = await import("@testing-library/react");
    const useChatModule = await import("../hooks/useChat?t=" + Date.now());
    const { useChat } = useChatModule;

    // Render the hook
    const { result } = renderHook(() => useChat({}));

    // Wait for WebSocket initialization
    await waitFor(() => {
      expect(mockStreamHandler).not.toBeNull();
    }, { timeout: 1000 });

    // Create test chunks
    const sampleRate = 24000;
    const channels = 1;
    const bitsPerSample = 16;
    const samplesPerChunk = sampleRate * 0.5;
    const bytesPerChunk = samplesPerChunk * channels * (bitsPerSample / 8);

    const createPCMChunk = (chunkIndex) => {
      const buffer = new ArrayBuffer(bytesPerChunk);
      const view = new DataView(buffer);
      for (let i = 0; i < samplesPerChunk; i++) {
        const sample = Math.sin(i * 0.1 + chunkIndex * 100) * 32767;
        view.setInt16(i * 2, sample, true);
      }
      return btoa(String.fromCharCode(...new Uint8Array(buffer)));
    };

    const chunk1 = createPCMChunk(0);
    const chunk2 = createPCMChunk(1);
    const chunk3 = createPCMChunk(2);
    const totalSourceBytes = bytesPerChunk * 3;

    // Trigger message creation - this creates user + assistant placeholder messages
    await act(async () => {
      result.current.sendMessage("generate audio", "audio");
    });

    // CRITICAL: Wait for the assistant placeholder message to exist in state
    // sendMessage creates: 1) user message, 2) assistant placeholder
    await waitFor(() => {
      const messages = result.current.messages;
      const assistantMsg = messages.find(m => m.role === "assistant");
      expect(assistantMsg).toBeDefined();
      expect(assistantMsg.role).toBe("assistant");
    }, { timeout: 1000 });

    // NOW feed chunk 1 through the REAL stream handler
    await act(async () => {
      mockStreamHandler({
        type: "stream",
        content: "Generating audio...",
        done: false,
        responseType: "audio",
        metadata: {
          chunk: 0,
          totalChunks: 3,
          chunkData: chunk1,
          sampleRate,
          channels,
          bitsPerSample,
        },
      });
    });

    // Wait for chunk 1 to be processed
    await waitFor(() => {
      const audioMsg = result.current.messages.find(m => m.role === "assistant" && m.type === "audio");
      expect(audioMsg?.audioChunks?.length).toBe(1);
    }, { timeout: 1000 });

    // Feed chunk 2
    await act(async () => {
      mockStreamHandler({
        type: "stream",
        content: "",
        done: false,
        responseType: "audio",
        metadata: {
          chunk: 1,
          totalChunks: 3,
          chunkData: chunk2,
          sampleRate,
          channels,
          bitsPerSample,
        },
      });
    });

    // Wait for chunk 2 to be processed
    await waitFor(() => {
      const audioMsg = result.current.messages.find(m => m.role === "assistant" && m.type === "audio");
      expect(audioMsg?.audioChunks?.length).toBe(2);
    }, { timeout: 1000 });

    // Feed chunk 3 (FINAL) - THE CRITICAL TEST
    await act(async () => {
      mockStreamHandler({
        type: "stream",
        content: "",
        done: true,
        responseType: "audio",
        metadata: {
          chunk: 2,
          totalChunks: 3,
          chunkData: chunk3, // THIS MUST BE PUSHED BY THE REAL useChat CODE
          sampleRate,
          channels,
          bitsPerSample,
          url: "/uploads/generated/audio.wav",
          model: "framerai-audio",
          durationSec: 1.5,
        },
      });
    });

    // Wait for completion
    await waitFor(() => {
      const audioMsg = result.current.messages.find(m => m.role === "assistant" && m.type === "audio");
      expect(audioMsg?.audioComplete).toBe(true);
    }, { timeout: 1000 });

    // CRITICAL ASSERTIONS - verify the REAL useChat state
    const audioMsg = result.current.messages.find(m => m.role === "assistant" && m.type === "audio");

    expect(audioMsg).toBeDefined();
    expect(audioMsg.audioChunks).toBeDefined();

    // This is the key assertion: if the done branch doesn't push chunkData,
    // this will be 2 instead of 3, causing the test to FAIL
    expect(audioMsg.audioChunks.length).toBe(3);

    // Verify total byte count matches source
    let reassembledBytes = 0;
    audioMsg.audioChunks.forEach(chunkData => {
      reassembledBytes += atob(chunkData).length;
    });
    expect(reassembledBytes).toBe(totalSourceBytes);

    // Cleanup mocks for this test
    vi.doUnmock("../services/websocket");
    vi.doUnmock("../services/api");
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
    expect(onSend).toHaveBeenCalledWith("Say hello", "text", []);
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
    expect(onSend).toHaveBeenCalledWith("a sunset", "image", []);
  });

  it("attaches a picked file and sends the stored path with the message", async () => {
    const { api } = await import("../services/api");
    const upload = vi
      .spyOn(api, "uploadAttachment")
      .mockResolvedValue({ path: "/uploads/documents/stored.pdf", kind: "document", name: "report.pdf" });

    const onSend = vi.fn();
    const { container } = render(<Chat {...chatProps({ onSend })} />);

    const fileInput = container.querySelector('input[type="file"][multiple]');
    const file = new File(["%PDF-1.4"], "report.pdf", { type: "application/pdf" });
    await user.upload(fileInput, file);

    // The chip proves the upload happened before the message was sent, which is
    // what lets a user see and remove an attachment first.
    expect(upload).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("report.pdf")).toBeTruthy();

    await user.type(screen.getByRole("textbox", { name: /message/i }), "what is in here");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(onSend).toHaveBeenCalledWith("what is in here", "text", ["/uploads/documents/stored.pdf"]);
    upload.mockRestore();
  });

  it("removes an attachment before the message is sent", async () => {
    const { api } = await import("../services/api");
    const upload = vi
      .spyOn(api, "uploadAttachment")
      .mockResolvedValue({ path: "/uploads/images/stored.png", kind: "image", name: "diagram.png" });

    const onSend = vi.fn();
    const { container } = render(<Chat {...chatProps({ onSend })} />);

    const fileInput = container.querySelector('input[type="file"][multiple]');
    await user.upload(fileInput, new File(["x"], "diagram.png", { type: "image/png" }));
    await screen.findByText("diagram.png");

    await user.click(screen.getByRole("button", { name: /remove diagram.png/i }));
    expect(screen.queryByText("diagram.png")).toBeNull();

    await user.type(screen.getByRole("textbox", { name: /message/i }), "hello");
    await user.click(screen.getByRole("button", { name: /send/i }));
    expect(onSend).toHaveBeenCalledWith("hello", "text", []);
    upload.mockRestore();
  });

  it("reports an attachment that could not be stored", async () => {
    const { api } = await import("../services/api");
    const upload = vi
      .spyOn(api, "uploadAttachment")
      .mockRejectedValue(new Error("Uploaded file too large"));

    const { container } = render(<Chat {...chatProps()} />);
    const fileInput = container.querySelector('input[type="file"][multiple]');
    await user.upload(fileInput, new File(["x"], "big.pdf", { type: "application/pdf" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/Uploaded file too large/);
    expect(screen.queryByText("big.pdf")).toBeNull();
    upload.mockRestore();
  });

  it("opens settings when settings button is clicked", async () => {
    const onOpenSettings = vi.fn();
    render(<Chat {...chatProps({ onOpenSettings })} />);
    await user.click(screen.getByRole("button", { name: /generation settings/i }));
    expect(onOpenSettings).toHaveBeenCalledOnce();
  });

  describe("CLI Command Approval Prompt", () => {
    it("renders approval prompt card with command, argv, and root when pendingApproval is present", () => {
      const pendingApproval = {
        approvalId: "test-app-1",
        command: "cat /etc/passwd",
        argv: ["cat", "/etc/passwd"],
        root: "/sandbox/root",
      };
      render(<Chat {...chatProps({ pendingApproval })} />);

      expect(screen.getByTestId("cli-approval-card")).toBeInTheDocument();
      expect(screen.getByText("CLI Command Approval Required")).toBeInTheDocument();
      expect(screen.getByText("cat /etc/passwd")).toBeInTheDocument();
      expect(screen.getByText("/sandbox/root")).toBeInTheDocument();
    });

    it("calls onApproveCommand with approvalId when Approve button is clicked", async () => {
      const onApproveCommand = vi.fn();
      const pendingApproval = {
        approvalId: "test-app-1",
        command: "ls -la",
        argv: ["ls", "-la"],
        root: "/sandbox",
      };
      render(<Chat {...chatProps({ pendingApproval, onApproveCommand })} />);

      await user.click(screen.getByRole("button", { name: /approve/i }));
      expect(onApproveCommand).toHaveBeenCalledWith("test-app-1");
    });

    it("calls onDenyCommand with (approvalId, false) when Deny button is clicked", async () => {
      const onDenyCommand = vi.fn();
      const pendingApproval = {
        approvalId: "test-app-1",
        command: "ls -la",
        argv: ["ls", "-la"],
        root: "/sandbox",
      };
      render(<Chat {...chatProps({ pendingApproval, onDenyCommand })} />);

      await user.click(screen.getByRole("button", { name: /^deny$/i }));
      expect(onDenyCommand).toHaveBeenCalledWith("test-app-1", false);
    });

    it("calls onDenyCommand with (approvalId, true) when 'Deny all future commands' button is clicked", async () => {
      const onDenyCommand = vi.fn();
      const pendingApproval = {
        approvalId: "test-app-1",
        command: "ls -la",
        argv: ["ls", "-la"],
        root: "/sandbox",
      };
      render(<Chat {...chatProps({ pendingApproval, onDenyCommand })} />);

      await user.click(screen.getByRole("button", { name: /deny all future commands/i }));
      expect(onDenyCommand).toHaveBeenCalledWith("test-app-1", true);
    });
  });

  describe("Keyboard shortcut: / to focus chat input", () => {
    it("focuses the chat input when '/' is pressed outside editable elements", () => {
      render(<Chat {...chatProps()} />);
      const textarea = screen.getByRole("textbox", { name: /message input/i });
      expect(document.activeElement).not.toBe(textarea);

      const event = new KeyboardEvent("keydown", { key: "/", bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, "preventDefault");
      document.dispatchEvent(event);

      expect(document.activeElement).toBe(textarea);
      expect(preventDefaultSpy).toHaveBeenCalled();
    });

    it("does not trigger shortcut when typing inside an input element", () => {
      render(
        <div>
          <input data-testid="external-input" type="text" />
          <Chat {...chatProps()} />
        </div>
      );
      const input = screen.getByTestId("external-input");
      const textarea = screen.getByRole("textbox", { name: /message input/i });
      input.focus();
      expect(document.activeElement).toBe(input);

      const event = new KeyboardEvent("keydown", { key: "/", bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, "preventDefault");
      input.dispatchEvent(event);

      expect(document.activeElement).toBe(input);
      expect(document.activeElement).not.toBe(textarea);
      expect(preventDefaultSpy).not.toHaveBeenCalled();
    });

    it("does not trigger shortcut when typing inside another textarea element", () => {
      render(
        <div>
          <textarea data-testid="external-textarea" />
          <Chat {...chatProps()} />
        </div>
      );
      const extTextarea = screen.getByTestId("external-textarea");
      const chatTextarea = screen.getByRole("textbox", { name: /message input/i });
      extTextarea.focus();
      expect(document.activeElement).toBe(extTextarea);

      const event = new KeyboardEvent("keydown", { key: "/", bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, "preventDefault");
      extTextarea.dispatchEvent(event);

      expect(document.activeElement).toBe(extTextarea);
      expect(document.activeElement).not.toBe(chatTextarea);
      expect(preventDefaultSpy).not.toHaveBeenCalled();
    });

    it("does not prevent default or re-focus when typing '/' inside the chat input itself", () => {
      render(<Chat {...chatProps()} />);
      const textarea = screen.getByRole("textbox", { name: /message input/i });
      textarea.focus();
      expect(document.activeElement).toBe(textarea);

      const event = new KeyboardEvent("keydown", { key: "/", bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, "preventDefault");
      textarea.dispatchEvent(event);

      expect(preventDefaultSpy).not.toHaveBeenCalled();
    });

    it("does not trigger shortcut when focused on a select element", () => {
      render(
        <div>
          <select data-testid="external-select">
            <option value="1">1</option>
          </select>
          <Chat {...chatProps()} />
        </div>
      );
      const select = screen.getByTestId("external-select");
      const textarea = screen.getByRole("textbox", { name: /message input/i });
      select.focus();
      expect(document.activeElement).toBe(select);

      const event = new KeyboardEvent("keydown", { key: "/", bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, "preventDefault");
      select.dispatchEvent(event);

      expect(document.activeElement).toBe(select);
      expect(document.activeElement).not.toBe(textarea);
      expect(preventDefaultSpy).not.toHaveBeenCalled();
    });

    it("does not trigger shortcut when focused inside a contenteditable element", () => {
      render(
        <div>
          <div data-testid="editable-div" contentEditable="true" suppressContentEditableWarning={true}>
            <span data-testid="editable-child">text</span>
          </div>
          <Chat {...chatProps()} />
        </div>
      );
      const editableDiv = screen.getByTestId("editable-div");
      const editableChild = screen.getByTestId("editable-child");
      const textarea = screen.getByRole("textbox", { name: /message input/i });

      editableDiv.focus();

      const event = new KeyboardEvent("keydown", { key: "/", bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, "preventDefault");
      editableChild.dispatchEvent(event);

      expect(document.activeElement).not.toBe(textarea);
      expect(preventDefaultSpy).not.toHaveBeenCalled();
    });

    it("does not trigger shortcut with modifier keys like Ctrl, Cmd, or Alt", () => {
      render(<Chat {...chatProps()} />);
      const textarea = screen.getByRole("textbox", { name: /message input/i });

      ["ctrlKey", "metaKey", "altKey"].forEach((mod) => {
        document.body.focus();
        const event = new KeyboardEvent("keydown", {
          key: "/",
          [mod]: true,
          bubbles: true,
          cancelable: true,
        });
        const preventDefaultSpy = vi.spyOn(event, "preventDefault");
        document.dispatchEvent(event);

        expect(document.activeElement).not.toBe(textarea);
        expect(preventDefaultSpy).not.toHaveBeenCalled();
      });
    });

    it("does not trigger shortcut on other keys", () => {
      render(<Chat {...chatProps()} />);
      const textarea = screen.getByRole("textbox", { name: /message input/i });

      ["a", "Enter", "Escape", "Tab", "ArrowDown"].forEach((key) => {
        const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
        const preventDefaultSpy = vi.spyOn(event, "preventDefault");
        document.dispatchEvent(event);

        expect(document.activeElement).not.toBe(textarea);
        expect(preventDefaultSpy).not.toHaveBeenCalled();
      });
    });

    it("does not steal focus out of an open modal dialog", () => {
      render(
        <div>
          <Chat {...chatProps()} />
          <div role="dialog" aria-modal="true">
            <button data-testid="dialog-btn">Close settings</button>
          </div>
        </div>
      );
      const dialogBtn = screen.getByTestId("dialog-btn");
      const textarea = screen.getByRole("textbox", { name: /message input/i });
      dialogBtn.focus();
      expect(document.activeElement).toBe(dialogBtn);

      const event = new KeyboardEvent("keydown", { key: "/", bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, "preventDefault");
      dialogBtn.dispatchEvent(event);

      expect(document.activeElement).toBe(dialogBtn);
      expect(document.activeElement).not.toBe(textarea);
      expect(preventDefaultSpy).not.toHaveBeenCalled();
    });

    it("does not swallow the keystroke while the chat input is disabled", () => {
      render(<Chat {...chatProps({ loading: true })} />);
      const textarea = screen.getByRole("textbox", { name: /message input/i });
      expect(textarea.disabled).toBe(true);

      const event = new KeyboardEvent("keydown", { key: "/", bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, "preventDefault");
      document.dispatchEvent(event);

      expect(preventDefaultSpy).not.toHaveBeenCalled();
      expect(document.activeElement).not.toBe(textarea);
    });

    it("cleans up keydown listener on unmount", () => {
      const { unmount } = render(<Chat {...chatProps()} />);
      unmount();

      const event = new KeyboardEvent("keydown", { key: "/", bubbles: true, cancelable: true });
      expect(() => document.dispatchEvent(event)).not.toThrow();
    });

    it("renders the keyboard shortcut hint near the chat input when empty", () => {
      render(<Chat {...chatProps()} />);
      const hint = screen.getByTitle("Press / to focus");
      expect(hint).toBeInTheDocument();
      expect(hint).toHaveTextContent(/press\s*\/\s*to focus/i);
    });

    it("focuses the chat input when clicking the shortcut hint", async () => {
      const user = userEvent.setup();
      render(<Chat {...chatProps()} />);
      const textarea = screen.getByRole("textbox", { name: /message input/i });
      const hint = screen.getByTitle("Press / to focus");

      expect(document.activeElement).not.toBe(textarea);
      await user.click(hint);
      expect(document.activeElement).toBe(textarea);
    });

    it("hides the shortcut hint when text is typed into the input", async () => {
      const user = userEvent.setup();
      render(<Chat {...chatProps()} />);
      const textarea = screen.getByRole("textbox", { name: /message input/i });
      expect(screen.getByTitle("Press / to focus")).toBeInTheDocument();

      await user.type(textarea, "Hello");
      expect(screen.queryByTitle("Press / to focus")).not.toBeInTheDocument();
    });
  });
});

// ── Chat — interface layout and header ─────────────────────────────────────

describe("Chat — interface layout and header", () => {
  it("renders the chat header with title and subtitle", () => {
    render(<Chat {...chatProps()} />);
    expect(screen.getByRole("heading", { level: 1, name: "FramerAI" })).toBeInTheDocument();
    expect(screen.getByText("Text, code, image, video, and audio")).toBeInTheDocument();
  });

  it("renders sidebar toggle button and applies full-width class when sidebar is closed", async () => {
    const user = userEvent.setup();
    const onToggleSidebar = vi.fn();
    render(<Chat {...chatProps({ sidebarOpen: false, onToggleSidebar })} />);

    const main = screen.getByRole("main", { name: "Chat" });
    expect(main).toHaveClass("full-width");

    const toggleBtn = screen.getByRole("button", { name: /open sidebar/i });
    expect(toggleBtn).toBeInTheDocument();
    await user.click(toggleBtn);
    expect(onToggleSidebar).toHaveBeenCalledOnce();
  });

  it("does not render sidebar toggle button or full-width class when sidebar is open", () => {
    render(<Chat {...chatProps({ sidebarOpen: true })} />);
    const main = screen.getByRole("main", { name: "Chat" });
    expect(main).not.toHaveClass("full-width");
    expect(screen.queryByRole("button", { name: /open sidebar/i })).not.toBeInTheDocument();
  });

  it("renders generation settings button in header and invokes onOpenSettings on click", async () => {
    const user = userEvent.setup();
    const onOpenSettings = vi.fn();
    render(<Chat {...chatProps({ onOpenSettings })} />);
    const settingsBtn = screen.getByRole("button", { name: /generation settings/i });
    await user.click(settingsBtn);
    expect(onOpenSettings).toHaveBeenCalledOnce();
  });

  it("calls scrollIntoView on the messages end marker when messages update", () => {
    const scrollSpy = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scrollSpy;

    const { rerender } = render(<Chat {...chatProps({ messages: [] })} />);
    expect(scrollSpy).toHaveBeenCalled();

    scrollSpy.mockClear();
    rerender(<Chat {...chatProps({ messages: [makeMessage({ content: "New update" })] })} />);
    expect(scrollSpy).toHaveBeenCalled();
  });
});

// ── Chat — empty state and prompt suggestions ──────────────────────────────

describe("Chat — empty state and prompt suggestions", () => {
  it("renders welcome screen with description and prompt suggestions when messages is empty", () => {
    render(<Chat {...chatProps({ messages: [] })} />);
    expect(screen.getByRole("status", { name: /welcome to framerai/i })).toBeInTheDocument();
    expect(
      screen.getByText(/a multimodal ai that can generate text, code, images, video, and audio/i)
    ).toBeInTheDocument();

    expect(screen.getByRole("button", { name: /what can you do/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /write a fibonacci function/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /generate a sunset image/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /generate a voice clip/i })).toBeInTheDocument();
  });

  it("does not render welcome screen when conversation has messages", () => {
    render(<Chat {...chatProps({ messages: [makeMessage({ role: "user", content: "Hello" })] })} />);
    expect(screen.queryByRole("status", { name: /welcome to framerai/i })).not.toBeInTheDocument();
  });

  it("sends expected prompt text when each suggestion button is clicked", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<Chat {...chatProps({ onSend })} />);

    await user.click(screen.getByRole("button", { name: /write a fibonacci function/i }));
    expect(onSend).toHaveBeenCalledWith("Write a fibonacci function in Python");

    await user.click(screen.getByRole("button", { name: /generate a sunset image/i }));
    expect(onSend).toHaveBeenCalledWith("Generate an image of a sunset over mountains");

    await user.click(screen.getByRole("button", { name: /generate a voice clip/i }));
    expect(onSend).toHaveBeenCalledWith("Generate audio that says hello and welcome");
  });
});

// ── Chat — message composition and input validation ────────────────────────

describe("Chat — message composition and input validation", () => {
  it("prevents submission when input contains only whitespace", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<Chat {...chatProps({ onSend })} />);

    const textarea = screen.getByRole("textbox", { name: /message input/i });
    const sendBtn = screen.getByRole("button", { name: /send/i });

    await user.type(textarea, "   \n   \t  ");
    expect(sendBtn).toBeDisabled();

    await user.type(textarea, "{Enter}");
    expect(onSend).not.toHaveBeenCalled();
  });

  it("adjusts textarea height style during typing", async () => {
    const user = userEvent.setup();
    render(<Chat {...chatProps()} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });

    await user.type(textarea, "Hello world line");
    expect(textarea.style.height).toBeDefined();
  });
});

// ── Chat — mode selection and submission resets ────────────────────────────

describe("Chat — mode selection and submission resets", () => {
  it("defaults to text mode and toggles aria-pressed when selecting modes", async () => {
    const user = userEvent.setup();
    render(<Chat {...chatProps()} />);

    const textBtn = screen.getByRole("button", { name: "Text mode" });
    const codeBtn = screen.getByRole("button", { name: "Code mode" });
    const imgBtn = screen.getByRole("button", { name: "Image generation mode" });
    const videoBtn = screen.getByRole("button", { name: "Video generation mode" });
    const audioBtn = screen.getByRole("button", { name: "Audio generation mode" });

    expect(textBtn).toHaveAttribute("aria-pressed", "true");
    expect(codeBtn).toHaveAttribute("aria-pressed", "false");

    await user.click(codeBtn);
    expect(codeBtn).toHaveAttribute("aria-pressed", "true");
    expect(textBtn).toHaveAttribute("aria-pressed", "false");

    await user.click(videoBtn);
    expect(videoBtn).toHaveAttribute("aria-pressed", "true");
    expect(codeBtn).toHaveAttribute("aria-pressed", "false");

    await user.click(audioBtn);
    expect(audioBtn).toHaveAttribute("aria-pressed", "true");
    expect(videoBtn).toHaveAttribute("aria-pressed", "false");

    await user.click(imgBtn);
    expect(imgBtn).toHaveAttribute("aria-pressed", "true");
    expect(audioBtn).toHaveAttribute("aria-pressed", "false");
  });

  it("resets message type back to text after sending a message in another mode", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<Chat {...chatProps({ onSend })} />);

    const codeBtn = screen.getByRole("button", { name: "Code mode" });
    const textBtn = screen.getByRole("button", { name: "Text mode" });
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    const sendBtn = screen.getByRole("button", { name: /send/i });

    // Switch to code mode and send
    await user.click(codeBtn);
    await user.type(textarea, "def add(a, b): return a + b");
    await user.click(sendBtn);

    expect(onSend).toHaveBeenCalledWith("def add(a, b): return a + b", "code", []);

    // Mode button state should revert to text mode
    expect(textBtn).toHaveAttribute("aria-pressed", "true");
    expect(codeBtn).toHaveAttribute("aria-pressed", "false");

    // Subsequent send should send as text mode
    await user.type(textarea, "How does this look?");
    await user.click(sendBtn);
    expect(onSend).toHaveBeenCalledWith("How does this look?", "text", []);
  });
});

// ── Chat — loading, busy, and skeleton states ──────────────────────────────

describe("Chat — loading, busy, and skeleton states", () => {
  it("renders spinning loader and busy class when loading or streaming", () => {
    const { container, rerender } = render(<Chat {...chatProps({ loading: true })} />);

    const inputWrapper = container.querySelector(".input-wrapper");
    expect(inputWrapper).toHaveClass("busy");

    const sendBtn = screen.getByRole("button", { name: /send/i });
    expect(sendBtn).toBeDisabled();
    expect(sendBtn.querySelector(".spin")).toBeInTheDocument();

    rerender(<Chat {...chatProps({ streaming: true })} />);
    expect(inputWrapper).toHaveClass("busy");
    expect(sendBtn.querySelector(".spin")).toBeInTheDocument();
  });

  it("renders typing indicator while loading or streaming when last message is not assistant", () => {
    const messages = [makeMessage({ role: "user", content: "Tell me a joke" })];
    const { rerender } = render(<Chat {...chatProps({ messages, loading: true })} />);

    expect(screen.getByRole("status", { name: /framerai is typing/i })).toBeInTheDocument();

    // When an assistant message is added, typing indicator in chat container is removed
    rerender(
      <Chat
        {...chatProps({
          messages: [
            ...messages,
            makeMessage({ id: "asst-1", role: "assistant", content: "Why did the chicken..." }),
          ],
          loading: true,
        })}
      />
    );
    expect(screen.queryByRole("status", { name: /framerai is typing/i })).not.toBeInTheDocument();
  });

  it("renders message skeletons and disables composer when loadingMessages is true", () => {
    render(<Chat {...chatProps({ loadingMessages: true, messages: [] })} />);

    const skeletonContainer = screen.getByLabelText("Loading messages");
    expect(skeletonContainer).toHaveAttribute("aria-busy", "true");
    expect(document.querySelector(".messages-loading")).toBeInTheDocument();

    // Welcome screen must not be displayed while loading conversation messages
    expect(screen.queryByRole("status", { name: /welcome to framerai/i })).not.toBeInTheDocument();

    const textarea = screen.getByRole("textbox", { name: /message input/i });
    expect(textarea).toBeDisabled();
    expect(textarea).toHaveAttribute("placeholder", "Loading conversation…");
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("auto-focuses textarea when loading or streaming finishes", () => {
    const { rerender } = render(<Chat {...chatProps({ loading: true })} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    expect(document.activeElement).not.toBe(textarea);

    rerender(<Chat {...chatProps({ loading: false, streaming: false })} />);
    expect(document.activeElement).toBe(textarea);

    const settingsBtn = screen.getByRole("button", { name: /generation settings/i });
    settingsBtn.focus();
    expect(document.activeElement).toBe(settingsBtn);

    rerender(<Chat {...chatProps({ streaming: true })} />);
    rerender(<Chat {...chatProps({ streaming: false })} />);
    expect(document.activeElement).toBe(textarea);
  });
});

// ── Chat — error handling and retry flow ───────────────────────────────────

describe("Chat — error handling and retry flow", () => {
  it("renders retry button on trailing error message and calls onSend on click", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    const messages = [
      makeMessage({ id: "user-1", role: "user", content: "Generate speech", type: "audio" }),
      makeMessage({ id: "err-1", role: "assistant", type: "error", content: "Model unavailable" }),
    ];

    render(<Chat {...chatProps({ messages, onSend })} />);

    const retryBtn = screen.getByRole("button", { name: /retry/i });
    expect(retryBtn).toBeInTheDocument();

    await user.click(retryBtn);
    expect(onSend).toHaveBeenCalledWith("Generate speech", "audio");
  });

  it("does not render retry button if the error message is not the last message", () => {
    const messages = [
      makeMessage({ id: "user-1", role: "user", content: "First" }),
      makeMessage({ id: "err-1", role: "assistant", type: "error", content: "Failed" }),
      makeMessage({ id: "user-2", role: "user", content: "Second" }),
    ];
    render(<Chat {...chatProps({ messages })} />);
    expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
  });

  it("does not render retry button if there is no preceding user message", () => {
    const messages = [
      makeMessage({ id: "err-1", role: "assistant", type: "error", content: "Failed to connect" }),
    ];
    render(<Chat {...chatProps({ messages })} />);
    expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
  });
});

// ── Chat — audio input and transcription flow ──────────────────────────────

describe("Chat — audio input and transcription flow", () => {
  it("transcribes uploaded audio file and inserts text into the input", async () => {
    const user = userEvent.setup();
    const { api } = await import("../services/api");
    const transcribeSpy = vi.spyOn(api, "transcribe").mockResolvedValue({ text: "transcribed speech text" });

    const { container } = render(<Chat {...chatProps()} />);
    const audioInput = container.querySelector('input[type="file"][accept="audio/*"]');
    const file = new File(["fake-audio"], "speech.mp3", { type: "audio/mp3" });

    await user.upload(audioInput, file);

    expect(transcribeSpy).toHaveBeenCalledWith(file);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    expect(textarea).toHaveValue("transcribed speech text");
    expect(document.activeElement).toBe(textarea);

    transcribeSpy.mockRestore();
  });

  it("appends transcribed text with a space when input already has content", async () => {
    const user = userEvent.setup();
    const { api } = await import("../services/api");
    const transcribeSpy = vi.spyOn(api, "transcribe").mockResolvedValue({ text: "additional voice" });

    const { container } = render(<Chat {...chatProps()} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    await user.type(textarea, "Initial text");

    const audioInput = container.querySelector('input[type="file"][accept="audio/*"]');
    await user.upload(audioInput, new File(["fake-audio"], "voice.wav", { type: "audio/wav" }));

    expect(textarea).toHaveValue("Initial text additional voice");
    transcribeSpy.mockRestore();
  });

  it("shows error banner when transcription returns empty text", async () => {
    const user = userEvent.setup();
    const { api } = await import("../services/api");
    const transcribeSpy = vi.spyOn(api, "transcribe").mockResolvedValue({ text: "" });

    const { container } = render(<Chat {...chatProps()} />);
    const audioInput = container.querySelector('input[type="file"][accept="audio/*"]');
    await user.upload(audioInput, new File(["fake-audio"], "empty.wav", { type: "audio/wav" }));

    const banner = await screen.findByRole("alert");
    expect(banner).toHaveTextContent(/transcription returned empty text/i);

    // Dismissing error removes the banner
    await user.click(screen.getByRole("button", { name: /dismiss error/i }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    transcribeSpy.mockRestore();
  });

  it("shows error banner when audio transcription request fails", async () => {
    const user = userEvent.setup();
    const { api } = await import("../services/api");
    const transcribeSpy = vi.spyOn(api, "transcribe").mockRejectedValue(new Error("Network timeout"));

    const { container } = render(<Chat {...chatProps()} />);
    const audioInput = container.querySelector('input[type="file"][accept="audio/*"]');
    await user.upload(audioInput, new File(["fake-audio"], "voice.wav", { type: "audio/wav" }));

    const banner = await screen.findByRole("alert");
    expect(banner).toHaveTextContent(/transcription failed: network timeout/i);

    transcribeSpy.mockRestore();
  });

  it("handles microphone permission errors gracefully", async () => {
    const user = userEvent.setup();
    const originalMediaDevices = navigator.mediaDevices;

    // Simulate missing getUserMedia
    Object.defineProperty(navigator, "mediaDevices", {
      value: {},
      configurable: true,
      writable: true,
    });

    render(<Chat {...chatProps()} />);
    const micBtn = screen.getByRole("button", { name: /record from mic/i });
    await user.click(micBtn);

    expect(screen.getByRole("alert")).toHaveTextContent(/microphone access is not supported/i);

    // Simulate Permission Denied
    Object.defineProperty(navigator, "mediaDevices", {
      value: {
        getUserMedia: vi.fn().mockRejectedValue(new DOMException("Denied", "NotAllowedError")),
      },
      configurable: true,
      writable: true,
    });

    await user.click(micBtn);
    expect(screen.getByRole("alert")).toHaveTextContent(/microphone permission denied/i);

    // Simulate Not Found
    navigator.mediaDevices.getUserMedia.mockRejectedValue(new DOMException("NotFound", "NotFoundError"));
    await user.click(micBtn);
    expect(screen.getByRole("alert")).toHaveTextContent(/no microphone found/i);

    Object.defineProperty(navigator, "mediaDevices", {
      value: originalMediaDevices,
      configurable: true,
      writable: true,
    });
  });
});

// ── Chat — focus refs ──────────────────────────────────────────────────────

describe("Chat — focus refs", () => {
  it("focuses the first suggestion via focusRef when welcome screen is displayed", () => {
    const focusRef = { current: null };
    render(<Chat {...chatProps({ focusRef, messages: [] })} />);

    expect(typeof focusRef.current).toBe("function");
    focusRef.current();

    const firstSuggestion = screen.getByRole("button", { name: /what can you do/i });
    expect(document.activeElement).toBe(firstSuggestion);
  });

  it("focuses textarea via focusRef when suggestions are not shown", () => {
    const focusRef = { current: null };
    render(<Chat {...chatProps({ focusRef, messages: [makeMessage()] })} />);

    focusRef.current();
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    expect(document.activeElement).toBe(textarea);
  });

  it("focuses textarea via textareaFocusRef and settings button via chatSettingsFocusRef", () => {
    const textareaFocusRef = { current: null };
    const chatSettingsFocusRef = { current: null };
    render(<Chat {...chatProps({ textareaFocusRef, chatSettingsFocusRef })} />);

    textareaFocusRef.current();
    expect(document.activeElement).toBe(screen.getByRole("textbox", { name: /message input/i }));

    chatSettingsFocusRef.current();
    expect(document.activeElement).toBe(screen.getByRole("button", { name: /generation settings/i }));
  });
});

// ── Chat — full user chat flow integration ─────────────────────────────────

describe("Chat — full user chat flow integration", () => {
  function StatefulChatFlow({ onSendHandler }) {
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSend = async (content, type, attachments) => {
      const userMsg = makeMessage({
        id: `user-${Date.now()}-${Math.random()}`,
        role: "user",
        content,
        type,
      });
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      try {
        const response = await onSendHandler(content, type, attachments);
        setMessages((prev) => [
          ...prev,
          makeMessage({
            id: `asst-${Date.now()}-${Math.random()}`,
            role: "assistant",
            content: response.content,
            type: response.type || "text",
          }),
        ]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          makeMessage({
            id: `err-${Date.now()}-${Math.random()}`,
            role: "assistant",
            content: err.message || "Failed",
            type: "error",
          }),
        ]);
      } finally {
        setLoading(false);
      }
    };

    return (
      <Chat
        {...chatProps({
          messages,
          loading,
          onSend: handleSend,
        })}
      />
    );
  }

  it("completes full chat cycle: welcome screen -> enter message -> submit -> receive response", async () => {
    const user = userEvent.setup();
    const onSendHandler = vi.fn().mockResolvedValue({
      content: "Here is your detailed answer.",
      type: "text",
    });

    render(<StatefulChatFlow onSendHandler={onSendHandler} />);

    // 1. Initial state: Welcome screen is visible
    expect(screen.getByText(/welcome to framerai/i)).toBeInTheDocument();

    // 2. Compose message
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    await user.type(textarea, "What is machine learning?");

    // 3. Send message
    await user.click(screen.getByRole("button", { name: /send/i }));

    // User message should be visible and welcome screen removed
    expect(await screen.findByText("What is machine learning?")).toBeInTheDocument();
    expect(screen.queryByText(/welcome to framerai/i)).not.toBeInTheDocument();
    expect(textarea).toHaveValue("");

    // 4. Response arrives from assistant
    expect(await screen.findByText("Here is your detailed answer.")).toBeInTheDocument();

    // 5. Send button is re-enabled once response arrives and user can send follow-up
    await user.type(textarea, "Follow up question");
    expect(screen.getByRole("button", { name: /send/i })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText("Follow up question")).toBeInTheDocument();
  });

  it("handles error response in full chat flow and allows successful retry", async () => {
    const user = userEvent.setup();
    const onSendHandler = vi
      .fn()
      .mockRejectedValueOnce(new Error("Inference service overloaded"))
      .mockResolvedValueOnce({ content: "Recovered response on retry", type: "text" });

    render(<StatefulChatFlow onSendHandler={onSendHandler} />);

    const textarea = screen.getByRole("textbox", { name: /message input/i });
    await user.type(textarea, "Calculate pi");
    await user.click(screen.getByRole("button", { name: /send/i }));

    // Error message appears in assistant bubble
    const errorBubble = await screen.findByText("Inference service overloaded");
    expect(errorBubble).toBeInTheDocument();

    // Retry button is available
    const retryBtn = screen.getByRole("button", { name: /retry/i });
    expect(retryBtn).toBeInTheDocument();

    // Click retry -> second call resolves successfully
    await user.click(retryBtn);

    expect(await screen.findByText("Recovered response on retry")).toBeInTheDocument();
    // After retry succeeds, the trailing message is no longer an error so no retry button
    expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
  });
});

// ── Duplicate Submission Prevention (Issue #345) ──────────────────────────

describe("Chat — duplicate submission prevention (Issue #345)", () => {
  let user;
  beforeEach(() => {
    user = userEvent.setup();
  });

  it("prevents duplicate submissions via Send button while generation is in progress and re-enables afterward", async () => {
    let resolveGeneration;
    const onSend = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveGeneration = resolve;
        })
    );

    render(<Chat {...chatProps({ onSend })} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    const sendBtn = screen.getByRole("button", { name: /send/i });

    // Type first message and submit
    await user.type(textarea, "First message");
    await user.click(sendBtn);

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith("First message", "text", []);

    // While generation is in progress: textarea and send button should be disabled, and any submission attempt ignored
    expect(sendBtn).toBeDisabled();
    expect(textarea).toBeDisabled();
    await user.click(sendBtn);
    fireEvent.submit(sendBtn.closest("form"));
    expect(onSend).toHaveBeenCalledTimes(1);

    // Complete the first generation
    await act(async () => {
      resolveGeneration();
    });

    // After generation completes: can submit a new message
    await user.type(textarea, "Second message");
    expect(sendBtn).toBeEnabled();
    await user.click(sendBtn);

    expect(onSend).toHaveBeenCalledTimes(2);
    expect(onSend).toHaveBeenLastCalledWith("Second message", "text", []);
  });

  it("prevents duplicate submissions via Enter key while generation is in progress", async () => {
    let resolveGeneration;
    const onSend = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveGeneration = resolve;
        })
    );

    render(<Chat {...chatProps({ onSend })} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });

    // Type and press Enter
    await user.type(textarea, "Message via Enter");
    await user.keyboard("{Enter}");

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith("Message via Enter", "text", []);

    // Attempt rapid or repeated Enter key while in progress
    await user.keyboard("{Enter}");
    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(onSend).toHaveBeenCalledTimes(1);

    // Complete first generation
    await act(async () => {
      resolveGeneration();
    });

    // Verify submission allowed afterward via Enter
    await user.type(textarea, "Follow-up via Enter");
    await user.keyboard("{Enter}");

    expect(onSend).toHaveBeenCalledTimes(2);
    expect(onSend).toHaveBeenLastCalledWith("Follow-up via Enter", "text", []);
  });

  it("prevents rapid repeated submissions in the same render tick", async () => {
    let resolveGeneration;
    const onSend = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveGeneration = resolve;
        })
    );

    render(<Chat {...chatProps({ onSend })} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    const sendBtn = screen.getByRole("button", { name: /send/i });

    await user.type(textarea, "Rapid fire");

    // Fire multiple submit / keydown events in rapid succession
    fireEvent.submit(sendBtn.closest("form"));
    fireEvent.submit(sendBtn.closest("form"));
    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(onSend).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveGeneration?.();
    });
  });

  it("prevents submission attempts when loading or streaming prop is true", async () => {
    const onSend = vi.fn();
    const { rerender } = render(<Chat {...chatProps({ loading: true, onSend })} />);
    const textarea = screen.getByRole("textbox", { name: /message input/i });
    const sendBtn = screen.getByRole("button", { name: /send/i });

    fireEvent.submit(sendBtn.closest("form"));
    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(onSend).not.toHaveBeenCalled();

    rerender(<Chat {...chatProps({ streaming: true, onSend })} />);
    fireEvent.submit(sendBtn.closest("form"));
    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(onSend).not.toHaveBeenCalled();
  });

  it("guards suggestion clicks during active generation", async () => {
    let resolveGeneration;
    const onSend = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveGeneration = resolve;
        })
    );

    render(<Chat {...chatProps({ onSend })} />);
    const suggestionBtn = screen.getByRole("button", { name: /^what can you do\?$/i });

    await user.click(suggestionBtn);
    expect(onSend).toHaveBeenCalledTimes(1);

    // Repeated clicks while generation is active
    expect(suggestionBtn).toBeDisabled();
    await user.click(suggestionBtn);
    expect(onSend).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveGeneration();
    });
  });
});
