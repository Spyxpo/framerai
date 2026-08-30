import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
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
});
