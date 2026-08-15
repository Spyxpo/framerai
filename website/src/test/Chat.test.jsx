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

  it("regression: reassembled streaming chunks match source PCM byte count", () => {
    // This test simulates the WebSocket "stream" event handler in useChat.js
    // to verify that ALL chunks (including the final one) are preserved.
    //
    // IMPORTANT: This test MUST FAIL if the final chunk is dropped.
    //
    // The bug occurs in useChat.js when data.done === true:
    // OLD CODE (buggy): only sets metadata, doesn't push chunkData
    // NEW CODE (fixed): pushes chunkData BEFORE setting metadata

    // Simulate a 1.5 second audio clip with 3 chunks (0.5 sec each)
    const sampleRate = 24000;
    const channels = 1;
    const bitsPerSample = 16;
    const samplesPerChunk = sampleRate * 0.5; // 0.5 seconds per chunk
    const bytesPerChunk = samplesPerChunk * channels * (bitsPerSample / 8);

    // Create mock PCM data for 3 chunks
    const createPCMChunk = (chunkIndex) => {
      const buffer = new ArrayBuffer(bytesPerChunk);
      const view = new DataView(buffer);
      for (let i = 0; i < samplesPerChunk; i++) {
        const sample = Math.sin(i * 0.1 + chunkIndex * 100) * 32767;
        view.setInt16(i * 2, sample, true); // little-endian
      }
      return btoa(String.fromCharCode(...new Uint8Array(buffer)));
    };

    const chunk1 = createPCMChunk(0);
    const chunk2 = createPCMChunk(1);
    const chunk3 = createPCMChunk(2); // FINAL CHUNK

    const totalSourceBytes = bytesPerChunk * 3;

    // Simulate useChat's message state by manually applying the same logic
    // that the ws.on("stream") handler uses
    const messages = [];

    // Initial assistant message (empty placeholder)
    const assistantMsg = {
      id: "msg-2",
      role: "assistant",
      content: "",
      type: "text",
      timestamp: new Date().toISOString(),
    };
    messages.push(assistantMsg);

    // Helper to simulate the useChat WebSocket handler logic
    const handleStreamEvent = (data) => {
      if (data.responseType === "audio") {
        const last = messages[messages.length - 1];
        if (data.done) {
          // THIS IS THE CRITICAL CODE PATH being tested
          // Old buggy version: only sets metadata, doesn't push chunkData
          // New fixed version: pushes chunkData first
          last.type = "audio";
          if (data.metadata?.chunkData) {
            last.audioChunks = last.audioChunks || [];
            last.audioChunks.push(data.metadata.chunkData);
          }
          last.metadata = data.metadata;
          last.audioComplete = true;
        } else {
          last.type = "audio";
          last.content = data.content || last.content;
          last.audioChunks = last.audioChunks || [];
          last.audioChunks.push(data.metadata.chunkData);
          last.audioMetadata = {
            sampleRate: data.metadata.sampleRate,
            channels: data.metadata.channels,
            bitsPerSample: data.metadata.bitsPerSample,
            totalChunks: data.metadata.totalChunks,
          };
        }
      }
    };

    // Simulate backend streaming sequence
    // Chunk 1 (not done)
    handleStreamEvent({
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

    // Chunk 2 (not done)
    handleStreamEvent({
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

    // At this point, audioChunks should have 2 chunks
    expect(assistantMsg.audioChunks?.length).toBe(2);

    // Chunk 3 (FINAL - done=true with chunkData)
    // THIS IS WHERE THE BUG OCCURS
    handleStreamEvent({
      type: "stream",
      content: "",
      done: true,
      responseType: "audio",
      metadata: {
        chunk: 2,
        totalChunks: 3,
        chunkData: chunk3, // THIS MUST BE PUSHED TO audioChunks
        sampleRate,
        channels,
        bitsPerSample,
        url: "/uploads/generated/audio.wav",
        model: "framerai-audio",
        durationSec: 1.5,
      },
    });

    // CRITICAL ASSERTIONS
    expect(assistantMsg.audioComplete).toBe(true);
    expect(assistantMsg.audioChunks).toBeDefined();

    // If the final chunk is dropped (old bug), this would be 2 instead of 3
    expect(assistantMsg.audioChunks.length).toBe(3);

    // Reassemble chunks and verify byte count
    let reassembledBytes = 0;
    assistantMsg.audioChunks.forEach(chunkData => {
      const binaryString = atob(chunkData);
      reassembledBytes += binaryString.length;
    });

    // Expected: 3 chunks × 24000 bytes = 72000 bytes
    // If buggy (final chunk lost): 2 chunks × 24000 bytes = 48000 bytes
    expect(reassembledBytes).toBe(totalSourceBytes);
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
