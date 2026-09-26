/**
 * Regression tests for Issue #366:
 * Prevent duplicate assistant messages when generation completion is processed multiple times.
 *
 * Requirements verified:
 * 1. One normal generation produces exactly one assistant message.
 * 2. Processing the same completion twice does NOT create two assistant messages.
 * 3. Duplicate completion/stream-finalization events are ignored safely.
 * 4. The original assistant response content is preserved.
 * 5. Normal streaming still produces exactly one final assistant message.
 * 6. Separate generation requests still produce separate assistant messages.
 * 7. Conversation/message persistence does not contain duplicates after completion.
 * 8. Existing error/retry behavior remains functional.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { sanitizeConversation, loadConversationsFromStorage, saveConversationsToStorage } from "../utils/storage";

describe("Duplicate Assistant Message Prevention (Issue #366)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
    localStorage.clear();
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
    localStorage.clear();
  });

  async function setupHook(options = {}) {
    const { wsReadyState = 1 } = options;
    let mockStreamHandler = null;
    let mockTypingHandler = null;
    let mockErrorHandler = null;
    let mockCloseHandler = null;
    let resolveRestSendMessage = null;
    let rejectRestSendMessage = null;
    const sentWsMessages = [];

    const MockWebSocketClient = class {
      constructor() {
        this.ws = { readyState: wsReadyState };
        this.listeners = new Map();
      }
      connect() {
        return Promise.resolve();
      }
      on(type, handler) {
        this.listeners.set(type, handler);
        if (type === "stream") mockStreamHandler = handler;
        if (type === "typing") mockTypingHandler = handler;
        if (type === "error") mockErrorHandler = handler;
        if (type === "close") mockCloseHandler = handler;
        return () => {};
      }
      send(payload) {
        sentWsMessages.push(payload);
      }
      disconnect() {}
    };

    const convStore = new Map();

    const mockApi = {
      createConversation: vi.fn(() => {
        const id = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        const c = { id, title: "New Chat", messages: [] };
        convStore.set(id, c);
        return Promise.resolve(c);
      }),
      listConversations: vi.fn(() => Promise.resolve([])),
      getConversation: vi.fn((id) => {
        const c = convStore.get(id) || { id, title: "Chat " + id, messages: [] };
        return Promise.resolve(c);
      }),
      deleteConversation: vi.fn((id) => {
        convStore.delete(id);
        return Promise.resolve();
      }),
      sendMessage: vi.fn(() => {
        return new Promise((resolve, reject) => {
          resolveRestSendMessage = resolve;
          rejectRestSendMessage = reject;
        });
      }),
    };

    vi.doMock("../services/api", () => ({ api: mockApi }));
    vi.doMock("../services/websocket", () => ({ WebSocketClient: MockWebSocketClient }));

    const { renderHook, act, waitFor } = await import("@testing-library/react");
    const { useChat } = await import("../hooks/useChat?t=" + Date.now());
    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    return {
      result,
      act,
      waitFor,
      mockApi,
      convStore,
      sentWsMessages,
      stream: (data) => mockStreamHandler?.(data),
      typing: (data) => mockTypingHandler?.(data),
      error: (data) => mockErrorHandler?.(data),
      close: () => mockCloseHandler?.(),
      resolveRest: (val) => resolveRestSendMessage?.(val),
      rejectRest: (err) => rejectRestSendMessage?.(err),
    };
  }

  // ─── 1. One normal generation produces exactly one assistant message ──────────
  it("Requirement 1: one normal generation produces exactly one assistant message", async () => {
    const { result, act, stream } = await setupHook({ wsReadyState: 1 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("Tell me a story");
    });

    // Check placeholder created
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].role).toBe("user");
    expect(result.current.messages[1].role).toBe("assistant");

    // Deliver normal streaming completion
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "Once upon a time in a digital realm...",
        done: true,
      });
    });

    const assistantMessages = result.current.messages.filter((m) => m.role === "assistant");
    expect(assistantMessages).toHaveLength(1);
    expect(assistantMessages[0].content).toBe("Once upon a time in a digital realm...");
    expect(result.current.streaming).toBe(false);
  });

  // ─── 2. Processing the same completion twice does NOT create two assistant messages ───
  it("Requirement 2: processing the same completion twice does NOT create two assistant messages", async () => {
    const { result, act, stream } = await setupHook({ wsReadyState: 1 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("Calculate 2+2");
    });

    // First completion
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "4",
        done: true,
      });
    });

    expect(result.current.messages.filter((m) => m.role === "assistant")).toHaveLength(1);

    // Second (duplicate) completion event processed
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "4",
        done: true,
      });
    });

    const assistantMessages = result.current.messages.filter((m) => m.role === "assistant");
    expect(assistantMessages).toHaveLength(1);
    expect(assistantMessages[0].content).toBe("4");
    expect(result.current.messages).toHaveLength(2); // 1 user + 1 assistant
  });

  // ─── 3. Duplicate completion/stream-finalization events are ignored safely ──────
  it("Requirement 3: duplicate completion/stream-finalization events are ignored safely", async () => {
    const { result, act, stream } = await setupHook({ wsReadyState: 1 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    // Test with audio streaming duplicate finalization
    await act(async () => {
      await result.current.sendMessage("Generate speech", "audio");
    });

    // First audio done frame
    await act(async () => {
      stream({
        type: "stream",
        responseType: "audio",
        conversationId: convId,
        content: "Audio speech",
        done: true,
        metadata: {
          chunk: 0,
          totalChunks: 1,
          chunkData: "base64audiochunk1",
          url: "/audio/test.wav",
        },
      });
    });

    let assistantMsg = result.current.messages.find((m) => m.role === "assistant");
    expect(assistantMsg).toBeDefined();
    expect(assistantMsg.audioChunks).toEqual(["base64audiochunk1"]);
    expect(assistantMsg.audioComplete).toBe(true);

    // Duplicate terminal audio frame with chunkData should be safely ignored
    await act(async () => {
      stream({
        type: "stream",
        responseType: "audio",
        conversationId: convId,
        content: "Audio speech",
        done: true,
        metadata: {
          chunk: 0,
          totalChunks: 1,
          chunkData: "base64audiochunk1",
          url: "/audio/test.wav",
        },
      });
    });

    assistantMsg = result.current.messages.find((m) => m.role === "assistant");
    expect(result.current.messages.filter((m) => m.role === "assistant")).toHaveLength(1);
    // audioChunks must NOT be duplicated to ["base64audiochunk1", "base64audiochunk1"]
    expect(assistantMsg.audioChunks).toEqual(["base64audiochunk1"]);
    expect(result.current.streaming).toBe(false);
  });

  // ─── 4. The original assistant response content is preserved ──────────────────
  it("Requirement 4: original assistant response content is preserved against duplicate completion with different/empty payload", async () => {
    const { result, act, stream } = await setupHook({ wsReadyState: 1 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("Explain recursion");
    });

    // First completion arrives with correct content
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "Recursion is a process in which a function calls itself as a subroutine.",
        done: true,
      });
    });

    // A duplicate completion event arrives with empty content or altered content
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "",
        done: true,
      });
    });

    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "Overwritten duplicate payload",
        done: true,
      });
    });

    const assistantMsg = result.current.messages.find((m) => m.role === "assistant");
    expect(assistantMsg.content).toBe("Recursion is a process in which a function calls itself as a subroutine.");
  });

  // ─── 5. Normal streaming still produces exactly one final assistant message ────
  it("Requirement 5: normal streaming still produces exactly one final assistant message", async () => {
    const { result, act, stream } = await setupHook({ wsReadyState: 1 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("Count to three");
    });

    // Streaming chunk 1
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "One",
        done: false,
      });
    });
    expect(result.current.messages.find((m) => m.role === "assistant").content).toBe("One");

    // Streaming chunk 2
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "One, two",
        done: false,
      });
    });
    expect(result.current.messages.find((m) => m.role === "assistant").content).toBe("One, two");

    // Terminal chunk
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "One, two, three.",
        done: true,
      });
    });

    const assistantMessages = result.current.messages.filter((m) => m.role === "assistant");
    expect(assistantMessages).toHaveLength(1);
    expect(assistantMessages[0].content).toBe("One, two, three.");
    expect(result.current.streaming).toBe(false);
  });

  // ─── 6. Separate generation requests still produce separate assistant messages ─
  it("Requirement 6: separate generation requests still produce separate assistant messages and late completions do not cross turns", async () => {
    const { result, act, stream } = await setupHook({ wsReadyState: 1 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    // Generation 1
    await act(async () => {
      await result.current.sendMessage("First question");
    });
    const asst1Id = result.current.messages[1].id;
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: asst1Id,
        content: "First answer",
        done: true,
      });
    });

    expect(result.current.messages).toHaveLength(2); // user 1 + asst 1
    expect(result.current.messages[1].content).toBe("First answer");

    // Generation 2
    await act(async () => {
      await result.current.sendMessage("Second question");
    });
    const asst2Id = result.current.messages[3].id;
    expect(result.current.messages).toHaveLength(4); // user 1, asst 1, user 2, asst 2 (empty)

    // Now simulate a late duplicate completion event from Generation 1 arriving during Generation 2
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: asst1Id,
        content: "First answer duplicate",
        done: true,
      });
    });

    // Turn 2 placeholder should NOT be corrupted by Turn 1's duplicate completion
    const asst2Before = result.current.messages[3];
    expect(asst2Before.content).toBe(""); // untouched!

    // Now Generation 2 completes normally
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: asst2Id,
        content: "Second answer",
        done: true,
      });
    });

    const assistantMessages = result.current.messages.filter((m) => m.role === "assistant");
    expect(assistantMessages).toHaveLength(2);
    expect(assistantMessages[0].content).toBe("First answer");
    expect(assistantMessages[1].content).toBe("Second answer");
  });

  // ─── 7. Conversation/message persistence does not contain duplicates after completion ─
  it("Requirement 7: conversation/message persistence does not contain duplicates after completion", async () => {
    // Test storage sanitization directly with duplicate message IDs
    const duplicateConversation = {
      id: "conv-storage-test",
      title: "Storage Test",
      updatedAt: new Date().toISOString(),
      messages: [
        { id: "u-1", role: "user", content: "Hello", type: "text" },
        { id: "a-1", role: "assistant", content: "Hi there!", type: "text" },
        // Simulate duplicate assistant message record
        { id: "a-1", role: "assistant", content: "Hi there!", type: "text" },
      ],
    };

    const sanitized = sanitizeConversation(duplicateConversation);
    expect(sanitized.messages).toHaveLength(2);
    expect(sanitized.messages.filter((m) => m.role === "assistant")).toHaveLength(1);
    expect(sanitized.messages[1].id).toBe("a-1");

    // Test full save and load cycle
    saveConversationsToStorage([duplicateConversation], "conv-storage-test");
    const loaded = loadConversationsFromStorage();
    expect(loaded.conversations[0].messages).toHaveLength(2);
    expect(loaded.conversations[0].messages.filter((m) => m.role === "assistant")).toHaveLength(1);
  });

  // ─── 8. Existing error/retry behavior remains functional ──────────────────────
  it("Requirement 8: existing error/retry behavior remains functional", async () => {
    const { result, act, stream } = await setupHook({ wsReadyState: 1 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    // Send generation that fails with an error
    await act(async () => {
      await result.current.sendMessage("Faulty prompt");
    });

    await act(async () => {
      stream({
        type: "error",
        conversationId: convId,
        message: "Model execution failed",
      });
    });

    let assistantMessages = result.current.messages.filter((m) => m.role === "assistant");
    expect(assistantMessages).toHaveLength(1);
    expect(assistantMessages[0].type).toBe("error");
    expect(assistantMessages[0].content).toBe("Model execution failed");
    expect(result.current.streaming).toBe(false);

    // Duplicate error event arrives — should be safely ignored and not duplicate message
    await act(async () => {
      stream({
        type: "error",
        conversationId: convId,
        message: "Duplicate error event",
      });
    });

    assistantMessages = result.current.messages.filter((m) => m.role === "assistant");
    expect(assistantMessages).toHaveLength(1);
    expect(assistantMessages[0].content).toBe("Model execution failed");

    // Retry: send message again
    await act(async () => {
      await result.current.sendMessage("Faulty prompt");
    });

    expect(result.current.messages).toHaveLength(4); // user 1, err 1, user 2, asst 2

    // Retry completes successfully
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "Successful response on retry",
        done: true,
      });
    });

    assistantMessages = result.current.messages.filter((m) => m.role === "assistant");
    expect(assistantMessages).toHaveLength(2);
    expect(assistantMessages[0].type).toBe("error");
    expect(assistantMessages[0].content).toBe("Model execution failed");
    expect(assistantMessages[1].type).toBe("text");
    expect(assistantMessages[1].content).toBe("Successful response on retry");
  });
});
