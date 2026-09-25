/**
 * Regression tests for Issue #354:
 * Prevent stale generation responses from mutating replaced conversation state.
 *
 * Scenarios covered:
 * 1. Start generation in Conversation A via REST, switch to Conversation B (with existing messages)
 *    before it completes. Stale REST response must NOT overwrite Conversation B's messages.
 * 2. Start generation in Conversation A via REST, switch to Conversation B. Late REST error must
 *    NOT mutate Conversation B, and Conversation A in background store must receive the error.
 * 3. Start generation in Conversation A via WebSocket, switch to Conversation B. Stale streaming
 *    chunks and completed frames for A must NOT mutate Conversation B.
 * 4. Stale WebSocket stream frame without conversationId must NOT mutate Conversation B.
 * 5. Stale WebSocket error frame for Conversation A must NOT mutate Conversation B.
 * 6. Switching conversations during generation does not leave broken loading/streaming state.
 * 7. Deleting or clearing conversation during generation safely ignores late responses.
 * 8. Valid generation updates Conversation A when A remains active.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

describe("Stale Generation Response Isolation (Issue #354)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  async function setupHook(options = {}) {
    const { wsReadyState = 1 } = options;
    let mockStreamHandler = null;
    let mockTypingHandler = null;
    let mockErrorHandler = null;
    let mockCloseHandler = null;
    let resolveRestSendMessage = null;
    let rejectRestSendMessage = null;

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
      send() {}
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
      stream: (data) => mockStreamHandler?.(data),
      typing: (data) => mockTypingHandler?.(data),
      error: (data) => mockErrorHandler?.(data),
      close: () => mockCloseHandler?.(),
      resolveRest: (val) => resolveRestSendMessage?.(val),
      rejectRest: (err) => rejectRestSendMessage?.(err),
    };
  }

  // ─── 1. REST: Late response must NOT overwrite Conversation B's messages ──────
  it("REGRESSION: late REST response from Conversation A must NOT overwrite Conversation B's messages", async () => {
    let resolveRestA = null;
    const { result, act, waitFor, mockApi } = await setupHook({ wsReadyState: 0 });

    // Step 1: Create Conversation B first with an existing message
    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;

    mockApi.sendMessage.mockImplementation((convId, content) => {
      if (content === "Hello B") {
        return Promise.resolve({
          content: "Original B message",
          type: "text",
          metadata: {},
        });
      }
      return new Promise((resolve) => {
        resolveRestA = resolve;
      });
    });

    await act(async () => {
      await result.current.sendMessage("Hello B", "text", []);
    });
    await waitFor(() => {
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[1].content).toBe("Original B message");
    });

    // Step 2: Create Conversation A and start a pending generation in A
    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;
    expect(convAId).not.toBe(convBId);

    await act(async () => {
      result.current.sendMessage("Request from A", "text", []);
    });
    await waitFor(() => {
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.loading).toBe(true);
    });

    // Step 3: Switch back to Conversation B while A's generation is in flight
    await act(async () => {
      await result.current.selectConversation(convBId);
    });
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages[1].content).toBe("Original B message");

    // Step 4: Deliver late REST response belonging to Conversation A
    await act(async () => {
      resolveRestA({
        content: "Completed response for A",
        type: "text",
        metadata: {},
      });
    });

    // Step 5: Verify Conversation B's message was NOT overwritten by Conversation A!
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].content).toBe("Original B message");

    // Step 6: Verify Conversation A was properly updated in background store
    const convA = result.current.conversations.find((c) => c.id === convAId);
    expect(convA).toBeDefined();
    expect(convA.messages).toHaveLength(2);
    expect(convA.messages[1].content).toBe("Completed response for A");

    // Step 7: Verify loading state is cleaned up
    expect(result.current.loading).toBe(false);
  });

  // ─── 2. REST: Late failure must NOT mutate Conversation B ─────────────────────
  it("REGRESSION: late REST error from Conversation A must NOT mutate Conversation B", async () => {
    const { result, act, waitFor, rejectRest } = await setupHook({ wsReadyState: 0 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    await act(async () => {
      result.current.sendMessage("Failing request from A", "text", []);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    // Switch to Conversation B
    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;
    expect(result.current.messages).toHaveLength(0);

    // Deliver late error for Conversation A
    await act(async () => {
      rejectRest(new Error("Network timeout in A"));
    });

    // Conversation B must NOT have error message bubble
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages).toHaveLength(0);

    // Conversation A receives the error in conversations store
    const convA = result.current.conversations.find((c) => c.id === convAId);
    expect(convA).toBeDefined();
    expect(convA.messages).toHaveLength(2);
    expect(convA.messages[1].type).toBe("error");
    expect(convA.messages[1].content).toBe("Network timeout in A");

    expect(result.current.loading).toBe(false);
  });

  // ─── 3. WebSocket: Stale chunk & completed response must NOT mutate Conv B ────
  it("REGRESSION: stale streaming chunks and completed responses must NOT mutate replaced conversation", async () => {
    const { result, act, waitFor, stream, typing } = await setupHook({ wsReadyState: 1 });

    // Step 1: Start generation in Conversation A
    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    await act(async () => {
      result.current.sendMessage("Streaming request from A", "text", []);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    await act(async () => {
      typing({ conversationId: convAId });
    });
    expect(result.current.streaming).toBe(true);

    // Step 2: Switch to Conversation B before completion
    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;
    expect(convBId).not.toBe(convAId);
    expect(result.current.messages).toHaveLength(0);

    // Step 3: Deliver stale streaming chunk belonging to Conversation A
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convAId,
        content: "Stale chunk from A",
        done: false,
        responseType: "text",
      });
    });

    // Step 4: Verify Conversation B is NOT modified
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages).toHaveLength(0);

    // Step 5: Deliver stale completed response belonging to Conversation A
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convAId,
        content: "Completed response from A",
        done: true,
        responseType: "text",
      });
    });

    // Step 6: Verify Conversation B is STILL not modified
    expect(result.current.messages).toHaveLength(0);

    // Step 7: Verify Conversation A has the full response preserved
    const convA = result.current.conversations.find((c) => c.id === convAId);
    expect(convA).toBeDefined();
    expect(convA.messages).toHaveLength(2);
    expect(convA.messages[1].content).toBe("Completed response from A");

    // Step 8: Generation cleanup works (streaming turns off when all streams finish)
    expect(result.current.streaming).toBe(false);
  });

  // ─── 4. WebSocket: Stale chunk without conversationId must NOT mutate Conv B ─
  it("REGRESSION: stream frame without conversationId must NOT mutate newly active conversation when A was in flight", async () => {
    const { result, act, waitFor, stream, typing } = await setupHook({ wsReadyState: 1 });

    // Step 1: Create Conversation B first with an existing message
    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;

    await act(async () => {
      result.current.sendMessage("Hello B", "text", []);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convBId,
        content: "Original B message",
        done: true,
        responseType: "text",
      });
    });
    expect(result.current.messages[1].content).toBe("Original B message");

    // Step 2: Create Conversation A and start generation in A
    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;
    expect(convAId).not.toBe(convBId);

    await act(async () => {
      result.current.sendMessage("Request in A", "text", []);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    await act(async () => {
      typing({ conversationId: convAId });
    });

    // Step 3: Switch back to Conversation B while A is in flight
    await act(async () => {
      await result.current.selectConversation(convBId);
    });
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages[1].content).toBe("Original B message");

    // Step 4: Deliver stream frame without conversationId (belongs to in-flight A)
    await act(async () => {
      stream({
        type: "stream",
        content: "Legacy chunk from A",
        done: false,
        responseType: "text",
      });
    });

    // Conversation B must NOT receive the chunk
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].content).toBe("Original B message");

    // Conversation A must receive the chunk in background store
    const convA = result.current.conversations.find((c) => c.id === convAId);
    expect(convA).toBeDefined();
    expect(convA.messages[1].content).toBe("Legacy chunk from A");
  });

  // ─── 5. WebSocket: Stale error frame for Conversation A must NOT mutate Conv B ─
  it("REGRESSION: WebSocket error frame for Conversation A must NOT mutate newly active Conversation B", async () => {
    const { result, act, waitFor, error, stream, typing } = await setupHook({ wsReadyState: 1 });

    // Step 1: Create Conversation B first with an existing message
    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;

    await act(async () => {
      result.current.sendMessage("Hello B", "text", []);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convBId,
        content: "Untouched message in B",
        done: true,
        responseType: "text",
      });
    });
    expect(result.current.messages[1].content).toBe("Untouched message in B");

    // Step 2: Create Conversation A and start generation
    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    await act(async () => {
      result.current.sendMessage("Request in A that fails", "text", []);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    await act(async () => {
      typing({ conversationId: convAId });
    });

    // Step 3: Switch back to Conversation B
    await act(async () => {
      await result.current.selectConversation(convBId);
    });
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages[1].content).toBe("Untouched message in B");

    // Step 4: Deliver WS error for Conversation A
    await act(async () => {
      error({
        type: "error",
        conversationId: convAId,
        message: "Model rate limit exceeded in A",
      });
    });

    // Conversation B must be untouched
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].content).toBe("Untouched message in B");

    // Conversation A must record the error
    const convA = result.current.conversations.find((c) => c.id === convAId);
    expect(convA).toBeDefined();
    expect(convA.messages[1].type).toBe("error");
    expect(convA.messages[1].content).toBe("Model rate limit exceeded in A");
    expect(result.current.streaming).toBe(false);
  });

  // ─── 5. Normal single-conversation generation works when A remains active ─────
  it("preserves normal generation updates when conversation remains active", async () => {
    const { result, act, waitFor, stream, typing } = await setupHook({ wsReadyState: 1 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      result.current.sendMessage("Normal prompt", "text", []);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    await act(async () => {
      typing({ conversationId: convId });
    });
    expect(result.current.streaming).toBe(true);

    // Stream chunk
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "Normal chunk 1",
        done: false,
        responseType: "text",
      });
    });
    expect(result.current.messages[1].content).toBe("Normal chunk 1");

    // Complete
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "Normal complete",
        done: true,
        responseType: "text",
      });
    });
    expect(result.current.messages[1].content).toBe("Normal complete");
    expect(result.current.streaming).toBe(false);
  });

  // ─── 6. Deleted/replaced conversation ignores stale responses safely ──────────
  it("cleans up generation state when active conversation is deleted during generation", async () => {
    const { result, act, waitFor, stream, typing } = await setupHook({ wsReadyState: 1 });

    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    await act(async () => {
      result.current.sendMessage("Prompt before delete", "text", []);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    await act(async () => {
      typing({ conversationId: convAId });
    });
    expect(result.current.streaming).toBe(true);

    // Delete conversation A while streaming
    await act(async () => {
      await result.current.deleteConversation(convAId);
    });

    // Streaming state must be cleaned up
    expect(result.current.streaming).toBe(false);

    // Stale late stream frame arriving for deleted convAId must not crash or recreate it
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convAId,
        content: "Late ghost chunk",
        done: true,
        responseType: "text",
      });
    });

    expect(result.current.conversations.find((c) => c.id === convAId)).toBeUndefined();
    expect(result.current.streaming).toBe(false);
  });

  // ─── 7. Concurrent REST requests keep loading active until all finish ─────────
  it("keeps loading state true until all concurrent REST generations finish", async () => {
    let resolveA = null;
    let resolveB = null;

    const mockApi = {
      createConversation: vi.fn(() =>
        Promise.resolve({
          id: `conv-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          title: "New Chat",
          messages: [],
        })
      ),
      listConversations: vi.fn(() => Promise.resolve([])),
      getConversation: vi.fn((id) => Promise.resolve({ id, title: "Chat " + id, messages: [] })),
      deleteConversation: vi.fn(() => Promise.resolve()),
      sendMessage: vi.fn(() => {
        return new Promise((resolve) => {
          if (!resolveA) resolveA = resolve;
          else resolveB = resolve;
        });
      }),
    };

    const MockWebSocketClient = class {
      constructor() {
        this.ws = { readyState: 0 }; // REST mode
        this.listeners = new Map();
      }
      connect() { return Promise.resolve(); }
      on(type, handler) { this.listeners.set(type, handler); return () => {}; }
      send() {}
      disconnect() {}
    };

    vi.doMock("../services/api", () => ({ api: mockApi }));
    vi.doMock("../services/websocket", () => ({ WebSocketClient: MockWebSocketClient }));

    const { renderHook, act, waitFor } = await import("@testing-library/react");
    const { useChat } = await import("../hooks/useChat?t=" + Date.now());
    const { result } = renderHook(() => useChat({}));

    await waitFor(() => expect(result.current.loadingConversations).toBe(false));

    // Conv A starts REST generation
    await act(async () => { await result.current.createConversation(); });
    const convAId = result.current.activeConversation;
    expect(convAId).toBeDefined();
    await act(async () => { result.current.sendMessage("A message", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    expect(result.current.loading).toBe(true);

    // Conv B starts REST generation
    await act(async () => { await result.current.createConversation(); });
    const convBId = result.current.activeConversation;
    expect(convBId).not.toBe(convAId);
    await act(async () => { result.current.sendMessage("B message", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    expect(result.current.loading).toBe(true);

    // A finishes first in background
    await act(async () => {
      resolveA({ content: "Response A", type: "text", metadata: {} });
    });

    // Loading MUST remain true because B is still in flight!
    expect(result.current.loading).toBe(true);

    // B finishes
    await act(async () => {
      resolveB({ content: "Response B", type: "text", metadata: {} });
    });

    // Now loading should be false
    expect(result.current.loading).toBe(false);
  });
});
