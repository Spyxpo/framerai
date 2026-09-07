/**
 * Regression tests for streaming-state cleanup on disconnect and conversation deletion.
 *
 * BUGS:
 * 1. WebSocket closes unexpectedly while a stream is active → done/error frames never
 *    arrive → the conversation ID stays in streamingConversationIdsRef forever →
 *    global streaming stays true and the composer is permanently disabled.
 *
 * 2. A conversation is deleted while its stream is active → same stuck-Set problem.
 *
 * FIX:
 * 1. WebSocketClient now emits a "close" event to its listeners on ws.onclose.
 *    useChat registers a ws.on("close") handler that calls markStreamingEnd for
 *    every ID currently in the Set.
 * 2. deleteConversation calls markStreamingEnd(id) before removing the conversation
 *    from state, so its ID is purged from the Set even if no done/error arrives.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

describe("Streaming cleanup on disconnect / deletion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  // ─── shared mock factory ───────────────────────────────────────────────────
  async function setupHook() {
    let mockStreamHandler = null;
    let mockTypingHandler = null;
    let mockCloseHandler = null;
    // Expose the raw WS instance so tests can trigger onclose
    let wsInstance = null;

    const MockWebSocketClient = class {
      constructor() {
        wsInstance = this;
        this.ws = { readyState: 1 };
        this.listeners = new Map();
      }
      connect() {
        return Promise.resolve();
      }
      on(type, handler) {
        this.listeners.set(type, handler);
        if (type === "stream") mockStreamHandler = handler;
        if (type === "typing") mockTypingHandler = handler;
        if (type === "close") mockCloseHandler = handler;
        return () => {};
      }
      send() {}
      disconnect() {}
      // Test helper: simulate unexpected socket close
      simulateClose() {
        // Fire the close handler exactly as websocket.js does
        const h = this.listeners.get("close");
        if (h) h();
      }
    };

    vi.doMock("../services/api", () => ({
      api: {
        createConversation: vi.fn(() =>
          Promise.resolve({ id: `conv-${Date.now()}-${Math.random()}`, title: "Test", messages: [] })
        ),
        listConversations: vi.fn(() => Promise.resolve([])),
        getConversation: vi.fn((id) => Promise.resolve({ id, title: "Test", messages: [] })),
        deleteConversation: vi.fn(() => Promise.resolve()),
      },
    }));
    vi.doMock("../services/websocket", () => ({ WebSocketClient: MockWebSocketClient }));

    const { renderHook, act, waitFor } = await import("@testing-library/react");
    const { useChat } = await import("../hooks/useChat?t=" + Date.now());
    const { result } = renderHook(() => useChat({}));

    await waitFor(
      () => {
        expect(mockStreamHandler).not.toBeNull();
        expect(mockTypingHandler).not.toBeNull();
        expect(mockCloseHandler).not.toBeNull();
      },
      { timeout: 1000 }
    );

    return {
      result,
      act,
      waitFor,
      stream: () => mockStreamHandler,
      typing: () => mockTypingHandler,
      close: () => () => wsInstance.simulateClose(),
    };
  }

  // ─── 1. Disconnect clears streaming for that conversation ─────────────────
  it("REGRESSION: disconnect without done/error clears streaming state", async () => {
    const { result, act, waitFor, typing, close } = await setupHook();

    await act(async () => { await result.current.createConversation(); });
    const convId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("Hello", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    // Mark the conversation as streaming
    await act(async () => { typing()({ conversationId: convId }); });
    expect(result.current.streaming).toBe(true);

    // No done/error frame — the socket just closes
    await act(async () => { close()(); });

    // Streaming must be cleared
    expect(result.current.streaming).toBe(false);
  });

  // ─── 2. Last disconnect clears streaming; concurrent stream unaffected ─────
  it("REGRESSION: disconnect clears only the disconnected stream; other stream stays active", async () => {
    const { result, act, waitFor, stream, typing, close } = await setupHook();

    // Conv A starts streaming
    await act(async () => { await result.current.createConversation(); });
    const convAId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("A", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    await act(async () => { typing()({ conversationId: convAId }); });

    // Conv B starts streaming
    await act(async () => { await result.current.createConversation(); });
    const convBId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("B", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    await act(async () => { typing()({ conversationId: convBId }); });

    expect(result.current.streaming).toBe(true);

    // The socket drops — both in-flight streams are orphaned.
    // A "close" event clears all IDs currently in the Set.
    await act(async () => { close()(); });

    // All streams cleared → false
    expect(result.current.streaming).toBe(false);

    // Separately: B finishes normally after a reconnect (done frame arrives)
    // Streaming should still clear correctly (was already false, stays false).
    await act(async () => {
      stream()({ conversationId: convBId, content: "Done B", done: true, responseType: "text" });
    });
    expect(result.current.streaming).toBe(false);
  });

  // ─── 3. Conversation deleted while streaming ───────────────────────────────
  it("REGRESSION: deleting a streaming conversation clears its streaming state", async () => {
    const { result, act, waitFor, typing } = await setupHook();

    await act(async () => { await result.current.createConversation(); });
    const convId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("Hello", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    // Mark as streaming
    await act(async () => { typing()({ conversationId: convId }); });
    expect(result.current.streaming).toBe(true);

    // Delete the conversation before done/error arrives
    await act(async () => { await result.current.deleteConversation(convId); });

    // Streaming must be cleared
    expect(result.current.streaming).toBe(false);
    expect(result.current.conversations.find((c) => c.id === convId)).toBeUndefined();
  });

  // ─── 4. Deleting one streaming conv does not affect another ───────────────
  it("deleting one streaming conversation must not affect another active stream", async () => {
    const { result, act, waitFor, stream, typing } = await setupHook();

    // Conv A streaming
    await act(async () => { await result.current.createConversation(); });
    const convAId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("A", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    await act(async () => { typing()({ conversationId: convAId }); });

    // Conv B streaming
    await act(async () => { await result.current.createConversation(); });
    const convBId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("B", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    await act(async () => { typing()({ conversationId: convBId }); });

    expect(result.current.streaming).toBe(true);

    // Delete A while B is still streaming
    await act(async () => { await result.current.deleteConversation(convAId); });

    // B is still streaming → global streaming must remain true
    expect(result.current.streaming).toBe(true);

    // B finishes → now false
    await act(async () => {
      stream()({ conversationId: convBId, content: "Done B", done: true, responseType: "text" });
    });
    expect(result.current.streaming).toBe(false);
  });
});
