/**
 * Regression test for Issue #253: Concurrent streaming state across multiple conversations
 *
 * THE BUG: streamingConversationIdRef tracked only one conversation ID.
 * When conversation A is streaming in the background and B (active) finishes first,
 * setStreaming(false) is called unconditionally because B is the active conversation.
 * A is still streaming but the UI incorrectly reports that streaming has stopped.
 *
 * Also: when both A and B start via sendMessage(), the ref is overwritten by B.
 * A finishes as a background conv — convAId !== ref(convBId), so no clear (fine).
 * B finishes as active conv — unconditionally clears streaming even if A still active.
 *
 * FIX: Use a Set (streamingConversationIdsRef) to track ALL active streaming conversations.
 * Only call setStreaming(false) when the Set becomes empty.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

describe("Concurrent Streaming State (Issue #253)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  /**
   * Shared mock factory — keeps each test self-contained.
   */
  async function setupHook() {
    let mockStreamHandler = null;
    let mockTypingHandler = null;

    const MockWebSocketClient = class {
      constructor() {
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
        return () => {};
      }
      send() {}
      disconnect() {}
    };

    vi.doMock("../services/api", () => ({
      api: {
        createConversation: vi.fn(() =>
          Promise.resolve({ id: `conv-${Date.now()}-${Math.random()}`, title: "Test", messages: [] })
        ),
        listConversations: vi.fn(() => Promise.resolve([])),
        getConversation: vi.fn((id) => Promise.resolve({ id, title: "Test", messages: [] })),
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
      },
      { timeout: 1000 }
    );

    return { result, act, waitFor, mockStreamHandler: () => mockStreamHandler, mockTypingHandler: () => mockTypingHandler };
  }

  // ─── THE KEY REGRESSION TEST ──────────────────────────────────────────────
  it("REGRESSION #253: active conv B finishes while background conv A still streams — streaming must stay true", async () => {
    const { result, act, waitFor, mockStreamHandler, mockTypingHandler } = await setupHook();
    const stream = () => mockStreamHandler();
    const typing = () => mockTypingHandler();

    // Create conversation A and begin its message
    await act(async () => { await result.current.createConversation(); });
    const convAId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("Hello from A", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    // Mark A as streaming
    await act(async () => { typing()({ conversationId: convAId }); });
    expect(result.current.streaming).toBe(true);

    // Switch to conversation B and begin its message
    await act(async () => { await result.current.createConversation(); });
    const convBId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("Hello from B", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    // Mark B as streaming — with old code this overwrites the ref, losing track of A
    await act(async () => { typing()({ conversationId: convBId }); });
    expect(result.current.streaming).toBe(true);

    // ── CRITICAL: B (active conv) finishes first ───────────────────────────
    await act(async () => {
      stream()({ conversationId: convBId, content: "Response B", done: true, responseType: "text" });
    });

    // With OLD code: setStreaming(false) called unconditionally because B is active → BUG
    // With NEW code: A still in the Set → streaming stays true
    expect(result.current.streaming).toBe(true); // A is still active!

    // ── Now A finishes ─────────────────────────────────────────────────────
    await act(async () => {
      stream()({ conversationId: convAId, content: "Response A", done: true, responseType: "text" });
    });

    // Both done → streaming clears
    expect(result.current.streaming).toBe(false);
  });

  // ─── Single conversation: existing behaviour unchanged ────────────────────
  it("single conversation streaming and finishing", async () => {
    const { result, act, waitFor, mockStreamHandler, mockTypingHandler } = await setupHook();
    const stream = () => mockStreamHandler();
    const typing = () => mockTypingHandler();

    await act(async () => { await result.current.createConversation(); });
    const convId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("Hello", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    await act(async () => { typing()({ conversationId: convId }); });
    expect(result.current.streaming).toBe(true);

    await act(async () => {
      stream()({ conversationId: convId, content: "Done", done: true, responseType: "text" });
    });
    expect(result.current.streaming).toBe(false);
  });

  // ─── A finishes before B; B is active ─────────────────────────────────────
  it("background conv A finishes while active conv B still streams — streaming stays true", async () => {
    const { result, act, waitFor, mockStreamHandler, mockTypingHandler } = await setupHook();
    const stream = () => mockStreamHandler();
    const typing = () => mockTypingHandler();

    await act(async () => { await result.current.createConversation(); });
    const convAId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("A", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    await act(async () => { typing()({ conversationId: convAId }); });

    await act(async () => { await result.current.createConversation(); });
    const convBId = result.current.activeConversation;
    await act(async () => { result.current.sendMessage("B", "text", []); });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    await act(async () => { typing()({ conversationId: convBId }); });

    expect(result.current.streaming).toBe(true);

    // A finishes in background — B still active
    await act(async () => {
      stream()({ conversationId: convAId, content: "Done A", done: true, responseType: "text" });
    });
    expect(result.current.streaming).toBe(true); // B still going

    // B finishes — last one, streaming clears
    await act(async () => {
      stream()({ conversationId: convBId, content: "Done B", done: true, responseType: "text" });
    });
    expect(result.current.streaming).toBe(false);
  });

  // ─── Conversation switching doesn't corrupt the Set ───────────────────────
  it("switching conversations while both are streaming preserves streaming state", async () => {
    const { result, act, mockStreamHandler, mockTypingHandler } = await setupHook();
    const stream = () => mockStreamHandler();
    const typing = () => mockTypingHandler();

    await act(async () => { await result.current.createConversation(); });
    const convAId = result.current.activeConversation;

    await act(async () => { await result.current.createConversation(); });
    const convBId = result.current.activeConversation;

    // Both marked as streaming
    await act(async () => {
      typing()({ conversationId: convAId });
      typing()({ conversationId: convBId });
    });
    expect(result.current.streaming).toBe(true);

    // Switch around
    await act(async () => { result.current.selectConversation(convAId); });
    expect(result.current.streaming).toBe(true);
    await act(async () => { result.current.selectConversation(convBId); });
    expect(result.current.streaming).toBe(true);

    // Finish A (background from B's perspective)
    await act(async () => {
      stream()({ conversationId: convAId, content: "Done A", done: true, responseType: "text" });
    });
    expect(result.current.streaming).toBe(true);

    // Finish B
    await act(async () => {
      stream()({ conversationId: convBId, content: "Done B", done: true, responseType: "text" });
    });
    expect(result.current.streaming).toBe(false);
  });
});
