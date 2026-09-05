/**
 * Regression test for Issue #250: Background conversation completion must not affect active streaming state
 *
 * VERIFIED BUG: Before the fix, when a background conversation finished streaming,
 * setStreaming(false) was called before the conversation routing check, incorrectly
 * turning off the streaming state for whichever conversation was currently active.
 *
 * FIX: setStreaming(false) now only executes when the completed stream belongs to
 * the currently active conversation (isActiveConv check happens BEFORE state update).
 */

import { describe, it, expect, vi } from "vitest";

describe("Streaming State Isolation (Issue #250)", () => {
  it("REGRESSION: background conversation finishing must NOT turn off active conversation's streaming state", async () => {
    // Set up isolated mock for this test only
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
        if (!this.listeners.has(type)) {
          this.listeners.set(type, []);
        }
        this.listeners.get(type).push(handler);
        if (type === "stream") {
          mockStreamHandler = handler;
        }
        if (type === "typing") {
          mockTypingHandler = handler;
        }
        return () => {};
      }
      send() {}
      disconnect() {}
    };

    // Mock the API module
    vi.doMock("../services/api", () => ({
      api: {
        createConversation: vi.fn(() =>
          Promise.resolve({
            id: `conv-${Date.now()}-${Math.random()}`,
            title: "Test Chat",
            messages: [],
          })
        ),
        listConversations: vi.fn(() => Promise.resolve([])),
        getConversation: vi.fn((id) => Promise.resolve({ id, title: "Test Chat", messages: [] })),
      },
    }));

    // Mock WebSocket
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
    await waitFor(
      () => {
        expect(mockStreamHandler).not.toBeNull();
      },
      { timeout: 1000 }
    );

    // STEP 1: Create conversation A and start streaming
    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    await act(async () => {
      result.current.sendMessage("Hello from A", "text", []);
    });

    // Wait for messages to be created (user + assistant placeholder)
    await waitFor(() => {
      expect(result.current.messages).toHaveLength(2);
    });

    // Simulate stream start for conversation A
    await act(async () => {
      mockStreamHandler({
        type: "stream",
        conversationId: convAId,
        content: "Starting response A",
        done: false,
        responseType: "text",
      });
    });

    // Trigger typing event to set streaming=true
    await act(async () => {
      mockTypingHandler();
    });

    expect(result.current.streaming).toBe(true);

    // STEP 2: Create conversation B and switch to it
    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;

    // Verify we're now viewing conversation B
    expect(convBId).not.toBe(convAId);
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages).toHaveLength(0); // B has no messages yet

    // STEP 3: Start streaming in conversation B (the currently active one)
    await act(async () => {
      result.current.sendMessage("Hello from B", "text", []);
    });

    await waitFor(() => {
      expect(result.current.messages).toHaveLength(2);
    });

    // Simulate B starting to stream
    await act(async () => {
      mockStreamHandler({
        type: "stream",
        conversationId: convBId,
        content: "Starting response B",
        done: false,
        responseType: "text",
      });
    });

    // Trigger typing for B to ensure streaming is true
    await act(async () => {
      mockTypingHandler();
    });

    expect(result.current.streaming).toBe(true);

    // STEP 4: CRITICAL TEST - Conversation A finishes in the background
    const streamingBeforeAFinishes = result.current.streaming;

    await act(async () => {
      mockStreamHandler({
        type: "stream",
        conversationId: convAId, // A's ID, not B's
        content: "Final response from A",
        done: true, // A is done
        responseType: "text",
      });
    });

    // CRITICAL ASSERTION: B's streaming state must NOT be affected by A finishing
    // WITHOUT THE FIX: streaming would be set to false even though B is active
    // WITH THE FIX: streaming state remains unchanged because only isActiveConv affects it
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.streaming).toBe(streamingBeforeAFinishes); // Should still be true!
    expect(result.current.streaming).toBe(true); // Explicitly verify it's still true

    // Verify A's content was updated correctly in the background
    const convA = result.current.conversations.find((c) => c.id === convAId);
    expect(convA).toBeDefined();
    expect(convA.messages).toHaveLength(2);
    expect(convA.messages[1].content).toBe("Final response from A");

    // STEP 5: Now finish conversation B (the active one)
    await act(async () => {
      mockStreamHandler({
        type: "stream",
        conversationId: convBId,
        content: "Final response from B",
        done: true,
        responseType: "text",
      });
    });

    // Verify B completed correctly
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].content).toBe("Final response from B");

    // When the ACTIVE conversation finishes, streaming SHOULD be turned off
    expect(result.current.streaming).toBe(false);
  });
});
