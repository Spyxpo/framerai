/**
 * Regression test for issue #241: Stream frames must only update their originating conversation.
 *
 * VERIFIED BUG: Before the fix, switching conversations mid-stream would cause tokens to appear
 * in whichever conversation was currently displayed, not the one that initiated the request.
 *
 * FIX: Stream frames are now routed to the correct conversation in the conversations array,
 * regardless of which conversation is currently active. All mutations are immutable.
 */

import { describe, it, expect, vi } from "vitest";

describe("Conversation Isolation (Issue #241)", () => {
  it("REGRESSION: stream frames must route to originating conversation, not active one", async () => {
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

    // Mock the API module
    vi.doMock("../services/api", () => ({
      api: {
        createConversation: vi.fn((id) => 
          Promise.resolve({ 
            id: id || `conv-${Date.now()}`, 
            title: "Test Chat", 
            messages: [] 
          })
        ),
        listConversations: vi.fn(() => Promise.resolve([])),
        getConversation: vi.fn((id) => 
          Promise.resolve({ id, title: "Test Chat", messages: [] })
        ),
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
    await waitFor(() => {
      expect(mockStreamHandler).not.toBeNull();
    }, { timeout: 1000 });

    // Create conversation A
    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    // Send message in conversation A - creates user + assistant placeholder
    await act(async () => {
      result.current.sendMessage("Hello from A", "text", []);
    });

    // Verify placeholder exists
    await waitFor(() => {
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[1].role).toBe("assistant");
      expect(result.current.messages[1].content).toBe("");
    });

    // Simulate stream frames for conversation A
    await act(async () => {
      mockStreamHandler({
        type: "stream",
        conversationId: convAId,
        content: "First chunk from A",
        done: false,
        responseType: "text",
      });
    });

    // Verify tokens appear in conversation A
    expect(result.current.messages[1].content).toBe("First chunk from A");

    // Create and switch to conversation B
    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;

    // Verify we're now in conversation B (empty messages)
    expect(convBId).not.toBe(convAId);
    expect(result.current.messages).toHaveLength(0);

    // CRITICAL TEST: Simulate MORE stream frames for conversation A arriving
    // while we're viewing conversation B
    await act(async () => {
      mockStreamHandler({
        type: "stream",
        conversationId: convAId,
        content: "Second chunk from A - should update A, not B",
        done: false,
        responseType: "text",
      });
    });

    // Conversation B should still be empty (not polluted with A's frames)
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages).toHaveLength(0);

    // Final frame for A
    await act(async () => {
      mockStreamHandler({
        type: "stream",
        conversationId: convAId,
        content: "Final chunk from A",
        done: true,
        responseType: "text",
      });
    });

    // Conversation B should STILL be empty
    expect(result.current.messages).toHaveLength(0);

    // Switch back to conversation A
    await act(async () => {
      await result.current.selectConversation(convAId);
    });

    // CRITICAL VERIFICATION: Conversation A must have the FULL response content
    // The fix routes frames to the conversations array, then selectConversation
    // loads from there, so A's full content should be preserved
    expect(result.current.activeConversation).toBe(convAId);
    
    // Find conversation A in the conversations array
    const convA = result.current.conversations.find(c => c.id === convAId);
    expect(convA).toBeDefined();
    expect(convA.messages).toHaveLength(2);
    expect(convA.messages[1].content).toBe("Final chunk from A");
    expect(convA.messages[1].role).toBe("assistant");
  });
});
