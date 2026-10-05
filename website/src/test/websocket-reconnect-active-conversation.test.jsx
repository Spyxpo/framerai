import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

describe("WebSocket Reconnection Active Conversation Subscription (Issue #410)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  async function setupHook() {
    let wsInstance = null;
    const sentFrames = [];

    const convStore = new Map();

    const MockWebSocketClient = class {
      constructor() {
        wsInstance = this;
        this.ws = { readyState: 1 };
        this.listeners = new Map();
        this.sent = sentFrames;
      }
      connect() {
        return Promise.resolve();
      }
      on(type, handler) {
        if (!this.listeners.has(type)) {
          this.listeners.set(type, []);
        }
        this.listeners.get(type).push(handler);
        return () => {
          const arr = this.listeners.get(type) || [];
          const idx = arr.indexOf(handler);
          if (idx >= 0) arr.splice(idx, 1);
        };
      }
      send(data) {
        sentFrames.push(data);
      }
      disconnect() {}
      simulateClose() {
        const handlers = this.listeners.get("close") || [];
        handlers.forEach((h) => h());
      }
      simulateReconnect() {
        this.ws.readyState = 1;
        const handlers = this.listeners.get("reconnect") || [];
        handlers.forEach((h) => h());
      }
    };

    const mockApi = {
      createConversation: vi.fn(() => {
        const id = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        const c = { id, title: "Test Chat", messages: [] };
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
      sendMessage: vi.fn(() => Promise.resolve({ id: "rest-msg", content: "rest response", type: "text" })),
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
      api: mockApi,
      convStore,
      sentFrames,
      wsInstance: () => wsInstance,
      stream: (data) => {
        const handlers = wsInstance?.listeners.get("stream") || [];
        handlers.forEach((h) => h(data));
      },
      ack: (data) => {
        const handlers = wsInstance?.listeners.get("ack") || [];
        handlers.forEach((h) => h(data));
      },
      typing: (data) => {
        const handlers = wsInstance?.listeners.get("typing") || [];
        handlers.forEach((h) => h(data));
      },
      close: () => wsInstance?.simulateClose(),
      reconnect: () => wsInstance?.simulateReconnect(),
    };
  }

  // ─── Test 1: Reconnect preserves active conversation ─────────────────────────
  it("Test 1 — Reconnect preserves active conversation and receives events", async () => {
    const { result, act, waitFor, stream, ack, typing, close, reconnect, sentFrames } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;
    expect(convAId).toBeTruthy();

    // Simulate connection loss
    await act(async () => {
      close();
    });

    // Simulate reconnect
    await act(async () => {
      reconnect();
    });

    // Conversation A remains active
    expect(result.current.activeConversation).toBe(convAId);

    // Generate new message after reconnect
    await act(async () => {
      await result.current.sendMessage("Hello post-reconnect", "text", []);
    });

    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    const lastSent = sentFrames[sentFrames.length - 1];
    expect(lastSent).toMatchObject({
      type: "chat",
      conversationId: convAId,
      content: "Hello post-reconnect",
    });

    // Deliver ack and stream frames
    const serverUserMsgId = "server-user-1";
    const serverAssistantMsgId = "server-ast-1";

    await act(async () => {
      ack({
        type: "ack",
        conversationId: convAId,
        messageId: serverUserMsgId,
        assistantMessageId: serverAssistantMsgId,
      });
      typing({ conversationId: convAId });
      stream({
        type: "stream",
        conversationId: convAId,
        messageId: serverAssistantMsgId,
        content: "Streaming response chunk",
        done: false,
        responseType: "text",
      });
    });

    expect(result.current.messages[1].content).toBe("Streaming response chunk");

    // Complete stream
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convAId,
        messageId: serverAssistantMsgId,
        content: "Completed response",
        done: true,
        responseType: "text",
      });
    });

    expect(result.current.messages[1].content).toBe("Completed response");
    expect(result.current.messages[1].completed).toBe(true);
    expect(result.current.streaming).toBe(false);
  });

  // ─── Test 2: Reconnect does not require conversation switching ────────────────
  it("Test 2 — Reconnect does not require conversation switching or refresh", async () => {
    const { result, act, waitFor, stream, ack, close, reconnect } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    // Disconnect and reconnect without switching
    await act(async () => {
      close();
      reconnect();
    });

    // Immediately send message in the active conversation
    await act(async () => {
      await result.current.sendMessage("Message immediately after reconnect", "text", []);
    });

    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "user-srv-id",
        assistantMessageId: "ast-srv-id",
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: "ast-srv-id",
        content: "Immediate reply",
        done: true,
        responseType: "text",
      });
    });

    expect(result.current.messages[1].content).toBe("Immediate reply");
    expect(result.current.messages[1].completed).toBe(true);
  });

  // ─── Test 3: Reconnect with conversation switch ──────────────────────────────
  it("Test 3 — Reconnect with conversation switch delivers events only to the new active conversation", async () => {
    const { result, act, stream, ack, close, reconnect, sentFrames } = await setupHook();

    // Create Conv A
    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    // Create Conv B
    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;
    expect(convBId).not.toBe(convAId);

    // Switch back to A
    await act(async () => {
      await result.current.selectConversation(convAId);
    });
    expect(result.current.activeConversation).toBe(convAId);

    // WebSocket disconnects while viewing A
    await act(async () => {
      close();
    });

    // Switch to B while disconnected
    await act(async () => {
      await result.current.selectConversation(convBId);
    });
    expect(result.current.activeConversation).toBe(convBId);

    // Reconnect while viewing B
    await act(async () => {
      reconnect();
    });

    // Send message in B
    await act(async () => {
      await result.current.sendMessage("Message in B", "text", []);
    });

    const lastSent = sentFrames[sentFrames.length - 1];
    expect(lastSent.conversationId).toBe(convBId);

    // Deliver events for B
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convBId,
        messageId: "user-b-srv",
        assistantMessageId: "ast-b-srv",
      });
      stream({
        type: "stream",
        conversationId: convBId,
        messageId: "ast-b-srv",
        content: "Reply in B",
        done: true,
        responseType: "text",
      });
    });

    // B should have received the reply
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages[1].content).toBe("Reply in B");

    // Conv A must be untouched
    const convA = result.current.conversations.find((c) => c.id === convAId);
    expect(convA.messages).toHaveLength(0);
  });

  // ─── Test 4: Reconnect race during conversation selection ─────────────────────
  it("Test 4 — Reconnect race during conversation switching routes to the final active conversation", async () => {
    const { result, act, stream, ack, close, reconnect } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    await act(async () => {
      await result.current.createConversation();
    });

    // Disconnect
    await act(async () => {
      close();
    });

    // Begin switching to A and simulate reconnect firing during the switch
    let pendingSelect;
    act(() => {
      pendingSelect = result.current.selectConversation(convAId);
      // Reconnect fires while selection is resolving
      reconnect();
    });

    await act(async () => {
      await pendingSelect;
    });

    expect(result.current.activeConversation).toBe(convAId);

    // Send message in the final active conversation (A)
    await act(async () => {
      await result.current.sendMessage("Message in A after race", "text", []);
    });

    await act(async () => {
      ack({
        type: "ack",
        conversationId: convAId,
        messageId: "user-a-srv",
        assistantMessageId: "ast-a-srv",
      });
      stream({
        type: "stream",
        conversationId: convAId,
        messageId: "ast-a-srv",
        content: "Reply in A",
        done: true,
        responseType: "text",
      });
    });

    expect(result.current.activeConversation).toBe(convAId);
    expect(result.current.messages[1].content).toBe("Reply in A");
  });

  // ─── Test 5: Multiple disconnect/reconnect cycles ─────────────────────────────
  it("Test 5 — Multiple reconnects do not create duplicate handlers or drop events", async () => {
    const { result, act, stream, ack, close, reconnect, wsInstance } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    // 3 disconnect and reconnect cycles
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        close();
        reconnect();
      });
    }

    // Verify stream listener count does not explode
    const streamHandlers = wsInstance().listeners.get("stream") || [];
    expect(streamHandlers.length).toBeLessThanOrEqual(1);

    // Message works normally after multiple reconnects
    await act(async () => {
      await result.current.sendMessage("Testing multiple reconnects", "text", []);
    });

    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "user-srv-multi",
        assistantMessageId: "ast-srv-multi",
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: "ast-srv-multi",
        content: "Multi reconnect success",
        done: true,
        responseType: "text",
      });
    });

    expect(result.current.messages[1].content).toBe("Multi reconnect success");
    expect(result.current.messages[1].completed).toBe(true);
  });

  // ─── Test 6: Disconnect mid-stream cleans up so post-reconnect message works ──
  it("Test 6 — Interrupted in-flight generation cleans up on disconnect and post-reconnect message streams cleanly", async () => {
    const { result, act, waitFor, stream, ack, typing, close, reconnect } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    // 1. Send first message
    await act(async () => {
      await result.current.sendMessage("First prompt", "text", []);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));

    await act(async () => {
      typing({ conversationId: convId });
    });
    expect(result.current.streaming).toBe(true);

    // 2. Disconnect happens mid-stream before completion
    await act(async () => {
      close();
    });

    expect(result.current.streaming).toBe(false);
    expect(result.current.messages[1].type).toBe("error");
    expect(result.current.messages[1].content).toBe("Connection lost. Please retry.");

    // 3. Socket reconnects
    await act(async () => {
      reconnect();
    });

    // 4. Send second message in the same active conversation
    await act(async () => {
      await result.current.sendMessage("Second prompt after reconnect", "text", []);
    });

    await waitFor(() => expect(result.current.messages).toHaveLength(4));
    expect(result.current.messages[2].content).toBe("Second prompt after reconnect");
    expect(result.current.messages[3].content).toBe("");

    // 5. Server sends ack for second message
    const secondUserSrvId = "second-user-srv-id";
    const secondAstSrvId = "second-ast-srv-id";

    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: secondUserSrvId,
        assistantMessageId: secondAstSrvId,
      });
    });

    // 6. Server streams chunks for second message
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: secondAstSrvId,
        content: "Second message chunk 1",
        done: false,
        responseType: "text",
      });
    });

    // Crucial assertion: the NEW assistant bubble must receive the chunk, NOT be empty!
    expect(result.current.messages[3].content).toBe("Second message chunk 1");

    // 7. Complete second message
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: secondAstSrvId,
        content: "Second message complete!",
        done: true,
        responseType: "text",
      });
    });

    expect(result.current.messages[3].content).toBe("Second message complete!");
    expect(result.current.messages[3].completed).toBe(true);
    expect(result.current.streaming).toBe(false);

    // The first message's error bubble must remain intact
    expect(result.current.messages[1].content).toBe("Connection lost. Please retry.");
    expect(result.current.messages[1].type).toBe("error");
  });
});
