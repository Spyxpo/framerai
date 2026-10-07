import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mergeMessages } from "../utils/dedupe";
import { loadConversationsFromStorage } from "../utils/storage";

describe("Issue #437 — Interrupted Generation Recovery & State Convergence", () => {
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
    let wsInstance = null;
    const sentFrames = [];
    const convStore = new Map();

    const MockWebSocketClient = class {
      constructor() {
        wsInstance = this;
        this.ws = { readyState: wsReadyState };
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
        this.ws.readyState = 3;
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
      listConversations: vi.fn(() => {
        return Promise.resolve(
          Array.from(convStore.values()).map((c) => ({
            id: c.id,
            title: c.title,
            messageCount: c.messages?.length || 0,
            updatedAt: c.updatedAt,
          }))
        );
      }),
      getConversation: vi.fn((id) => {
        const c = convStore.get(id) || { id, title: "Chat " + id, messages: [] };
        return Promise.resolve(JSON.parse(JSON.stringify(c)));
      }),
      deleteConversation: vi.fn((id) => {
        convStore.delete(id);
        return Promise.resolve({ success: true });
      }),
      sendMessage: vi.fn(() => Promise.resolve({ id: "rest-msg", content: "rest response", type: "text" })),
      branchConversation: vi.fn((convId, messageId) => {
        const parent = convStore.get(convId);
        const idx = parent?.messages?.findIndex((m) => m.id === messageId);
        const msgs = idx >= 0 ? parent.messages.slice(0, idx + 1) : [];
        const branchId = `branch-${Date.now()}`;
        const branchConv = {
          id: branchId,
          title: "Branch Chat",
          parentConversationId: convId,
          branchedFromMessageId: messageId,
          messages: JSON.parse(JSON.stringify(msgs)),
        };
        convStore.set(branchId, branchConv);
        return Promise.resolve(branchConv);
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

  // ─── Test 1: Reconnect after partial streaming recovers full response from backend ───
  it("Test 1 — Reconnect recovers interrupted partial response from backend persistence", async () => {
    const { result, act, waitFor, stream, ack, close, reconnect, convStore } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    // Send first message
    await act(async () => {
      await result.current.sendMessage("Tell me about Paris", "text", []);
    });

    // Server acknowledges with persistent IDs
    const userSrvId = "srv-user-1";
    const astSrvId = "srv-ast-1";
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: userSrvId,
        assistantMessageId: astSrvId,
      });
    });

    // Stream arrives partially
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: astSrvId,
        content: "Paris is the capital of",
        done: false,
      });
    });

    expect(result.current.messages[1].content).toBe("Paris is the capital of");

    // Network connection disconnects mid-stream
    await act(async () => {
      close();
    });

    // In the backend store, the model completed generation and saved the full response
    convStore.set(convId, {
      id: convId,
      title: "Tell me about Paris",
      messages: [
        { id: userSrvId, role: "user", content: "Tell me about Paris", type: "text" },
        { id: astSrvId, role: "assistant", content: "Paris is the capital of France.", type: "text" },
      ],
    });

    // Connection reconnects
    await act(async () => {
      reconnect();
    });

    // After reconnect recovery, the conversation state MUST converge with the backend
    await waitFor(() => {
      expect(result.current.messages[1].content).toBe("Paris is the capital of France.");
      expect(result.current.messages[1].completed).toBe(true);
    });

    // Must NOT have duplicate assistant messages
    const assistantMsgs = result.current.messages.filter((m) => m.role === "assistant");
    expect(assistantMsgs).toHaveLength(1);

    // Local storage must also reflect the recovered state
    const stored = loadConversationsFromStorage(localStorage);
    const storedConv = stored.conversations.find((c) => c.id === convId);
    expect(storedConv.messages[1].content).toBe("Paris is the capital of France.");
    expect(storedConv.messages[1].completed).toBe(true);
  });

  // ─── Test 2: Stale frame from interrupted generation does NOT overwrite newer generation ───
  it("Test 2 — Stale frame from interrupted generation does not overwrite newer generation", async () => {
    const { result, act, stream, ack, close, reconnect } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    // Send Generation 1
    await act(async () => {
      await result.current.sendMessage("Prompt 1", "text", []);
    });
    const gen1AstId = "gen1-ast-srv";
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "gen1-user-srv",
        assistantMessageId: gen1AstId,
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: gen1AstId,
        content: "Gen 1 partial",
        done: false,
      });
    });

    // Disconnect occurs mid-stream
    await act(async () => {
      close();
    });

    // Reconnect occurs
    await act(async () => {
      reconnect();
    });

    // Send Generation 2 after reconnect
    await act(async () => {
      await result.current.sendMessage("Prompt 2", "text", []);
    });
    expect(result.current.messages).toHaveLength(4);

    const gen2AstId = "gen2-ast-srv";
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "gen2-user-srv",
        assistantMessageId: gen2AstId,
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: gen2AstId,
        content: "Gen 2 response chunk",
        done: false,
      });
    });

    expect(result.current.messages[3].content).toBe("Gen 2 response chunk");

    // Now a STALE frame from Gen 1 arrives (either naming gen1AstId or omitting messageId)
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: gen1AstId,
        content: "Late stale frame from gen 1",
        done: false,
      });
    });

    // Gen 2 must NOT be corrupted!
    expect(result.current.messages[3].content).toBe("Gen 2 response chunk");

    // Complete Gen 2
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: gen2AstId,
        content: "Gen 2 completed!",
        done: true,
      });
    });

    expect(result.current.messages[3].content).toBe("Gen 2 completed!");
    expect(result.current.messages[3].completed).toBe(true);
  });

  // ─── Test 3: mergeMessages correctly reconciles turns with unadopted client IDs without duplicates ───
  it("Test 3 — mergeMessages reconciles local messages without creating duplicates or losing completion", () => {
    const local = [
      { id: "local-u1", role: "user", content: "Hello" },
      { id: "local-a1", role: "assistant", content: "Hi par...", completed: false },
    ];
    const remote = [
      { id: "srv-u1", role: "user", content: "Hello" },
      { id: "srv-a1", role: "assistant", content: "Hi! How can I help you today?" },
    ];

    const merged = mergeMessages(local, remote);
    expect(merged).toHaveLength(2);
    expect(merged[0].role).toBe("user");
    expect(merged[0].content).toBe("Hello");
    expect(merged[1].role).toBe("assistant");
    expect(merged[1].content).toBe("Hi! How can I help you today?");
    expect(merged[1].completed).toBe(true);
    expect(merged[1].id).toBe("srv-a1");
  });

  // ─── Test 4: Conversation switch during interrupted generation preserves isolation ───
  it("Test 4 — Conversation switch during interrupted generation isolates recovery to original conversation", async () => {
    const { result, act, waitFor, stream, ack, close, reconnect, convStore } = await setupHook();

    // Create Conversation A
    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    // Create Conversation B
    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;

    // Switch back to A and start streaming
    await act(async () => {
      await result.current.selectConversation(convAId);
    });

    await act(async () => {
      await result.current.sendMessage("Query in A", "text", []);
    });

    await act(async () => {
      ack({
        type: "ack",
        conversationId: convAId,
        messageId: "u-a-srv",
        assistantMessageId: "ast-a-srv",
      });
      stream({
        type: "stream",
        conversationId: convAId,
        messageId: "ast-a-srv",
        content: "Reply in A part...",
        done: false,
      });
    });

    // Disconnect while A is streaming
    await act(async () => {
      close();
    });

    // Switch to B before reconnect
    await act(async () => {
      await result.current.selectConversation(convBId);
    });
    expect(result.current.activeConversation).toBe(convBId);

    // Backend completed A's response
    convStore.set(convAId, {
      id: convAId,
      title: "Query in A",
      messages: [
        { id: "u-a-srv", role: "user", content: "Query in A", type: "text" },
        { id: "ast-a-srv", role: "assistant", content: "Reply in A full answer.", type: "text" },
      ],
    });

    // Reconnect fires while B is active
    await act(async () => {
      reconnect();
    });

    // B remains active and unaffected
    expect(result.current.activeConversation).toBe(convBId);

    // Switch back to A: A must display the recovered state
    await act(async () => {
      await result.current.selectConversation(convAId);
    });

    await waitFor(() => {
      expect(result.current.activeConversation).toBe(convAId);
      expect(result.current.messages[1].content).toBe("Reply in A full answer.");
    });

    const astList = result.current.messages.filter((m) => m.role === "assistant");
    expect(astList).toHaveLength(1);
  });

  // ─── Test 5: Duplicate completion arriving after recovery is safely ignored ───
  it("Test 5 — Duplicate completion frame arriving after recovery is idempotent", async () => {
    const { result, act, waitFor, stream, ack, close, reconnect, convStore } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("Prompt", "text", []);
    });

    const astId = "ast-srv-dup";
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "u-srv-dup",
        assistantMessageId: astId,
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: astId,
        content: "Partial",
        done: false,
      });
    });

    await act(async () => {
      close();
    });

    convStore.set(convId, {
      id: convId,
      title: "Prompt",
      messages: [
        { id: "u-srv-dup", role: "user", content: "Prompt", type: "text" },
        { id: astId, role: "assistant", content: "Recovered complete answer", type: "text" },
      ],
    });

    await act(async () => {
      reconnect();
    });

    await waitFor(() => {
      expect(result.current.messages[1].content).toBe("Recovered complete answer");
    });

    // Now a redundant duplicate completion frame arrives over websocket
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: astId,
        content: "Duplicate completion text",
        done: true,
      });
    });

    // Message must remain the recovered response and not create duplicate or overwrite
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].content).toBe("Recovered complete answer");
    expect(result.current.messages[1].completed).toBe(true);
  });

  // ─── Test 6: Branching from an interrupted turn uses the recovered content ───
  it("Test 6 — Branching from an interrupted turn includes the recovered response", async () => {
    const { result, act, waitFor, stream, ack, close, reconnect, convStore } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("Root message", "text", []);
    });

    const astId = "ast-srv-branch";
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "u-srv-branch",
        assistantMessageId: astId,
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: astId,
        content: "Partial before branch",
        done: false,
      });
    });

    await act(async () => {
      close();
    });

    convStore.set(convId, {
      id: convId,
      title: "Root message",
      messages: [
        { id: "u-srv-branch", role: "user", content: "Root message", type: "text" },
        { id: astId, role: "assistant", content: "Full root reply from backend", type: "text" },
      ],
    });

    await act(async () => {
      reconnect();
    });

    await waitFor(() => {
      expect(result.current.messages[1].content).toBe("Full root reply from backend");
    });

    // Branch from the recovered assistant message
    let branchResult;
    await act(async () => {
      branchResult = await result.current.branchConversation(astId);
    });

    expect(branchResult).not.toBeNull();
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].content).toBe("Full root reply from backend");
  });

  // ─── Test 7: Multiple reconnect cycles do not duplicate or resurrect generation state ───
  it("Test 7 — Multiple reconnect cycles do not duplicate or resurrect generation state", async () => {
    const { result, act, waitFor, stream, ack, close, reconnect, convStore } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("Hello multi-reconnect", "text", []);
    });

    const astId = "ast-srv-multi";
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "u-srv-multi",
        assistantMessageId: astId,
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: astId,
        content: "Multi-reconnect partial",
        done: false,
      });
    });

    convStore.set(convId, {
      id: convId,
      title: "Hello multi-reconnect",
      messages: [
        { id: "u-srv-multi", role: "user", content: "Hello multi-reconnect", type: "text" },
        { id: astId, role: "assistant", content: "Multi-reconnect recovered full", type: "text" },
      ],
    });

    // 3 disconnect and reconnect cycles
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        close();
        reconnect();
      });
    }

    await waitFor(() => {
      expect(result.current.messages[1].content).toBe("Multi-reconnect recovered full");
      expect(result.current.messages[1].completed).toBe(true);
    });

    // Verify no duplicates created across multiple reconnects
    expect(result.current.messages).toHaveLength(2);
    const astMsgs = result.current.messages.filter((m) => m.role === "assistant");
    expect(astMsgs).toHaveLength(1);
    expect(result.current.streaming).toBe(false);
  });

  // ─── Test 8: Stale frame without messageId after reconnect is dropped when not streaming ───
  it("Test 8 — Stale frame without messageId after reconnect does not corrupt conversation state", async () => {
    const { result, act, waitFor, stream, ack, close, reconnect, convStore } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("Initial prompt", "text", []);
    });

    const astId = "ast-srv-test8";
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "u-srv-test8",
        assistantMessageId: astId,
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: astId,
        content: "Initial partial",
        done: false,
      });
    });

    await act(async () => {
      close();
    });

    convStore.set(convId, {
      id: convId,
      title: "Initial prompt",
      messages: [
        { id: "u-srv-test8", role: "user", content: "Initial prompt", type: "text" },
        { id: astId, role: "assistant", content: "Initial recovered response", type: "text" },
      ],
    });

    await act(async () => {
      reconnect();
    });

    await waitFor(() => {
      expect(result.current.messages[1].content).toBe("Initial recovered response");
    });

    // An unaddressed stale frame arrives while no streaming is active
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        content: "Orphaned stale chunk with no messageId",
        done: false,
      });
    });

    // Conversation state remains intact
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].content).toBe("Initial recovered response");
    expect(result.current.streaming).toBe(false);
  });

  // ─── Test 9: Terminal completion frame from interrupted generation arriving after recovery is dropped ───
  it("Test 9 — Terminal completion frame from interrupted generation after recovery does not overwrite", async () => {
    const { result, act, waitFor, stream, ack, close, reconnect, convStore } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("First message", "text", []);
    });

    const astId = "ast-srv-test9";
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "u-srv-test9",
        assistantMessageId: astId,
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: astId,
        content: "First partial",
        done: false,
      });
    });

    await act(async () => {
      close();
    });

    convStore.set(convId, {
      id: convId,
      title: "First message",
      messages: [
        { id: "u-srv-test9", role: "user", content: "First message", type: "text" },
        { id: astId, role: "assistant", content: "Recovered complete response 9", type: "text" },
      ],
    });

    await act(async () => {
      reconnect();
    });

    await waitFor(() => {
      expect(result.current.messages[1].content).toBe("Recovered complete response 9");
      expect(result.current.messages[1].completed).toBe(true);
    });

    // A stale terminal frame arrives with older/different text
    await act(async () => {
      stream({
        type: "stream",
        conversationId: convId,
        messageId: astId,
        content: "Stale terminal content that should be rejected",
        done: true,
      });
    });

    // Response must NOT be overwritten by the stale terminal frame
    expect(result.current.messages[1].content).toBe("Recovered complete response 9");
    expect(result.current.messages[1].completed).toBe(true);
  });

  // ─── Test 10: Interrupted generation where backend has no response stays marked completed ───
  it("Test 10 — Interrupted generation with partial content marks message completed without hanging in limbo", async () => {
    const { result, act, stream, ack, close, reconnect } = await setupHook();

    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("No backend response prompt", "text", []);
    });

    const astId = "ast-srv-test10";
    await act(async () => {
      ack({
        type: "ack",
        conversationId: convId,
        messageId: "u-srv-test10",
        assistantMessageId: astId,
      });
      stream({
        type: "stream",
        conversationId: convId,
        messageId: astId,
        content: "Partial content before crash",
        done: false,
      });
    });

    // Disconnect happens
    await act(async () => {
      close();
    });

    // Message must be marked completed so composer and next turns work cleanly
    expect(result.current.messages[1].content).toBe("Partial content before crash");
    expect(result.current.messages[1].completed).toBe(true);
    expect(result.current.streaming).toBe(false);

    // Reconnect happens
    await act(async () => {
      reconnect();
    });

    // Subsequent message works smoothly without conflict
    await act(async () => {
      await result.current.sendMessage("Next turn prompt", "text", []);
    });

    expect(result.current.messages).toHaveLength(4);
    expect(result.current.messages[2].content).toBe("Next turn prompt");
  });
});
