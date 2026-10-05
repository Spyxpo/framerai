/**
 * Regression tests for Issue #408:
 * Concurrent message updates leaving conversation messages out of order.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import { api } from "../services/api";
import {
  saveConversationsToStorage,
  loadConversationsFromStorage,
  clearConversationsFromStorage,
} from "../utils/storage";

const doubles = vi.hoisted(() => ({ sockets: [] }));

vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(),
    createConversation: vi.fn(),
    getConversation: vi.fn(),
    deleteConversation: vi.fn(),
    sendMessage: vi.fn(),
    branchConversation: vi.fn(),
  },
}));

vi.mock("../services/websocket", () => ({
  WebSocketClient: class {
    constructor() {
      this.ws = { readyState: 1 };
      this.handlers = new Map();
      this.sent = [];
      doubles.sockets.push(this);
    }
    connect() {
      return Promise.resolve();
    }
    on(type, handler) {
      if (!this.handlers.has(type)) this.handlers.set(type, []);
      this.handlers.get(type).push(handler);
      return () => {};
    }
    emit(type, data) {
      (this.handlers.get(type) || []).forEach((handler) => handler(data));
    }
    send(payload) {
      this.sent.push(payload);
    }
    disconnect() {}
  },
}));

describe("Concurrent Message Ordering (Issue #408)", () => {
  let socket;

  beforeEach(() => {
    vi.clearAllMocks();
    doubles.sockets.length = 0;
    clearConversationsFromStorage();

    api.listConversations.mockResolvedValue([]);
    api.createConversation.mockImplementation(() =>
      Promise.resolve({
        id: crypto.randomUUID(),
        title: "New Chat",
        messages: [],
      })
    );
  });

  afterEach(() => {
    clearConversationsFromStorage();
  });

  async function startChat() {
    const hook = renderHook(() => useChat({}));
    await waitFor(() => {
      expect(hook.result.current.loadingConversations).toBe(false);
    });
    socket = doubles.sockets[doubles.sockets.length - 1];
    await act(async () => {
      await hook.result.current.createConversation();
    });
    return hook;
  }

  it("Test 1 — Out-of-order completion via REST: Operation A starts first, B starts second; B completes first, A later", async () => {
    let resolveReqA;
    let resolveReqB;

    const { result } = await startChat();
    socket.ws.readyState = 3; // Force REST fallback

    api.sendMessage.mockImplementation((convId, content) => {
      if (content === "msgA") {
        return new Promise((resolve) => {
          resolveReqA = () =>
            resolve({
              id: "server-a1",
              userMessageId: "server-u1",
              role: "assistant",
              content: "reply to msgA",
              type: "text",
            });
        });
      }
      if (content === "msgB") {
        return new Promise((resolve) => {
          resolveReqB = () =>
            resolve({
              id: "server-a2",
              userMessageId: "server-u2",
              role: "assistant",
              content: "reply to msgB",
              type: "text",
            });
        });
      }
      return Promise.resolve({
        id: "server-reply",
        userMessageId: "server-user",
        role: "assistant",
        content: `reply to: ${content}`,
        type: "text",
      });
    });

    // 1. Operation A starts first
    let pA;
    act(() => {
      pA = result.current.sendMessage("msgA");
    });

    // 2. Operation B starts second
    let pB;
    act(() => {
      pB = result.current.sendMessage("msgB");
    });

    // Four messages visible on screen (userA, placeholderA, userB, placeholderB)
    expect(result.current.messages).toHaveLength(4);
    expect(result.current.messages[0].content).toBe("msgA");
    expect(result.current.messages[2].content).toBe("msgB");

    // 3. Operation B completes FIRST
    await act(async () => {
      resolveReqB();
      await pB;
    });

    // msgB reply is completed, msgA reply is still pending
    expect(result.current.messages[3].completed).toBe(true);
    expect(result.current.messages[3].content).toBe("reply to msgB");
    expect(result.current.messages[1].completed).toBeFalsy();

    // 4. Operation A completes LATER
    await act(async () => {
      resolveReqA();
      await pA;
    });

    // Verify final message order: [msgA, replyA, msgB, replyB]
    const msgs = result.current.messages;
    expect(msgs).toHaveLength(4);
    expect(msgs[0].content).toBe("msgA");
    expect(msgs[0].id).toBe("server-u1");
    expect(msgs[1].content).toBe("reply to msgA");
    expect(msgs[1].id).toBe("server-a1");
    expect(msgs[2].content).toBe("msgB");
    expect(msgs[2].id).toBe("server-u2");
    expect(msgs[3].content).toBe("reply to msgB");
    expect(msgs[3].id).toBe("server-a2");
  });

  it("Test 2 & 4 — Multiple concurrent messages with WebSocket / REST interaction and controlled completion order", async () => {
    const { result } = await startChat();
    const convId = result.current.activeConversation;

    // Send msg1 over WebSocket
    await act(async () => {
      await result.current.sendMessage("msg1");
    });

    // Send msg2 over WebSocket
    await act(async () => {
      await result.current.sendMessage("msg2");
    });
    expect(result.current.messages).toHaveLength(4);

    // Acknowledge msg1 and msg2
    await act(async () => {
      socket.emit("ack", {
        type: "ack",
        conversationId: convId,
        messageId: "server-u1",
        assistantMessageId: "server-a1",
      });
      socket.emit("ack", {
        type: "ack",
        conversationId: convId,
        messageId: "server-u2",
        assistantMessageId: "server-a2",
      });
    });

    // Complete msg2 FIRST
    await act(async () => {
      socket.emit("stream", {
        type: "stream",
        conversationId: convId,
        messageId: "server-a2",
        content: "reply to msg2",
        done: true,
      });
    });

    expect(result.current.messages[3].content).toBe("reply to msg2");
    expect(result.current.messages[3].completed).toBe(true);

    // Complete msg1 LATER
    await act(async () => {
      socket.emit("stream", {
        type: "stream",
        conversationId: convId,
        messageId: "server-a1",
        content: "reply to msg1",
        done: true,
      });
    });

    const msgs = result.current.messages;
    expect(msgs).toHaveLength(4);
    expect(msgs[0].content).toBe("msg1");
    expect(msgs[0].id).toBe("server-u1");
    expect(msgs[1].content).toBe("reply to msg1");
    expect(msgs[1].id).toBe("server-a1");
    expect(msgs[2].content).toBe("msg2");
    expect(msgs[2].id).toBe("server-u2");
    expect(msgs[3].content).toBe("reply to msg2");
    expect(msgs[3].id).toBe("server-a2");
  });

  it("Test 3 — Persistence race: an older persistence result cannot overwrite newer conversation state", () => {
    const convId = "conv-persisted";
    const initialList = [
      {
        id: convId,
        title: "Test Chat",
        messages: [
          { id: "u1", role: "user", content: "hello" },
          { id: "a1", role: "assistant", content: "world" },
        ],
      },
    ];

    // Newer state saved at t = 2000
    saveConversationsToStorage(
      [
        {
          id: convId,
          title: "Test Chat",
          messages: [
            { id: "u1", role: "user", content: "hello" },
            { id: "a1", role: "assistant", content: "world" },
            { id: "u2", role: "user", content: "newer" },
            { id: "a2", role: "assistant", content: "newest" },
          ],
        },
      ],
      convId,
      { savedAt: 2000 }
    );

    // Stale older persistence attempt completing at t = 1000
    const olderSaved = saveConversationsToStorage(initialList, convId, { savedAt: 1000 });

    // Older save must be rejected
    expect(olderSaved).toBe(false);

    // Storage must still hold the newer 4 messages
    const loaded = loadConversationsFromStorage();
    expect(loaded.messages).toHaveLength(4);
    expect(loaded.messages[2].content).toBe("newer");
    expect(loaded.messages[3].content).toBe("newest");
  });

  it("Test 5 — Conversation switching: pending operations in A complete without corrupting B or A", async () => {
    let resolveConvA;

    const { result } = await startChat();
    const convAId = result.current.activeConversation;
    socket.ws.readyState = 3; // REST fallback

    api.sendMessage.mockImplementation((convId, content) => {
      if (content === "for-A") {
        return new Promise((resolve) => {
          resolveConvA = () =>
            resolve({
              id: "server-reply-A",
              userMessageId: "server-user-A",
              role: "assistant",
              content: "answer for A",
              type: "text",
            });
        });
      }
      return Promise.resolve({
        id: "server-reply-B",
        userMessageId: "server-user-B",
        role: "assistant",
        content: `reply to: ${content}`,
        type: "text",
      });
    });

    // Start operation in Conversation A
    let pA;
    act(() => {
      pA = result.current.sendMessage("for-A");
    });

    // Create and switch to Conversation B
    const convBId = crypto.randomUUID();
    api.createConversation.mockResolvedValueOnce({
      id: convBId,
      title: "Chat B",
      messages: [{ id: "b1", role: "user", content: "existing in B" }],
    });

    await act(async () => {
      await result.current.createConversation();
    });

    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages).toHaveLength(0);

    // Allow Conversation A's operation to complete while B is active
    await act(async () => {
      resolveConvA();
      await pA;
    });

    // Conversation B's messages must NOT be corrupted
    expect(result.current.activeConversation).toBe(convBId);
    expect(result.current.messages).toHaveLength(0);

    // Conversation A in conversations list must contain the completed response
    const storedConvA = result.current.conversations.find((c) => c.id === convAId);
    expect(storedConvA).toBeDefined();
    expect(storedConvA.messages).toHaveLength(2);
    expect(storedConvA.messages[0].content).toBe("for-A");
    expect(storedConvA.messages[1].content).toBe("answer for A");
    expect(storedConvA.messages[1].completed).toBe(true);
  });

  it("Test 6 — Branching during pending message operations retains correct ordering and isolation", async () => {
    let resolveReqA;

    const { result } = await startChat();
    const parentId = result.current.activeConversation;
    socket.ws.readyState = 3;

    api.sendMessage.mockImplementation((convId, content) => {
      if (content === "pending-in-parent") {
        return new Promise((resolve) => {
          resolveReqA = () =>
            resolve({
              id: "server-reply-pending",
              userMessageId: "server-user-pending",
              role: "assistant",
              content: "reply to pending",
              type: "text",
            });
        });
      }
      return Promise.resolve({
        id: "server-reply-1",
        userMessageId: "server-u1",
        role: "assistant",
        content: "initial answer",
        type: "text",
      });
    });

    // Send first message and complete it
    await act(async () => {
      await result.current.sendMessage("initial");
    });
    expect(result.current.messages).toHaveLength(2);
    const initialUserMsgId = result.current.messages[0].id;

    // Start pending message in parent
    let pA;
    act(() => {
      pA = result.current.sendMessage("pending-in-parent");
    });
    expect(result.current.messages).toHaveLength(4);

    // Branch from initial message while pending operation is in flight
    api.branchConversation.mockResolvedValue({
      id: "branch-conv-1",
      title: "New Chat (Branch)",
      parentConversationId: parentId,
      branchedFromMessageId: initialUserMsgId,
      messages: [{ id: initialUserMsgId, role: "user", content: "initial" }],
    });

    await act(async () => {
      await result.current.branchConversation(initialUserMsgId);
    });

    expect(result.current.activeConversation).toBe("branch-conv-1");
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].id).toBe(initialUserMsgId);

    // Now resolve parent's pending operation
    await act(async () => {
      resolveReqA();
      await pA;
    });

    // Branch must remain isolated with 1 message
    expect(result.current.activeConversation).toBe("branch-conv-1");
    expect(result.current.messages).toHaveLength(1);

    // Parent conversation must contain all 4 completed messages
    const parentConv = result.current.conversations.find((c) => c.id === parentId);
    expect(parentConv).toBeDefined();
    expect(parentConv.messages).toHaveLength(4);
    expect(parentConv.messages[0].content).toBe("initial");
    expect(parentConv.messages[1].content).toBe("initial answer");
    expect(parentConv.messages[2].content).toBe("pending-in-parent");
    expect(parentConv.messages[3].content).toBe("reply to pending");
  });
});
