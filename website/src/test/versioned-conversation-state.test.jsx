import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import { api } from "../services/api";
import {
  loadConversationsFromStorage,
  saveConversationsToStorage,
  sanitizeConversation,
} from "../utils/storage";

let mockWsHandlers = new Map();
let lastWsSent = [];
let lastWsClient = null;

vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(() => Promise.resolve([])),
    createConversation: vi.fn((data) => {
      const id = data?.id || "conv-1";
      return Promise.resolve({ id, title: "New Chat", version: 1, messages: [] });
    }),
    getConversation: vi.fn((id) => Promise.resolve({ id, title: "Chat " + id, version: 1, messages: [] })),
    updateConversation: vi.fn((id, updates) =>
      Promise.resolve({ id, title: updates.title, version: (updates.expectedVersion || 1) + 1, messages: [] })
    ),
    deleteConversation: vi.fn(() => Promise.resolve({ success: true })),
    sendMessage: vi.fn((convId, content, type, attachments, settings, expectedVersion) =>
      Promise.resolve({
        id: "asst-1",
        userMessageId: "user-1",
        content: "Echo reply",
        type: "text",
        version: (expectedVersion || 1) + 1,
      })
    ),
    branchConversation: vi.fn((convId, msgId, expectedVersion) =>
      Promise.resolve({
        id: "branch-1",
        title: "Branched Chat",
        version: 1,
        parentVersion: expectedVersion || 1,
        parentConversationId: convId,
        messages: [{ id: msgId, role: "user", content: "root" }],
      })
    ),
    health: vi.fn(() => Promise.resolve({ ok: true })),
  },
}));

vi.mock("../services/websocket", () => ({
  WebSocketClient: class {
    constructor() {
      this.ws = { readyState: 1 };
      mockWsHandlers = new Map();
      lastWsSent = [];
      lastWsClient = this;
    }
    connect() {
      return Promise.resolve();
    }
    on(event, handler) {
      if (!mockWsHandlers.has(event)) {
        mockWsHandlers.set(event, []);
      }
      mockWsHandlers.get(event).push(handler);
      return () => {};
    }
    send(data) {
      lastWsSent.push(data);
    }
    sendApprovalResponse() {}
    disconnect() {}
  },
}));

function emitWs(event, payload) {
  const handlers = mockWsHandlers.get(event) || [];
  handlers.forEach((h) => h(payload));
}

describe("Issue #438 — Versioned Conversation State & Concurrency", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    lastWsSent = [];
    api.listConversations.mockReset();
    api.listConversations.mockImplementation(() => Promise.resolve([]));
    api.getConversation.mockReset();
    api.getConversation.mockImplementation((id) =>
      Promise.resolve({ id, title: "Chat " + id, version: 1, messages: [] })
    );
    api.deleteConversation.mockReset();
    api.deleteConversation.mockImplementation(() => Promise.resolve({ success: true }));
    api.sendMessage.mockReset();
    api.sendMessage.mockImplementation((convId, content, type, attachments, settings, expectedVersion) =>
      Promise.resolve({
        id: "asst-1",
        userMessageId: "user-1",
        content: "Echo reply",
        type: "text",
        version: (expectedVersion || 1) + 1,
      })
    );
    api.updateConversation.mockReset();
    api.updateConversation.mockImplementation((id, updates) =>
      Promise.resolve({ id, title: updates.title, version: (updates.expectedVersion || 1) + 1, messages: [] })
    );
    api.branchConversation.mockReset();
    api.branchConversation.mockImplementation((convId, msgId, expectedVersion) =>
      Promise.resolve({
        id: "branch-1",
        title: "Branched Chat",
        version: 1,
        parentVersion: expectedVersion || 1,
        parentConversationId: convId,
        messages: [{ id: msgId, role: "user", content: "root" }],
      })
    );
  });

  it("initializes conversations with version 1 and increments version on mutations", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
    });

    const conv = result.current.conversations.find((c) => c.id === result.current.activeConversation);
    expect(conv).toBeDefined();
    expect(conv.version).toBe(1);

    // Rename increments version
    await act(async () => {
      result.current.renameConversation(conv.id, "Updated Title");
    });

    const renamed = result.current.conversations.find((c) => c.id === conv.id);
    expect(renamed.title).toBe("Updated Title");
    expect(renamed.version).toBe(2);
    expect(api.updateConversation).toHaveBeenCalledWith(conv.id, {
      title: "Updated Title",
      expectedVersion: 1,
    });
  });

  it("passes expectedVersion to WebSocket chat frame and updates version on stream done", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
    });

    const convId = result.current.activeConversation;

    await act(async () => {
      await result.current.sendMessage("Hello over socket");
    });

    expect(lastWsSent.length).toBeGreaterThan(0);
    const sentFrame = lastWsSent[lastWsSent.length - 1];
    expect(sentFrame.type).toBe("chat");
    expect(sentFrame.expectedVersion).toBe(1);
    expect(sentFrame.conversationId).toBe(convId);

    // Simulate completion with updated version 3
    await act(async () => {
      emitWs("stream", {
        conversationId: convId,
        content: "Streaming response completed",
        done: true,
        version: 3,
      });
    });

    const updated = result.current.conversations.find((c) => c.id === convId);
    expect(updated.version).toBe(3);
  });

  it("reconciles state when a WebSocket VERSION_CONFLICT is received", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
    });

    const convId = result.current.activeConversation;

    api.getConversation.mockResolvedValueOnce({
      id: convId,
      title: "Authoritative Remote Title",
      version: 5,
      messages: [{ id: "m-remote", role: "user", content: "from server" }],
    });

    await act(async () => {
      emitWs("stream", {
        conversationId: convId,
        type: "error",
        code: "VERSION_CONFLICT",
        message: "Stale conversation version",
      });
    });

    await waitFor(() => {
      expect(api.getConversation).toHaveBeenCalledWith(convId);
    });

    await waitFor(() => {
      const conv = result.current.conversations.find((c) => c.id === convId);
      expect(conv.version).toBe(5);
      expect(conv.title).toBe("Authoritative Remote Title");
      expect(result.current.error).toContain("version conflict");
    });
  });

  it("handles branch conversation with parentVersion and version 1", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
    });

    const parentId = result.current.activeConversation;

    // Send a message
    await act(async () => {
      await result.current.sendMessage("Initial message");
    });

    const userMsg = result.current.messages[0];
    expect(userMsg).toBeDefined();

    // Acknowledge and complete turn with version 2
    await act(async () => {
      emitWs("ack", {
        conversationId: parentId,
        messageId: userMsg.id,
        version: 2,
      });
      emitWs("stream", {
        conversationId: parentId,
        content: "Turn complete",
        done: true,
        version: 2,
      });
    });

    let branchResult;
    await act(async () => {
      branchResult = await result.current.branchConversation(userMsg.id, parentId, 2);
    });

    expect(branchResult).toBeDefined();
    expect(branchResult.version).toBe(1);
    expect(branchResult.parentVersion).toBe(2);
    expect(branchResult.parentConversationId).toBe(parentId);
  });

  it("reconciles state when REST sendMessage encounters a 409 conflict", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
    });

    const convId = result.current.activeConversation;

    // Disconnect WS mock so REST path is taken
    if (lastWsClient) {
      lastWsClient.ws.readyState = 3;
    }

    const conflictErr = new Error("Conversation version mismatch");
    conflictErr.status = 409;
    conflictErr.code = "VERSION_CONFLICT";
    api.sendMessage.mockRejectedValueOnce(conflictErr);

    api.getConversation.mockResolvedValueOnce({
      id: convId,
      title: "Reconciled REST Title",
      version: 4,
      messages: [{ id: "m-server", role: "assistant", content: "server content" }],
    });

    await act(async () => {
      await result.current.sendMessage("trigger rest conflict");
    });

    await waitFor(() => {
      expect(result.current.error).toContain("version conflict");
    });
  });

  it("reconciles state when rename encounters a 409 conflict", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
    });

    const convId = result.current.activeConversation;

    const conflictErr = new Error("Conflict");
    conflictErr.status = 409;
    conflictErr.code = "VERSION_CONFLICT";
    api.updateConversation.mockRejectedValueOnce(conflictErr);

    api.getConversation.mockResolvedValueOnce({
      id: convId,
      title: "Authoritative Remote Name",
      version: 10,
      messages: [],
    });

    await act(async () => {
      result.current.renameConversation(convId, "My Local Rename");
    });

    await waitFor(() => {
      expect(api.getConversation).toHaveBeenCalledWith(convId);
      const conv = result.current.conversations.find((c) => c.id === convId);
      expect(conv.version).toBe(10);
      expect(conv.title).toBe("Authoritative Remote Name");
    });
  });

  it("backward compatibility: unversioned conversations default to version 1 and persist version", () => {
    // Storing legacy unversioned data
    const legacy = [
      {
        id: "legacy-1",
        title: "Old Chat",
        messages: [{ id: "m1", role: "user", content: "hi" }],
        createdAt: new Date().toISOString(),
      },
    ];

    saveConversationsToStorage(legacy, "legacy-1");

    const loaded = loadConversationsFromStorage();
    expect(loaded.conversations[0].version).toBe(1);

    // Sanitization retains version
    const sanitized = sanitizeConversation({ id: "test", title: "Test", version: 7 });
    expect(sanitized.version).toBe(7);
  });

  it("TEST 1: Successful mutation increments version correctly", async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;
    const initial = result.current.conversations.find((c) => c.id === convId);
    expect(initial.version).toBe(1);

    if (lastWsClient) lastWsClient.ws.readyState = 3; // REST path
    api.sendMessage.mockResolvedValueOnce({
      id: "asst-ok",
      userMessageId: "user-ok",
      content: "Reply OK",
      type: "text",
      version: 2,
    });

    await act(async () => {
      await result.current.sendMessage("Hello valid send");
    });

    const updated = result.current.conversations.find((c) => c.id === convId);
    expect(updated.version).toBe(2);
    expect(api.sendMessage).toHaveBeenCalledWith(
      convId,
      "Hello valid send",
      "text",
      [],
      undefined,
      1
    );

    const stored = loadConversationsFromStorage();
    const storedConv = stored.conversations.find((c) => c.id === convId);
    expect(storedConv.version).toBe(2);
  });

  it("TEST 2: Server rejection before persistence does NOT permanently bump local version", async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;
    expect(result.current.conversations.find((c) => c.id === convId).version).toBe(1);

    if (lastWsClient) lastWsClient.ws.readyState = 3;
    const rateLimitErr = new Error("Rate limit exceeded");
    rateLimitErr.status = 429;
    api.sendMessage.mockRejectedValueOnce(rateLimitErr);

    await act(async () => {
      await result.current.sendMessage("Will be rejected");
    });

    const convAfter = result.current.conversations.find((c) => c.id === convId);
    expect(convAfter.version).toBe(1);

    const stored = loadConversationsFromStorage();
    expect(stored.conversations.find((c) => c.id === convId).version).toBe(1);
  });

  it("TEST 3: Failed send followed by successful send works", async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    if (lastWsClient) lastWsClient.ws.readyState = 3;
    // First send fails with rate limit
    const rateLimitErr = new Error("Too Many Requests");
    rateLimitErr.status = 429;
    api.sendMessage.mockRejectedValueOnce(rateLimitErr);

    await act(async () => {
      await result.current.sendMessage("Fail message");
    });

    expect(result.current.conversations.find((c) => c.id === convId).version).toBe(1);

    // Second send succeeds
    api.sendMessage.mockResolvedValueOnce({
      id: "asst-2",
      userMessageId: "user-2",
      content: "Success on retry",
      type: "text",
      version: 2,
    });

    await act(async () => {
      await result.current.sendMessage("Retry message");
    });

    expect(api.sendMessage).toHaveBeenLastCalledWith(
      convId,
      "Retry message",
      "text",
      [],
      undefined,
      1 // MUST send expectedVersion 1, avoiding conflict!
    );
    expect(result.current.conversations.find((c) => c.id === convId).version).toBe(2);
  });

  it("TEST 4: Failed send followed by reload works", async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    if (lastWsClient) lastWsClient.ws.readyState = 3;
    const rateLimitErr = new Error("Rate limited");
    rateLimitErr.status = 429;
    api.sendMessage.mockRejectedValueOnce(rateLimitErr);

    await act(async () => {
      await result.current.sendMessage("Message before reload");
    });

    // Verify localStorage has version 1
    const stored = loadConversationsFromStorage();
    expect(stored.conversations.find((c) => c.id === convId).version).toBe(1);

    // Simulate reload: fresh hook instance restores from localStorage
    const reloaded = renderHook(() => useChat());
    const reloadedConv = reloaded.result.current.conversations.find((c) => c.id === convId);
    expect(reloadedConv).toBeDefined();
    expect(reloadedConv.version).toBe(1);

    if (lastWsClient) lastWsClient.ws.readyState = 3;
    api.sendMessage.mockResolvedValueOnce({
      id: "asst-after-reload",
      userMessageId: "user-after-reload",
      content: "Reply after reload",
      type: "text",
      version: 2,
    });

    await act(async () => {
      await reloaded.result.current.sendMessage("Message after reload");
    });

    expect(api.sendMessage).toHaveBeenLastCalledWith(
      convId,
      "Message after reload",
      "text",
      [],
      undefined,
      1 // Sent expectedVersion 1 after reload, avoiding conflict!
    );
    expect(reloaded.result.current.conversations.find((c) => c.id === convId).version).toBe(2);
  });

  it("TEST 5: Rate-limit/rejected request does not poison the conversation version", async () => {
    const initial = [
      {
        id: "conv-v3",
        title: "Active Chat",
        version: 3,
        messages: [{ id: "m1", role: "user", content: "hello" }],
        createdAt: new Date().toISOString(),
      },
    ];
    saveConversationsToStorage(initial, "conv-v3");

    const { result } = renderHook(() => useChat());
    expect(result.current.conversations.find((c) => c.id === "conv-v3").version).toBe(3);

    // WebSocket emits RATE_LIMITED error frame
    api.getConversation.mockResolvedValueOnce({
      id: "conv-v3",
      title: "Active Chat",
      version: 3,
      messages: [{ id: "m1", role: "user", content: "hello" }],
    });

    await act(async () => {
      emitWs("error", {
        conversationId: "conv-v3",
        code: "RATE_LIMITED",
        message: "Too many generation requests. Try again in 2s.",
      });
    });

    await waitFor(() => {
      const conv = result.current.conversations.find((c) => c.id === "conv-v3");
      expect(conv.version).toBe(3);
    });

    const stored = loadConversationsFromStorage();
    expect(stored.conversations.find((c) => c.id === "conv-v3").version).toBe(3);
  });

  it("TEST 6: Network failure does not permanently create a false conflict", async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    if (lastWsClient) lastWsClient.ws.readyState = 3;
    const networkErr = new TypeError("Failed to fetch");
    api.sendMessage.mockRejectedValueOnce(networkErr);
    // Remote fetch also fails (device is offline)
    api.getConversation.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    await act(async () => {
      await result.current.sendMessage("Send during offline");
    });

    const convOffline = result.current.conversations.find((c) => c.id === convId);
    expect(convOffline.version).toBe(1);

    // Network recovers, server was never mutated, so server is at version 1
    api.sendMessage.mockResolvedValueOnce({
      id: "asst-online",
      userMessageId: "user-online",
      content: "Now online",
      type: "text",
      version: 2,
    });

    await act(async () => {
      await result.current.sendMessage("Send after network recovered");
    });

    expect(api.sendMessage).toHaveBeenLastCalledWith(
      convId,
      "Send after network recovered",
      "text",
      [],
      undefined,
      1
    );
    expect(result.current.conversations.find((c) => c.id === convId).version).toBe(2);
  });

  it("TEST 7: A legitimate committed mutation is NOT rolled back by a later failed operation", async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    if (lastWsClient) lastWsClient.ws.readyState = 3;
    // Successful mutation commits version 2
    api.sendMessage.mockResolvedValueOnce({
      id: "asst-1",
      userMessageId: "user-1",
      content: "Reply 1",
      type: "text",
      version: 2,
    });

    await act(async () => {
      await result.current.sendMessage("Msg 1");
    });

    expect(result.current.conversations.find((c) => c.id === convId).version).toBe(2);

    // Subsequent operation fails
    const rateLimitErr = new Error("Rate limit");
    rateLimitErr.status = 429;
    api.sendMessage.mockRejectedValueOnce(rateLimitErr);
    api.getConversation.mockResolvedValueOnce({
      id: convId,
      title: "New Chat",
      version: 2,
      messages: [{ id: "user-1", role: "user", content: "Msg 1" }],
    });

    await act(async () => {
      await result.current.sendMessage("Msg 2 fails");
    });

    // Version must NOT be rolled back below 2
    expect(result.current.conversations.find((c) => c.id === convId).version).toBe(2);
    const stored = loadConversationsFromStorage();
    expect(stored.conversations.find((c) => c.id === convId).version).toBe(2);
  });

  it("TEST 8: Concurrent mutation failure/success resolves correctly", async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.createConversation();
    });
    const convId = result.current.activeConversation;

    if (lastWsClient) lastWsClient.ws.readyState = 3;

    let serverVersion = 1;
    api.sendMessage.mockImplementation((id, content) => {
      if (content === "Op A") {
        const err = new Error("Rate limit");
        err.status = 429;
        return Promise.reject(err);
      }
      serverVersion = 2;
      return Promise.resolve({
        id: "asst-b",
        userMessageId: "user-b",
        content: "Op B succeeded",
        type: "text",
        version: 2,
      });
    });
    api.getConversation.mockImplementation(() =>
      Promise.resolve({
        id: convId,
        title: "New Chat",
        version: serverVersion,
        messages: [],
      })
    );

    await act(async () => {
      const p1 = result.current.sendMessage("Op A");
      const p2 = result.current.sendMessage("Op B");
      await Promise.allSettled([p1, p2]);
    });

    expect(result.current.conversations.find((c) => c.id === convId).version).toBe(2);
  });

  it("TEST 9: No duplicate message is created during recovery/reconciliation", async () => {
    const initial = [
      {
        id: "conv-dedupe",
        title: "Test Dedupe",
        version: 1,
        messages: [{ id: "msg-1", role: "user", content: "hello" }],
        createdAt: new Date().toISOString(),
      },
    ];
    saveConversationsToStorage(initial, "conv-dedupe");

    const { result } = renderHook(() => useChat());

    api.getConversation.mockResolvedValueOnce({
      id: "conv-dedupe",
      title: "Test Dedupe",
      version: 2,
      messages: [
        { id: "msg-1", role: "user", content: "hello" },
        { id: "msg-2", role: "assistant", content: "world" },
      ],
    });

    if (lastWsClient) lastWsClient.ws.readyState = 3;
    const conflictErr = new Error("Version mismatch");
    conflictErr.status = 409;
    conflictErr.code = "VERSION_CONFLICT";
    api.sendMessage.mockRejectedValueOnce(conflictErr);

    await act(async () => {
      await result.current.sendMessage("trigger conflict");
    });

    const conv = result.current.conversations.find((c) => c.id === "conv-dedupe");
    expect(conv.version).toBe(2);
    const msgIds = conv.messages.map((m) => m.id);
    const uniqueIds = new Set(msgIds);
    expect(msgIds.length).toBe(uniqueIds.size);
    expect(msgIds).toContain("msg-1");
    expect(msgIds).toContain("msg-2");
  });

  it("TEST 10: The client eventually matches authoritative server version", async () => {
    // Local state somehow drifted to version 5, but server is at version 2
    const initial = [
      {
        id: "conv-drift",
        title: "Drifted Chat",
        version: 5,
        messages: [{ id: "m1", role: "user", content: "hi" }],
        createdAt: new Date().toISOString(),
      },
    ];
    saveConversationsToStorage(initial, "conv-drift");

    const { result } = renderHook(() => useChat());
    expect(result.current.conversations.find((c) => c.id === "conv-drift").version).toBe(5);

    if (lastWsClient) lastWsClient.ws.readyState = 3;
    const conflictErr = new Error("Version mismatch");
    conflictErr.status = 409;
    conflictErr.code = "VERSION_CONFLICT";
    api.sendMessage.mockRejectedValueOnce(conflictErr);

    api.getConversation.mockResolvedValueOnce({
      id: "conv-drift",
      title: "Authoritative Server Title",
      version: 2,
      messages: [{ id: "m1", role: "user", content: "hi" }],
    });

    await act(async () => {
      await result.current.sendMessage("send with drifted version");
    });

    // Reconciled to server's authoritative version 2!
    const reconciled = result.current.conversations.find((c) => c.id === "conv-drift");
    expect(reconciled.version).toBe(2);
    expect(reconciled.title).toBe("Authoritative Server Title");

    // Next send sends expectedVersion: 2 and succeeds
    api.sendMessage.mockResolvedValueOnce({
      id: "asst-ok",
      userMessageId: "user-ok",
      content: "Reply OK",
      type: "text",
      version: 3,
    });

    await act(async () => {
      await result.current.sendMessage("send after reconciliation");
    });

    expect(api.sendMessage).toHaveBeenLastCalledWith(
      "conv-drift",
      "send after reconciliation",
      "text",
      [],
      undefined,
      2
    );
    expect(result.current.conversations.find((c) => c.id === "conv-drift").version).toBe(3);
  });
});
