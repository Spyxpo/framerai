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
    api.listConversations.mockImplementation(() => Promise.resolve([]));
    api.deleteConversation.mockImplementation(() => Promise.resolve({ success: true }));
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
});
