import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import { api } from "../services/api";
import {
  computeConversationDiff,
  reconcileConversationDiff,
  applyConversationDiff,
  mergeMessages,
  reconcileConversation,
  dedupeMessages,
} from "../utils/dedupe";
import {
  loadConversationsFromStorage,
  saveConversationsToStorage,
  sanitizeConversation,
  STORAGE_KEY,
  STORAGE_VERSION,
} from "../utils/storage";

let mockWsHandlers = new Map();

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
    deleteMessage: vi.fn(() => Promise.resolve({ success: true, version: 2 })),
    syncConversation: vi.fn((id, payload) =>
      Promise.resolve({
        status: "synchronized",
        version: (payload.clientVersion || 1) + 1,
        serverVersion: (payload.clientVersion || 1) + 1,
        clientVersion: payload.clientVersion,
        conversation: {
          id,
          title: payload.title || "Synced Chat",
          version: (payload.clientVersion || 1) + 1,
          messages: payload.messages || [],
        },
      })
    ),
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
        branchedFromMessageId: msgId,
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
    send() {}
    sendApprovalResponse() {}
    disconnect() {}
  },
}));

function emitWs(event, payload) {
  const handlers = mockWsHandlers.get(event) || [];
  handlers.forEach((h) => h(payload));
}

describe("Issue #449 — Differential Conversation Synchronization & Conflict Resolution", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    api.listConversations.mockImplementation(() => Promise.resolve([]));
    api.getConversation.mockImplementation((id) =>
      Promise.resolve({ id, title: "Chat " + id, version: 1, messages: [] })
    );
  });

  // 1. No differences -> no unnecessary changes
  it("Scenario 1: No differences -> diff reports in_sync and no changes are made", () => {
    const conv = {
      id: "c-1",
      title: "Clean Chat",
      version: 2,
      messages: [{ id: "m-1", role: "user", content: "Hello", type: "text" }],
    };
    const diff = computeConversationDiff(conv, conv);
    expect(diff.hasDifferences).toBe(false);
    expect(diff.status).toBe("in_sync");

    const reconciled = reconcileConversation(conv, conv);
    expect(reconciled.version).toBe(2);
    expect(reconciled.messages).toHaveLength(1);
    expect(reconciled.messages[0].id).toBe("m-1");
  });

  // 2. Client-only message -> correctly reconciled
  it("Scenario 2: Client-only message -> correctly reconciled without losing state", () => {
    const clientConv = {
      id: "c-1",
      version: 1,
      messages: [
        { id: "m-1", role: "user", content: "Existing", type: "text" },
        { id: "m-2", role: "user", content: "Offline Turn", type: "text" },
      ],
    };
    const serverConv = {
      id: "c-1",
      version: 1,
      messages: [{ id: "m-1", role: "user", content: "Existing", type: "text" }],
    };

    const diff = computeConversationDiff(clientConv, serverConv);
    expect(diff.status).toBe("client_ahead");
    expect(diff.messages.clientOnly).toHaveLength(1);
    expect(diff.messages.clientOnly[0].id).toBe("m-2");

    const reconciled = reconcileConversation(clientConv, serverConv);
    expect(reconciled.messages).toHaveLength(2);
    expect(reconciled.messages[1].id).toBe("m-2");
  });

  // 3. Server-only message -> correctly reconciled
  it("Scenario 3: Server-only message -> correctly reconciled and merged into conversation", () => {
    const clientConv = {
      id: "c-1",
      version: 1,
      messages: [{ id: "m-1", role: "user", content: "Query", type: "text" }],
    };
    const serverConv = {
      id: "c-1",
      version: 2,
      messages: [
        { id: "m-1", role: "user", content: "Query", type: "text" },
        { id: "m-2", role: "assistant", content: "Reply", type: "text", completed: true },
      ],
    };

    const diff = computeConversationDiff(clientConv, serverConv);
    expect(diff.status).toBe("server_ahead");
    expect(diff.messages.serverOnly).toHaveLength(1);
    expect(diff.messages.serverOnly[0].id).toBe("m-2");

    const reconciled = reconcileConversation(clientConv, serverConv);
    expect(reconciled.version).toBe(2);
    expect(reconciled.messages).toHaveLength(2);
    expect(reconciled.messages[1].id).toBe("m-2");
  });

  // 4. Both sides add different messages -> deterministic merge
  it("Scenario 4: Both sides add different messages -> deterministic merge preserving stable IDs", () => {
    const clientConv = {
      id: "c-1",
      version: 2,
      messages: [
        { id: "m-1", role: "user", content: "A" },
        { id: "m-2", role: "assistant", content: "B" },
        { id: "m-3", role: "user", content: "C" },
        { id: "m-client", role: "user", content: "D" },
      ],
    };
    const serverConv = {
      id: "c-1",
      version: 3,
      messages: [
        { id: "m-1", role: "user", content: "A" },
        { id: "m-2", role: "assistant", content: "B" },
        { id: "m-3", role: "user", content: "C" },
        { id: "m-server", role: "assistant", content: "E" },
      ],
    };

    const diff = computeConversationDiff(clientConv, serverConv);
    expect(diff.status).toBe("diverged");
    expect(diff.messages.clientOnly.map((m) => m.id)).toEqual(["m-client"]);
    expect(diff.messages.serverOnly.map((m) => m.id)).toEqual(["m-server"]);

    const reconciled = reconcileConversation(clientConv, serverConv);
    expect(reconciled.messages).toHaveLength(5);
    const ids = reconciled.messages.map((m) => m.id);
    expect(ids).toContain("m-client");
    expect(ids).toContain("m-server");
    expect(new Set(ids).size).toBe(5);
  });

  // 5. Same message ID on both sides -> correct deduplication
  it("Scenario 5: Same message ID on both sides -> correct deduplication without duplicated turns", () => {
    const clientConv = {
      id: "c-1",
      version: 1,
      messages: [{ id: "same-id", role: "user", content: "Question", type: "text" }],
    };
    const serverConv = {
      id: "c-1",
      version: 1,
      messages: [{ id: "same-id", role: "user", content: "Question", type: "text" }],
    };

    const diff = computeConversationDiff(clientConv, serverConv);
    expect(diff.messages.identicalCount).toBe(1);
    const reconciled = reconcileConversation(clientConv, serverConv);
    expect(reconciled.messages).toHaveLength(1);
    expect(reconciled.messages[0].id).toBe("same-id");
  });

  // 6. Message update conflict
  it("Scenario 6: Message update conflict -> completed assistant message resolves over in-flight", () => {
    const clientConv = {
      id: "c-1",
      version: 1,
      messages: [{ id: "asst-1", role: "assistant", content: "Partial str...", completed: false }],
    };
    const serverConv = {
      id: "c-1",
      version: 2,
      messages: [{ id: "asst-1", role: "assistant", content: "Complete response.", completed: true, metadata: { cost: 10 } }],
    };

    const reconciled = reconcileConversation(clientConv, serverConv);
    expect(reconciled.messages).toHaveLength(1);
    expect(reconciled.messages[0].content).toBe("Complete response.");
    expect(reconciled.messages[0].completed).toBe(true);
    expect(reconciled.messages[0].metadata).toEqual({ cost: 10 });
  });

  // 7. Delete vs update
  it("Scenario 7: Delete vs update -> deleted message is never silently resurrected", () => {
    const deletedId = "m-deleted";
    const clientConv = {
      id: "c-1",
      version: 2,
      messages: [
        { id: deletedId, role: "user", content: "Deleted message" },
        { id: "m-alive", role: "user", content: "Surviving message" },
      ],
      deletedMessageIds: [deletedId],
    };
    const serverConv = {
      id: "c-1",
      version: 3,
      messages: [
        { id: deletedId, role: "user", content: "Deleted message on server too" },
        { id: "m-alive", role: "user", content: "Surviving message" },
      ],
      deletedMessageIds: [deletedId],
    };

    const reconciled = reconcileConversation(clientConv, serverConv, 0, { deletedMessageIds: [deletedId] });
    expect(reconciled.messages.some((m) => m.id === deletedId)).toBe(false);
    expect(reconciled.messages.some((m) => m.id === "m-alive")).toBe(true);
  });

  // 8. Rename + message update
  it("Scenario 8: Rename + message update -> preserves both independent valid changes", () => {
    const localTime = Date.now();
    const clientConv = {
      id: "c-1",
      title: "New Local Title",
      titleUpdatedAt: localTime + 1000,
      version: 1,
      messages: [{ id: "m-1", role: "user", content: "Q1" }],
    };
    const serverConv = {
      id: "c-1",
      title: "Old Server Title",
      titleUpdatedAt: localTime,
      version: 2,
      messages: [
        { id: "m-1", role: "user", content: "Q1" },
        { id: "m-2", role: "assistant", content: "Answer 1", completed: true },
      ],
    };

    const reconciled = reconcileConversation(clientConv, serverConv, localTime + 1000);
    expect(reconciled.title).toBe("New Local Title");
    expect(reconciled.titleUpdatedAt).toBe(localTime + 1000);
    expect(reconciled.messages).toHaveLength(2);
    expect(reconciled.messages[1].id).toBe("m-2");
  });

  // 9. Concurrent message creation
  it("Scenario 9: Concurrent message creation -> deterministic monotonic ordering", () => {
    const t1 = "2026-10-01T10:00:00.000Z";
    const t2 = "2026-10-01T10:00:05.000Z";
    const localMsgs = [{ id: "c-msg", role: "user", content: "Client concurrent", timestamp: t2 }];
    const remoteMsgs = [{ id: "s-msg", role: "user", content: "Server concurrent", timestamp: t1 }];

    const merged = mergeMessages(localMsgs, remoteMsgs);
    expect(merged).toHaveLength(2);
    expect(merged[0].id).toBe("s-msg");
    expect(merged[1].id).toBe("c-msg");
  });

  // 10. WebSocket + REST divergence
  it("Scenario 10: WebSocket + REST divergence -> VERSION_CONFLICT triggers differential sync", async () => {
    const convId = "conv-ws-rest";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: convId,
          title: "Initial Chat",
          version: 2,
          messages: [{ id: "m-1", role: "user", content: "Initial" }],
        },
      ],
      activeConversationId: convId,
      messages: [{ id: "m-1", role: "user", content: "Initial" }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    api.getConversation.mockImplementation((id) =>
      Promise.resolve({
        id,
        title: "Authoritative Chat",
        version: 5,
        messages: [
          { id: "m-1", role: "user", content: "Initial" },
          { id: "m-rest", role: "assistant", content: "Rest Turn", completed: true },
        ],
      })
    );

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    // Simulate WebSocket VERSION_CONFLICT error
    act(() => {
      emitWs("error", {
        type: "error",
        code: "VERSION_CONFLICT",
        conversationId: convId,
        currentVersion: 5,
        expectedVersion: 2,
      });
    });

    await waitFor(() => {
      const conv = result.current.conversations.find((c) => c.id === convId);
      expect(conv.version).toBe(5);
      expect(conv.messages.some((m) => m.id === "m-rest")).toBe(true);
    });
  });

  // 11. Reconnect after divergence
  it("Scenario 11: Reconnect after divergence -> syncs state with server without duplicating turns", async () => {
    const convId = "conv-reconnect-sync";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: convId,
          title: "Reconnect Chat",
          version: 1,
          messages: [{ id: "m-1", role: "user", content: "Offline Question" }],
        },
      ],
      activeConversationId: convId,
      messages: [{ id: "m-1", role: "user", content: "Offline Question" }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    api.syncConversation.mockImplementationOnce((id, _payload) =>
      Promise.resolve({
        status: "synchronized",
        version: 3,
        conversation: {
          id,
          title: "Reconnect Chat",
          version: 3,
          messages: [
            { id: "m-1", role: "user", content: "Offline Question" },
            { id: "m-2", role: "assistant", content: "Server Replied", completed: true },
          ],
        },
      })
    );

    api.getConversation.mockImplementation((id) =>
      Promise.resolve({
        id,
        title: "Reconnect Chat",
        version: 3,
        messages: [
          { id: "m-1", role: "user", content: "Offline Question" },
          { id: "m-2", role: "assistant", content: "Server Replied", completed: true },
        ],
      })
    );

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    // Trigger reconnect handler
    act(() => {
      emitWs("reconnect", {});
    });

    await waitFor(() => {
      const conv = result.current.conversations.find((c) => c.id === convId);
      expect(conv.version).toBe(3);
      expect(conv.messages).toHaveLength(2);
      expect(conv.messages[1].id).toBe("m-2");
    });
  });

  // 12. Branch divergence
  it("Scenario 12: Branch divergence -> branch maintains independent version and parent metadata", () => {
    const parentConv = {
      id: "parent-1",
      title: "Parent Conversation",
      version: 5,
      messages: [
        { id: "m-1", role: "user", content: "Root" },
        { id: "m-2", role: "assistant", content: "Parent turn" },
      ],
    };
    const branchConv = {
      id: "branch-1",
      title: "Parent Conversation (Branch)",
      parentConversationId: "parent-1",
      parentVersion: 2,
      branchedFromMessageId: "m-1",
      version: 1,
      messages: [
        { id: "m-1", role: "user", content: "Root" },
        { id: "m-b1", role: "user", content: "Branch alternate direction" },
      ],
    };

    const diff = computeConversationDiff(branchConv, parentConv);
    expect(diff.status).toBe("diverged");

    const reconciledBranch = reconcileConversation(branchConv, branchConv);
    expect(reconciledBranch.parentConversationId).toBe("parent-1");
    expect(reconciledBranch.parentVersion).toBe(2);
    expect(reconciledBranch.branchedFromMessageId).toBe("m-1");
    expect(reconciledBranch.messages).toHaveLength(2);
  });

  // 13. Nested / sibling branches
  it("Scenario 13: Nested / sibling branches -> preserves hierarchical lineages", () => {
    const branchA = {
      id: "branch-A",
      parentConversationId: "parent-1",
      parentVersion: 2,
      branchedFromMessageId: "m-root",
      version: 2,
      messages: [{ id: "m-root" }, { id: "m-a1" }],
    };
    const branchB = {
      id: "branch-B",
      parentConversationId: "branch-A",
      parentVersion: 2,
      branchedFromMessageId: "m-a1",
      version: 1,
      messages: [{ id: "m-root" }, { id: "m-a1" }, { id: "m-b1" }],
    };

    const diffNested = computeConversationDiff(branchB, branchA);
    expect(diffNested.status).toBe("client_ahead");

    const reconciledNested = reconcileConversation(branchB, branchB);
    expect(reconciledNested.parentConversationId).toBe("branch-A");
    expect(reconciledNested.branchedFromMessageId).toBe("m-a1");
  });

  // 14. Persistence after synchronization
  it("Scenario 14: Persistence after synchronization -> sanitized state stored in localStorage", () => {
    const conv = {
      id: "c-persist",
      title: "Persisted Chat",
      version: 4,
      deletedMessageIds: ["m-del-1"],
      messages: [
        { id: "m-1", role: "user", content: "Clean message", type: "text" },
      ],
    };
    saveConversationsToStorage([conv], conv.id);

    const stored = loadConversationsFromStorage();
    expect(stored.conversations).toHaveLength(1);
    expect(stored.conversations[0].id).toBe("c-persist");
    expect(stored.conversations[0].version).toBe(4);
    expect(stored.conversations[0].deletedMessageIds).toEqual(["m-del-1"]);
  });

  // 15. Reload after synchronization
  it("Scenario 15: Reload after synchronization -> exact same final state restored", () => {
    const conv = {
      id: "c-reload",
      title: "Reloaded Chat",
      version: 3,
      messages: [
        { id: "m-1", role: "user", content: "Turn 1", type: "text" },
        { id: "m-2", role: "assistant", content: "Turn 2", type: "text", completed: true },
      ],
    };
    saveConversationsToStorage([conv], conv.id);

    const loaded = loadConversationsFromStorage();
    const sanitized = sanitizeConversation(loaded.conversations[0]);
    expect(sanitized.id).toBe("c-reload");
    expect(sanitized.version).toBe(3);
    expect(sanitized.messages).toHaveLength(2);
    expect(sanitized.messages[1].content).toBe("Turn 2");
  });

  // 16. Repeated synchronization is idempotent
  it("Scenario 16: Repeated synchronization is idempotent -> identical state and no mutations", () => {
    const client = {
      id: "c-idem",
      title: "Idempotent Chat",
      version: 2,
      messages: [{ id: "m-1", role: "user", content: "Hello" }],
    };
    const server = {
      id: "c-idem",
      title: "Idempotent Chat",
      version: 2,
      messages: [{ id: "m-1", role: "user", content: "Hello" }],
    };

    const first = reconcileConversation(client, server);
    const second = reconcileConversation(first, server);

    expect(second).toEqual(first);
    expect(second.version).toBe(first.version);
    expect(second.messages).toEqual(first.messages);

    const diff = computeConversationDiff(client, server);
    const fromReconcileDiff = reconcileConversationDiff(client, server, diff);
    expect(fromReconcileDiff.reconciled.messages).toEqual(first.messages);
    const applied = applyConversationDiff(client, diff);
    expect(applied.messages).toEqual(first.messages);
  });

  // 17. Stale version cannot overwrite newer state
  it("Scenario 17: Stale version cannot overwrite newer state", () => {
    const staleClient = {
      id: "c-stale",
      version: 1,
      messages: [],
    };
    const newerServer = {
      id: "c-stale",
      version: 10,
      messages: [
        { id: "m-1", role: "user", content: "Committed turn 1" },
        { id: "m-2", role: "assistant", content: "Committed turn 2" },
      ],
    };

    const reconciled = reconcileConversation(staleClient, newerServer);
    expect(reconciled.version).toBe(10);
    expect(reconciled.messages).toHaveLength(2);
    expect(reconciled.messages[0].id).toBe("m-1");
  });

  // 18. Large conversation with small diff
  it("Scenario 18: Large conversation with small diff -> linear time O(N) diff", () => {
    const count = 500;
    const base = [];
    for (let i = 0; i < count; i++) {
      base.push({
        id: `msg-${i}`,
        role: i % 2 === 0 ? "user" : "assistant",
        content: `Content ${i}`,
        type: "text",
      });
    }

    const clientConv = { id: "large-1", version: 10, messages: [...base, { id: "new-client", role: "user", content: "Delta" }] };
    const serverConv = { id: "large-1", version: 10, messages: base };

    const start = performance.now();
    const diff = computeConversationDiff(clientConv, serverConv);
    const reconciled = reconcileConversation(clientConv, serverConv);
    const duration = performance.now() - start;

    expect(diff.messages.clientOnly).toHaveLength(1);
    expect(reconciled.messages).toHaveLength(count + 1);
    expect(duration).toBeLessThan(500); // Must be fast O(N)
  });

  // 19. Partial synchronization failure
  it("Scenario 19: Partial synchronization failure -> safe recovery fallback without corrupted state", async () => {
    const convId = "conv-partial-fail";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: convId,
          title: "Safe Chat",
          version: 2,
          messages: [{ id: "m-1", role: "user", content: "Local query" }],
        },
      ],
      activeConversationId: convId,
      messages: [{ id: "m-1", role: "user", content: "Local query" }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    api.syncConversation.mockImplementation(() => Promise.reject(new Error("Network Error")));
    api.getConversation.mockImplementation(() => Promise.reject(new Error("Offline")));

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    let syncResult;
    await act(async () => {
      syncResult = await result.current.syncConversation(convId);
    });

    // Conversation state is preserved safely
    expect(syncResult).toBeTruthy();
    expect(syncResult.id).toBe(convId);
    expect(syncResult.version).toBe(2);
    expect(syncResult.messages).toHaveLength(1);
  });

  // 20. Duplicate events / messages do not create duplicates
  it("Scenario 20: Duplicate events / messages do not create duplicates", () => {
    const raw = [
      { id: "dup-1", role: "user", content: "Text" },
      { id: "dup-1", role: "user", content: "Text" },
      { id: "dup-2", role: "assistant", content: "Reply" },
      { id: "dup-2", role: "assistant", content: "Reply" },
    ];
    const deduped = dedupeMessages(raw);
    expect(deduped).toHaveLength(2);
    expect(deduped[0].id).toBe("dup-1");
    expect(deduped[1].id).toBe("dup-2");

    const merged = mergeMessages(raw, raw);
    expect(merged).toHaveLength(2);
  });
});
