import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import { api } from "../services/api";
import {
  STORAGE_KEY,
  STORAGE_VERSION,
  loadConversationsFromStorage,
} from "../utils/storage";

let mockWsHandlers = new Map();

vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(() => Promise.resolve([])),
    createConversation: vi.fn((data) => {
      const id = data?.id || `conv-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      return Promise.resolve({ id, title: "New Chat", messages: [] });
    }),
    getConversation: vi.fn((id) => Promise.resolve({ id, title: "Chat " + id, messages: [] })),
    deleteConversation: vi.fn(() => Promise.resolve({ success: true })),
    sendMessage: vi.fn(() => Promise.resolve({ content: "Echo reply", type: "text" })),
    branchConversation: vi.fn(),
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

describe("Issue #404 — Conversation State Divergence Regression Tests", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    api.listConversations.mockImplementation(() => Promise.resolve([]));
    api.deleteConversation.mockImplementation(() => Promise.resolve({ success: true }));
    api.getConversation.mockImplementation((id) => Promise.resolve({ id, title: "Chat " + id, messages: [] }));
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("Test 1 — Message + Rename Race: older message operation must not overwrite newer title", async () => {
    const convId = "conv-rename-race";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: convId,
          title: "Original Title",
          messages: [{ id: "m-1", role: "user", content: "Initial query", type: "text" }],
          updatedAt: "2026-10-01T10:00:00.000Z",
        },
      ],
      activeConversationId: convId,
      messages: [{ id: "m-1", role: "user", content: "Initial query", type: "text" }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    let resolveGetConv;
    const pendingGetConv = new Promise((resolve) => {
      resolveGetConv = resolve;
    });

    api.getConversation.mockImplementation((id) => {
      if (id === convId) return pendingGetConv;
      return Promise.resolve({ id, title: "Other", messages: [] });
    });

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    expect(result.current.activeConversation).toBe(convId);
    expect(result.current.conversations[0].title).toBe("Original Title");

    // 1. Message/getConversation fetch starts for the conversation
    let selectPromise;
    act(() => {
      selectPromise = result.current.selectConversation(convId);
    });

    // 2. User renames the conversation while the fetch is pending
    act(() => {
      result.current.renameConversation(convId, "My Renamed Title");
    });

    expect(result.current.conversations.find((c) => c.id === convId)?.title).toBe("My Renamed Title");

    // 3. Older fetch completes later with the older title
    await act(async () => {
      resolveGetConv({
        id: convId,
        title: "Original Title",
        messages: [
          { id: "m-1", role: "user", content: "Initial query", type: "text" },
          { id: "m-2", role: "assistant", content: "Server reply", type: "text" },
        ],
      });
      await selectPromise;
    });

    // Expected: latest messages AND latest title ("My Renamed Title")
    const updatedConv = result.current.conversations.find((c) => c.id === convId);
    expect(updatedConv.title).toBe("My Renamed Title");
    expect(updatedConv.messages).toHaveLength(2);
    expect(updatedConv.messages[1].content).toBe("Server reply");

    // Verify localStorage also reflects the latest title
    const stored = loadConversationsFromStorage(localStorage);
    const storedConv = stored.conversations.find((c) => c.id === convId);
    expect(storedConv.title).toBe("My Renamed Title");
  });

  it("Test 2 — Conversation Switch Race: Conversation A operation must only update A, B remains unchanged", async () => {
    const convA = "conv-switch-A";
    const convB = "conv-switch-B";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: convA,
          title: "Chat A",
          messages: [{ id: "m-a1", role: "user", content: "Msg A1", type: "text" }],
          updatedAt: "2026-10-01T10:00:00.000Z",
        },
        {
          id: convB,
          title: "Chat B",
          messages: [{ id: "m-b1", role: "user", content: "Msg B1", type: "text" }],
          updatedAt: "2026-10-01T10:00:00.000Z",
        },
      ],
      activeConversationId: convA,
      messages: [{ id: "m-a1", role: "user", content: "Msg A1", type: "text" }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    expect(result.current.activeConversation).toBe(convA);

    // 1. Conversation A operation starts (sendMessage)
    act(() => {
      result.current.sendMessage("Query in A");
    });

    // 2. User switches to Conversation B while A's operation is pending
    await act(async () => {
      await result.current.selectConversation(convB);
    });

    expect(result.current.activeConversation).toBe(convB);
    expect(result.current.messages.map((m) => m.content)).toEqual(["Msg B1"]);

    // 3. Operation A completes over WebSocket
    act(() => {
      emitWs("stream", {
        type: "stream",
        conversationId: convA,
        content: "Reply in A",
        done: true,
        messageId: "server-reply-a",
      });
    });

    // Expected: Active conversation B is completely unchanged
    expect(result.current.activeConversation).toBe(convB);
    expect(result.current.messages.map((m) => m.content)).toEqual(["Msg B1"]);

    const convBInState = result.current.conversations.find((c) => c.id === convB);
    expect(convBInState.messages.map((m) => m.content)).toEqual(["Msg B1"]);

    // Conversation A in conversations is updated with its new turn
    const convAInState = result.current.conversations.find((c) => c.id === convA);
    expect(convAInState.messages.some((m) => m.content === "Reply in A")).toBe(true);
  });

  it("Test 3 — Delete + Pending Operation: deleted conversation must not be resurrected", async () => {
    const convId = "conv-to-delete";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: convId,
          title: "To Be Deleted",
          messages: [{ id: "m-1", role: "user", content: "Hi", type: "text" }],
          updatedAt: "2026-10-01T10:00:00.000Z",
        },
      ],
      activeConversationId: convId,
      messages: [{ id: "m-1", role: "user", content: "Hi", type: "text" }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    let resolveGetConv;
    const pendingGetConv = new Promise((resolve) => {
      resolveGetConv = resolve;
    });
    api.getConversation.mockImplementation(() => pendingGetConv);

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    // 1. Conversation operation starts (e.g. selectConversation initiates getConversation)
    let selectPromise;
    act(() => {
      selectPromise = result.current.selectConversation(convId);
    });

    // 2. Conversation is deleted while operation is pending
    await act(async () => {
      await result.current.deleteConversation(convId);
    });

    expect(result.current.conversations).toHaveLength(0);
    expect(result.current.activeConversation).toBeNull();

    // 3. Older operation completes
    await act(async () => {
      resolveGetConv({
        id: convId,
        title: "Resurrect Attempt",
        messages: [{ id: "m-1", role: "user", content: "Hi", type: "text" }],
      });
      await selectPromise;
    });

    // Expected: Conversation remains deleted and is not resurrected
    expect(result.current.conversations).toHaveLength(0);
    expect(result.current.activeConversation).toBeNull();

    const stored = loadConversationsFromStorage(localStorage);
    expect(stored.conversations).toHaveLength(0);
  });

  it("Test 3b — Delete + listConversations Race: initial listConversations must not resurrect deleted conversation", async () => {
    const convId = "conv-deleted-during-list";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: convId,
          title: "Initial Chat",
          messages: [{ id: "m-1", role: "user", content: "Hi", type: "text" }],
          updatedAt: "2026-10-01T10:00:00.000Z",
        },
      ],
      activeConversationId: convId,
      messages: [{ id: "m-1", role: "user", content: "Hi", type: "text" }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    let resolveList;
    const pendingList = new Promise((resolve) => {
      resolveList = resolve;
    });
    api.listConversations.mockImplementation(() => pendingList);

    const { result } = renderHook(() => useChat({}));

    // Before listConversations finishes, delete the conversation
    await act(async () => {
      await result.current.deleteConversation(convId);
    });

    expect(result.current.conversations).toHaveLength(0);

    // Now listConversations returns containing the deleted conversation
    await act(async () => {
      resolveList([
        {
          id: convId,
          title: "Initial Chat from Remote",
          createdAt: "2026-10-01T10:00:00.000Z",
          messages: [],
        },
      ]);
    });

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    // The deleted conversation must NOT be restored to conversations
    expect(result.current.conversations).toHaveLength(0);
    const stored = loadConversationsFromStorage(localStorage);
    expect(stored.conversations).toHaveLength(0);
  });

  it("Test 4 — Branch + Parent Update Race: parent retains latest state and branch relationship remains correct", async () => {
    const parentId = "parent-conv-1";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: parentId,
          title: "Parent Conversation",
          messages: [
            { id: "m-p1", role: "user", content: "Step 1", type: "text" },
            { id: "m-p2", role: "assistant", content: "Answer 1", type: "text" },
          ],
          updatedAt: "2026-10-01T10:00:00.000Z",
        },
      ],
      activeConversationId: parentId,
      messages: [
        { id: "m-p1", role: "user", content: "Step 1", type: "text" },
        { id: "m-p2", role: "assistant", content: "Answer 1", type: "text" },
      ],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    let resolveBranch;
    const pendingBranch = new Promise((resolve) => {
      resolveBranch = resolve;
    });
    api.branchConversation.mockImplementation(() => pendingBranch);

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    // 1. Branching starts from message m-p1
    let branchPromise;
    act(() => {
      branchPromise = result.current.branchConversation("m-p1", parentId);
    });

    // 2. Parent receives an update while branching is in-flight
    act(() => {
      result.current.renameConversation(parentId, "Parent Updated Title");
    });

    // 3. Branch resolves
    const branchId = "branch-conv-new";
    await act(async () => {
      resolveBranch({
        id: branchId,
        title: "Parent Conversation (Branch)",
        parentConversationId: parentId,
        branchedFromMessageId: "m-p1",
        messages: [{ id: "m-p1", role: "user", content: "Step 1", type: "text" }],
      });
      await branchPromise;
    });

    // Expected:
    // - Branch conversation exists and is active
    expect(result.current.activeConversation).toBe(branchId);
    const branchInState = result.current.conversations.find((c) => c.id === branchId);
    expect(branchInState).toBeDefined();
    expect(branchInState.parentConversationId).toBe(parentId);
    expect(branchInState.branchedFromMessageId).toBe("m-p1");

    // - Parent retains latest state ("Parent Updated Title")
    const parentInState = result.current.conversations.find((c) => c.id === parentId);
    expect(parentInState.title).toBe("Parent Updated Title");

    // - Storage reflects latest parent title and branch relationship
    const stored = loadConversationsFromStorage(localStorage);
    const storedParent = stored.conversations.find((c) => c.id === parentId);
    expect(storedParent.title).toBe("Parent Updated Title");
  });

  it("Test 5 — Multiple Out-of-Order Updates: latest logical state wins over late-resolving promises", async () => {
    const convId = "conv-out-of-order";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: convId,
          title: "Chat Out Of Order",
          messages: [{ id: "m-1", role: "user", content: "First", type: "text" }],
          updatedAt: "2026-10-01T10:00:00.000Z",
        },
      ],
      activeConversationId: convId,
      messages: [{ id: "m-1", role: "user", content: "First", type: "text" }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    let resolveOp1, resolveOp2;
    const promise1 = new Promise((resolve) => {
      resolveOp1 = resolve;
    });
    const promise2 = new Promise((resolve) => {
      resolveOp2 = resolve;
    });

    let callCount = 0;
    api.getConversation.mockImplementation((id) => {
      if (id === convId) {
        callCount++;
        return callCount === 1 ? promise1 : promise2;
      }
      return Promise.resolve({ id, title: "Other", messages: [] });
    });

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    // Start Op 1 (snapshot with 1 message)
    let p1;
    act(() => {
      p1 = result.current.selectConversation(convId);
    });

    // Start Op 2 (snapshot with 3 messages)
    let p2;
    act(() => {
      p2 = result.current.selectConversation(convId);
    });

    // Resolve Op 2 FIRST (latest logical state)
    await act(async () => {
      resolveOp2({
        id: convId,
        title: "Chat Out Of Order",
        messages: [
          { id: "m-1", role: "user", content: "First", type: "text" },
          { id: "m-2", role: "assistant", content: "Second", type: "text" },
          { id: "m-3", role: "user", content: "Third", type: "text" },
        ],
      });
      await p2;
    });

    expect(result.current.messages).toHaveLength(3);

    // Now resolve Op 1 LATER with stale snapshot (1 message)
    await act(async () => {
      resolveOp1({
        id: convId,
        title: "Chat Out Of Order",
        messages: [{ id: "m-1", role: "user", content: "First", type: "text" }],
      });
      await p1;
    });

    // Expected: Stale Op 1 must NOT revert or overwrite state back to 1 message!
    expect(result.current.messages).toHaveLength(3);
    expect(result.current.messages.map((m) => m.content)).toEqual(["First", "Second", "Third"]);

    const convInState = result.current.conversations.find((c) => c.id === convId);
    expect(convInState.messages).toHaveLength(3);
  });
});
