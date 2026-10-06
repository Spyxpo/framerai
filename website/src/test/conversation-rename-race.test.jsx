import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import { api } from "../services/api";
import {
  STORAGE_KEY,
  STORAGE_VERSION,
  loadConversationsFromStorage,
} from "../utils/storage";

let mockWsCallbacks = {};
let wsConnected = true;

vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(),
    createConversation: vi.fn(),
    getConversation: vi.fn(),
    deleteConversation: vi.fn(),
    branchConversation: vi.fn(),
    sendMessage: vi.fn(),
  },
}));

vi.mock("../services/websocket", () => {
  return {
    WebSocketClient: class {
      constructor() {
        mockWsCallbacks = {};
      }
      connect() {
        return Promise.resolve();
      }
      disconnect() {
        mockWsCallbacks = {};
      }
      on(event, cb) {
        mockWsCallbacks[event] = cb;
        return () => {
          delete mockWsCallbacks[event];
        };
      }
      send(data) {
        this.lastSent = data;
      }
      isConnected() {
        return wsConnected;
      }
      get ws() {
        return { readyState: wsConnected ? 1 : 3 };
      }
    },
  };
});

describe("Issue #431 — Conversation Rename State Protection during Concurrent Updates", () => {
  beforeEach(() => {
    localStorage.clear();
    mockWsCallbacks = {};
    wsConnected = true;
    vi.clearAllMocks();

    api.listConversations.mockResolvedValue([]);
    api.createConversation.mockImplementation(async () => ({
      id: "conv-default",
      title: "New Chat",
      messages: [],
      createdAt: new Date().toISOString(),
    }));
    api.getConversation.mockImplementation(async (id) => ({
      id,
      title: "Backend Snapshot Title",
      messages: [],
      createdAt: new Date().toISOString(),
    }));
    api.deleteConversation.mockResolvedValue({ success: true });
    api.branchConversation.mockImplementation(async (parentId, msgId) => ({
      id: `branch-${parentId}`,
      title: "Backend Parent Title (Branch)",
      parentConversationId: parentId,
      branchedFromMessageId: msgId,
      messages: [],
      createdAt: new Date().toISOString(),
    }));
    api.sendMessage.mockResolvedValue({
      id: "reply-1",
      role: "assistant",
      content: "Assistant reply content",
      type: "text",
      userMessageId: "user-1",
    });
  });

  it("Scenario A: Rename + immediate message send (REST) preserves latest title", async () => {
    wsConnected = false;
    const convId = "conv-scenario-a";
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [
          {
            id: convId,
            title: "Original Title",
            messages: [],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
        ],
        activeConversationId: convId,
      })
    );

    let resolveSendMessage;
    api.sendMessage.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveSendMessage = resolve;
        })
    );

    const { result } = renderHook(() => useChat());
    await act(async () => {});

    // 1. User renames the conversation
    act(() => {
      result.current.renameConversation(convId, "User Renamed Title A");
    });

    expect(result.current.conversations.find((c) => c.id === convId)?.title).toBe(
      "User Renamed Title A"
    );

    // 2. User immediately sends a message
    act(() => {
      result.current.sendMessage("A question for assistant");
    });

    // 3. Backend message operation resolves later
    await act(async () => {
      resolveSendMessage({
        id: "msg-assistant-1",
        role: "assistant",
        content: "Detailed response",
        type: "text",
        userMessageId: "msg-user-1",
      });
    });

    // Expected: The renamed title MUST NOT be overwritten
    const currentConv = result.current.conversations.find((c) => c.id === convId);
    expect(currentConv?.title).toBe("User Renamed Title A");

    // Local storage must also reflect the latest renamed title
    const stored = loadConversationsFromStorage(localStorage);
    expect(stored.conversations.find((c) => c.id === convId)?.title).toBe(
      "User Renamed Title A"
    );
  });

  it("Scenario A2: Rename + WebSocket message streaming preserves renamed title", async () => {
    wsConnected = true;
    const convId = "conv-scenario-a2";
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [
          {
            id: convId,
            title: "Original Title A2",
            messages: [],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
        ],
        activeConversationId: convId,
      })
    );

    const { result } = renderHook(() => useChat());
    await act(async () => {});

    // 1. User renames
    act(() => {
      result.current.renameConversation(convId, "Renamed WS Title");
    });

    // 2. User immediately sends a message via WS
    act(() => {
      result.current.sendMessage("A question via WS");
    });

    // 3. WS stream completes
    act(() => {
      if (mockWsCallbacks["stream"]) {
        mockWsCallbacks["stream"]({
          conversationId: convId,
          content: "WS response text",
          done: true,
        });
      }
    });

    expect(result.current.conversations.find((c) => c.id === convId)?.title).toBe(
      "Renamed WS Title"
    );

    const stored = loadConversationsFromStorage(localStorage);
    expect(stored.conversations.find((c) => c.id === convId)?.title).toBe(
      "Renamed WS Title"
    );
  });

  it("Scenario B: Rename during active generation/streaming preserves renamed title upon completion", async () => {
    const convId = "conv-scenario-b";
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [
          {
            id: convId,
            title: "Initial Streaming Chat",
            messages: [
              { id: "m-u1", role: "user", content: "Tell me a long story", type: "text" },
              { id: "m-a1", role: "assistant", content: "", type: "text", completed: false },
            ],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
        ],
        activeConversationId: convId,
      })
    );

    const { result } = renderHook(() => useChat());
    await act(async () => {});

    // Simulate streaming chunks arriving
    act(() => {
      if (mockWsCallbacks["stream"]) {
        mockWsCallbacks["stream"]({
          conversationId: convId,
          messageId: "m-a1",
          content: "Once upon a time",
          done: false,
        });
      }
    });

    // User renames conversation during active streaming
    act(() => {
      result.current.renameConversation(convId, "Story Project 2026");
    });

    expect(result.current.conversations.find((c) => c.id === convId)?.title).toBe(
      "Story Project 2026"
    );

    // Stream completes
    act(() => {
      if (mockWsCallbacks["stream"]) {
        mockWsCallbacks["stream"]({
          conversationId: convId,
          messageId: "m-a1",
          content: "Once upon a time in a galaxy far away. The End.",
          done: true,
        });
      }
    });

    // Final title must remain the renamed title
    const updated = result.current.conversations.find((c) => c.id === convId);
    expect(updated?.title).toBe("Story Project 2026");

    const stored = loadConversationsFromStorage(localStorage);
    expect(stored.conversations.find((c) => c.id === convId)?.title).toBe(
      "Story Project 2026"
    );
  });

  it("Scenario C: Rename + conversation switching preserves title when returning", async () => {
    const convA = "conv-switch-A";
    const convB = "conv-switch-B";
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [
          {
            id: convA,
            title: "Original Title A",
            messages: [{ id: "ma-1", role: "user", content: "Hi A", type: "text" }],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
          {
            id: convB,
            title: "Original Title B",
            messages: [{ id: "mb-1", role: "user", content: "Hi B", type: "text" }],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
        ],
        activeConversationId: convA,
      })
    );

    let resolveGetConvA;
    api.getConversation.mockImplementation((id) => {
      if (id === convA) {
        return new Promise((resolve) => {
          resolveGetConvA = resolve;
        });
      }
      return Promise.resolve({
        id,
        title: "Stale Server B",
        messages: [{ id: "mb-1", role: "user", content: "Hi B", type: "text" }],
      });
    });

    const { result } = renderHook(() => useChat());
    await act(async () => {});

    // 1. Rename conversation A
    act(() => {
      result.current.renameConversation(convA, "Renamed Conversation A");
    });
    expect(result.current.conversations.find((c) => c.id === convA)?.title).toBe(
      "Renamed Conversation A"
    );

    // 2. Switch to conversation B
    await act(async () => {
      await result.current.selectConversation(convB);
    });
    expect(result.current.activeConversation).toBe(convB);

    // 3. Switch back to conversation A (initiates getConversation for A)
    let selectPromiseA;
    act(() => {
      selectPromiseA = result.current.selectConversation(convA);
    });

    // 4. Remote getConversation(convA) returns with backend's stale title
    await act(async () => {
      resolveGetConvA({
        id: convA,
        title: "Stale Backend Snapshot A",
        messages: [{ id: "ma-1", role: "user", content: "Hi A", type: "text" }],
      });
      await selectPromiseA;
    });

    // Expected: Renamed title for A is preserved
    const convAState = result.current.conversations.find((c) => c.id === convA);
    expect(convAState?.title).toBe("Renamed Conversation A");
  });

  it("Scenario D: Rename parent conversation + branch preserves parent title and derives branch title", async () => {
    const parentId = "parent-conv-1";
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [
          {
            id: parentId,
            title: "Initial Parent Title",
            messages: [
              { id: "m-p1", role: "user", content: "First turn", type: "text" },
              { id: "m-p2", role: "assistant", content: "First answer", type: "text", completed: true },
            ],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
        ],
        activeConversationId: parentId,
      })
    );

    // Backend branchConversation returns branch title using its own stale parent title
    api.branchConversation.mockResolvedValue({
      id: "branch-new-1",
      title: "Initial Parent Title (Branch)",
      parentConversationId: parentId,
      branchedFromMessageId: "m-p1",
      messages: [{ id: "m-p1", role: "user", content: "First turn", type: "text" }],
      createdAt: new Date().toISOString(),
    });

    const { result } = renderHook(() => useChat());
    await act(async () => {});

    // 1. Rename parent conversation
    act(() => {
      result.current.renameConversation(parentId, "Master Architecture Plan");
    });

    expect(result.current.conversations.find((c) => c.id === parentId)?.title).toBe(
      "Master Architecture Plan"
    );

    // 2. Branch from parent message
    await act(async () => {
      await result.current.branchConversation("m-p1", parentId);
    });

    // Verify parent title is preserved and branch title incorporates parent's updated title
    const parent = result.current.conversations.find((c) => c.id === parentId);
    const branch = result.current.conversations.find((c) => c.id === "branch-new-1");

    expect(parent?.title).toBe("Master Architecture Plan");
    expect(branch?.title).toBe("Master Architecture Plan (Branch)");
  });

  it("Scenario E: Multiple rapid renames ensure latest valid rename wins", async () => {
    const convId = "rapid-rename-conv";
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [
          {
            id: convId,
            title: "Initial",
            messages: [],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
        ],
        activeConversationId: convId,
      })
    );

    const { result } = renderHook(() => useChat());
    await act(async () => {});

    act(() => {
      result.current.renameConversation(convId, "Rename 1");
      result.current.renameConversation(convId, "Rename 2");
      result.current.renameConversation(convId, "Rename 3");
      result.current.renameConversation(convId, "Final Intended Title");
    });

    expect(result.current.conversations.find((c) => c.id === convId)?.title).toBe(
      "Final Intended Title"
    );

    const stored = loadConversationsFromStorage(localStorage);
    expect(stored.conversations.find((c) => c.id === convId)?.title).toBe(
      "Final Intended Title"
    );
  });

  it("Scenario F: Refresh/reload preserves renamed title against subsequent getConversation", async () => {
    const convId = "refresh-preservation-conv";
    // 1. Initial render: user renames conversation
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [
          {
            id: convId,
            title: "Original Before Reload",
            messages: [{ id: "m1", role: "user", content: "Hi", type: "text" }],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
        ],
        activeConversationId: convId,
      })
    );

    const first = renderHook(() => useChat());
    await act(async () => {});
    act(() => {
      first.result.current.renameConversation(convId, "Persisted Custom Title");
    });
    expect(first.result.current.conversations[0].title).toBe("Persisted Custom Title");
    first.unmount();

    // 2. Simulate page reload / new mount
    api.getConversation.mockResolvedValue({
      id: convId,
      title: "Stale Backend Original",
      messages: [
        { id: "m1", role: "user", content: "Hi", type: "text" },
        { id: "m2", role: "assistant", content: "Stored reply", type: "text" },
      ],
    });

    const second = renderHook(() => useChat());
    await act(async () => {});

    // Trigger selectConversation on the reloaded hook
    await act(async () => {
      await second.result.current.selectConversation(convId);
    });

    // The persisted renamed title MUST win over stale backend snapshot
    const reloadedConv = second.result.current.conversations.find((c) => c.id === convId);
    expect(reloadedConv?.title).toBe("Persisted Custom Title");
    expect(reloadedConv?.messages).toHaveLength(2);

    const storedAfterReload = loadConversationsFromStorage(localStorage);
    expect(storedAfterReload.conversations.find((c) => c.id === convId)?.title).toBe(
      "Persisted Custom Title"
    );
  });

  it("Scenario G: Out-of-order delete nextId fetch cannot overwrite renamed nextId title", async () => {
    const conv1 = "conv-to-delete";
    const nextConv = "conv-next-id";
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [
          {
            id: conv1,
            title: "To Be Deleted",
            messages: [{ id: "m-del", role: "user", content: "Bye" }],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
          {
            id: nextConv,
            title: "Old Next Title",
            messageCount: 1,
            messages: [],
            updatedAt: "2026-10-01T10:00:00.000Z",
          },
        ],
        activeConversationId: conv1,
      })
    );

    let resolveGetNextConv;
    api.getConversation.mockImplementation((id) => {
      if (id === nextConv) {
        return new Promise((resolve) => {
          resolveGetNextConv = resolve;
        });
      }
      return Promise.resolve({ id, messages: [] });
    });

    const { result } = renderHook(() => useChat());
    await act(async () => {});

    // User renames nextConv right before or during deletion
    act(() => {
      result.current.renameConversation(nextConv, "Important Next Project");
    });

    // Delete conv1, which activates nextConv and triggers api.getConversation(nextConv)
    let deletePromise;
    act(() => {
      deletePromise = result.current.deleteConversation(conv1);
    });

    // Stale getConversation resolves with backend's old title
    await act(async () => {
      resolveGetNextConv({
        id: nextConv,
        title: "Old Next Title From Server",
        messages: [{ id: "m-next-1", role: "user", content: "Hello next" }],
      });
      await deletePromise;
    });

    const nextConvState = result.current.conversations.find((c) => c.id === nextConv);
    expect(nextConvState?.title).toBe("Important Next Project");
    expect(nextConvState?.messages).toHaveLength(1);
  });
});
