import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act, waitFor, renderHook } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useChat } from "../hooks/useChat";
import Sidebar from "../components/Sidebar/Sidebar";
import Chat from "../components/Chat/Chat";
import { api } from "../services/api";
import {
  STORAGE_KEY,
  STORAGE_VERSION,
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

describe("Issue #396 — Conversation Deletion & Selected State Management", () => {
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

  it("1. Reproduction: deleting active conversation switches to remaining conversation and clears stale messages", async () => {
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "conv-1",
          title: "Conversation 1",
          messages: [
            { id: "m-1", role: "user", content: "Message from conversation 1", type: "text" },
            { id: "m-2", role: "assistant", content: "Reply in conversation 1", type: "text" },
          ],
        },
        {
          id: "conv-2",
          title: "Conversation 2",
          messages: [
            { id: "m-3", role: "user", content: "Message from conversation 2", type: "text" },
          ],
        },
      ],
      activeConversationId: "conv-1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    expect(result.current.activeConversation).toBe("conv-1");
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].content).toBe("Message from conversation 1");

    // Delete the active conversation
    await act(async () => {
      await result.current.deleteConversation("conv-1");
    });

    // 1. Immediately removed from list
    expect(result.current.conversations.map((c) => c.id)).toEqual(["conv-2"]);

    // 2. Active conversation switched to remaining conversation
    expect(result.current.activeConversation).toBe("conv-2");

    // 3. Stale messages from conv-1 are NOT visible; conv-2 messages are shown
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].content).toBe("Message from conversation 2");

    // 4. Persisted storage updated
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    expect(saved.conversations).toHaveLength(1);
    expect(saved.conversations[0].id).toBe("conv-2");
    expect(saved.activeConversationId).toBe("conv-2");
  });

  it("2. Reproduction: deleting the last remaining conversation clears active state, messages, and storage", async () => {
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "last-conv",
          title: "Only Chat",
          messages: [
            { id: "m-1", role: "user", content: "Sole message", type: "text" },
          ],
        },
      ],
      activeConversationId: "last-conv",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    expect(result.current.activeConversation).toBe("last-conv");
    expect(result.current.messages).toHaveLength(1);

    await act(async () => {
      await result.current.deleteConversation("last-conv");
    });

    // State is fully cleared
    expect(result.current.conversations).toEqual([]);
    expect(result.current.activeConversation).toBeNull();
    expect(result.current.messages).toEqual([]);
    expect(result.current.loadingMessages).toBe(false);

    // Persisted storage is completely cleared
    const stored = localStorage.getItem(STORAGE_KEY);
    expect(stored).toBeNull();
  });

  it("3. Loading state cleanup: deleting a conversation while message loading is active resets loadingMessages", async () => {
    let resolveGet = null;
    api.getConversation.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveGet = resolve;
        })
    );

    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        { id: "c1", title: "Chat 1", messages: [] },
        { id: "c2", title: "Chat 2", messages: [] },
      ],
      activeConversationId: "c1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    // Trigger selectConversation on c1 which starts loading
    act(() => {
      result.current.selectConversation("c1");
    });
    expect(result.current.loadingMessages).toBe(true);

    // Delete c1 while it is loading messages
    await act(async () => {
      await result.current.deleteConversation("c1");
    });

    // loadingMessages must NOT stay stuck at true
    expect(result.current.loadingMessages).toBe(false);
    expect(result.current.activeConversation).toBe("c2");

    // If getConversation resolves late, it should not resurrect c1 or set loadingMessages
    if (resolveGet) {
      await act(async () => {
        resolveGet({ id: "c1", title: "Chat 1", messages: [{ id: "late-m", role: "assistant", content: "Late" }] });
      });
    }
    expect(result.current.loadingMessages).toBe(false);
    expect(result.current.conversations.some((c) => c.id === "c1")).toBe(false);
    expect(result.current.activeConversation).toBe("c2");
  });

  it("4. Pending approval cleanup: deleting a conversation clears any pending approval for that conversation", async () => {
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        { id: "conv-with-approval", title: "Approval Chat", messages: [] },
      ],
      activeConversationId: "conv-with-approval",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    // Simulate an approval request arriving over WebSocket
    act(() => {
      emitWs("approval_request", {
        approvalId: "appr-123",
        conversationId: "conv-with-approval",
        command: "rm -rf /tmp/test",
        argv: ["rm", "-rf", "/tmp/test"],
        root: "/tmp",
      });
    });

    expect(result.current.pendingApproval).not.toBeNull();
    expect(result.current.pendingApproval.approvalId).toBe("appr-123");

    // Delete the conversation
    await act(async () => {
      await result.current.deleteConversation("conv-with-approval");
    });

    // Pending approval must be dismissed
    expect(result.current.pendingApproval).toBeNull();
  });

  it("5. Generation cleanup: late stream frames for deleted conversation do not bleed into the new active conversation", async () => {
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "stream-conv",
          title: "Streaming Chat",
          messages: [{ id: "m-user", role: "user", content: "Tell me a joke", type: "text" }],
        },
        {
          id: "target-conv",
          title: "Clean Chat",
          messages: [
            { id: "m-target", role: "user", content: "Clean message", type: "text" },
            { id: "m-target-ast", role: "assistant", content: "Existing reply", type: "text", completed: false },
          ],
        },
      ],
      activeConversationId: "stream-conv",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    // Mark as streaming
    act(() => {
      emitWs("typing", { conversationId: "stream-conv" });
    });
    expect(result.current.streaming).toBe(true);

    // Delete stream-conv while streaming
    await act(async () => {
      await result.current.deleteConversation("stream-conv");
    });

    expect(result.current.streaming).toBe(false);
    expect(result.current.activeConversation).toBe("target-conv");

    // Stale late stream frame arrives without conversationId (or with stream-conv)
    act(() => {
      emitWs("stream", {
        content: "Ghost message from deleted conversation",
        done: false,
      });
    });

    // The active conversation target-conv must NOT receive the ghost chunk
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].content).toBe("Clean message");
    expect(result.current.messages[1].content).toBe("Existing reply");
  });

  it("6. Optimistic deletion: state updates immediately even if backend delete API is delayed", async () => {
    let resolveDelete = null;
    api.deleteConversation.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveDelete = resolve;
        })
    );

    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        { id: "c-slow", title: "Slow Delete Chat", messages: [{ id: "m1", role: "user", content: "Slow" }] },
        { id: "c-next", title: "Next Chat", messages: [{ id: "m2", role: "user", content: "Next" }] },
      ],
      activeConversationId: "c-slow",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    // Trigger deletion without waiting for promise to resolve
    let deletePromise;
    act(() => {
      deletePromise = result.current.deleteConversation("c-slow");
    });

    // Immediately, before delete API resolves:
    expect(result.current.conversations.map((c) => c.id)).toEqual(["c-next"]);
    expect(result.current.activeConversation).toBe("c-next");
    expect(result.current.messages[0].content).toBe("Next");

    // Clean up delayed promise
    await act(async () => {
      resolveDelete({ success: true });
      await deletePromise;
    });
  });

  it("7. Rapid deletion: consecutive rapid deletes correctly transition to next and then empty state", async () => {
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        { id: "rapid-1", title: "Rapid 1", messages: [{ id: "m1", role: "user", content: "R1" }] },
        { id: "rapid-2", title: "Rapid 2", messages: [{ id: "m2", role: "user", content: "R2" }] },
        { id: "rapid-3", title: "Rapid 3", messages: [{ id: "m3", role: "user", content: "R3" }] },
      ],
      activeConversationId: "rapid-1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    await act(async () => {
      // Rapidly delete rapid-1
      result.current.deleteConversation("rapid-1");
      // And immediately delete rapid-2
      result.current.deleteConversation("rapid-2");
    });

    expect(result.current.conversations.map((c) => c.id)).toEqual(["rapid-3"]);
    expect(result.current.activeConversation).toBe("rapid-3");
    expect(result.current.messages[0].content).toBe("R3");

    // Delete the final one
    await act(async () => {
      await result.current.deleteConversation("rapid-3");
    });

    expect(result.current.conversations).toEqual([]);
    expect(result.current.activeConversation).toBeNull();
    expect(result.current.messages).toEqual([]);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("8. Deleting a non-active conversation preserves the active conversation and its messages", async () => {
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        { id: "active-c", title: "Active", messages: [{ id: "ma", role: "user", content: "Stay active" }] },
        { id: "inactive-c", title: "Inactive", messages: [{ id: "mi", role: "user", content: "Delete me" }] },
      ],
      activeConversationId: "active-c",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    await act(async () => {
      await result.current.deleteConversation("inactive-c");
    });

    expect(result.current.conversations.map((c) => c.id)).toEqual(["active-c"]);
    expect(result.current.activeConversation).toBe("active-c");
    expect(result.current.messages[0].content).toBe("Stay active");

    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    expect(saved.conversations).toHaveLength(1);
    expect(saved.activeConversationId).toBe("active-c");
  });

  it("9. Switching to remaining conversation with unloaded messages fetches them from backend", async () => {
    api.listConversations.mockImplementation(() =>
      Promise.resolve([
        { id: "conv-current", title: "Current Chat", messageCount: 1 },
        { id: "conv-remote", title: "Remote Chat", messageCount: 1 },
      ])
    );
    api.getConversation.mockImplementation((id) => {
      if (id === "conv-remote") {
        return Promise.resolve({
          id: "conv-remote",
          title: "Remote Chat",
          messages: [{ id: "m-remote", role: "assistant", content: "Remote messages fetched" }],
        });
      }
      return Promise.resolve({
        id: "conv-current",
        title: "Current Chat",
        messages: [{ id: "m-curr", role: "user", content: "Current content" }],
      });
    });

    const { result } = renderHook(() => useChat({}));

    await waitFor(() => {
      expect(result.current.loadingConversations).toBe(false);
    });

    // Select conv-current to populate it
    await act(async () => {
      await result.current.selectConversation("conv-current");
    });
    expect(result.current.activeConversation).toBe("conv-current");
    expect(result.current.messages[0].content).toBe("Current content");

    await act(async () => {
      await result.current.deleteConversation("conv-current");
    });

    expect(result.current.activeConversation).toBe("conv-remote");

    await waitFor(() => {
      expect(result.current.messages).toHaveLength(1);
      expect(result.current.messages[0].content).toBe("Remote messages fetched");
    });
  });

  it("10. UI integration: deleting active conversation via Sidebar delete button updates UI and message list", async () => {
    const user = userEvent.setup();
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "ui-c1",
          title: "First UI Chat",
          messages: [{ id: "m1", role: "user", content: "First chat message", type: "text" }],
        },
        {
          id: "ui-c2",
          title: "Second UI Chat",
          messages: [{ id: "m2", role: "user", content: "Second chat message", type: "text" }],
        },
      ],
      activeConversationId: "ui-c1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    function TestApp() {
      const chat = useChat({});
      return (
        <div>
          <Sidebar
            open={true}
            conversations={chat.conversations}
            activeId={chat.activeConversation}
            loadingConversations={chat.loadingConversations}
            onSelect={chat.selectConversation}
            onDelete={chat.deleteConversation}
            onNew={chat.createConversation}
          />
          <Chat
            messages={chat.messages}
            loading={chat.loading}
            streaming={chat.streaming}
            loadingMessages={chat.loadingMessages}
            error={chat.error}
            pendingApproval={chat.pendingApproval}
            onSend={chat.sendMessage}
          />
        </div>
      );
    }

    render(<TestApp />);

    await waitFor(() => {
      expect(screen.queryByLabelText("Loading conversations")).not.toBeInTheDocument();
    });

    // Initial state: first chat is active and shows its message
    expect(screen.getByText("First chat message")).toBeInTheDocument();
    expect(screen.queryByText("Second chat message")).not.toBeInTheDocument();

    // Click the delete button on First UI Chat
    const deleteBtn = screen.getByRole("button", { name: /delete conversation: first ui chat/i });
    await user.click(deleteBtn);

    // After deletion: First UI Chat is gone, Second UI Chat is active and its message is displayed
    await waitFor(() => {
      expect(screen.queryByText("First chat message")).not.toBeInTheDocument();
      expect(screen.getByText("Second chat message")).toBeInTheDocument();
    });

    const secondItem = screen.getByRole("button", { name: /second ui chat, currently active/i });
    expect(secondItem).toBeInTheDocument();
  });

  it("11. UI integration: deleting the last conversation transitions UI to empty sidebar and welcome screen", async () => {
    const user = userEvent.setup();
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "single-c",
          title: "Sole Conversation",
          messages: [{ id: "m-sole", role: "user", content: "Sole message in chat", type: "text" }],
        },
      ],
      activeConversationId: "single-c",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    function TestApp() {
      const chat = useChat({});
      return (
        <div>
          <Sidebar
            open={true}
            conversations={chat.conversations}
            activeId={chat.activeConversation}
            loadingConversations={chat.loadingConversations}
            onSelect={chat.selectConversation}
            onDelete={chat.deleteConversation}
            onNew={chat.createConversation}
          />
          <Chat
            messages={chat.messages}
            loading={chat.loading}
            streaming={chat.streaming}
            loadingMessages={chat.loadingMessages}
            error={chat.error}
            pendingApproval={chat.pendingApproval}
            onSend={chat.sendMessage}
          />
        </div>
      );
    }

    render(<TestApp />);

    await waitFor(() => {
      expect(screen.queryByLabelText("Loading conversations")).not.toBeInTheDocument();
    });

    expect(screen.getByText("Sole message in chat")).toBeInTheDocument();

    const deleteBtn = screen.getByRole("button", { name: /delete conversation: sole conversation/i });
    await user.click(deleteBtn);

    await waitFor(() => {
      expect(screen.queryByText("Sole message in chat")).not.toBeInTheDocument();
      // Sidebar shows empty state
      expect(screen.getByText("No conversations yet")).toBeInTheDocument();
      // Chat shows welcome screen
      expect(screen.getByText("Welcome to FramerAI")).toBeInTheDocument();
    });
  });
});
