import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderHook } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import Sidebar from "../components/Sidebar/Sidebar";
import MessageBubble from "../components/Chat/MessageBubble";
import { api } from "../services/api";
import {
  STORAGE_KEY,
  STORAGE_VERSION,
  loadConversationsFromStorage,
} from "../utils/storage";

vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(() => Promise.resolve([])),
    createConversation: vi.fn(() =>
      Promise.resolve({ id: "parent-1", title: "Parent Chat", messages: [] })
    ),
    getConversation: vi.fn((id) =>
      Promise.resolve({ id, title: "Chat", messages: [] })
    ),
    deleteConversation: vi.fn(() => Promise.resolve({ success: true })),
    sendMessage: vi.fn(() =>
      Promise.resolve({ content: "reply", type: "text" })
    ),
    branchConversation: vi.fn((conversationId, messageId) =>
      Promise.resolve({
        id: "branch-1",
        title: "Parent Chat (Branch)",
        parentConversationId: conversationId,
        branchedFromMessageId: messageId,
        messages: [
          {
            id: messageId,
            role: "user",
            content: "branch point message",
            type: "text",
            timestamp: "2026-09-29T10:00:00.000Z",
          },
        ],
        createdAt: "2026-09-29T10:05:00.000Z",
      })
    ),
  },
}));

vi.mock("../services/websocket", () => ({
  WebSocketClient: class {
    constructor() {
      this.ws = { readyState: 1 };
      this.handlers = new Map();
    }
    connect() {
      return Promise.resolve();
    }
    on(event, handler) {
      if (!this.handlers.has(event)) {
        this.handlers.set(event, []);
      }
      this.handlers.get(event).push(handler);
      return () => {};
    }
    send() {}
    disconnect() {}
  },
}));

describe("Conversation Branching — UI Components", () => {
  it("1. Branch action renders correctly on assistant and user messages", () => {
    const onBranch = vi.fn();
    const assistantMsg = {
      id: "msg-ast-1",
      role: "assistant",
      content: "Hello from assistant",
      type: "text",
    };
    const userMsg = {
      id: "msg-usr-1",
      role: "user",
      content: "Hello from user",
      type: "text",
    };

    const { rerender } = render(
      <MessageBubble message={assistantMsg} onBranch={onBranch} />
    );
    const branchBtnAssistant = screen.getByRole("button", {
      name: /branch from here/i,
    });
    expect(branchBtnAssistant).toBeInTheDocument();

    rerender(<MessageBubble message={userMsg} onBranch={onBranch} />);
    const branchBtnUser = screen.getByRole("button", {
      name: /branch from here/i,
    });
    expect(branchBtnUser).toBeInTheDocument();
  });

  it("2 & 3. Clicking Branch from here passes correct message ID", async () => {
    const user = userEvent.setup();
    const onBranch = vi.fn();
    const msg = {
      id: "target-msg-id-123",
      role: "user",
      content: "branch target",
      type: "text",
    };

    render(<MessageBubble message={msg} onBranch={onBranch} />);
    const branchBtn = screen.getByRole("button", { name: /branch from here/i });
    await user.click(branchBtn);

    expect(onBranch).toHaveBeenCalledTimes(1);
    expect(onBranch).toHaveBeenCalledWith("target-msg-id-123");
  });

  it("4 & 5. Loading state disables button and repeated clicks are guarded", async () => {
    const onBranch = vi.fn();
    const msg = {
      id: "target-msg-id-456",
      role: "assistant",
      content: "branch target 2",
      type: "text",
    };

    render(
      <MessageBubble message={msg} onBranch={onBranch} isBranching={true} />
    );
    const branchBtn = screen.getByRole("button", { name: /branch from here/i });
    expect(branchBtn).toBeDisabled();
  });

  it("8. Branch metadata and indicator render in Sidebar", () => {
    const conversations = [
      { id: "parent-id", title: "Main Chat" },
      {
        id: "branch-id",
        title: "Main Chat (Branch)",
        parentConversationId: "parent-id",
        branchedFromMessageId: "msg-1",
      },
    ];

    render(
      <Sidebar
        open={true}
        conversations={conversations}
        activeId="branch-id"
        loadingConversations={false}
        onToggle={vi.fn()}
        onNew={vi.fn()}
        onSelect={vi.fn()}
        onDelete={vi.fn()}
      />
    );

    expect(screen.getByText("Main Chat")).toBeInTheDocument();
    expect(screen.getByText("Main Chat (Branch)")).toBeInTheDocument();
    expect(screen.getByText("branch")).toBeInTheDocument();
  });
});

describe("Conversation Branching — useChat hook & State Management", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it("6 & 7. Successful branch creation opens the branch and preserves parent", async () => {
    const parentMsg1 = { id: "m1", role: "user", content: "first question" };
    const parentMsg2 = { id: "m2", role: "assistant", content: "first answer" };
    const parentMsg3 = { id: "m3", role: "user", content: "second question" };

    const initialStorage = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "parent-c1",
          title: "Parent Conversation",
          messages: [parentMsg1, parentMsg2, parentMsg3],
          updatedAt: "2026-09-29T10:00:00.000Z",
        },
      ],
      activeConversationId: "parent-c1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialStorage));

    api.branchConversation.mockResolvedValueOnce({
      id: "branch-c2",
      title: "Parent Conversation (Branch)",
      parentConversationId: "parent-c1",
      branchedFromMessageId: "m2",
      messages: [parentMsg1, parentMsg2],
      createdAt: "2026-09-29T10:05:00.000Z",
    });

    const { result } = renderHook(() => useChat({}));

    await act(async () => {
      await result.current.branchConversation("m2");
    });

    // 6. Branch is now active
    expect(result.current.activeConversation).toBe("branch-c2");
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages.map((m) => m.id)).toEqual(["m1", "m2"]);

    // 7. Parent remains in conversations list with all 3 messages intact
    const parentInList = result.current.conversations.find((c) => c.id === "parent-c1");
    expect(parentInList).toBeDefined();
    expect(parentInList.messages).toHaveLength(3);
    expect(parentInList.messages.map((m) => m.id)).toEqual(["m1", "m2", "m3"]);
  });

  it("9. Switching between parent and branch works cleanly", async () => {
    const parentConv = {
      id: "parent-c1",
      title: "Parent",
      messages: [
        { id: "m1", role: "user", content: "parent msg" },
        { id: "m2", role: "assistant", content: "parent reply" },
      ],
    };
    const branchConv = {
      id: "branch-c2",
      title: "Branch",
      parentConversationId: "parent-c1",
      branchedFromMessageId: "m1",
      messages: [{ id: "m1", role: "user", content: "parent msg" }],
    };

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [parentConv, branchConv],
        activeConversationId: "branch-c2",
      })
    );

    api.getConversation.mockImplementation((id) => {
      if (id === "parent-c1") return Promise.resolve(parentConv);
      if (id === "branch-c2") return Promise.resolve(branchConv);
      return Promise.reject(new Error("Not found"));
    });

    const { result } = renderHook(() => useChat({}));

    // Initially in branch
    expect(result.current.activeConversation).toBe("branch-c2");
    expect(result.current.messages).toHaveLength(1);

    // Switch to parent
    await act(async () => {
      await result.current.selectConversation("parent-c1");
    });

    expect(result.current.activeConversation).toBe("parent-c1");
    expect(result.current.messages).toHaveLength(2);

    // Switch back to branch
    await act(async () => {
      await result.current.selectConversation("branch-c2");
    });

    expect(result.current.activeConversation).toBe("branch-c2");
    expect(result.current.messages).toHaveLength(1);
  });

  it("10. Refresh/reload restores the branch and metadata from localStorage", () => {
    const payload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "branch-saved-id",
          title: "My Branch",
          parentConversationId: "parent-saved-id",
          branchedFromMessageId: "branched-msg-id",
          createdAt: "2026-09-29T10:00:00.000Z",
          updatedAt: "2026-09-29T10:01:00.000Z",
          messages: [
            { id: "msg-1", role: "user", content: "hello", type: "text" },
          ],
        },
      ],
      activeConversationId: "branch-saved-id",
    };

    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    const loaded = loadConversationsFromStorage(localStorage);

    expect(loaded.conversations).toHaveLength(1);
    const restored = loaded.conversations[0];
    expect(restored.id).toBe("branch-saved-id");
    expect(restored.parentConversationId).toBe("parent-saved-id");
    expect(restored.branchedFromMessageId).toBe("branched-msg-id");
    expect(restored.createdAt).toBe("2026-09-29T10:00:00.000Z");
  });

  it("11. Generation in branch does not mutate parent state", async () => {
    const parentConv = {
      id: "parent-c1",
      title: "Parent",
      messages: [
        { id: "m1", role: "user", content: "root 1" },
        { id: "m2", role: "assistant", content: "reply 1" },
      ],
    };
    const branchConv = {
      id: "branch-c2",
      title: "Branch",
      parentConversationId: "parent-c1",
      branchedFromMessageId: "m1",
      messages: [{ id: "m1", role: "user", content: "root 1" }],
    };

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [parentConv, branchConv],
        activeConversationId: "branch-c2",
      })
    );

    api.sendMessage.mockResolvedValueOnce({
      id: "reply-in-branch",
      content: "branch assistant response",
      type: "text",
    });

    const { result } = renderHook(() => useChat({}));

    // Send a message in branch
    await act(async () => {
      await result.current.sendMessage("new question in branch");
    });

    // Branch has new user message + assistant reply
    expect(result.current.messages.some((m) => m.content === "new question in branch")).toBe(true);

    // Parent in conversations still has only its original 2 messages
    const parentAfter = result.current.conversations.find((c) => c.id === "parent-c1");
    expect(parentAfter.messages).toHaveLength(2);
    expect(parentAfter.messages.some((m) => m.content === "new question in branch")).toBe(false);
  });

  it("12. Errors restore usable UI state and reset branching flag", async () => {
    const conv = {
      id: "conv-1",
      title: "Chat",
      messages: [{ id: "m1", role: "user", content: "question" }],
    };

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [conv],
        activeConversationId: "conv-1",
      })
    );

    api.branchConversation.mockRejectedValueOnce(
      new Error("Server error occurred")
    );

    const { result } = renderHook(() => useChat({}));

    await act(async () => {
      await result.current.branchConversation("m1");
    });

    expect(result.current.branching).toBe(false);
    expect(result.current.error).toBe("Server error occurred");
    expect(result.current.activeConversation).toBe("conv-1");
  });

  it("Offline fallback creates local branch when API is unreachable", async () => {
    const conv = {
      id: "conv-offline",
      title: "Offline Chat",
      messages: [
        { id: "m1", role: "user", content: "q1" },
        { id: "m2", role: "assistant", content: "a1" },
      ],
    };

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [conv],
        activeConversationId: "conv-offline",
      })
    );

    // Network / server connection error
    api.branchConversation.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    const { result } = renderHook(() => useChat({}));

    let branchResult;
    await act(async () => {
      branchResult = await result.current.branchConversation("m1");
    });

    expect(branchResult).toBeDefined();
    expect(branchResult.parentConversationId).toBe("conv-offline");
    expect(result.current.branching).toBe(false);
    expect(result.current.activeConversation).not.toBe("conv-offline");
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].id).toBe("m1");

    const newBranch = result.current.conversations.find((c) => c.id === result.current.activeConversation);
    expect(newBranch).toBeDefined();
    expect(newBranch.parentConversationId).toBe("conv-offline");
    expect(newBranch.branchedFromMessageId).toBe("m1");
  });

  it("Guards against rapid repeated branch clicks while branch creation is in-flight", async () => {
    const conv = {
      id: "conv-inflight",
      title: "Chat",
      messages: [{ id: "m1", role: "user", content: "hello" }],
    };

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [conv],
        activeConversationId: "conv-inflight",
      })
    );

    let resolveApi;
    api.branchConversation.mockImplementation(
      () =>
        new Promise((res) => {
          resolveApi = res;
        })
    );

    const { result } = renderHook(() => useChat({}));

    // Trigger first branch call (in-flight)
    let p1, p2;
    act(() => {
      p1 = result.current.branchConversation("m1");
      // Immediate rapid second click
      p2 = result.current.branchConversation("m1");
    });

    expect(result.current.branching).toBe(true);
    expect(api.branchConversation).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveApi({
        id: "branch-resolved",
        title: "Chat (Branch)",
        parentConversationId: "conv-inflight",
        branchedFromMessageId: "m1",
        messages: [{ id: "m1", role: "user", content: "hello" }],
      });
      await p1;
      await p2;
    });

    expect(result.current.branching).toBe(false);
    expect(result.current.activeConversation).toBe("branch-resolved");
    expect(api.branchConversation).toHaveBeenCalledTimes(1);
  });

  it("Data integrity: sibling branches remain independent and nested branches use correct parent", async () => {
    const convRoot = {
      id: "root-id",
      title: "Root",
      messages: [
        { id: "m1", role: "user", content: "turn 1" },
        { id: "m2", role: "assistant", content: "turn 2" },
      ],
    };

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: STORAGE_VERSION,
        conversations: [convRoot],
        activeConversationId: "root-id",
      })
    );

    // Create Sibling Branch A
    api.branchConversation.mockResolvedValueOnce({
      id: "branch-A",
      title: "Root (Branch A)",
      parentConversationId: "root-id",
      branchedFromMessageId: "m1",
      messages: [{ id: "m1", role: "user", content: "turn 1" }],
    });

    const { result } = renderHook(() => useChat({}));

    await act(async () => {
      await result.current.branchConversation("m1", "root-id");
    });
    expect(result.current.activeConversation).toBe("branch-A");

    // Create Sibling Branch B from same root
    api.branchConversation.mockResolvedValueOnce({
      id: "branch-B",
      title: "Root (Branch B)",
      parentConversationId: "root-id",
      branchedFromMessageId: "m2",
      messages: [
        { id: "m1", role: "user", content: "turn 1" },
        { id: "m2", role: "assistant", content: "turn 2" },
      ],
    });

    await act(async () => {
      await result.current.branchConversation("m2", "root-id");
    });
    expect(result.current.activeConversation).toBe("branch-B");

    // Sibling A and Sibling B both point to root-id
    const branchA = result.current.conversations.find((c) => c.id === "branch-A");
    const branchB = result.current.conversations.find((c) => c.id === "branch-B");
    expect(branchA.parentConversationId).toBe("root-id");
    expect(branchB.parentConversationId).toBe("root-id");

    // Nested Branch: Branch B.1 branched from Branch B
    api.branchConversation.mockResolvedValueOnce({
      id: "branch-B-1",
      title: "Root (Branch B) (Branch)",
      parentConversationId: "branch-B",
      branchedFromMessageId: "m2",
      messages: [
        { id: "m1", role: "user", content: "turn 1" },
        { id: "m2", role: "assistant", content: "turn 2" },
      ],
    });

    await act(async () => {
      await result.current.branchConversation("m2", "branch-B");
    });
    expect(result.current.activeConversation).toBe("branch-B-1");

    const branchB1 = result.current.conversations.find((c) => c.id === "branch-B-1");
    expect(branchB1.parentConversationId).toBe("branch-B");
  });
});
