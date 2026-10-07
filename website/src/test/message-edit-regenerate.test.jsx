import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderHook } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import MessageBubble from "../components/Chat/MessageBubble";
import { api } from "../services/api";
import { STORAGE_KEY, STORAGE_VERSION, loadConversationsFromStorage } from "../utils/storage";

let mockStreamHandler = null;
let mockAckHandler = null;
let mockErrorHandler = null;
let lastSentWsFrame = null;
let mockWsReadyState = 1;

vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(() => Promise.resolve([])),
    createConversation: vi.fn(() =>
      Promise.resolve({ id: "c1", title: "New Chat", messages: [] })
    ),
    getConversation: vi.fn((id) =>
      Promise.resolve({ id, title: "Chat", messages: [] })
    ),
    deleteConversation: vi.fn(() => Promise.resolve({ success: true })),
    sendMessage: vi.fn((convId, content, type, attachments, settings, options) =>
      Promise.resolve({
        id: "reply-1",
        userMessageId: options?.editMessageId || "u-ack-1",
        role: "assistant",
        content: `Mock reply to: ${content}`,
        type: "text",
        metadata: {},
      })
    ),
    branchConversation: vi.fn((convId, messageId) =>
      Promise.resolve({
        id: "branch-1",
        title: "Parent Chat (Branch)",
        parentConversationId: convId,
        branchedFromMessageId: messageId,
        messages: [{ id: messageId, role: "user", content: "Original turn" }],
        createdAt: new Date().toISOString(),
      })
    ),
  },
}));

vi.mock("../services/websocket", () => ({
  WebSocketClient: class {
    constructor() {
      this.ws = {
        get readyState() {
          return mockWsReadyState;
        },
      };
      this.listeners = new Map();
    }
    connect() {
      return Promise.resolve();
    }
    isConnected() {
      return mockWsReadyState === 1;
    }
    on(type, handler) {
      this.listeners.set(type, handler);
      if (type === "stream") mockStreamHandler = handler;
      if (type === "ack") mockAckHandler = handler;
      if (type === "error") mockErrorHandler = handler;
      return () => {};
    }
    send(data) {
      lastSentWsFrame = data;
    }
    disconnect() {}
  },
}));

describe("Issue #430: Message Edit, Regeneration & Continuation", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    mockStreamHandler = null;
    mockAckHandler = null;
    mockErrorHandler = null;
    lastSentWsFrame = null;
    mockWsReadyState = 1;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // 1. UI Components: MessageBubble Edit, Regenerate, and Continue
  // -------------------------------------------------------------------------
  describe("UI Components (MessageBubble)", () => {
    it("renders Edit and Continue buttons on user messages, but not Regenerate", () => {
      const userMsg = { id: "u1", role: "user", content: "Hello world", type: "text" };
      const onEdit = vi.fn();
      const onRegenerate = vi.fn();
      const onContinue = vi.fn();

      render(
        <MessageBubble
          message={userMsg}
          onEdit={onEdit}
          onRegenerate={onRegenerate}
          onContinue={onContinue}
        />
      );

      expect(screen.getByRole("button", { name: /edit message/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /continue from here/i })).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /regenerate response/i })).not.toBeInTheDocument();
    });

    it("renders Regenerate and Continue buttons on assistant messages, but not Edit", () => {
      const assistantMsg = { id: "a1", role: "assistant", content: "Bot reply", type: "text" };
      const onEdit = vi.fn();
      const onRegenerate = vi.fn();
      const onContinue = vi.fn();

      render(
        <MessageBubble
          message={assistantMsg}
          onEdit={onEdit}
          onRegenerate={onRegenerate}
          onContinue={onContinue}
        />
      );

      expect(screen.getByRole("button", { name: /regenerate response/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /continue from here/i })).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /edit message/i })).not.toBeInTheDocument();
    });

    it("clicking Edit opens textarea with Save & Send, Branch, and Cancel buttons", async () => {
      const user = userEvent.setup();
      const userMsg = { id: "u1", role: "user", content: "Original message", type: "text" };
      const onEdit = vi.fn();

      render(<MessageBubble message={userMsg} onEdit={onEdit} />);

      await user.click(screen.getByRole("button", { name: /edit message/i }));

      const textarea = screen.getByRole("textbox");
      expect(textarea).toBeInTheDocument();
      expect(textarea).toHaveValue("Original message");

      const saveBtn = screen.getByRole("button", { name: /save and send/i });
      const branchBtn = screen.getByRole("button", { name: /branch and send/i });
      const cancelBtn = screen.getByRole("button", { name: /cancel edit/i });

      expect(saveBtn).toBeInTheDocument();
      expect(branchBtn).toBeInTheDocument();
      expect(cancelBtn).toBeInTheDocument();

      // Cancel button exits edit mode without calling onEdit
      await user.click(cancelBtn);
      expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
      expect(onEdit).not.toHaveBeenCalled();
    });

    it("saving an edit calls onEdit with updated content and branch: false", async () => {
      const user = userEvent.setup();
      const userMsg = { id: "u1", role: "user", content: "Original prompt", type: "text" };
      const onEdit = vi.fn();

      render(<MessageBubble message={userMsg} onEdit={onEdit} />);
      await user.click(screen.getByRole("button", { name: /edit message/i }));

      const textarea = screen.getByRole("textbox");
      await user.clear(textarea);
      await user.type(textarea, "Edited prompt");

      await user.click(screen.getByRole("button", { name: /save and send/i }));
      expect(onEdit).toHaveBeenCalledWith("u1", "Edited prompt", { branch: false });
    });

    it("branching an edit calls onEdit with updated content and branch: true", async () => {
      const user = userEvent.setup();
      const userMsg = { id: "u1", role: "user", content: "Original prompt", type: "text" };
      const onEdit = vi.fn();

      render(<MessageBubble message={userMsg} onEdit={onEdit} />);
      await user.click(screen.getByRole("button", { name: /edit message/i }));

      const textarea = screen.getByRole("textbox");
      await user.clear(textarea);
      await user.type(textarea, "Branched prompt");

      await user.click(screen.getByRole("button", { name: /branch and send/i }));
      expect(onEdit).toHaveBeenCalledWith("u1", "Branched prompt", { branch: true });
    });

    it("regenerate button calls onRegenerate with assistant message id", async () => {
      const user = userEvent.setup();
      const assistantMsg = { id: "a1", role: "assistant", content: "Bot reply", type: "text" };
      const onRegenerate = vi.fn();

      render(<MessageBubble message={assistantMsg} onRegenerate={onRegenerate} />);
      await user.click(screen.getByRole("button", { name: /regenerate response/i }));

      expect(onRegenerate).toHaveBeenCalledWith("a1");
    });
  });

  // -------------------------------------------------------------------------
  // 2. Hook and State Management (useChat)
  // -------------------------------------------------------------------------
  describe("useChat hook: Editing, Regeneration, and Continuation", () => {
    function setupChatHook(initialStorage = null, wsReadyState = 1) {
      if (initialStorage) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(initialStorage));
      }
      mockWsReadyState = wsReadyState;

      const { result } = renderHook(() => useChat({}));

      return {
        result,
        emitStream: (data) => mockStreamHandler?.(data),
        emitAck: (data) => mockAckHandler?.(data),
        emitError: (data) => mockErrorHandler?.(data),
      };
    }

    it("edit latest user message: in-place update replaces subsequent assistant reply", async () => {
      const u1 = { id: "u1", role: "user", content: "Tell me a joke" };
      const a1 = { id: "a1", role: "assistant", content: "Why did the chicken cross the road?", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Joke Chat", messages: [u1, a1] }],
        activeConversationId: "c1",
      };

      const { result, emitAck, emitStream } = setupChatHook(storage, 1);

      await act(async () => {
        await result.current.editMessage("u1", "Tell me a poem", { branch: false });
      });

      // User message is updated, placeholder assistant bubble created, obsolete a1 removed
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[0].content).toBe("Tell me a poem");
      expect(result.current.messages[1].role).toBe("assistant");
      expect(result.current.messages[1].completed).toBe(false);

      // Verify WebSocket frame sent with editMessageId
      expect(lastSentWsFrame).toMatchObject({
        type: "chat",
        conversationId: "c1",
        editMessageId: "u1",
        content: "Tell me a poem",
      });

      // Stream frames arrive
      act(() => {
        emitAck({ conversationId: "c1", messageId: "u1", assistantMessageId: "a2" });
        emitStream({ conversationId: "c1", content: "Roses are red", done: false });
        emitStream({ conversationId: "c1", content: "Roses are red, violets are blue", done: true, messageId: "a2" });
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[0].content).toBe("Tell me a poem");
      expect(result.current.messages[1].content).toBe("Roses are red, violets are blue");
      expect(result.current.messages[1].completed).toBe(true);
    });

    it("edit earlier user message: truncates all obsolete messages after edit point", async () => {
      const u1 = { id: "u1", role: "user", content: "First question" };
      const a1 = { id: "a1", role: "assistant", content: "First answer", completed: true };
      const u2 = { id: "u2", role: "user", content: "Second question" };
      const a2 = { id: "a2", role: "assistant", content: "Second answer", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Two Turn Chat", messages: [u1, a1, u2, a2] }],
        activeConversationId: "c1",
      };

      const { result, emitStream } = setupChatHook(storage, 1);

      // Edit turn 1 user message
      await act(async () => {
        await result.current.editMessage("u1", "Rewritten first question", { branch: false });
      });

      // All turns after u1 must be truncated!
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[0].id).toBe("u1");
      expect(result.current.messages[0].content).toBe("Rewritten first question");
      expect(result.current.messages.some((m) => m.id === "u2" || m.id === "a2")).toBe(false);

      // Complete stream
      act(() => {
        emitStream({ conversationId: "c1", content: "New answer to rewritten first", done: true });
      });

      expect(result.current.messages[1].content).toBe("New answer to rewritten first");
    });

    it("edit with branching: creates new branch from user message and preserves parent", async () => {
      const u1 = { id: "u1", role: "user", content: "Original turn" };
      const a1 = { id: "a1", role: "assistant", content: "Original response", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Parent Chat", messages: [u1, a1] }],
        activeConversationId: "c1",
      };

      api.branchConversation.mockResolvedValueOnce({
        id: "branch-1",
        title: "Parent Chat (Branch)",
        parentConversationId: "c1",
        branchedFromMessageId: "u1",
        messages: [{ ...u1 }],
      });

      const { result } = setupChatHook(storage, 0); // REST mode

      await act(async () => {
        await result.current.editMessage("u1", "Branched turn content", { branch: true });
      });

      // Branch API was called
      expect(api.branchConversation).toHaveBeenCalledWith("c1", "u1");

      // Active conversation is now the branch
      expect(result.current.activeConversation).toBe("branch-1");

      // Parent conversation is intact with original content in conversations list
      const parentConv = result.current.conversations.find((c) => c.id === "c1");
      expect(parentConv).toBeDefined();
      expect(parentConv.messages[0].content).toBe("Original turn");
    });

    it("regenerate assistant response (in-place): replaces response using previous user turn", async () => {
      const u1 = { id: "u1", role: "user", content: "Explain quantum mechanics" };
      const a1 = { id: "a1", role: "assistant", content: "It's physics.", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Physics Chat", messages: [u1, a1] }],
        activeConversationId: "c1",
      };

      const { result, emitStream } = setupChatHook(storage, 1);

      await act(async () => {
        await result.current.regenerateResponse("a1", { branch: false });
      });

      // Assistant message is replaced with a new streaming assistant bubble
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[0].content).toBe("Explain quantum mechanics");
      expect(result.current.messages[1].completed).toBe(false);

      // Verify WS frame carried regenerateMessageId
      expect(lastSentWsFrame).toMatchObject({
        type: "chat",
        conversationId: "c1",
        regenerateMessageId: "a1",
        content: "Explain quantum mechanics",
      });

      // Complete generation
      act(() => {
        emitStream({
          conversationId: "c1",
          content: "Quantum mechanics describes nature at microscopic scales.",
          done: true,
        });
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[1].content).toBe("Quantum mechanics describes nature at microscopic scales.");
      expect(result.current.messages[1].completed).toBe(true);
    });

    it("regenerate assistant response with branching: branches from preceding user message", async () => {
      const u1 = { id: "u1", role: "user", content: "Explain gravity" };
      const a1 = { id: "a1", role: "assistant", content: "Things fall down.", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Gravity Chat", messages: [u1, a1] }],
        activeConversationId: "c1",
      };

      api.branchConversation.mockResolvedValueOnce({
        id: "branch-1",
        title: "Gravity Chat (Branch)",
        parentConversationId: "c1",
        branchedFromMessageId: "u1",
        messages: [{ ...u1 }],
      });

      const { result } = setupChatHook(storage, 0);

      await act(async () => {
        await result.current.regenerateResponse("a1", { branch: true });
      });

      // Branch API called with preceding user message id
      expect(api.branchConversation).toHaveBeenCalledWith("c1", "u1");
      expect(result.current.activeConversation).toBe("branch-1");

      // Parent conversation is intact
      const parentConv = result.current.conversations.find((c) => c.id === "c1");
      expect(parentConv.messages).toHaveLength(2);
      expect(parentConv.messages[1].content).toBe("Things fall down.");
    });

    it("continueFromMessage: branches by default and preserves parent/child relationship", async () => {
      const u1 = { id: "u1", role: "user", content: "Step 1" };
      const a1 = { id: "a1", role: "assistant", content: "Step 1 done", completed: true };
      const u2 = { id: "u2", role: "user", content: "Step 2" };
      const a2 = { id: "a2", role: "assistant", content: "Step 2 done", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Workflow", messages: [u1, a1, u2, a2] }],
        activeConversationId: "c1",
      };

      api.branchConversation.mockResolvedValueOnce({
        id: "branch-1",
        title: "Workflow (Branch)",
        parentConversationId: "c1",
        branchedFromMessageId: "a1",
        messages: [u1, a1],
      });

      const { result } = setupChatHook(storage, 0);

      // Continue from a1 with a prompt
      await act(async () => {
        await result.current.continueFromMessage("a1", "Alternative step 2", { branch: true });
      });

      expect(api.branchConversation).toHaveBeenCalledWith("c1", "a1");
      expect(result.current.activeConversation).toBe("branch-1");
    });

    it("continueFromMessage with branch: false truncates in place and continues", async () => {
      const u1 = { id: "u1", role: "user", content: "Step 1" };
      const a1 = { id: "a1", role: "assistant", content: "Step 1 done", completed: true };
      const u2 = { id: "u2", role: "user", content: "Step 2" };
      const a2 = { id: "a2", role: "assistant", content: "Step 2 done", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Workflow", messages: [u1, a1, u2, a2] }],
        activeConversationId: "c1",
      };

      const { result } = setupChatHook(storage, 0);

      await act(async () => {
        await result.current.continueFromMessage("a1", { branch: false });
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages.map((m) => m.id)).toEqual(["u1", "a1"]);
    });

    it("stale previous generation cannot overwrite regenerated response", async () => {
      const u1 = { id: "u1", role: "user", content: "Question" };
      const a1 = { id: "a1", role: "assistant", content: "Old reply", completed: false };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Stale Chat", messages: [u1, a1] }],
        activeConversationId: "c1",
      };

      const { result, emitStream } = setupChatHook(storage, 1);

      // Trigger regenerate
      await act(async () => {
        await result.current.regenerateResponse("a1", { branch: false });
      });

      const newAssistantId = result.current.messages[1].id;
      expect(newAssistantId).not.toBe("a1");

      // Late stream frame targeting the old a1 must be ignored
      act(() => {
        emitStream({ conversationId: "c1", messageId: "a1", content: "Stale junk", done: false });
      });
      expect(result.current.messages[1].content).toBe("");

      // Chunks for the new generation succeed
      act(() => {
        emitStream({ conversationId: "c1", content: "Fresh valid reply", done: true, messageId: newAssistantId });
      });
      expect(result.current.messages[1].content).toBe("Fresh valid reply");
    });

    it("switching conversations during generation isolates streaming", async () => {
      const c1 = {
        id: "c1",
        title: "Chat 1",
        messages: [{ id: "u1", role: "user", content: "Q1" }, { id: "a1", role: "assistant", content: "Ans 1", completed: true }],
      };
      const c2 = {
        id: "c2",
        title: "Chat 2",
        messages: [{ id: "u2", role: "user", content: "Q2" }, { id: "a2", role: "assistant", content: "Ans 2", completed: true }],
      };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [c1, c2],
        activeConversationId: "c1",
      };

      api.getConversation.mockImplementation((id) => {
        if (id === "c1") return Promise.resolve(c1);
        if (id === "c2") return Promise.resolve(c2);
        return Promise.reject(new Error("Not found"));
      });

      const { result, emitStream } = setupChatHook(storage, 1);

      // Start regenerating in c1
      await act(async () => {
        await result.current.regenerateResponse("a1", { branch: false });
      });

      // Switch to c2 while c1 is streaming
      await act(async () => {
        await result.current.selectConversation("c2");
      });
      expect(result.current.activeConversation).toBe("c2");
      expect(result.current.messages[0].content).toBe("Q2");

      // Stream frames arrive for c1
      act(() => {
        emitStream({ conversationId: "c1", content: "New answer for c1", done: true });
      });

      // c2 messages are unaffected!
      expect(result.current.messages[0].content).toBe("Q2");
      expect(result.current.messages[1].content).toBe("Ans 2");

      // In background conversations list, c1 has the completed reply
      const backgroundC1 = result.current.conversations.find((c) => c.id === "c1");
      expect(backgroundC1.messages[1].content).toBe("New answer for c1");
    });

    it("persistence: edits and regenerations survive storage reloads", async () => {
      const u1 = { id: "u1", role: "user", content: "Persist prompt" };
      const a1 = { id: "a1", role: "assistant", content: "Initial reply", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Storage Chat", messages: [u1, a1] }],
        activeConversationId: "c1",
      };

      const { result, emitStream } = setupChatHook(storage, 1);

      await act(async () => {
        await result.current.editMessage("u1", "Persisted edit", { branch: false });
      });

      act(() => {
        emitStream({ conversationId: "c1", content: "Persisted assistant reply", done: true });
      });

      // Read directly from localStorage
      const loaded = loadConversationsFromStorage();
      const conv = loaded.conversations.find((c) => c.id === "c1");
      expect(conv.messages[0].content).toBe("Persisted edit");
      expect(conv.messages[1].content).toBe("Persisted assistant reply");
    });

    it("edge case: empty edited message is rejected", async () => {
      const u1 = { id: "u1", role: "user", content: "Something" };
      const a1 = { id: "a1", role: "assistant", content: "Reply", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "Validation Chat", messages: [u1, a1] }],
        activeConversationId: "c1",
      };

      const { result } = setupChatHook(storage, 1);

      await act(async () => {
        const res = await result.current.editMessage("u1", "   ", { branch: false });
        expect(res).toBeNull();
      });

      // Error message set
      expect(result.current.error).toMatch(/empty/i);
      // Messages unchanged
      expect(result.current.messages[0].content).toBe("Something");
    });

    it("edge case: deleting conversation during edit generation cleans up cleanly", async () => {
      const u1 = { id: "u1", role: "user", content: "To be deleted" };
      const a1 = { id: "a1", role: "assistant", content: "Reply", completed: true };

      const storage = {
        version: STORAGE_VERSION,
        conversations: [{ id: "c1", title: "To Delete", messages: [u1, a1] }],
        activeConversationId: "c1",
      };

      const { result, emitStream } = setupChatHook(storage, 1);

      await act(async () => {
        await result.current.editMessage("u1", "New edit before delete", { branch: false });
      });

      await act(async () => {
        await result.current.deleteConversation("c1");
      });

      // Late stream frame for deleted conversation does not crash or re-add conversation
      act(() => {
        emitStream({ conversationId: "c1", content: "Late chunk", done: true });
      });

      expect(result.current.conversations.find((c) => c.id === "c1")).toBeUndefined();
    });
  });
});
