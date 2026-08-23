import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  STORAGE_KEY,
  STORAGE_VERSION,
  loadConversationsFromStorage,
  saveConversationsToStorage,
  clearConversationsFromStorage,
  evictOldestConversations,
} from "../utils/storage";
import { useChat } from "../hooks/useChat";
import Sidebar from "../components/Sidebar/Sidebar";

// Mock external services so hooks operate predictably in tests
vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(() => Promise.resolve([])),
    createConversation: vi.fn(() => Promise.resolve({ id: "api-c1", title: "API Chat", messages: [] })),
    getConversation: vi.fn((id) => Promise.resolve({ id, messages: [] })),
    deleteConversation: vi.fn(() => Promise.resolve()),
    sendMessage: vi.fn(() => Promise.resolve({ content: "API reply", type: "text" })),
  },
}));

vi.mock("../services/websocket", () => ({
  WebSocketClient: class {
    connect() {
      return Promise.resolve();
    }
    on() {
      return () => {};
    }
    send() {}
    disconnect() {}
  },
}));

describe("Storage utility — loadConversationsFromStorage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("1. no stored payload -> empty state", () => {
    const result = loadConversationsFromStorage();
    expect(result).toEqual({
      conversations: [],
      activeConversationId: null,
      messages: [],
    });
  });

  it("2. valid stored payload -> conversations restored", () => {
    const payload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "c1",
          title: "First Chat",
          updatedAt: "2026-08-23T10:00:00.000Z",
          messages: [
            { id: "m1", role: "user", content: "Hello", type: "text", timestamp: "2026-08-23T10:00:00.000Z" },
            { id: "m2", role: "assistant", content: "Hi", type: "text", timestamp: "2026-08-23T10:00:01.000Z" },
          ],
        },
      ],
      activeConversationId: "c1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

    const result = loadConversationsFromStorage();
    expect(result.conversations).toHaveLength(1);
    expect(result.conversations[0].id).toBe("c1");
    expect(result.conversations[0].title).toBe("First Chat");
    expect(result.conversations[0].messages).toHaveLength(2);
    expect(result.messages).toHaveLength(2);
  });

  it("3. active conversation restored", () => {
    const payload = {
      version: STORAGE_VERSION,
      conversations: [
        { id: "c1", title: "Chat 1", messages: [{ id: "m1", role: "user", content: "Msg 1" }] },
        { id: "c2", title: "Chat 2", messages: [{ id: "m2", role: "user", content: "Msg 2" }] },
      ],
      activeConversationId: "c2",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

    const result = loadConversationsFromStorage();
    expect(result.activeConversationId).toBe("c2");
    expect(result.messages[0].content).toBe("Msg 2");
  });

  it("4. malformed JSON -> empty state", () => {
    localStorage.setItem(STORAGE_KEY, "{ invalid json ... ");

    const result = loadConversationsFromStorage();
    expect(result).toEqual({
      conversations: [],
      activeConversationId: null,
      messages: [],
    });
  });

  it("5. wrong storage version -> empty state", () => {
    const oldVersionPayload = {
      version: 0,
      conversations: [{ id: "c1", title: "Old Chat", messages: [] }],
      activeConversationId: "c1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(oldVersionPayload));

    const result = loadConversationsFromStorage();
    expect(result).toEqual({
      conversations: [],
      activeConversationId: null,
      messages: [],
    });

    const futureVersionPayload = {
      version: 99,
      conversations: [{ id: "c1", title: "Future Chat", messages: [] }],
      activeConversationId: "c1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(futureVersionPayload));

    expect(loadConversationsFromStorage()).toEqual({
      conversations: [],
      activeConversationId: null,
      messages: [],
    });
  });

  it("6. invalid payload structure -> empty state", () => {
    // Missing conversations array
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, activeConversationId: "c1" }));
    expect(loadConversationsFromStorage()).toEqual({
      conversations: [],
      activeConversationId: null,
      messages: [],
    });

    // Conversations is not an array
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, conversations: "not an array" }));
    expect(loadConversationsFromStorage()).toEqual({
      conversations: [],
      activeConversationId: null,
      messages: [],
    });

    // Primitive value instead of object
    localStorage.setItem(STORAGE_KEY, "12345");
    expect(loadConversationsFromStorage()).toEqual({
      conversations: [],
      activeConversationId: null,
      messages: [],
    });
  });
});

describe("Storage utility — eviction & sanitization", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("7. storage size exceeded -> oldest conversations evicted", () => {
    const conv1 = { id: "c1", title: "Old Chat", messages: [{ id: "m1", role: "user", content: "A".repeat(500) }] };
    const conv2 = { id: "c2", title: "Medium Chat", messages: [{ id: "m2", role: "user", content: "B".repeat(500) }] };
    const conv3 = { id: "c3", title: "New Chat", messages: [{ id: "m3", role: "user", content: "C".repeat(500) }] };

    const conversations = [conv1, conv2, conv3];

    // Set maxBytes small enough to force eviction of oldest conversation (c1)
    const { conversations: evicted } = evictOldestConversations(conversations, "c3", 1200);

    // Oldest non-active conversation (c1) should be evicted
    expect(evicted.some((c) => c.id === "c1")).toBe(false);
    expect(evicted.some((c) => c.id === "c3")).toBe(true);
  });

  it("sanitizes transient fields from messages before saving", () => {
    const conv = {
      id: "c1",
      title: "Audio Chat",
      messages: [
        {
          id: "m1",
          role: "assistant",
          content: "Audio message",
          type: "audio",
          timestamp: "2026-08-23T10:00:00.000Z",
          audioChunks: ["chunk1", "chunk2"], // Transient field
          audioMetadata: { sampleRate: 24000 }, // Transient field
          audioComplete: true, // Transient field
        },
      ],
    };

    saveConversationsToStorage([conv], "c1");

    const loaded = loadConversationsFromStorage();
    const message = loaded.conversations[0].messages[0];

    expect(message.id).toBe("m1");
    expect(message.content).toBe("Audio message");
    expect(message.audioChunks).toBeUndefined();
    expect(message.audioMetadata).toBeUndefined();
    expect(message.audioComplete).toBeUndefined();
  });

  it("removes storage key via clearConversationsFromStorage", () => {
    localStorage.setItem(STORAGE_KEY, "test-data");
    expect(localStorage.getItem(STORAGE_KEY)).toBe("test-data");
    clearConversationsFromStorage();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });
});

describe("useChat integration with localStorage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("8. delete conversation -> state and storage updated", async () => {
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        { id: "c1", title: "Chat One", messages: [{ id: "m1", role: "user", content: "Hi 1" }] },
        { id: "c2", title: "Chat Two", messages: [{ id: "m2", role: "user", content: "Hi 2" }] },
      ],
      activeConversationId: "c1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    const { result } = renderHook(() => useChat({}));

    expect(result.current.conversations).toHaveLength(2);
    expect(result.current.activeConversation).toBe("c1");

    await act(async () => {
      await result.current.deleteConversation("c1");
    });

    expect(result.current.conversations).toHaveLength(1);
    expect(result.current.activeConversation).toBe("c2");

    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    expect(saved.conversations).toHaveLength(1);
    expect(saved.conversations[0].id).toBe("c2");
    expect(saved.activeConversationId).toBe("c2");
  });

  it("9. clear all -> state and storage cleared", async () => {
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        { id: "c1", title: "Chat One", messages: [{ id: "m1", role: "user", content: "Hi" }] },
      ],
      activeConversationId: "c1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    let result;
    await act(async () => {
      const rendered = renderHook(() => useChat({}));
      result = rendered.result;
    });

    expect(result.current.conversations).toHaveLength(1);

    await act(async () => {
      result.current.clearAllConversations();
    });

    expect(result.current.conversations).toEqual([]);
    expect(result.current.activeConversation).toBeNull();
    expect(result.current.messages).toEqual([]);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("10. localStorage.getItem throwing -> application still works", async () => {
    const getItemSpy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("SecurityError: Access denied");
    });

    let result;
    await act(async () => {
      const rendered = renderHook(() => useChat({}));
      result = rendered.result;
    });

    expect(result.current.conversations).toEqual([]);
    expect(result.current.activeConversation).toBeNull();
    expect(result.current.messages).toEqual([]);

    getItemSpy.mockRestore();
  });

  it("11. localStorage.setItem throwing -> application still works", async () => {
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("QuotaExceededError: Storage full");
    });

    const { result } = renderHook(() => useChat({}));

    await act(async () => {
      await result.current.createConversation();
    });

    expect(result.current.conversations).toHaveLength(1);
    expect(result.current.activeConversation).toBeDefined();

    setItemSpy.mockRestore();
  });
});

describe("Sidebar Clear History UI", () => {
  it("renders clear history button when conversations exist", () => {
    const conversations = [{ id: "c1", title: "Test Chat" }];
    render(
      <Sidebar
        open={true}
        conversations={conversations}
        activeId="c1"
        loadingConversations={false}
        onToggle={vi.fn()}
        onNew={vi.fn()}
        onSelect={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
        onOpenSettings={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: /clear all conversations/i })).toBeInTheDocument();
  });

  it("calls onClearAll when clear history button is clicked and confirmed", async () => {
    const user = userEvent.setup();
    const onClearAll = vi.fn();
    const conversations = [{ id: "c1", title: "Test Chat" }];

    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(
      <Sidebar
        open={true}
        conversations={conversations}
        activeId="c1"
        loadingConversations={false}
        onToggle={vi.fn()}
        onNew={vi.fn()}
        onSelect={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={onClearAll}
        onOpenSettings={vi.fn()}
      />
    );

    const clearBtn = screen.getByRole("button", { name: /clear all conversations/i });
    await user.click(clearBtn);

    expect(onClearAll).toHaveBeenCalledOnce();
  });

  it("does not call onClearAll when clear history is cancelled", async () => {
    const user = userEvent.setup();
    const onClearAll = vi.fn();
    const conversations = [{ id: "c1", title: "Test Chat" }];

    vi.spyOn(window, "confirm").mockReturnValue(false);

    render(
      <Sidebar
        open={true}
        conversations={conversations}
        activeId="c1"
        loadingConversations={false}
        onToggle={vi.fn()}
        onNew={vi.fn()}
        onSelect={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={onClearAll}
        onOpenSettings={vi.fn()}
      />
    );

    const clearBtn = screen.getByRole("button", { name: /clear all conversations/i });
    await user.click(clearBtn);

    expect(onClearAll).not.toHaveBeenCalled();
  });
});
