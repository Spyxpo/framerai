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
import { api } from "../services/api";
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
    const conv3 = { id: "c3", title: "Newest Chat", updatedAt: "2026-08-23T12:00:00.000Z", messages: [{ id: "m3", role: "user", content: "C".repeat(500) }] };
    const conv2 = { id: "c2", title: "Medium Chat", updatedAt: "2026-08-23T11:00:00.000Z", messages: [{ id: "m2", role: "user", content: "B".repeat(500) }] };
    const conv1 = { id: "c1", title: "Oldest Chat", updatedAt: "2026-08-23T10:00:00.000Z", messages: [{ id: "m1", role: "user", content: "A".repeat(500) }] };

    // Newest-first ordering in array: [conv3, conv2, conv1]
    const conversations = [conv3, conv2, conv1];

    // Set maxBytes small enough to force eviction of oldest conversation (c1)
    const { conversations: evicted } = evictOldestConversations(conversations, "c3", 1500);

    // Oldest non-active conversation (c1) should be evicted
    expect(evicted.some((c) => c.id === "c1")).toBe(false);
    // Newest conversation (c3 - active) and middle conversation (c2) should be preserved
    expect(evicted.some((c) => c.id === "c3")).toBe(true);
    expect(evicted.some((c) => c.id === "c2")).toBe(true);
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

// Issue #453: a conversation too large to be saved even on its own made the
// eviction protect it while evicting every other conversation, then give up
// with an empty list, which the save wrote as "nothing saved".
describe("Storage utility — an oversized conversation (Issue #453)", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  // What the eviction measures: the stored payload without savedAt.
  const payloadLength = (conversations, activeConversationId) =>
    JSON.stringify({ version: STORAGE_VERSION, conversations, activeConversationId }).length;

  const chat = (id, hour, chars) => ({
    id,
    title: id,
    updatedAt: `2026-08-23T${String(hour).padStart(2, "0")}:00:00.000Z`,
    messages: [{ id: `${id}-m1`, role: "user", content: "x".repeat(chars) }],
  });

  const storedIds = () => {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw === null ? null : JSON.parse(raw).conversations.map((c) => c.id);
  };

  it("#453: oldest-first eviction is unchanged when every conversation fits on its own", () => {
    const list = [chat("c4", 13, 1000), chat("c3", 12, 1000), chat("c2", 11, 1000), chat("c1", 10, 1000)];

    // Exactly at the limit, nothing is evicted.
    const whole = payloadLength(list, "c4");
    expect(evictOldestConversations(list, "c4", whole).conversations.map((c) => c.id)).toEqual(["c4", "c3", "c2", "c1"]);

    // One over, only the oldest goes, and the rest keep their order.
    expect(evictOldestConversations(list, "c4", whole - 1).conversations.map((c) => c.id)).toEqual(["c4", "c3", "c2"]);

    // Room for two: the oldest are evicted first, the active one is kept even
    // though it is the newest only by chance here.
    const two = payloadLength([list[0], list[1]], "c4");
    expect(evictOldestConversations(list, "c4", two).conversations.map((c) => c.id)).toEqual(["c4", "c3"]);
    expect(evictOldestConversations(list, "c1", payloadLength([list[0], list[3]], "c1")).conversations.map((c) => c.id)).toEqual(["c4", "c1"]);
  });

  it("#453: an oversized active conversation is left out and every other conversation is kept", () => {
    const big = chat("big", 14, 6000);
    const small = [chat("s3", 12, 300), chat("s2", 11, 300), chat("s1", 10, 300)];

    const result = evictOldestConversations([big, ...small], "big", 4096);

    expect(result.conversations.map((c) => c.id)).toEqual(["s3", "s2", "s1"]);
    // The active one is not stored, so the stored pointer names one that is,
    // the same fallback an evicted active conversation already gets.
    expect(result.activeConversationId).toBe("s3");
  });

  it("#453: an oversized conversation that is not active is left out too", () => {
    // The newest, so oldest-first eviction alone would reach it last.
    const big = chat("big", 14, 6000);
    const small = [chat("s3", 12, 300), chat("s2", 11, 300), chat("s1", 10, 300)];

    const result = evictOldestConversations([big, ...small], "s2", 4096);

    expect(result.conversations.map((c) => c.id)).toEqual(["s3", "s2", "s1"]);
    expect(result.activeConversationId).toBe("s2");
  });

  it("#453: once the oversized one is left out, the rest are still evicted oldest-first if they do not fit", () => {
    const big = chat("big", 14, 6000);
    const small = [chat("s3", 12, 1000), chat("s2", 11, 1000), chat("s1", 10, 1000)];
    const room = payloadLength([small[0], small[1]], "s3");

    const result = evictOldestConversations([big, ...small], "big", room);

    expect(result.conversations.map((c) => c.id)).toEqual(["s3", "s2"]);
  });

  it("#453: a conversation exactly at the limit on its own is kept, one character over is left out", () => {
    const maxBytes = 4096;
    const sized = (chars) => chat("big", 14, chars);
    const exact = maxBytes - payloadLength([sized(0)], "big");
    expect(payloadLength([sized(exact)], "big")).toBe(maxBytes);
    const small = chat("s1", 10, 300);

    // At the limit it fits on its own, so it is the active one that is kept and
    // the other that is evicted, as before.
    const atLimit = evictOldestConversations([sized(exact), small], "big", maxBytes);
    expect(atLimit.conversations.map((c) => c.id)).toEqual(["big"]);

    // One over, it can never be saved, so it is the one left out.
    const over = evictOldestConversations([sized(exact + 1), small], "big", maxBytes);
    expect(over.conversations.map((c) => c.id)).toEqual(["s1"]);
  });

  it("#453: saving with an oversized active conversation keeps the conversations already saved", () => {
    const small = [chat("s3", 12, 300), chat("s2", 11, 300), chat("s1", 10, 300)];
    expect(saveConversationsToStorage(small, "s1", { maxBytes: 8192 })).toBe(true);
    expect(storedIds()).toEqual(["s3", "s2", "s1"]);

    const big = chat("big", 14, 12000);
    expect(saveConversationsToStorage([big, ...small], "big", { maxBytes: 8192 })).toBe(true);

    expect(storedIds()).toEqual(["s3", "s2", "s1"]);
    const reloaded = loadConversationsFromStorage();
    expect(reloaded.conversations.map((c) => c.id)).toEqual(["s3", "s2", "s1"]);
    expect(reloaded.conversations.every((c) => c.messages.length === 1)).toBe(true);
  });

  it("#453: when not one conversation fits, nothing is saved rather than a stale copy", () => {
    // Storage mirrors the conversations it is given, so keeping the old key here
    // would bring back conversations deleted since it was written.
    saveConversationsToStorage([chat("old", 9, 300)], "old", { maxBytes: 8192 });
    expect(storedIds()).toEqual(["old"]);

    expect(saveConversationsToStorage([chat("big", 14, 12000)], "big", { maxBytes: 8192 })).toBe(true);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("#453: a failed write still leaves what was saved untouched", () => {
    const small = [chat("s2", 11, 300), chat("s1", 10, 300)];
    saveConversationsToStorage(small, "s1", { maxBytes: 8192 });
    const before = localStorage.getItem(STORAGE_KEY);

    const setItemSpy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("QuotaExceededError: Storage full");
    });
    try {
      // An ordinary save that cannot be written.
      expect(saveConversationsToStorage([chat("s3", 12, 300), ...small], "s3", { maxBytes: 8192 })).toBe(false);
      expect(localStorage.getItem(STORAGE_KEY)).toBe(before);

      // The same with an oversized conversation open: still a failed write,
      // never a cleared key.
      expect(saveConversationsToStorage([chat("big", 14, 12000), ...small], "big", { maxBytes: 8192 })).toBe(false);
      expect(localStorage.getItem(STORAGE_KEY)).toBe(before);
    } finally {
      setItemSpy.mockRestore();
    }
  });

  it("#453: opening a conversation over the real 2 MB limit keeps every saved conversation", async () => {
    const saved = Array.from({ length: 5 }, (_, i) => ({
      id: `s${i}`,
      title: `Saved ${i}`,
      updatedAt: `2026-08-23T1${i}:00:00.000Z`,
      messages: [
        { id: `s${i}-u`, role: "user", content: `question ${i}`, type: "text", timestamp: "2026-08-23T10:00:00.000Z" },
        { id: `s${i}-a`, role: "assistant", content: `answer ${i}`, type: "text", timestamp: "2026-08-23T10:00:01.000Z" },
      ],
    }));
    // The long conversation is known only from the server list, as it is in the
    // app before it is opened.
    const summary = { id: "big", title: "Long chat", updatedAt: "2026-08-23T09:00:00.000Z", messages: [] };
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ version: STORAGE_VERSION, conversations: [...saved, summary], activeConversationId: "s0" })
    );

    const messages = Array.from({ length: 270 }, (_, i) => ({
      id: `big-${i}`,
      role: i % 2 ? "assistant" : "user",
      content: "lorem ipsum ".repeat(667).slice(0, 8000),
      type: "text",
      timestamp: "2026-08-23T09:00:00.000Z",
    }));
    const bigConv = { id: "big", title: "Long chat", version: 271, updatedAt: "2026-08-23T15:00:00.000Z", messages };
    expect(JSON.stringify(bigConv).length).toBeGreaterThan(2 * 1024 * 1024);

    api.getConversation.mockImplementation((id) => Promise.resolve(id === "big" ? bigConv : { id, messages: [] }));
    try {
      let result;
      await act(async () => {
        result = renderHook(() => useChat({})).result;
      });
      // Opened like a click: the local copy is shown first, then the server's
      // replaces it. One act around both would batch them out of that order.
      let pending;
      act(() => {
        pending = result.current.selectConversation("big");
      });
      await act(async () => {
        await pending;
      });

      // The conversation is open and whole in the app ...
      expect(result.current.activeConversation).toBe("big");
      expect(result.current.messages).toHaveLength(270);

      // ... and every conversation that was saved is still saved, whole.
      const reloaded = loadConversationsFromStorage();
      expect(reloaded.conversations.map((c) => c.id).sort()).toEqual(["s0", "s1", "s2", "s3", "s4"]);
      expect(reloaded.conversations.every((c) => c.messages.length === 2)).toBe(true);
    } finally {
      api.getConversation.mockImplementation((id) => Promise.resolve({ id, messages: [] }));
    }
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

  it("reload does not change updatedAt timestamp", async () => {
    const initialUpdatedAt = "2026-08-23T10:00:00.000Z";
    const initialPayload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "c1",
          title: "Persisted Chat",
          updatedAt: initialUpdatedAt,
          messages: [{ id: "m1", role: "user", content: "Existing message" }],
        },
      ],
      activeConversationId: "c1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(initialPayload));

    let result;
    await act(async () => {
      const rendered = renderHook(() => useChat({}));
      result = rendered.result;
    });

    expect(result.current.conversations[0].updatedAt).toBe(initialUpdatedAt);

    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    expect(saved.conversations[0].updatedAt).toBe(initialUpdatedAt);
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
