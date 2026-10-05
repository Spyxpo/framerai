import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import { searchConversations, SEARCH_SCOPES } from "../utils/search";

let mockWsHandlers = new Map();

vi.mock("../services/api", () => ({
  api: {
    health: vi.fn().mockResolvedValue({ status: "ok", model: "framerai-v1" }),
    listConversations: vi.fn().mockResolvedValue([]),
    getConversation: vi.fn().mockImplementation((id) =>
      Promise.resolve({ id, title: `Backend Title ${id}`, messages: [] })
    ),
    createConversation: vi.fn().mockImplementation(() =>
      Promise.resolve({
        id: `conv-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        title: "New Chat",
        messages: [],
      })
    ),
    deleteConversation: vi.fn().mockResolvedValue({ ok: true }),
    sendMessage: vi.fn().mockImplementation((convId, content) =>
      Promise.resolve({
        id: `srv-msg-${Date.now()}`,
        userMessageId: `srv-user-${Date.now()}`,
        content: `Reply to ${content}`,
        type: "text",
      })
    ),
    branchConversation: vi.fn().mockImplementation((parentId, messageId) =>
      Promise.resolve({
        id: `branch-from-${parentId}`,
        title: "Branched Chat (Branch)",
        parentConversationId: parentId,
        branchedFromMessageId: messageId,
      })
    ),
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
    disconnect() {}
    send() {}
    on(event, handler) {
      if (!mockWsHandlers.has(event)) {
        mockWsHandlers.set(event, []);
      }
      mockWsHandlers.get(event).push(handler);
      return () => {};
    }
    off() {}
  },
}));

if (!window.HTMLElement.prototype.scrollIntoView) {
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
}

describe("Issue #406 — Conversation Search Results Staleness Regression Tests", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it("Test 1 — Newly Added Message becomes searchable immediately without refresh", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
    });

    const activeId = result.current.activeConversation;
    expect(activeId).toBeDefined();

    // 1. Initial search: no matching message
    let searchRes = searchConversations({
      conversations: result.current.conversations,
      activeConversationId: activeId,
      query: "quantum-superposition-concept",
    });
    expect(searchRes).toHaveLength(0);

    // 2. Add matching message
    await act(async () => {
      await result.current.sendMessage("Explain quantum-superposition-concept in physics");
    });

    // 3. Search same query immediately
    searchRes = searchConversations({
      conversations: result.current.conversations,
      activeConversationId: result.current.activeConversation,
      query: "quantum-superposition-concept",
    });

    // Expected: new message appears in search results
    expect(searchRes.length).toBeGreaterThan(0);
    const msgMatch = searchRes.find((r) => r.type === "message");
    expect(msgMatch).toBeDefined();
    expect(msgMatch.content).toContain("quantum-superposition-concept");
  });

  it("Test 2 — Rename conversation reflects immediately and is protected from stale backend overwrite", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
    });

    const convId = result.current.activeConversation;

    // Search old title
    let searchOld = searchConversations({
      conversations: result.current.conversations,
      query: "New Chat",
    });
    expect(searchOld.some((r) => r.conversationId === convId)).toBe(true);

    // Rename conversation
    expect(typeof result.current.renameConversation).toBe("function");
    act(() => {
      result.current.renameConversation(convId, "Quantum Mechanics Deep Dive");
    });

    // Search new title immediately
    let searchNew = searchConversations({
      conversations: result.current.conversations,
      query: "Quantum Mechanics",
    });
    expect(searchNew.length).toBeGreaterThan(0);
    expect(searchNew[0].conversationTitle).toBe("Quantum Mechanics Deep Dive");

    // Old title should no longer represent the current conversation
    searchOld = searchConversations({
      conversations: result.current.conversations,
      query: "New Chat",
    });
    expect(searchOld.some((r) => r.conversationId === convId)).toBe(false);

    // Create another conversation and switch away
    await act(async () => {
      await result.current.createConversation();
    });

    // Now switch back to renamed conversation, triggering api.getConversation(convId)
    await act(async () => {
      await result.current.selectConversation(convId);
    });

    searchNew = searchConversations({
      conversations: result.current.conversations,
      query: "Quantum Mechanics",
    });
    expect(searchNew.length).toBeGreaterThan(0);
    expect(searchNew[0].conversationTitle).toBe("Quantum Mechanics Deep Dive");
  });

  it("Test 3 — Delete conversation immediately removes it and prevents resurrection from background fetches", async () => {
    const { api } = await import("../services/api");
    let resolveList;
    const listPromise = new Promise((resolve) => {
      resolveList = resolve;
    });
    api.listConversations.mockReturnValue(listPromise);

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
    });

    const convId = result.current.activeConversation;

    // Search finds conversation
    let searchRes = searchConversations({
      conversations: result.current.conversations,
      query: "New Chat",
    });
    expect(searchRes.some((r) => r.conversationId === convId)).toBe(true);

    // Delete conversation
    await act(async () => {
      await result.current.deleteConversation(convId);
    });

    // Search again: deleted conversation is absent
    searchRes = searchConversations({
      conversations: result.current.conversations,
      query: "New Chat",
    });
    expect(searchRes.some((r) => r.conversationId === convId)).toBe(false);

    // Out-of-order: background listConversations resolves with the deleted conversation
    await act(async () => {
      resolveList([{ id: convId, title: "New Chat", messageCount: 0 }]);
    });

    // Must still NOT appear in search results
    searchRes = searchConversations({
      conversations: result.current.conversations,
      query: "New Chat",
    });
    expect(searchRes.some((r) => r.conversationId === convId)).toBe(false);
  });

  it("Test 4 — Branch creation is searchable with branch semantics", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
      await result.current.sendMessage("Seed message for branching");
    });

    const parentId = result.current.activeConversation;
    const parentMsg = result.current.conversations[0]?.messages?.find((m) =>
      m.content?.includes("Seed message")
    );
    expect(parentMsg).toBeDefined();

    // Branch from this message
    let branch;
    await act(async () => {
      branch = await result.current.branchConversation(parentMsg.id, parentId);
    });
    expect(branch).toBeDefined();

    // Search for seed message across all conversations
    const searchRes = searchConversations({
      conversations: result.current.conversations,
      query: "Seed message",
      scope: SEARCH_SCOPES.ALL,
    });

    // Should find results in both parent and branch with correct branch semantics
    expect(searchRes.length).toBeGreaterThanOrEqual(2);
    const branchRes = searchRes.find((r) => r.conversationId === branch.id);
    expect(branchRes).toBeDefined();
    expect(branchRes.isBranch).toBe(true);
    expect(branchRes.parentConversationId).toBe(parentId);
  });

  it("Test 5 — Multiple Mutations (rename → add message → branch → delete another conversation)", async () => {
    const { result } = renderHook(() => useChat());

    // 1. Create two conversations: Conv A and Conv B
    await act(async () => {
      await result.current.createConversation();
    });
    const convAId = result.current.activeConversation;

    await act(async () => {
      await result.current.createConversation();
    });
    const convBId = result.current.activeConversation;

    // 2. Rename Conv A
    expect(typeof result.current.renameConversation).toBe("function");
    act(() => {
      result.current.renameConversation(convAId, "Renamed Alpha Conversation");
    });

    // 3. Switch to Conv A and add message
    await act(async () => {
      await result.current.selectConversation(convAId);
    });
    await act(async () => {
      await result.current.sendMessage("Unique content in Alpha turn");
    });

    // 4. Branch Conv A
    const alphaMsg = result.current.conversations
      .find((c) => c.id === convAId)
      ?.messages?.find((m) => m.content?.includes("Unique content in Alpha"));
    expect(alphaMsg).toBeDefined();

    let branchConv;
    await act(async () => {
      branchConv = await result.current.branchConversation(alphaMsg.id, convAId);
    });
    expect(branchConv).toBeDefined();

    // 5. Delete Conv B
    await act(async () => {
      await result.current.deleteConversation(convBId);
    });

    // Now search across everything:
    // a. Renamed title exists
    const titleRes = searchConversations({
      conversations: result.current.conversations,
      query: "Renamed Alpha",
    });
    expect(titleRes.length).toBeGreaterThan(0);

    // b. Added message exists
    const msgRes = searchConversations({
      conversations: result.current.conversations,
      query: "Unique content in Alpha",
    });
    expect(msgRes.length).toBeGreaterThanOrEqual(2); // In Conv A and Branch

    // c. Deleted Conv B is nowhere to be found
    const delRes = searchConversations({
      conversations: result.current.conversations,
      query: "Conv " + convBId,
    });
    expect(delRes.some((r) => r.conversationId === convBId)).toBe(false);

    // d. Branch conversation is active and distinct
    const branchRes = searchConversations({
      conversations: result.current.conversations,
      activeConversationId: branchConv.id,
      query: "Unique content in Alpha",
      scope: SEARCH_SCOPES.CURRENT,
    });
    expect(branchRes.every((r) => r.conversationId === branchConv.id)).toBe(true);
  });

  it("Test 6 — Repeated Search remains deterministic and current before and after mutations", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.createConversation();
      await result.current.sendMessage("Stable deterministic query message");
    });

    const activeId = result.current.activeConversation;

    // Search 1
    const search1 = searchConversations({
      conversations: result.current.conversations,
      activeConversationId: activeId,
      query: "deterministic query",
    });

    // Search 2 (repeated)
    const search2 = searchConversations({
      conversations: result.current.conversations,
      activeConversationId: activeId,
      query: "deterministic query",
    });

    expect(search1).toEqual(search2);

    // Also test id-less / legacy messages determinism
    const legacyConv = [
      {
        id: "legacy-conv-1",
        title: "Legacy Chat",
        messages: [{ role: "user", content: "Legacy message for deterministic query" }],
      },
    ];
    const leg1 = searchConversations({ conversations: legacyConv, query: "deterministic query" });
    const leg2 = searchConversations({ conversations: legacyConv, query: "deterministic query" });
    expect(leg1).toEqual(leg2);

    // Mutate: add another message
    await act(async () => {
      await result.current.sendMessage("Another deterministic query message");
    });

    // Search 3 (after mutation)
    const search3 = searchConversations({
      conversations: result.current.conversations,
      activeConversationId: activeId,
      query: "deterministic query",
    });

    // Search 4 (repeated after mutation)
    const search4 = searchConversations({
      conversations: result.current.conversations,
      activeConversationId: activeId,
      query: "deterministic query",
    });

    expect(search3.length).toBe(search1.length + 1);
    expect(search3).toEqual(search4);
  });

  it("Test 7 — Asynchronous out-of-order state updates preserve authoritative search results", async () => {
    const { api } = await import("../services/api");
    let resolveSlowFetch;
    api.getConversation.mockImplementation((id) => {
      if (id === "slow-conv") {
        return new Promise((resolve) => {
          resolveSlowFetch = resolve;
        });
      }
      return Promise.resolve({
        id,
        title: "Fast Conv",
        messages: [{ id: "fast-1", role: "user", content: "Fast message content" }],
      });
    });

    const { result } = renderHook(() => useChat());

    // User starts slow selection
    act(() => {
      result.current.selectConversation("slow-conv");
    });

    // Before slow selection finishes, user switches to fast-conv and sends a message
    await act(async () => {
      await result.current.selectConversation("fast-conv");
    });
    await act(async () => {
      await result.current.sendMessage("Authoritative latest turn");
    });

    // User deletes slow-conv while fetch is still in flight
    await act(async () => {
      await result.current.deleteConversation("slow-conv");
    });

    // Now slow selection finishes out-of-order with stale/outdated data
    await act(async () => {
      resolveSlowFetch({
        id: "slow-conv",
        title: "Stale Slow Title",
        messages: [{ id: "slow-1", role: "user", content: "Stale message content" }],
      });
    });

    // slow-conv must NOT be resurrected into search results
    const resurrectedResults = searchConversations({
      conversations: result.current.conversations,
      query: "Stale message",
      scope: SEARCH_SCOPES.ALL,
    });
    expect(resurrectedResults).toHaveLength(0);

    // Current conversation must be fast-conv, and search in current scope must NOT be contaminated by slow-conv
    const currentResults = searchConversations({
      conversations: result.current.conversations,
      activeConversationId: result.current.activeConversation,
      query: "Stale message",
      scope: SEARCH_SCOPES.CURRENT,
    });
    expect(currentResults).toHaveLength(0);

    const fastResults = searchConversations({
      conversations: result.current.conversations,
      activeConversationId: result.current.activeConversation,
      query: "Authoritative latest",
      scope: SEARCH_SCOPES.CURRENT,
    });
    expect(fastResults.length).toBeGreaterThan(0);
  });
});
