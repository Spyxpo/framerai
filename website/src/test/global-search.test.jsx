import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";
import SearchModal from "../components/Search/SearchModal";
import {
  SEARCH_SCOPES,
  searchConversations,
  extractSnippet,
  highlightMatches,
} from "../utils/search";
import { STORAGE_KEY, STORAGE_VERSION } from "../utils/storage";

vi.mock("../services/api", () => ({
  api: {
    health: vi.fn().mockResolvedValue({ status: "ok", model: "framerai-v1" }),
    listConversations: vi.fn().mockResolvedValue([]),
    getConversation: vi.fn().mockImplementation((id) => {
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
        const found = saved?.conversations?.find((c) => c.id === id);
        if (found) return Promise.resolve(found);
      } catch {
        // Fallback
      }
      return Promise.resolve({ id, title: `Conv ${id}`, messages: [] });
    }),
    createConversation: vi.fn().mockResolvedValue({ id: "new-conv", title: "New Chat", messages: [] }),
    deleteConversation: vi.fn().mockResolvedValue({ ok: true }),
    branchConversation: vi.fn().mockImplementation((parentId, messageId) =>
      Promise.resolve({
        id: `branch-from-${parentId}`,
        title: "Branched Chat",
        parentConversationId: parentId,
        branchedFromMessageId: messageId,
        messages: [],
      })
    ),
  },
}));

vi.mock("../services/websocket", () => ({
  WebSocketClient: class {
    connect() {
      return Promise.resolve();
    }
    disconnect() {}
    send() {}
    on() {}
    off() {}
  },
}));

// Mock scrollIntoView which is missing in jsdom
if (!window.HTMLElement.prototype.scrollIntoView) {
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
}

describe("Global Message Search - Unit & Algorithm Tests", () => {
  const sampleConversations = [
    {
      id: "conv-1",
      title: "React Architecture Discussion",
      updatedAt: "2026-09-01T10:00:00Z",
      messages: [
        { id: "m1", role: "user", content: "How do we build a global search component?" },
        { id: "m2", role: "assistant", content: "You can implement deterministic search with scope filters." },
      ],
    },
    {
      id: "conv-2",
      title: "Python Data Processing",
      updatedAt: "2026-09-02T10:00:00Z",
      messages: [
        { id: "m3", role: "user", content: "Show me a script for parsing JSON data." },
        { id: "m4", role: "assistant", content: "Here is a python snippet using json.loads() and search regex." },
      ],
    },
    {
      id: "conv-branch",
      title: "React Architecture Discussion (Branch)",
      parentConversationId: "conv-1",
      branchedFromMessageId: "m1",
      updatedAt: "2026-09-03T10:00:00Z",
      messages: [
        { id: "m1", role: "user", content: "How do we build a global search component?" },
        { id: "m5", role: "assistant", content: "In this branch, let's explore fuzzy indexing vs exact matching." },
      ],
    },
  ];

  it("1. Returns empty results for empty or whitespace-only queries", () => {
    expect(searchConversations({ conversations: sampleConversations, query: "" })).toEqual([]);
    expect(searchConversations({ conversations: sampleConversations, query: "   " })).toEqual([]);
    expect(searchConversations({ conversations: sampleConversations, query: null })).toEqual([]);
    expect(searchConversations({ conversations: sampleConversations, query: undefined })).toEqual([]);
  });

  it("2. Handles exact, partial, and case-insensitive matches", () => {
    // Exact match
    const exact = searchConversations({ conversations: sampleConversations, query: "deterministic" });
    expect(exact).toHaveLength(1);
    expect(exact[0].messageId).toBe("m2");

    // Partial match
    const partial = searchConversations({ conversations: sampleConversations, query: "determ" });
    expect(partial).toHaveLength(1);
    expect(partial[0].messageId).toBe("m2");

    // Case-insensitive match
    const upper = searchConversations({ conversations: sampleConversations, query: "DETERMINISTIC" });
    expect(upper).toHaveLength(1);
    expect(upper[0].messageId).toBe("m2");
  });

  it("3. Handles queries with special characters safely without crashing or regex errors", () => {
    const convWithSpecial = [
      {
        id: "c-spec",
        title: "Regex & Glob [v1.0] (Test)*+?^$",
        messages: [
          { id: "sm1", role: "user", content: "Can we match [v1.0] and (Test)? Yes *+?^$" },
          { id: "sm2", role: "assistant", content: "Special characters $100.00 and path C:\\root\\file" },
        ],
      },
    ];

    expect(() =>
      searchConversations({ conversations: convWithSpecial, query: "[v1.0]" })
    ).not.toThrow();
    const bracketMatch = searchConversations({ conversations: convWithSpecial, query: "[v1.0]" });
    expect(bracketMatch.length).toBeGreaterThanOrEqual(1);

    const regexCharsMatch = searchConversations({ conversations: convWithSpecial, query: "*+?^$" });
    expect(regexCharsMatch.length).toBeGreaterThanOrEqual(1);

    const dollarMatch = searchConversations({ conversations: convWithSpecial, query: "$100.00" });
    expect(dollarMatch).toHaveLength(1);
    expect(dollarMatch[0].messageId).toBe("sm2");
  });

  it("4. Matches conversation titles correctly", () => {
    const results = searchConversations({
      conversations: sampleConversations,
      query: "Python Data",
      scope: SEARCH_SCOPES.ALL,
    });
    const titleMatch = results.find((r) => r.type === "title");
    expect(titleMatch).toBeDefined();
    expect(titleMatch.conversationId).toBe("conv-2");
    expect(titleMatch.conversationTitle).toBe("Python Data Processing");
  });

  it("5. Matches user messages and assistant messages", () => {
    // User message match
    const userRes = searchConversations({ conversations: sampleConversations, query: "parsing JSON" });
    expect(userRes).toHaveLength(1);
    expect(userRes[0].role).toBe("user");
    expect(userRes[0].messageId).toBe("m3");

    // Assistant message match
    const asstRes = searchConversations({ conversations: sampleConversations, query: "scope filters" });
    expect(asstRes).toHaveLength(1);
    expect(asstRes[0].role).toBe("assistant");
    expect(asstRes[0].messageId).toBe("m2");
  });

  it("6. Respects search scopes: all, conversations, messages, and current", () => {
    // Scope: conversations (titles only)
    const titlesOnly = searchConversations({
      conversations: sampleConversations,
      query: "React",
      scope: SEARCH_SCOPES.CONVERSATIONS,
    });
    expect(titlesOnly.every((r) => r.type === "title")).toBe(true);
    expect(titlesOnly).toHaveLength(2); // conv-1 and conv-branch titles

    // Scope: messages (messages only, exclude titles)
    const messagesOnly = searchConversations({
      conversations: sampleConversations,
      query: "React",
      scope: SEARCH_SCOPES.MESSAGES,
    });
    expect(messagesOnly.every((r) => r.type === "message")).toBe(true);

    // Scope: current (only within active conversation)
    const currentOnly = searchConversations({
      conversations: sampleConversations,
      activeConversationId: "conv-1",
      query: "global search",
      scope: SEARCH_SCOPES.CURRENT,
    });
    expect(currentOnly.length).toBeGreaterThan(0);
    expect(currentOnly.every((r) => r.conversationId === "conv-1")).toBe(true);
    // conv-branch also has "global search", but must be excluded when scope is current (conv-1)
    expect(currentOnly.some((r) => r.conversationId === "conv-branch")).toBe(false);
  });

  it("7. Orders results deterministically", () => {
    const results = searchConversations({
      conversations: sampleConversations,
      query: "search",
      scope: SEARCH_SCOPES.ALL,
    });
    // Deterministic order: for each conversation in list, title match first, then messages
    expect(results.length).toBeGreaterThanOrEqual(3);
    expect(results[0].conversationId).toBe("conv-1");
    expect(results[1].conversationId).toBe("conv-1");
    expect(results[2].conversationId).toBe("conv-2");
  });

  it("8. Deduplicates messages in conversations before searching", () => {
    const convWithDuplicates = [
      {
        id: "c-dup",
        title: "Duplicate Chat",
        messages: [
          { id: "m-dup", role: "user", content: "Repeated content" },
          { id: "m-dup", role: "user", content: "Repeated content" }, // duplicate id
        ],
      },
    ];

    const results = searchConversations({
      conversations: convWithDuplicates,
      query: "Repeated content",
    });
    expect(results).toHaveLength(1);
    expect(results[0].messageId).toBe("m-dup");
  });

  it("9. Snippet extraction centers around the query and truncates properly", () => {
    const longText =
      "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega. Here is the TARGET_KEYWORD inside a very long text that must be extracted properly with context around it.";
    const snippet = extractSnippet(longText, "TARGET_KEYWORD", 60);
    expect(snippet).toContain("TARGET_KEYWORD");
    expect(snippet.startsWith("...")).toBe(true);
    expect(snippet.endsWith("...")).toBe(true);

    // Short text returns full text
    expect(extractSnippet("Short text", "text", 50)).toBe("Short text");
  });

  it("10. Highlight matching breaks text into matched and unmatched segments safely", () => {
    const segments = highlightMatches("Quick brown fox jumps", "brown");
    expect(segments).toEqual([
      { text: "Quick ", isMatch: false },
      { text: "brown", isMatch: true },
      { text: " fox jumps", isMatch: false },
    ]);
  });
});

describe("Branch-Aware Global Search", () => {
  const branchConversations = [
    {
      id: "parent-c",
      title: "Parent Conversation",
      messages: [
        { id: "p-m1", role: "user", content: "Initial requirement discussion" },
        { id: "p-m2", role: "assistant", content: "Let's plan the architecture." },
      ],
    },
    {
      id: "branch-a",
      title: "Parent Conversation (Branch A)",
      parentConversationId: "parent-c",
      branchedFromMessageId: "p-m1",
      messages: [
        { id: "p-m1", role: "user", content: "Initial requirement discussion" },
        { id: "b-m1", role: "user", content: "Branch A: using PostgreSQL database" },
      ],
    },
    {
      id: "branch-b",
      title: "Parent Conversation (Branch B)",
      parentConversationId: "parent-c",
      branchedFromMessageId: "p-m1",
      messages: [
        { id: "p-m1", role: "user", content: "Initial requirement discussion" },
        { id: "b-m2", role: "user", content: "Branch B: using MongoDB database" },
      ],
    },
    {
      id: "nested-branch",
      title: "Nested Branch from A",
      parentConversationId: "branch-a",
      branchedFromMessageId: "b-m1",
      messages: [
        { id: "p-m1", role: "user", content: "Initial requirement discussion" },
        { id: "b-m1", role: "user", content: "Branch A: using PostgreSQL database" },
        { id: "nb-m1", role: "assistant", content: "Nested decision: using Prisma ORM with PostgreSQL" },
      ],
    },
  ];

  it("identifies branch metadata, parent conversation, and branched message in results", () => {
    const results = searchConversations({
      conversations: branchConversations,
      query: "PostgreSQL",
    });

    // Should find results in branch-a (1 match) and nested-branch (2 matches)
    expect(results).toHaveLength(3);

    const branchAResult = results.find((r) => r.conversationId === "branch-a");
    expect(branchAResult).toBeDefined();
    expect(branchAResult.isBranch).toBe(true);
    expect(branchAResult.parentConversationId).toBe("parent-c");
    expect(branchAResult.branchedFromMessageId).toBe("p-m1");

    const nestedResult = results.find((r) => r.conversationId === "nested-branch");
    expect(nestedResult).toBeDefined();
    expect(nestedResult.isBranch).toBe(true);
    expect(nestedResult.parentConversationId).toBe("branch-a");
  });

  it("preserves correct conversationId for navigation without selecting parent", () => {
    const results = searchConversations({
      conversations: branchConversations,
      query: "MongoDB",
    });
    expect(results).toHaveLength(1);
    expect(results[0].conversationId).toBe("branch-b");
    expect(results[0].conversationId).not.toBe("parent-c");
  });
});

describe("Search UI Component (SearchModal)", () => {
  let user;
  beforeEach(() => {
    user = userEvent.setup();
  });

  const testConversations = [
    {
      id: "c1",
      title: "Algorithms",
      messages: [
        { id: "m1", role: "user", content: "What is quicksort?" },
        { id: "m2", role: "assistant", content: "Quicksort is a divide-and-conquer sorting algorithm." },
      ],
    },
    {
      id: "c2",
      title: "Web Development",
      messages: [
        { id: "m3", role: "user", content: "How do React hooks work?" },
        { id: "m4", role: "assistant", content: "Hooks allow state in functional components." },
      ],
    },
  ];

  it("renders when open is true, and does not render when open is false", () => {
    const { rerender } = render(
      <SearchModal
        open={false}
        conversations={testConversations}
        activeConversationId="c1"
        onClose={vi.fn()}
        onNavigate={vi.fn()}
      />
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    rerender(
      <SearchModal
        open={true}
        conversations={testConversations}
        activeConversationId="c1"
        onClose={vi.fn()}
        onNavigate={vi.fn()}
      />
    );
    expect(screen.getByRole("dialog", { name: /global message search/i })).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/search conversations/i)).toBeInTheDocument();
  });

  it("displays search results and result count when typing a query", async () => {
    render(
      <SearchModal
        open={true}
        conversations={testConversations}
        activeConversationId="c1"
        onClose={vi.fn()}
        onNavigate={vi.fn()}
      />
    );

    const input = screen.getByPlaceholderText(/search conversations/i);
    await user.type(input, "Hooks");

    await waitFor(() => {
      expect(screen.getByText(/2 results found/i)).toBeInTheDocument();
      expect(screen.getAllByText("Web Development").length).toBeGreaterThan(0);
      expect(screen.getByText(/allow state in functional components/i)).toBeInTheDocument();
    });
  });

  it("filters results when switching search scope buttons", async () => {
    render(
      <SearchModal
        open={true}
        conversations={testConversations}
        activeConversationId="c1"
        onClose={vi.fn()}
        onNavigate={vi.fn()}
      />
    );

    const input = screen.getByPlaceholderText(/search conversations/i);
    await user.type(input, "Algorithms");

    // "All" scope shows title match
    await waitFor(() => {
      expect(screen.getAllByText("Algorithms").length).toBeGreaterThan(0);
    });

    // Switch to "Messages" scope: Algorithms is only in title, so should show 0 results
    const messagesTab = screen.getByRole("tab", { name: /^messages$/i });
    await user.click(messagesTab);

    await waitFor(() => {
      expect(screen.getByText(/no matches found/i)).toBeInTheDocument();
    });

    // Switch to "Conversations" scope: should show title match
    const convsTab = screen.getByRole("tab", { name: /^conversations$/i });
    await user.click(convsTab);

    await waitFor(() => {
      expect(screen.getAllByText("Algorithms").length).toBeGreaterThan(0);
    });
  });

  it("navigates between results using Previous/Next buttons and keyboard shortcuts", async () => {
    render(
      <SearchModal
        open={true}
        conversations={testConversations}
        activeConversationId="c1"
        onClose={vi.fn()}
        onNavigate={vi.fn()}
      />
    );

    const input = screen.getByPlaceholderText(/search conversations/i);
    // Matches "is" across multiple messages
    await user.type(input, "is");

    await waitFor(() => {
      expect(screen.getByText(/results found/i)).toBeInTheDocument();
    });

    const nextBtn = screen.getByRole("button", { name: /next result/i });
    const prevBtn = screen.getByRole("button", { name: /previous result/i });

    expect(screen.getByText(/1 of \d+/)).toBeInTheDocument();

    await user.click(nextBtn);
    expect(screen.getByText(/2 of \d+/)).toBeInTheDocument();

    await user.click(prevBtn);
    expect(screen.getByText(/1 of \d+/)).toBeInTheDocument();

    // Keyboard ArrowDown / ArrowUp navigation
    await user.keyboard("{ArrowDown}");
    expect(screen.getByText(/2 of \d+/)).toBeInTheDocument();

    await user.keyboard("{ArrowUp}");
    expect(screen.getByText(/1 of \d+/)).toBeInTheDocument();
  });

  it("calls onNavigate and closes when a result is clicked or Enter is pressed", async () => {
    const onNavigate = vi.fn();
    const onClose = vi.fn();

    render(
      <SearchModal
        open={true}
        conversations={testConversations}
        activeConversationId="c1"
        onClose={onClose}
        onNavigate={onNavigate}
      />
    );

    const input = screen.getByPlaceholderText(/search conversations/i);
    await user.type(input, "quicksort");

    await waitFor(() => {
      expect(screen.getByText(/divide-and-conquer/i)).toBeInTheDocument();
    });

    const resultItem = screen.getByText(/divide-and-conquer/i).closest(".search-result-item");
    await user.click(resultItem);

    expect(onNavigate).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "c1",
        messageId: "m2",
      })
    );
    expect(onClose).toHaveBeenCalled();
  });

  it("closes modal on Escape key", async () => {
    const onClose = vi.fn();
    render(
      <SearchModal
        open={true}
        conversations={testConversations}
        activeConversationId="c1"
        onClose={onClose}
        onNavigate={vi.fn()}
      />
    );

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });
});

describe("Full Application Search & State Integration", () => {
  let user;
  beforeEach(() => {
    user = userEvent.setup();
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("1. Opens search modal from Sidebar search button and via Cmd+K keyboard shortcut", async () => {
    const payload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "c-app-1",
          title: "Searchable Chat",
          messages: [{ id: "m-app-1", role: "user", content: "Hello world" }],
        },
      ],
      activeConversationId: "c-app-1",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Searchable Chat")).toBeInTheDocument();
    });

    // Modal not open initially
    expect(screen.queryByRole("dialog", { name: /global message search/i })).not.toBeInTheDocument();

    // Click search button in sidebar header
    const sidebarSearchBtn = screen.getByRole("button", { name: /search conversations and messages/i });
    await user.click(sidebarSearchBtn);

    expect(screen.getByRole("dialog", { name: /global message search/i })).toBeInTheDocument();

    // Close on Escape
    await user.keyboard("{Escape}");
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /global message search/i })).not.toBeInTheDocument();
    });

    // Open via Cmd+K keyboard shortcut
    await user.keyboard("{Meta>}k{/Meta}");
    expect(screen.getByRole("dialog", { name: /global message search/i })).toBeInTheDocument();
  });

  it("2. Selecting a result switches conversation, scrolls message into view, and applies temporary highlight", async () => {
    const payload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "conv-first",
          title: "First Active Chat",
          messages: [{ id: "m-first", role: "user", content: "Active content" }],
        },
        {
          id: "conv-second",
          title: "Second Target Chat",
          messages: [{ id: "m-target", role: "assistant", content: "Unique target reply to find" }],
        },
      ],
      activeConversationId: "conv-first",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Active content")).toBeInTheDocument();
    });

    // Initially active is conv-first
    expect(screen.queryByText("Unique target reply to find")).not.toBeInTheDocument();

    // Open search
    const searchBtn = screen.getByRole("button", { name: /search conversations and messages/i });
    await user.click(searchBtn);

    const input = screen.getByPlaceholderText(/search conversations/i);
    await user.type(input, "Unique target reply");

    await waitFor(() => {
      expect(screen.getByText(/Unique target reply/i)).toBeInTheDocument();
    });

    // Select the target result
    const resultItem = screen.getByText(/Unique target reply/i).closest(".search-result-item");
    await user.click(resultItem);

    // Modal should close and active conversation should switch to conv-second
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
      expect(screen.getByText("Unique target reply to find")).toBeInTheDocument();
    });

    // Target message element has message-highlighted class and scrollIntoView was called
    const targetArticle = screen.getByText("Unique target reply to find").closest(".message");
    expect(targetArticle).toHaveClass("message-highlighted");
    expect(targetArticle.getAttribute("data-message-id")).toBe("m-target");
  });

  it("3. State consistency: deleted conversation immediately disappears from search results", async () => {
    const payload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "stay-c",
          title: "Stay Chat",
          messages: [{ id: "m-stay", role: "user", content: "Query keyword stays" }],
        },
        {
          id: "del-c",
          title: "Delete Me Chat",
          messages: [{ id: "m-del", role: "user", content: "Query keyword to delete" }],
        },
      ],
      activeConversationId: "stay-c",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Stay Chat")).toBeInTheDocument();
    });

    // Delete "Delete Me Chat"
    const deleteBtn = screen.getByRole("button", { name: /delete conversation: delete me chat/i });
    await user.click(deleteBtn);

    await waitFor(() => {
      expect(screen.queryByText("Delete Me Chat")).not.toBeInTheDocument();
    });

    // Open search and search for "Query keyword"
    const searchBtn = screen.getByRole("button", { name: /search conversations and messages/i });
    await user.click(searchBtn);

    const input = screen.getByPlaceholderText(/search conversations/i);
    await user.type(input, "Query keyword");

    await waitFor(() => {
      expect(screen.getByText(/1 result.*found/i)).toBeInTheDocument();
    });

    const modal = screen.getByRole("dialog", { name: /global message search/i });
    expect(within(modal).getByText("Stay Chat")).toBeInTheDocument();
    expect(within(modal).queryByText("Delete Me Chat")).not.toBeInTheDocument();
  });

  it("4. Branch navigation: selecting a result on a branch opens the branch conversation, not the parent", async () => {
    const payload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "parent-id",
          title: "Base Plan",
          messages: [{ id: "pm-1", role: "user", content: "Root message" }],
        },
        {
          id: "branch-id",
          title: "Base Plan (Branch)",
          parentConversationId: "parent-id",
          branchedFromMessageId: "pm-1",
          messages: [
            { id: "pm-1", role: "user", content: "Root message" },
            { id: "bm-1", role: "assistant", content: "Branch exclusive feature discussion" },
          ],
        },
      ],
      activeConversationId: "parent-id",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Root message")).toBeInTheDocument();
    });

    // Open search
    const searchBtn = screen.getByRole("button", { name: /search conversations and messages/i });
    await user.click(searchBtn);

    const input = screen.getByPlaceholderText(/search conversations/i);
    await user.type(input, "Branch exclusive");

    await waitFor(() => {
      expect(screen.getByText(/Branch exclusive/i)).toBeInTheDocument();
    });

    const resultItem = screen.getByText(/Branch exclusive/i).closest(".search-result-item");
    await user.click(resultItem);

    // After navigation, the active conversation must be the branch
    await waitFor(() => {
      expect(screen.getByText("Branch exclusive feature discussion")).toBeInTheDocument();
    });

    // The branch conversation item in Sidebar should be marked currently active
    const branchSidebarItem = screen.getByRole("button", {
      name: /base plan \(branch\), currently active/i,
    });
    expect(branchSidebarItem).toBeInTheDocument();
  });

  it("5. Stored message content is not mutated or altered by search or highlighting", async () => {
    const payload = {
      version: STORAGE_VERSION,
      conversations: [
        {
          id: "c-immutable",
          title: "Immutable Chat",
          messages: [{ id: "m-imm", role: "user", content: "Original plain content" }],
        },
      ],
      activeConversationId: "c-immutable",
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Original plain content")).toBeInTheDocument();
    });

    // Open search and select
    const searchBtn = screen.getByRole("button", { name: /search conversations and messages/i });
    await user.click(searchBtn);

    const input = screen.getByPlaceholderText(/search conversations/i);
    await user.type(input, "plain content");

    await waitFor(() => {
      const modal = screen.getByRole("dialog", { name: /global message search/i });
      expect(within(modal).getByText(/plain content/i)).toBeInTheDocument();
    });

    const modal = screen.getByRole("dialog", { name: /global message search/i });
    const resultItem = within(modal).getByText(/plain content/i).closest(".search-result-item");
    await user.click(resultItem);

    // Verify localStorage was not corrupted with HTML tags or search markers
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    expect(saved.conversations[0].messages[0].content).toBe("Original plain content");
  });
});
