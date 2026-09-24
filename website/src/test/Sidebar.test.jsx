import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Sidebar from "../components/Sidebar/Sidebar";

// ── Helpers ────────────────────────────────────────────────────────────────

function makeConv(id, title = "Test Chat") {
  return { id, title };
}

/** Minimal props to render <Sidebar> in open state without crashes */
function sidebarProps(overrides = {}) {
  return {
    open: true,
    conversations: [],
    activeId: null,
    loadingConversations: false,
    onToggle: vi.fn(),
    onNew: vi.fn(),
    onSelect: vi.fn(),
    onDelete: vi.fn(),
    onClearAll: vi.fn(),
    onOpenSettings: vi.fn(),
    onFocusChat: vi.fn(),
    onFocusChatSettings: vi.fn(),
    focusRef: { current: null },
    footerSettingsFocusRef: { current: null },
    ...overrides,
  };
}

// ── Sidebar rendering ──────────────────────────────────────────────────────

describe("Sidebar — rendering", () => {
  it("renders the New Chat button", () => {
    render(<Sidebar {...sidebarProps()} />);
    expect(screen.getByRole("button", { name: /create new conversation/i })).toBeInTheDocument();
  });

  it("renders empty state when there are no conversations", () => {
    render(<Sidebar {...sidebarProps()} />);
    expect(screen.getByText(/no conversations yet/i)).toBeInTheDocument();
  });

  it("renders a list of conversations", () => {
    const conversations = [makeConv("1", "First Chat"), makeConv("2", "Second Chat")];
    render(<Sidebar {...sidebarProps({ conversations })} />);
    expect(screen.getByText("First Chat")).toBeInTheDocument();
    expect(screen.getByText("Second Chat")).toBeInTheDocument();
  });

  it("marks the active conversation with aria-current", () => {
    const conversations = [makeConv("1", "Active Chat"), makeConv("2", "Other Chat")];
    render(<Sidebar {...sidebarProps({ conversations, activeId: "1" })} />);
    const activeItem = screen.getByRole("button", { name: /active chat.*currently active/i });
    expect(activeItem).toHaveAttribute("aria-current", "true");
  });

  it("does not mark non-active conversations with aria-current", () => {
    const conversations = [makeConv("1", "Chat A"), makeConv("2", "Chat B")];
    render(<Sidebar {...sidebarProps({ conversations, activeId: "1" })} />);
    const otherItem = screen.getByRole("button", { name: /^chat b$/i });
    expect(otherItem).not.toHaveAttribute("aria-current");
  });

  it("shows loading skeletons when loadingConversations is true", () => {
    render(<Sidebar {...sidebarProps({ loadingConversations: true })} />);
    expect(screen.getByLabelText(/loading conversations/i)).toBeInTheDocument();
  });

  it("renders closed toggle button when open is false", () => {
    render(<Sidebar {...sidebarProps({ open: false })} />);
    expect(screen.getByRole("button", { name: /open sidebar/i })).toBeInTheDocument();
  });

  it("renders settings button in footer", () => {
    render(<Sidebar {...sidebarProps()} />);
    expect(screen.getByRole("button", { name: /generation settings/i })).toBeInTheDocument();
  });

  it("renders close sidebar button in header", () => {
    render(<Sidebar {...sidebarProps()} />);
    expect(screen.getByRole("button", { name: /close sidebar/i })).toBeInTheDocument();
  });
});

// ── Sidebar — conversation switching ──────────────────────────────────────

describe("Sidebar — conversation switching", () => {
  let user;
  beforeEach(() => {
    user = userEvent.setup();
  });

  it("calls onSelect with the conversation id when clicked", async () => {
    const onSelect = vi.fn();
    const conversations = [makeConv("abc", "My Chat")];
    render(<Sidebar {...sidebarProps({ conversations, onSelect })} />);
    // The row div is role=button; delete btn inside it also has role=button —
    // use getAllByRole and take the first (the row itself)
    const [row] = screen.getAllByRole("button", { name: /my chat/i });
    await user.click(row);
    expect(onSelect).toHaveBeenCalledWith("abc");
  });

  it("calls onSelect when Enter is pressed on a conversation item", async () => {
    const onSelect = vi.fn();
    const conversations = [makeConv("abc", "My Chat")];
    render(<Sidebar {...sidebarProps({ conversations, onSelect })} />);
    const [row] = screen.getAllByRole("button", { name: /my chat/i });
    row.focus();
    await user.keyboard("{Enter}");
    expect(onSelect).toHaveBeenCalledWith("abc");
  });

  it("calls onSelect when Space is pressed on a conversation item", async () => {
    const onSelect = vi.fn();
    const conversations = [makeConv("abc", "My Chat")];
    render(<Sidebar {...sidebarProps({ conversations, onSelect })} />);
    const [row] = screen.getAllByRole("button", { name: /my chat/i });
    row.focus();
    await user.keyboard(" ");
    expect(onSelect).toHaveBeenCalledWith("abc");
  });

  it("switches active conversation when a different item is clicked", async () => {
    const onSelect = vi.fn();
    const conversations = [makeConv("1", "Chat One"), makeConv("2", "Chat Two")];
    render(<Sidebar {...sidebarProps({ conversations, activeId: "1", onSelect })} />);
    const [row] = screen.getAllByRole("button", { name: /chat two/i });
    await user.click(row);
    expect(onSelect).toHaveBeenCalledWith("2");
  });

  it("calls onNew when New Chat button is clicked", async () => {
    const onNew = vi.fn();
    render(<Sidebar {...sidebarProps({ onNew })} />);
    await user.click(screen.getByRole("button", { name: /create new conversation/i }));
    expect(onNew).toHaveBeenCalledOnce();
  });

  it("calls onToggle when close sidebar button is clicked", async () => {
    const onToggle = vi.fn();
    render(<Sidebar {...sidebarProps({ onToggle })} />);
    await user.click(screen.getByRole("button", { name: /close sidebar/i }));
    expect(onToggle).toHaveBeenCalledOnce();
  });

  it("calls onToggle when open sidebar button is clicked (closed state)", async () => {
    const onToggle = vi.fn();
    render(<Sidebar {...sidebarProps({ open: false, onToggle })} />);
    await user.click(screen.getByRole("button", { name: /open sidebar/i }));
    expect(onToggle).toHaveBeenCalledOnce();
  });

  it("calls onOpenSettings when footer settings button is clicked", async () => {
    const onOpenSettings = vi.fn();
    render(<Sidebar {...sidebarProps({ onOpenSettings })} />);
    await user.click(screen.getByRole("button", { name: /generation settings/i }));
    expect(onOpenSettings).toHaveBeenCalledOnce();
  });
});

// ── Sidebar — delete ───────────────────────────────────────────────────────

describe("Sidebar — delete", () => {
  let user;
  beforeEach(() => {
    user = userEvent.setup();
  });

  it("shows delete button on hover (focus-within)", async () => {
    const conversations = [makeConv("1", "Deletable")];
    render(<Sidebar {...sidebarProps({ conversations })} />);
    const [row] = screen.getAllByRole("button", { name: /deletable/i });
    row.focus();
    expect(screen.getByRole("button", { name: /delete conversation: deletable/i })).toBeInTheDocument();
  });

  it("calls onDelete when delete button is clicked", async () => {
    const onDelete = vi.fn();
    const conversations = [makeConv("1", "Deletable")];
    render(<Sidebar {...sidebarProps({ conversations, onDelete })} />);
    const [row] = screen.getAllByRole("button", { name: /deletable/i });
    row.focus();
    await user.click(screen.getByRole("button", { name: /delete conversation: deletable/i }));
    expect(onDelete).toHaveBeenCalledWith("1");
  });

  it("calls onDelete when Delete key is pressed on a conversation item", async () => {
    const onDelete = vi.fn();
    const conversations = [makeConv("1", "Deletable")];
    render(<Sidebar {...sidebarProps({ conversations, onDelete })} />);
    const [row] = screen.getAllByRole("button", { name: /deletable/i });
    row.focus();
    await user.keyboard("{Delete}");
    expect(onDelete).toHaveBeenCalledWith("1");
  });

  it("calls onDelete when Backspace key is pressed on a conversation item", async () => {
    const onDelete = vi.fn();
    const conversations = [makeConv("1", "Deletable")];
    render(<Sidebar {...sidebarProps({ conversations, onDelete })} />);
    const [row] = screen.getAllByRole("button", { name: /deletable/i });
    row.focus();
    await user.keyboard("{Backspace}");
    expect(onDelete).toHaveBeenCalledWith("1");
  });

  it("does not call onSelect when delete button is clicked", async () => {
    const onSelect = vi.fn();
    const onDelete = vi.fn();
    const conversations = [makeConv("1", "Deletable")];
    render(<Sidebar {...sidebarProps({ conversations, onSelect, onDelete })} />);
    const [row] = screen.getAllByRole("button", { name: /deletable/i });
    row.focus();
    await user.click(screen.getByRole("button", { name: /delete conversation: deletable/i }));
    expect(onSelect).not.toHaveBeenCalled();
  });
});

// ── Additional Rendering & Edge Cases ─────────────────────────────────────

describe("Sidebar — detailed rendering & states", () => {
  it("renders FramerAI brand logo and title", () => {
    render(<Sidebar {...sidebarProps()} />);
    expect(screen.getByAltText("FramerAI logo")).toBeInTheDocument();
    expect(screen.getByText("FramerAI")).toBeInTheDocument();
  });

  it("renders model info badge in footer", () => {
    render(<Sidebar {...sidebarProps()} />);
    expect(screen.getByText("FramerAI v1.0")).toBeInTheDocument();
  });

  it("falls back to 'New Chat' title and aria-label when conversation title is empty", () => {
    const conversations = [makeConv("c1", ""), makeConv("c2", null)];
    render(<Sidebar {...sidebarProps({ conversations })} />);
    const items = screen.getAllByText("New Chat");
    // "New Chat" appears on the new chat button plus both conversation title spans
    expect(items.length).toBeGreaterThanOrEqual(2);
    const deleteButtons = screen.getAllByRole("button", { name: /^delete conversation: new chat$/i });
    expect(deleteButtons).toHaveLength(2);
  });

  it("applies the active CSS class to the selected conversation item", () => {
    const conversations = [makeConv("1", "Chat 1"), makeConv("2", "Chat 2")];
    render(<Sidebar {...sidebarProps({ conversations, activeId: "1" })} />);
    const [activeRow] = screen.getAllByRole("button", { name: /chat 1.*currently active/i });
    const [otherRow] = screen.getAllByRole("button", { name: /^chat 2$/i });
    expect(activeRow).toHaveClass("active");
    expect(otherRow).not.toHaveClass("active");
  });

  it("does not render conversation list or empty state when loadingConversations is true", () => {
    const conversations = [makeConv("1", "Chat 1")];
    render(<Sidebar {...sidebarProps({ loadingConversations: true, conversations })} />);
    expect(screen.getByLabelText(/loading conversations/i)).toBeInTheDocument();
    expect(screen.queryByText("Chat 1")).not.toBeInTheDocument();
    expect(screen.queryByText(/no conversations yet/i)).not.toBeInTheDocument();
  });

  it("does not render navigation or header controls when closed", () => {
    const conversations = [makeConv("1", "Chat 1")];
    render(<Sidebar {...sidebarProps({ open: false, conversations })} />);
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /create new conversation/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /close sidebar/i })).not.toBeInTheDocument();
    expect(screen.queryByText("Chat 1")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /open sidebar/i })).toBeInTheDocument();
  });

  it("does not render the clear conversations button when there are no conversations", () => {
    render(<Sidebar {...sidebarProps({ conversations: [] })} />);
    expect(screen.queryByRole("button", { name: /clear all conversations/i })).not.toBeInTheDocument();
  });

  it("renders the clear conversations button when conversations exist", () => {
    const conversations = [makeConv("1", "Chat 1")];
    render(<Sidebar {...sidebarProps({ conversations })} />);
    expect(screen.getByRole("button", { name: /clear all conversations/i })).toBeInTheDocument();
  });
});

// ── Clear All Conversations ───────────────────────────────────────────────

describe("Sidebar — clear all conversations", () => {
  let user;
  beforeEach(() => {
    user = userEvent.setup();
  });

  it("prompts for confirmation and calls onClearAll when confirmed", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const onClearAll = vi.fn();
    const conversations = [makeConv("1", "Chat 1")];
    render(<Sidebar {...sidebarProps({ conversations, onClearAll })} />);

    const clearBtn = screen.getByRole("button", { name: /clear all conversations/i });
    await user.click(clearBtn);

    expect(confirmSpy).toHaveBeenCalledWith("Are you sure you want to clear all conversation history?");
    expect(onClearAll).toHaveBeenCalledOnce();
    confirmSpy.mockRestore();
  });

  it("does not call onClearAll when user cancels the confirmation", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const onClearAll = vi.fn();
    const conversations = [makeConv("1", "Chat 1")];
    render(<Sidebar {...sidebarProps({ conversations, onClearAll })} />);

    const clearBtn = screen.getByRole("button", { name: /clear all conversations/i });
    await user.click(clearBtn);

    expect(confirmSpy).toHaveBeenCalled();
    expect(onClearAll).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it("calls onClearAll even if window.confirm is undefined", async () => {
    const originalConfirm = window.confirm;
    delete window.confirm;
    const onClearAll = vi.fn();
    const conversations = [makeConv("1", "Chat 1")];
    render(<Sidebar {...sidebarProps({ conversations, onClearAll })} />);

    const clearBtn = screen.getByRole("button", { name: /clear all conversations/i });
    await user.click(clearBtn);

    expect(onClearAll).toHaveBeenCalledOnce();
    window.confirm = originalConfirm;
  });

  it("does not throw an error if onClearAll is not provided", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const conversations = [makeConv("1", "Chat 1")];
    render(<Sidebar {...sidebarProps({ conversations, onClearAll: undefined })} />);

    const clearBtn = screen.getByRole("button", { name: /clear all conversations/i });
    await expect(user.click(clearBtn)).resolves.not.toThrow();
    confirmSpy.mockRestore();
  });
});

// ── Keyboard Navigation ───────────────────────────────────────────────────

describe("Sidebar — keyboard navigation", () => {
  let user;
  beforeEach(() => {
    user = userEvent.setup();
  });

  it("moves focus between conversation items with ArrowDown and ArrowUp", async () => {
    const conversations = [
      makeConv("1", "First"),
      makeConv("2", "Second"),
      makeConv("3", "Third"),
    ];
    render(<Sidebar {...sidebarProps({ conversations })} />);

    const [item1] = screen.getAllByRole("button", { name: /^first$/i });
    const [item2] = screen.getAllByRole("button", { name: /^second$/i });
    const [item3] = screen.getAllByRole("button", { name: /^third$/i });

    item1.focus();
    expect(document.activeElement).toBe(item1);

    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(item2);

    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(item3);

    await user.keyboard("{ArrowUp}");
    expect(document.activeElement).toBe(item2);
  });

  it("moves focus from the last conversation item to footer settings button on ArrowDown", async () => {
    const conversations = [makeConv("1", "Single Item")];
    render(<Sidebar {...sidebarProps({ conversations })} />);

    const [item] = screen.getAllByRole("button", { name: /single item/i });
    const settingsBtn = screen.getByRole("button", { name: /generation settings/i });

    item.focus();
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(settingsBtn);
  });

  it("moves focus from the first conversation item to New Chat button on ArrowUp", async () => {
    const conversations = [makeConv("1", "First Item"), makeConv("2", "Second Item")];
    render(<Sidebar {...sidebarProps({ conversations })} />);

    const [item1] = screen.getAllByRole("button", { name: /first item/i });
    const newChatBtn = screen.getByRole("button", { name: /create new conversation/i });

    item1.focus();
    await user.keyboard("{ArrowUp}");
    expect(document.activeElement).toBe(newChatBtn);
  });

  it("navigates to first item on Home and last item on End", async () => {
    const conversations = [
      makeConv("1", "First"),
      makeConv("2", "Middle"),
      makeConv("3", "Last"),
    ];
    render(<Sidebar {...sidebarProps({ conversations })} />);

    const [item1] = screen.getAllByRole("button", { name: /^first$/i });
    const [item2] = screen.getAllByRole("button", { name: /^middle$/i });
    const [item3] = screen.getAllByRole("button", { name: /^last$/i });

    item2.focus();
    await user.keyboard("{Home}");
    expect(document.activeElement).toBe(item1);

    await user.keyboard("{End}");
    expect(document.activeElement).toBe(item3);
  });

  it("moves focus to the row delete button on ArrowRight from a conversation item", async () => {
    const conversations = [makeConv("1", "Focus Delete")];
    render(<Sidebar {...sidebarProps({ conversations })} />);

    const [item] = screen.getAllByRole("button", { name: /focus delete/i });
    const deleteBtn = screen.getByRole("button", { name: /delete conversation: focus delete/i });

    item.focus();
    await user.keyboard("{ArrowRight}");
    expect(document.activeElement).toBe(deleteBtn);
  });

  it("returns focus from delete button to conversation row on ArrowLeft or Escape", async () => {
    const conversations = [makeConv("1", "Deletable")];
    render(<Sidebar {...sidebarProps({ conversations })} />);

    const [item] = screen.getAllByRole("button", { name: /deletable/i });
    const deleteBtn = screen.getByRole("button", { name: /delete conversation: deletable/i });

    deleteBtn.focus();
    await user.keyboard("{ArrowLeft}");
    expect(document.activeElement).toBe(item);

    deleteBtn.focus();
    await user.keyboard("{Escape}");
    expect(document.activeElement).toBe(item);
  });

  it("calls onFocusChat on ArrowRight from delete button", async () => {
    const onFocusChat = vi.fn();
    const conversations = [makeConv("1", "Deletable")];
    render(<Sidebar {...sidebarProps({ conversations, onFocusChat })} />);

    const deleteBtn = screen.getByRole("button", { name: /delete conversation: deletable/i });
    deleteBtn.focus();
    await user.keyboard("{ArrowRight}");
    expect(onFocusChat).toHaveBeenCalledOnce();
  });

  it("navigates from header toggle button via keyboard", async () => {
    const onFocusChatSettings = vi.fn();
    render(<Sidebar {...sidebarProps({ onFocusChatSettings })} />);

    const toggleBtn = screen.getByRole("button", { name: /close sidebar/i });
    const newChatBtn = screen.getByRole("button", { name: /create new conversation/i });

    toggleBtn.focus();
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(newChatBtn);

    toggleBtn.focus();
    await user.keyboard("{ArrowRight}");
    expect(onFocusChatSettings).toHaveBeenCalledOnce();
  });

  it("navigates from New Chat button via keyboard", async () => {
    const onFocusChat = vi.fn();
    const conversations = [makeConv("1", "First Conv")];
    render(<Sidebar {...sidebarProps({ conversations, onFocusChat })} />);

    const newChatBtn = screen.getByRole("button", { name: /create new conversation/i });
    const toggleBtn = screen.getByRole("button", { name: /close sidebar/i });
    const [firstItem] = screen.getAllByRole("button", { name: /first conv/i });

    newChatBtn.focus();
    await user.keyboard("{ArrowUp}");
    expect(document.activeElement).toBe(toggleBtn);

    newChatBtn.focus();
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(firstItem);

    newChatBtn.focus();
    await user.keyboard("{ArrowRight}");
    expect(onFocusChat).toHaveBeenCalledOnce();
  });

  it("moves focus from New Chat button to footer settings button when no conversations exist", async () => {
    render(<Sidebar {...sidebarProps({ conversations: [] })} />);

    const newChatBtn = screen.getByRole("button", { name: /create new conversation/i });
    const settingsBtn = screen.getByRole("button", { name: /generation settings/i });

    newChatBtn.focus();
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(settingsBtn);
  });

  it("navigates from footer settings button via keyboard", async () => {
    const onFocusChatSettings = vi.fn();
    const conversations = [makeConv("1", "Chat 1"), makeConv("2", "Chat 2")];
    render(<Sidebar {...sidebarProps({ conversations, onFocusChatSettings })} />);

    const settingsBtn = screen.getByRole("button", { name: /generation settings/i });
    const [lastItem] = screen.getAllByRole("button", { name: /^chat 2$/i });

    settingsBtn.focus();
    await user.keyboard("{ArrowUp}");
    expect(document.activeElement).toBe(lastItem);

    settingsBtn.focus();
    await user.keyboard("{ArrowRight}");
    expect(onFocusChatSettings).toHaveBeenCalledOnce();
  });

  it("moves focus from footer settings to New Chat button on ArrowUp when no conversations exist", async () => {
    render(<Sidebar {...sidebarProps({ conversations: [] })} />);

    const settingsBtn = screen.getByRole("button", { name: /generation settings/i });
    const newChatBtn = screen.getByRole("button", { name: /create new conversation/i });

    settingsBtn.focus();
    await user.keyboard("{ArrowUp}");
    expect(document.activeElement).toBe(newChatBtn);
  });
});

// ── Focus References ──────────────────────────────────────────────────────

describe("Sidebar — focus references", () => {
  it("populates focusRef and focuses New Chat button when invoked", () => {
    const focusRef = { current: null };
    render(<Sidebar {...sidebarProps({ focusRef })} />);

    expect(typeof focusRef.current).toBe("function");
    focusRef.current();

    const newChatBtn = screen.getByRole("button", { name: /create new conversation/i });
    expect(document.activeElement).toBe(newChatBtn);
  });

  it("populates footerSettingsFocusRef and focuses settings button when invoked", () => {
    const footerSettingsFocusRef = { current: null };
    render(<Sidebar {...sidebarProps({ footerSettingsFocusRef })} />);

    expect(typeof footerSettingsFocusRef.current).toBe("function");
    footerSettingsFocusRef.current();

    const settingsBtn = screen.getByRole("button", { name: /generation settings/i });
    expect(document.activeElement).toBe(settingsBtn);
  });

  it("gracefully mounts when focus refs are null or undefined", () => {
    expect(() => {
      render(<Sidebar {...sidebarProps({ focusRef: null, footerSettingsFocusRef: null })} />);
    }).not.toThrow();
  });
});
