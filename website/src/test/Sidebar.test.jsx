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
