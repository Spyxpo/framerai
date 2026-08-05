import React, { useRef, useEffect } from "react";
import { Plus, MessageSquare, Trash2, PanelLeftClose, PanelLeft, Settings } from "lucide-react";

export default function Sidebar({
  open,
  conversations,
  activeId,
  loadingConversations,
  onToggle,
  onNew,
  onSelect,
  onDelete,
  onOpenSettings,
  onFocusChat,
  onFocusChatSettings,
  focusRef,
  footerSettingsFocusRef,
}) {
  const listRef = useRef(null);
  const newChatBtnRef = useRef(null);
  const toggleBtnRef = useRef(null);
  const footerSettingsBtnRef = useRef(null);

  // Expose focus fns to App
  useEffect(() => {
    if (focusRef) {
      focusRef.current = () => newChatBtnRef.current?.focus();
    }
    if (footerSettingsFocusRef) {
      footerSettingsFocusRef.current = () => footerSettingsBtnRef.current?.focus();
    }
  }, [focusRef, footerSettingsFocusRef]);

  if (!open) {
    return (
      <div className="sidebar-closed">
        <button className="icon-btn" onClick={onToggle} aria-label="Open sidebar">
          <PanelLeft size={20} aria-hidden="true" />
        </button>
      </div>
    );
  }
  const getItems = () =>
    listRef.current
      ? Array.from(listRef.current.querySelectorAll(".conversation-item"))
      : [];

  const handleItemKeyDown = (e, convId) => {
    const items = getItems();
    const currentIndex = items.indexOf(e.currentTarget);

    switch (e.key) {
      case "Enter":
      case " ":
        e.preventDefault();
        e.stopPropagation();
        onSelect(convId);
        break;
      case "Delete":
      case "Backspace":
        e.preventDefault();
        onDelete(convId);
        break;
      case "ArrowRight":
        e.preventDefault();
        // Focus the delete button inside this row
        e.currentTarget.querySelector(".delete-btn")?.focus();
        break;
      case "ArrowDown":
        e.preventDefault();
        if (currentIndex === items.length - 1) {
          footerSettingsBtnRef.current?.focus();
        } else {
          (items[currentIndex + 1] || items[0]).focus();
        }
        break;
      case "ArrowUp":
        e.preventDefault();
        if (currentIndex === 0) {
          newChatBtnRef.current?.focus();
        } else {
          items[currentIndex - 1].focus();
        }
        break;
      case "Home":
        e.preventDefault();
        items[0]?.focus();
        break;
      case "End":
        e.preventDefault();
        items[items.length - 1]?.focus();
        break;
      default:
        break;
    }
  };

  const handleDeleteKeyDown = (e, convId, itemEl) => {
    // Stop bubbling so the parent div's onKeyDown doesn't also fire
    e.stopPropagation();
    if (e.key === "ArrowLeft" || e.key === "Escape") {
      e.preventDefault();
      itemEl?.focus();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      onFocusChat?.();
    }
  };

  const renderList = () => {
    // Loading state — show skeleton items
    // aria-live="polite" announces loading state to screen readers (improvement b)
    if (loadingConversations) {
      return (
        <div className="conversation-list-loading" aria-label="Loading conversations" aria-busy="true" aria-live="polite">
          {[1, 2, 3].map((n) => (
            <div key={n} className="conversation-skeleton" aria-hidden="true">
              <div className="skeleton-icon" />
              <div className="skeleton-line" style={{ width: `${60 + n * 10}%` }} />
            </div>
          ))}
        </div>
      );
    }

    // Empty state — truly no conversations
    // role="status" + aria-live="polite" lets screen readers announce it (improvement c)
    if (conversations.length === 0) {
      return (
        <div className="empty-state" role="status" aria-live="polite" aria-label="No conversations yet">
          <MessageSquare size={24} className="empty-state-icon" aria-hidden="true" />
          <p>No conversations yet</p>
          <span>Click &ldquo;New Chat&rdquo; to get started</span>
        </div>
      );
    }

    // Normal list
    return conversations.map((conv) => (
      <div
        key={conv.id}
        className={`conversation-item ${conv.id === activeId ? "active" : ""}`}
        role="button"
        tabIndex={0}
        aria-current={conv.id === activeId ? "true" : undefined}
        aria-label={`${conv.title || "New Chat"}${conv.id === activeId ? ", currently active" : ""}`}
        onClick={() => onSelect(conv.id)}
        onKeyDown={(e) => handleItemKeyDown(e, conv.id)}
      >
        <MessageSquare size={16} aria-hidden="true" />
        <span className="conversation-title">{conv.title || "New Chat"}</span>
        <button
          className="delete-btn"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(conv.id);
          }}
          onKeyDown={(e) => handleDeleteKeyDown(e, conv.id, e.currentTarget.closest(".conversation-item"))}
          aria-label={`Delete conversation: ${conv.title || "New Chat"}`}
        >
          <Trash2 size={14} aria-hidden="true" />
        </button>
      </div>
    ));
  };

  return (
    <aside className="sidebar" aria-label="Conversations sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <img src="/logo.svg" alt="FramerAI logo" className="sidebar-logo" />
          <span className="sidebar-title" aria-hidden="true">FramerAI</span>
        </div>
        <button
          className="icon-btn"
          ref={toggleBtnRef}
          onClick={onToggle}
          aria-label="Close sidebar"
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              newChatBtnRef.current?.focus();
            } else if (e.key === "ArrowRight") {
              e.preventDefault();
              onFocusChatSettings?.();
            }
          }}
        >
          <PanelLeftClose size={20} aria-hidden="true" />
        </button>
      </div>

      <button
        className="new-chat-btn"
        ref={newChatBtnRef}
        onClick={onNew}
        aria-label="Create new conversation"
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            const first = listRef.current?.querySelector(".conversation-item");
            if (first) first.focus();
            else footerSettingsBtnRef.current?.focus();
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            toggleBtnRef.current?.focus();
          } else if (e.key === "ArrowRight") {
            e.preventDefault();
            onFocusChat?.();
          }
        }}
      >
        <Plus size={18} aria-hidden="true" />
        <span>New Chat</span>
      </button>

      <nav ref={listRef} className="conversation-list" aria-label="Your conversations">
        {renderList()}
      </nav>

      <div className="sidebar-footer">
        <div className="model-info">
          <div className="model-badge">FramerAI v1.0</div>
        </div>
        <button
          className="icon-btn"
          ref={footerSettingsBtnRef}
          onClick={onOpenSettings}
          aria-label="Generation settings"
          title="Generation settings"
          onKeyDown={(e) => {
            if (e.key === "ArrowUp") {
              e.preventDefault();
              const items = getItems();
              if (items.length > 0) {
                items[items.length - 1].focus();
              } else {
                newChatBtnRef.current?.focus();
              }
            } else if (e.key === "ArrowRight") {
              e.preventDefault();
              onFocusChatSettings?.();
            }
          }}
        >
          <Settings size={18} aria-hidden="true" />
        </button>
      </div>
    </aside>
  );
}
