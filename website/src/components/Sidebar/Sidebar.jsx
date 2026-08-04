import React from "react";
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
}) {
  if (!open) {
    return (
      <div className="sidebar-closed">
        <button className="icon-btn" onClick={onToggle} title="Open sidebar">
          <PanelLeft size={20} />
        </button>
      </div>
    );
  }

  const renderList = () => {
    // Loading state — show skeleton items
    if (loadingConversations) {
      return (
        <div className="conversation-list-loading" aria-label="Loading conversations">
          {[1, 2, 3].map((n) => (
            <div key={n} className="conversation-skeleton">
              <div className="skeleton-icon" />
              <div className="skeleton-line" style={{ width: `${60 + n * 10}%` }} />
            </div>
          ))}
        </div>
      );
    }

    // Empty state — truly no conversations
    if (conversations.length === 0) {
      return (
        <div className="empty-state">
          <MessageSquare size={24} className="empty-state-icon" />
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
        onClick={() => onSelect(conv.id)}
      >
        <MessageSquare size={16} />
        <span className="conversation-title">{conv.title || "New Chat"}</span>
        <button
          className="delete-btn"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(conv.id);
          }}
          title="Delete"
        >
          <Trash2 size={14} />
        </button>
      </div>
    ));
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <img src="/logo.svg" alt="FramerAI" className="sidebar-logo" />
          <span className="sidebar-title">FramerAI</span>
        </div>
        <button className="icon-btn" onClick={onToggle} title="Close sidebar">
          <PanelLeftClose size={20} />
        </button>
      </div>

      <button className="new-chat-btn" onClick={onNew}>
        <Plus size={18} />
        <span>New Chat</span>
      </button>

      <div className="conversation-list">
        {renderList()}
      </div>

      <div className="sidebar-footer">
        <div className="model-info">
          <div className="model-badge">FramerAI v1.0</div>
        </div>
        <button className="icon-btn" onClick={onOpenSettings} title="Generation settings">
          <Settings size={18} />
        </button>
      </div>
    </aside>
  );
}
