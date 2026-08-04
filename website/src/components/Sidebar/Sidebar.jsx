import React from "react";
import { Plus, MessageSquare, Trash2, PanelLeftClose, PanelLeft } from "lucide-react";

export default function Sidebar({
  open,
  conversations,
  activeId,
  loadingConversations,
  onToggle,
  onNew,
  onSelect,
  onDelete,
}) {
  if (!open) {
    return (
      <div className="sidebar-closed">
        <button
          className="icon-btn"
          onClick={onToggle}
          title="Open sidebar"
          aria-label="Open sidebar"
          aria-expanded={false}
          aria-controls="sidebar"
        >
          <PanelLeft size={20} aria-hidden="true" />
        </button>
      </div>
    );
  }

  const renderList = () => {
    // Loading state — show skeleton items
    if (loadingConversations) {
      return (
        <div className="conversation-list-loading">
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
          <MessageSquare size={24} className="empty-state-icon" aria-hidden="true" />
          <p>No conversations yet</p>
          <span>Click &ldquo;New Chat&rdquo; to get started</span>
        </div>
      );
    }

    // Normal list
    return (
      <ul className="conversation-items">
        {conversations.map((conv) => {
          const title = conv.title || "New Chat";
          const isActive = conv.id === activeId;
          return (
            <li key={conv.id} className={`conversation-item ${isActive ? "active" : ""}`}>
              <button
                type="button"
                className="conversation-select"
                onClick={() => onSelect(conv.id)}
                aria-current={isActive ? "true" : undefined}
              >
                <MessageSquare size={16} aria-hidden="true" />
                <span className="conversation-title">{title}</span>
              </button>
              <button
                type="button"
                className="delete-btn"
                onClick={() => onDelete(conv.id)}
                title="Delete"
                aria-label={`Delete conversation: ${title}`}
              >
                <Trash2 size={14} aria-hidden="true" />
              </button>
            </li>
          );
        })}
      </ul>
    );
  };

  return (
    <aside className="sidebar" id="sidebar" aria-label="Conversations">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <img src="/logo.svg" alt="" className="sidebar-logo" />
          <span className="sidebar-title">FramerAI</span>
        </div>
        <button
          className="icon-btn"
          onClick={onToggle}
          title="Close sidebar"
          aria-label="Close sidebar"
          aria-expanded={true}
          aria-controls="sidebar"
        >
          <PanelLeftClose size={20} aria-hidden="true" />
        </button>
      </div>

      <button className="new-chat-btn" onClick={onNew}>
        <Plus size={18} aria-hidden="true" />
        <span>New Chat</span>
      </button>

      <nav className="conversation-list" aria-label="Conversation history" aria-busy={loadingConversations}>
        {loadingConversations && <p className="sr-only">Loading conversations</p>}
        {renderList()}
      </nav>

      <div className="sidebar-footer">
        <div className="model-info">
          <div className="model-badge">FramerAI v1.0</div>
        </div>
      </div>
    </aside>
  );
}
