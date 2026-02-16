import React from "react";
import { Plus, MessageSquare, Trash2, PanelLeftClose, PanelLeft } from "lucide-react";

export default function Sidebar({ open, conversations, activeId, onToggle, onNew, onSelect, onDelete }) {
  if (!open) {
    return (
      <div className="sidebar-closed">
        <button className="icon-btn" onClick={onToggle} title="Open sidebar">
          <PanelLeft size={20} />
        </button>
      </div>
    );
  }

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
        {conversations.map((conv) => (
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
        ))}

        {conversations.length === 0 && (
          <div className="empty-state">
            <p>No conversations yet</p>
          </div>
        )}
      </div>

      <div className="sidebar-footer">
        <div className="model-info">
          <div className="model-badge">FramerAI v1.0</div>
        </div>
      </div>
    </aside>
  );
}
