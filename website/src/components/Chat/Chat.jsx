import React, { useState, useRef, useEffect } from "react";
import { Send, PanelLeft, Image, Video, Code, Loader2 } from "lucide-react";
import MessageBubble from "./MessageBubble";

export default function Chat({ messages, loading, streaming, sidebarOpen, onSend, onToggleSidebar }) {
  const [input, setInput] = useState("");
  const [messageType, setMessageType] = useState("text");
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading || streaming) return;
    onSend(input, messageType);
    setInput("");
    setMessageType("text");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInput = (e) => {
    setInput(e.target.value);
    // Auto-resize textarea
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  };

  return (
    <main className={`chat-container ${sidebarOpen ? "" : "full-width"}`}>
      {/* Header */}
      <header className="chat-header">
        {!sidebarOpen && (
          <button className="icon-btn" onClick={onToggleSidebar}>
            <PanelLeft size={20} />
          </button>
        )}
        <h1 className="chat-title">FramerAI</h1>
        <div className="chat-subtitle">Multimodal AI Assistant</div>
      </header>

      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-screen">
            <img src="/logo.svg" alt="FramerAI" className="welcome-logo" />
            <h2>Welcome to FramerAI</h2>
            <p>A multimodal AI that can generate text, code, images, and videos.</p>
            <div className="suggestions">
              <button className="suggestion" onClick={() => onSend("Hello! What can you do?")}>
                What can you do?
              </button>
              <button className="suggestion" onClick={() => onSend("Write a fibonacci function in Python")}>
                Write a fibonacci function
              </button>
              <button className="suggestion" onClick={() => onSend("Generate an image of a sunset over mountains")}>
                Generate a sunset image
              </button>
              <button className="suggestion" onClick={() => onSend("Create a video of ocean waves")}>
                Create an ocean video
              </button>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={msg.id || i} message={msg} isStreaming={streaming && i === messages.length - 1} />
        ))}

        {(loading || streaming) && messages[messages.length - 1]?.role !== "assistant" && (
          <div className="message assistant">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form className="input-container" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <div className="input-modes">
            <button
              type="button"
              className={`mode-btn ${messageType === "text" ? "active" : ""}`}
              onClick={() => setMessageType("text")}
              title="Text"
            >
              <MessageIcon size={16} />
            </button>
            <button
              type="button"
              className={`mode-btn ${messageType === "code" ? "active" : ""}`}
              onClick={() => setMessageType("code")}
              title="Code"
            >
              <Code size={16} />
            </button>
            <button
              type="button"
              className={`mode-btn ${messageType === "image" ? "active" : ""}`}
              onClick={() => setMessageType("image")}
              title="Image"
            >
              <Image size={16} />
            </button>
            <button
              type="button"
              className={`mode-btn ${messageType === "video" ? "active" : ""}`}
              onClick={() => setMessageType("video")}
              title="Video"
            >
              <Video size={16} />
            </button>
          </div>
          <textarea
            ref={textareaRef}
            className="chat-input"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Message FramerAI..."
            rows={1}
            disabled={loading || streaming}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={!input.trim() || loading || streaming}
          >
            {loading || streaming ? <Loader2 size={20} className="spin" /> : <Send size={20} />}
          </button>
        </div>
        <p className="input-hint">
          FramerAI is an open-source multimodal model. Train it with <code>python build.py --mode all</code>
        </p>
      </form>
    </main>
  );
}

function MessageIcon({ size }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}
