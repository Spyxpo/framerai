import React from "react";
import { User, Bot, Copy, Check, AlertCircle, RefreshCw } from "lucide-react";
import CodeBlock from "../CodeBlock/CodeBlock";

export default function MessageBubble({ message, isStreaming, onRetry }) {
  const [copied, setCopied] = React.useState(false);
  const isUser = message.role === "user";
  const isError = message.type === "error";

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderContent = (content) => {
    // Empty content during streaming = typing indicator
    if (!content) {
      return (
        <div className="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      );
    }

    // Error messages get their own renderer
    if (isError) {
      return (
        <div className="error-message-content">
          <AlertCircle size={15} className="error-icon" />
          <span>{content}</span>
        </div>
      );
    }

    // Parse code blocks
    const parts = content.split(/(```[\s\S]*?```)/g);
    return parts.map((part, i) => {
      if (part.startsWith("```")) {
        const match = part.match(/```(\w+)?\n?([\s\S]*?)```/);
        if (match) {
          return <CodeBlock key={i} language={match[1] || "text"} code={match[2].trim()} />;
        }
      }

      // Parse inline formatting
      return (
        <div key={i} className="text-content" dangerouslySetInnerHTML={{
          __html: part
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/`(.*?)`/g, '<code class="inline-code">$1</code>')
            .replace(/\n/g, "<br/>"),
        }} />
      );
    });
  };

  const renderMedia = () => {
    if (isError) return null;
    const url = message.metadata?.url;
    if (!url) return null;
    if (message.type === "image") {
      return <img className="message-media" src={url} alt={message.metadata?.prompt || "Generated image"} />;
    }
    if (message.type === "video") {
      // Generated video is served as an animated GIF.
      return <img className="message-media" src={url} alt={message.metadata?.prompt || "Generated video"} />;
    }
    if (message.type === "audio") {
      return <audio className="message-audio" src={url} controls />;
    }
    return null;
  };

  return (
    <div className={`message ${isUser ? "user" : "assistant"} ${isStreaming ? "streaming" : ""} ${isError ? "error" : ""}`}>
      <div className="message-avatar">
        {isUser ? <User size={18} /> : isError ? <AlertCircle size={18} /> : <Bot size={18} />}
      </div>
      <div className="message-body">
        <div className="message-content">{renderContent(message.content)}</div>
        {renderMedia()}
        {!isUser && message.content && !isError && (
          <div className="message-actions">
            <button className="action-btn" onClick={handleCopy} title="Copy">
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
            {message.metadata?.model && (
              <span className="model-tag">{message.metadata.model}</span>
            )}
          </div>
        )}
        {isError && onRetry && (
          <div className="message-actions">
            <button className="action-btn retry-btn" onClick={onRetry} title="Retry">
              <RefreshCw size={14} />
              <span>Retry</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
