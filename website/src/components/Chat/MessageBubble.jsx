import React from "react";
import { User, Bot, Copy, Check } from "lucide-react";
import CodeBlock from "../CodeBlock/CodeBlock";

export default function MessageBubble({ message, isStreaming }) {
  const [copied, setCopied] = React.useState(false);
  const isUser = message.role === "user";

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderContent = (content) => {
    if (!content) {
      return (
        <div className="typing-indicator">
          <span></span><span></span><span></span>
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

  return (
    <div className={`message ${isUser ? "user" : "assistant"} ${isStreaming ? "streaming" : ""}`}>
      <div className="message-avatar">
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>
      <div className="message-body">
        <div className="message-content">{renderContent(message.content)}</div>
        {!isUser && message.content && (
          <div className="message-actions">
            <button className="action-btn" onClick={handleCopy} title="Copy">
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
            {message.metadata?.model && (
              <span className="model-tag">{message.metadata.model}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
