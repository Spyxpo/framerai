import React from "react";
import ReactMarkdown from "react-markdown";
import { User, Bot, Copy, Check, AlertCircle, RefreshCw, GitBranch, Loader2, Pencil, X } from "lucide-react";
import CodeBlock from "../CodeBlock/CodeBlock";
import StreamingAudioPlayer from "../AudioPlayer/StreamingAudioPlayer";
import CognitionTrace from "./CognitionTrace";

export default function MessageBubble({
  message,
  isStreaming,
  onRetry,
  onBranch,
  onEdit,
  onRegenerate,
  onContinue,
  isBranching,
  isHighlighted,
}) {
  const [copied, setCopied] = React.useState(false);
  const [isEditing, setIsEditing] = React.useState(false);
  const [editContent, setEditContent] = React.useState(message.content || "");
  const isUser = message.role === "user";
  const isError = message.type === "error";

  const handleStartEdit = () => {
    setEditContent(message.content || "");
    setIsEditing(true);
  };

  const handleSaveEdit = async () => {
    if (!editContent.trim()) return;
    setIsEditing(false);
    if (onEdit && message.id) {
      await onEdit(message.id, editContent, { branch: false });
    }
  };

  const handleBranchEdit = async () => {
    if (!editContent.trim()) return;
    setIsEditing(false);
    if (onEdit && message.id) {
      await onEdit(message.id, editContent, { branch: true });
    }
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditContent(message.content || "");
  };

  const handleEditKeyDown = (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      handleCancelEdit();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSaveEdit();
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderContent = (content) => {
    // Empty content during streaming = typing indicator
    if (!content) {
      return (
        <div className="typing-indicator" aria-hidden="true">
          <span></span><span></span><span></span>
        </div>
      );
    }

    // Error messages get their own renderer
    if (isError) {
      return (
        <div className="error-message-content">
          <AlertCircle size={15} className="error-icon" aria-hidden="true" />
          <span>{content}</span>
        </div>
      );
    }

    // Render Markdown safely via react-markdown (never uses dangerouslySetInnerHTML).
    // All HTML in the source content is escaped by react-markdown's default
    // behaviour, preventing XSS from user-supplied or model-generated content.
    return (
      <div className="text-content">
        <ReactMarkdown
          components={{
            // Fenced code blocks in react-markdown v10 render as <pre><code>.
            // We intercept at the <pre> level and delegate to CodeBlock so that
            // syntax highlighting and copy behaviour are preserved.
            pre({ children }) {
              // The direct child is a React element for <code> with a
              // className like "language-python".
              const child = React.Children.only(children);
              if (React.isValidElement(child)) {
                const { className, children: codeText } = child.props;
                const language = /language-(\w+)/.exec(className || "")?.[1] || "text";
                return <CodeBlock language={language} code={String(codeText).trimEnd()} />;
              }
              return <pre>{children}</pre>;
            },
            // Inline backtick spans — render as a plain styled <code>
            // (react-markdown does NOT wrap these in <pre>)
            code({ className, children, ...props }) {
              return <code className={`inline-code${className ? ` ${className}` : ""}`} {...props}>{children}</code>;
            },
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  };

  const renderMedia = () => {
    if (isError) return null;

    // Handle streaming audio
    if (message.type === "audio" && message.audioChunks && message.audioMetadata) {
      return (
        <StreamingAudioPlayer
          chunks={message.audioChunks}
          sampleRate={message.audioMetadata.sampleRate}
          channels={message.audioMetadata.channels}
          bitsPerSample={message.audioMetadata.bitsPerSample}
        />
      );
    }

    // Handle complete audio URL (fallback or REST response)
    const url = message.metadata?.url;
    if (!url) return null;
    if (message.type === "image") {
      return <img className="message-media" src={url} alt={message.metadata?.prompt || "Generated image"} />;
    }
    if (message.type === "video") {
      // Written as a real container now rather than as an animated GIF, so it
      // needs a video element and the controls that come with one. A GIF from
      // the fallback path still renders here.
      if (url.endsWith(".gif")) {
        return <img className="message-media" src={url} alt={message.metadata?.prompt || "Generated video"} />;
      }
      return (
        <video
          className="message-media"
          src={url}
          controls
          loop
          playsInline
          aria-label={message.metadata?.prompt || "Generated video"}
        />
      );
    }
    if (message.type === "audio") {
      return <audio className="message-audio" src={url} controls aria-label="Generated audio" />;
    }
    return null;
  };

  const roleLabel = isUser ? "Your message" : isError ? "Error from FramerAI" : "FramerAI response";

  return (
    <article
      id={message.id ? `message-${message.id}` : undefined}
      data-message-id={message.id || message.clientId}
      className={`message ${isUser ? "user" : "assistant"} ${isStreaming ? "streaming" : ""} ${isError ? "error" : ""} ${isHighlighted ? "message-highlighted" : ""}`}
      aria-label={roleLabel}
    >
      <div className="message-avatar" aria-hidden="true">
        {isUser ? <User size={18} /> : isError ? <AlertCircle size={18} /> : <Bot size={18} />}
      </div>
      <div className="message-body">
        {isEditing ? (
          <div className="message-edit-container">
            <textarea
              className="message-edit-input"
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              onKeyDown={handleEditKeyDown}
              aria-label="Edit message content"
              rows={3}
              autoFocus
            />
            <div className="message-edit-actions">
              <button
                type="button"
                className="action-btn submit-edit-btn"
                onClick={handleSaveEdit}
                disabled={!editContent.trim()}
                aria-label="Save and send"
              >
                <Check size={14} aria-hidden="true" />
                <span>Save & Send</span>
              </button>
              <button
                type="button"
                className="action-btn branch-edit-btn"
                onClick={handleBranchEdit}
                disabled={!editContent.trim()}
                aria-label="Branch and send"
                title="Send as a new branch"
              >
                <GitBranch size={14} aria-hidden="true" />
                <span>Branch</span>
              </button>
              <button
                type="button"
                className="action-btn cancel-edit-btn"
                onClick={handleCancelEdit}
                aria-label="Cancel edit"
              >
                <X size={14} aria-hidden="true" />
                <span>Cancel</span>
              </button>
            </div>
          </div>
        ) : (
          <div
            className="message-content"
            role={isError ? "alert" : undefined}
            aria-live={isError ? "assertive" : undefined}
          >
            {renderContent(message.content)}
          </div>
        )}
        {!isEditing && renderMedia()}
        {!isUser && !isEditing && message.metadata?.trace && (
          <CognitionTrace trace={message.metadata.trace} />
        )}
        {!isUser && message.content && !isError && (
          <div className="message-actions">
            <button
              className="action-btn"
              onClick={handleCopy}
              aria-label={copied ? "Copied" : "Copy message"}
            >
              {copied ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}
            </button>
            {onRegenerate && message.id && (
              <button
                className="action-btn regenerate-btn"
                onClick={() => onRegenerate(message.id)}
                disabled={isStreaming || isBranching}
                aria-label="Regenerate response"
                title="Regenerate response"
              >
                <RefreshCw size={14} aria-hidden="true" />
                <span>Regenerate</span>
              </button>
            )}
            {onContinue && message.id && (
              <button
                className="action-btn continue-btn"
                onClick={() => onContinue(message.id)}
                disabled={isStreaming || isBranching}
                aria-label="Continue from here"
                title="Continue from here"
              >
                <GitBranch size={14} aria-hidden="true" />
                <span>Continue</span>
              </button>
            )}
            {onBranch && message.id && (
              <button
                className="action-btn branch-btn"
                onClick={() => onBranch(message.id)}
                disabled={isBranching || isStreaming}
                aria-label="Branch from here"
                title="Branch from here"
              >
                {isBranching ? (
                  <Loader2 size={14} className="spin" aria-hidden="true" />
                ) : (
                  <GitBranch size={14} aria-hidden="true" />
                )}
                <span>Branch</span>
              </button>
            )}
            {message.metadata?.model && (
              <span className="model-tag">{message.metadata.model}</span>
            )}
          </div>
        )}
        {isUser && message.content && !isError && !isEditing && (
          <div className="message-actions">
            {onEdit && message.id && (
              <button
                className="action-btn edit-btn"
                onClick={handleStartEdit}
                disabled={isStreaming || isBranching}
                aria-label="Edit message"
                title="Edit message"
              >
                <Pencil size={14} aria-hidden="true" />
                <span>Edit</span>
              </button>
            )}
            {onContinue && message.id && (
              <button
                className="action-btn continue-btn"
                onClick={() => onContinue(message.id)}
                disabled={isStreaming || isBranching}
                aria-label="Continue from here"
                title="Continue from here"
              >
                <GitBranch size={14} aria-hidden="true" />
                <span>Continue</span>
              </button>
            )}
            {onBranch && message.id && (
              <button
                className="action-btn branch-btn"
                onClick={() => onBranch(message.id)}
                disabled={isBranching}
                aria-label="Branch from here"
                title="Branch from here"
              >
                {isBranching ? (
                  <Loader2 size={14} className="spin" aria-hidden="true" />
                ) : (
                  <GitBranch size={14} aria-hidden="true" />
                )}
                <span>Branch</span>
              </button>
            )}
          </div>
        )}
        {isError && onRetry && (
          <div className="message-actions">
            <button className="action-btn retry-btn" onClick={onRetry} aria-label="Retry sending message">
              <RefreshCw size={14} aria-hidden="true" />
              <span>Retry</span>
            </button>
          </div>
        )}
      </div>
    </article>
  );
}
