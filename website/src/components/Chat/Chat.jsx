import React, { useState, useRef, useEffect } from "react";
import { Send, PanelLeft, Image, Video, Code, AudioLines, Mic, Loader2, X, AlertTriangle } from "lucide-react";
import MessageBubble from "./MessageBubble";
import { api } from "../../services/api";

const MODES = [
  { id: "text", label: "Text", Icon: MessageIcon },
  { id: "code", label: "Code", Icon: Code },
  { id: "image", label: "Image", Icon: Image },
  { id: "video", label: "Video", Icon: Video },
  { id: "audio", label: "Audio", Icon: AudioLines },
];

export default function Chat({
  messages,
  loading,
  streaming,
  loadingMessages,
  error,
  sidebarOpen,
  onSend,
  onToggleSidebar,
  onDismissError,
}) {
  const [input, setInput] = useState("");
  const [messageType, setMessageType] = useState("text");
  const [transcribing, setTranscribing] = useState(false);
  const [transcribeError, setTranscribeError] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const audioInputRef = useRef(null);

  const isBusy = loading || streaming || loadingMessages;
  const wasBusy = useRef(isBusy);

  const handleAudioUpload = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setTranscribing(true);
    setTranscribeError(null);
    try {
      const result = await api.transcribe(file);
      setInput((prev) => (prev ? `${prev} ${result.text}` : result.text));
      textareaRef.current?.focus();
    } catch (err) {
      setTranscribeError(`Transcription failed: ${err.message}`);
    } finally {
      setTranscribing(false);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // The composer is disabled while a response is in flight, which drops focus.
  // Put it back once the app is idle again so typing can continue uninterrupted.
  useEffect(() => {
    if (wasBusy.current && !isBusy) textareaRef.current?.focus();
    wasBusy.current = isBusy;
  }, [isBusy]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isBusy) return;
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
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  };

  // Find the last user message so we can offer retry on the error below it
  const lastUserMessage = [...messages].reverse().find((m) => m.role === "user");
  const handleRetry = lastUserMessage
    ? () => onSend(lastUserMessage.content, lastUserMessage.type || "text")
    : undefined;

  // Single spoken description of what the app is doing right now.
  const status = loadingMessages
    ? "Loading conversation"
    : transcribing
      ? "Transcribing audio"
      : loading || streaming
        ? "FramerAI is responding"
        : "";

  return (
    <main className={`chat-container ${sidebarOpen ? "" : "full-width"}`}>
      <a className="skip-link" href="#message-input">
        Skip to message input
      </a>

      {/* Header */}
      <header className="chat-header">
        {!sidebarOpen && (
          <button
            className="icon-btn"
            onClick={onToggleSidebar}
            aria-label="Open sidebar"
            aria-expanded={false}
            aria-controls="sidebar"
          >
            <PanelLeft size={20} aria-hidden="true" />
          </button>
        )}
        <h1 className="chat-title">FramerAI</h1>
        <div className="chat-subtitle">Text, code, image, video, and audio</div>
      </header>

      {/* Global error banner */}
      {error && (
        <div className="error-banner" role="alert">
          <AlertTriangle size={15} className="error-banner-icon" aria-hidden="true" />
          <span>{error}</span>
          <button className="error-banner-dismiss" onClick={onDismissError} aria-label="Dismiss error">
            <X size={14} aria-hidden="true" />
          </button>
        </div>
      )}

      {/* Transcription error banner */}
      {transcribeError && (
        <div className="error-banner" role="alert">
          <AlertTriangle size={15} className="error-banner-icon" aria-hidden="true" />
          <span>{transcribeError}</span>
          <button className="error-banner-dismiss" onClick={() => setTranscribeError(null)} aria-label="Dismiss error">
            <X size={14} aria-hidden="true" />
          </button>
        </div>
      )}

      {/* Spoken status for anything that has no visible text of its own */}
      <p className="sr-only" role="status">{status}</p>

      {/* Messages */}
      <div
        className="messages-container"
        role="log"
        aria-label="Conversation"
        aria-busy={loadingMessages}
        tabIndex={0}
      >
        {/* Loading skeleton — shown while fetching an existing conversation */}
        {loadingMessages ? (
          <div className="messages-loading">
            <div className="message-skeleton">
              <div className="skeleton-avatar" />
              <div className="skeleton-body">
                <div className="skeleton-line long" />
                <div className="skeleton-line medium" />
              </div>
            </div>
            <div className="message-skeleton user">
              <div className="skeleton-body right">
                <div className="skeleton-line short" />
              </div>
              <div className="skeleton-avatar" />
            </div>
            <div className="message-skeleton">
              <div className="skeleton-avatar" />
              <div className="skeleton-body">
                <div className="skeleton-line long" />
                <div className="skeleton-line short" />
                <div className="skeleton-line medium" />
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Empty / welcome state — only when not loading */}
            {messages.length === 0 && (
              <div className="welcome-screen">
                <img src="/logo.svg" alt="" className="welcome-logo" />
                <h2>Welcome to FramerAI</h2>
                <p>A multimodal AI that can generate text, code, images, video, and audio.</p>
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
                  <button className="suggestion" onClick={() => onSend("Generate audio that says hello and welcome")}>
                    Generate a voice clip
                  </button>
                </div>
              </div>
            )}

            {messages.map((msg, i) => {
              const isLastMsg = i === messages.length - 1;
              const isErrorMsg = msg.type === "error";
              return (
                <MessageBubble
                  key={msg.id || i}
                  message={msg}
                  isStreaming={streaming && isLastMsg}
                  onRetry={isErrorMsg && isLastMsg ? handleRetry : undefined}
                />
              );
            })}

            {/* Typing indicator — only when the assistant hasn't replied yet */}
            {(loading || streaming) && messages[messages.length - 1]?.role !== "assistant" && (
              <div className="message assistant">
                <div className="message-content">
                  <div className="typing-indicator" aria-hidden="true">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form className="input-container" onSubmit={handleSubmit}>
        <div className={`input-wrapper ${isBusy ? "busy" : ""}`}>
          <div className="input-modes" role="group" aria-label="Response type">
            {MODES.map(({ id, label, Icon }) => (
              <button
                key={id}
                type="button"
                className={`mode-btn ${messageType === id ? "active" : ""}`}
                onClick={() => setMessageType(id)}
                title={label}
                aria-label={label}
                aria-pressed={messageType === id}
              >
                <Icon size={16} aria-hidden="true" />
              </button>
            ))}
            <button
              type="button"
              className="mode-btn"
              onClick={() => audioInputRef.current?.click()}
              title="Upload audio to transcribe"
              aria-label="Upload audio to transcribe"
              disabled={transcribing}
            >
              {transcribing ? (
                <Loader2 size={16} className="spin" aria-hidden="true" />
              ) : (
                <Mic size={16} aria-hidden="true" />
              )}
            </button>
            <input
              ref={audioInputRef}
              type="file"
              accept="audio/*"
              onChange={handleAudioUpload}
              className="sr-only"
              tabIndex={-1}
              aria-hidden="true"
            />
          </div>
          <label className="sr-only" htmlFor="message-input">
            Message FramerAI
          </label>
          <textarea
            id="message-input"
            ref={textareaRef}
            className="chat-input"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={loadingMessages ? "Loading conversation…" : "Message FramerAI…"}
            rows={1}
            disabled={isBusy}
            aria-describedby="input-hint"
          />
          <button
            type="submit"
            className="send-btn"
            disabled={!input.trim() || isBusy}
            aria-label="Send message"
          >
            {loading || streaming ? (
              <Loader2 size={20} className="spin" aria-hidden="true" />
            ) : (
              <Send size={20} aria-hidden="true" />
            )}
          </button>
        </div>
        <p className="input-hint" id="input-hint">
          FramerAI is an open-source multimodal model. Train it with <code>python build.py --mode all</code>
        </p>
      </form>
    </main>
  );
}

function MessageIcon({ size, ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}
