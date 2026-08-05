import React, { useState, useRef, useEffect } from "react";
import { Send, PanelLeft, Image, Video, Code, AudioLines, Mic, Loader2, X, WifiOff, AlertTriangle, SlidersHorizontal } from "lucide-react";
import MessageBubble from "./MessageBubble";
import { api } from "../../services/api";

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
  onOpenSettings,
  focusRef,
  textareaFocusRef,
  onFocusSidebar,
  onFocusSidebarSettings,
  chatSettingsFocusRef,
}) {
  const [input, setInput] = useState("");
  const [messageType, setMessageType] = useState("text");
  const [transcribing, setTranscribing] = useState(false);
  const [transcribeError, setTranscribeError] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const audioInputRef = useRef(null);
  const suggestionsRef = useRef(null);
  const inputModesRef = useRef(null);
  const sendBtnRef = useRef(null);
  const chatSettingsBtnRef = useRef(null);

  // Expose focus fns to App
  useEffect(() => {
    if (focusRef) {
      focusRef.current = () => {
        const firstSuggestion = suggestionsRef.current?.querySelector(".suggestion");
        if (firstSuggestion) firstSuggestion.focus();
        else textareaRef.current?.focus();
      };
    }
    if (textareaFocusRef) {
      textareaFocusRef.current = () => textareaRef.current?.focus();
    }
    if (chatSettingsFocusRef) {
      chatSettingsFocusRef.current = () => chatSettingsBtnRef.current?.focus();
    }
  }, [focusRef, textareaFocusRef, chatSettingsFocusRef]);

  // Auto-focus textarea when AI response finishes
  const prevStreaming = useRef(false);
  const prevLoading = useRef(false);
  useEffect(() => {
    const wasActive = prevStreaming.current || prevLoading.current;
    const isNowDone = !streaming && !loading;
    if (wasActive && isNowDone) {
      textareaRef.current?.focus();
    }
    prevStreaming.current = streaming;
    prevLoading.current = loading;
  }, [streaming, loading]);

  // Suggestion buttons: ←→ between them, ↑↓ to input modes, ← at first → sidebar, ↑ → chat settings
  const handleSuggestionKeyDown = (e) => {
    if (!suggestionsRef.current) return;
    const btns = Array.from(suggestionsRef.current.querySelectorAll(".suggestion"));
    const currentIndex = btns.indexOf(document.activeElement);
    if (e.key === "ArrowRight") {
      e.preventDefault();
      (btns[currentIndex + 1] || btns[0]).focus();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      if (currentIndex === 0) {
        onFocusSidebar?.();
      } else {
        btns[currentIndex - 1].focus();
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      chatSettingsBtnRef.current?.focus();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      inputModesRef.current?.querySelector(".mode-btn")?.focus();
    }
  };

  // Mode buttons: ←→ between them, first ← → sidebar, last → → textarea, ↑ → suggestions/chat settings, ↓ → textarea
  const handleInputModesKeyDown = (e) => {
    if (!inputModesRef.current) return;
    const btns = Array.from(inputModesRef.current.querySelectorAll(".mode-btn"));
    const currentIndex = btns.indexOf(document.activeElement);
    if (e.key === "ArrowRight") {
      e.preventDefault();
      if (currentIndex === btns.length - 1) {
        textareaRef.current?.focus();
      } else {
        btns[currentIndex + 1].focus();
      }
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      if (currentIndex === 0) {
        onFocusSidebar?.();
      } else {
        btns[currentIndex - 1].focus();
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      const lastSuggestion = suggestionsRef.current?.querySelector(".suggestion:last-child");
      if (lastSuggestion) lastSuggestion.focus();
      else chatSettingsBtnRef.current?.focus();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      textareaRef.current?.focus();
    }
  };

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

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading || streaming || loadingMessages) return;
    onSend(input, messageType);
    setInput("");
    setMessageType("text");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
      return;
    }
    if (e.key === "ArrowRight") {
      const el = e.target;
      if (el.selectionStart === el.value.length) {
        e.preventDefault();
        sendBtnRef.current?.focus();
      }
    }
    if (e.key === "ArrowLeft") {
      const el = e.target;
      if (el.selectionStart === 0) {
        e.preventDefault();
        const btns = inputModesRef.current?.querySelectorAll(".mode-btn");
        if (btns?.length) btns[btns.length - 1].focus();
      }
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

  const isBusy = loading || streaming || loadingMessages;

  return (
    <main className={`chat-container ${sidebarOpen ? "" : "full-width"}`} aria-label="Chat">
      {/* Header */}
      <header className="chat-header">
        {!sidebarOpen && (
          <button className="icon-btn" onClick={onToggleSidebar} aria-label="Open sidebar">
            <PanelLeft size={20} aria-hidden="true" />
          </button>
        )}
        <h1 className="chat-title">FramerAI</h1>
        <div className="chat-subtitle" aria-hidden="true">Text, code, image, video, and audio</div>
        <button
          className="icon-btn chat-settings-btn"
          ref={chatSettingsBtnRef}
          onClick={onOpenSettings}
          title="Generation settings"
          aria-label="Generation settings"
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft") {
              e.preventDefault();
              onFocusSidebarSettings?.();
            } else if (e.key === "ArrowDown") {
              e.preventDefault();
              const firstSuggestion = suggestionsRef.current?.querySelector(".suggestion");
              if (firstSuggestion) firstSuggestion.focus();
              else inputModesRef.current?.querySelector(".mode-btn")?.focus();
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              sendBtnRef.current?.focus();
            }
          }}
        >
          <SlidersHorizontal size={18} aria-hidden="true" />
        </button>
      </header>

      {/* Global error banner */}
      {error && (
        <div className="error-banner" role="alert" aria-live="assertive">
          <AlertTriangle size={15} className="error-banner-icon" aria-hidden="true" />
          <span>{error}</span>
          <button className="error-banner-dismiss" onClick={onDismissError} aria-label="Dismiss error">
            <X size={14} aria-hidden="true" />
          </button>
        </div>
      )}

      {/* Transcription error banner */}
      {transcribeError && (
        <div className="error-banner" role="alert" aria-live="assertive">
          <AlertTriangle size={15} className="error-banner-icon" aria-hidden="true" />
          <span>{transcribeError}</span>
          <button className="error-banner-dismiss" onClick={() => setTranscribeError(null)} aria-label="Dismiss error">
            <X size={14} aria-hidden="true" />
          </button>
        </div>
      )}

      {/* Messages */}
      <div
        className="messages-container"
        role="log"
        aria-label="Conversation messages"
        aria-live="polite"
        aria-relevant="additions"
      >
        {loadingMessages ? (
          // Loading skeleton — shown while fetching an existing conversation
          // aria-live="polite" announces to screen readers when loading starts/ends (improvement b)
          <div className="messages-loading" aria-label="Loading messages" aria-busy="true" aria-live="polite">
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
            {/* aria-live="polite" + role="status" lets screen readers announce (improvement c) */}
            {messages.length === 0 && (
              <div className="welcome-screen" aria-label="Welcome to FramerAI" aria-live="polite" role="status">
                <img src="/logo.svg" alt="FramerAI logo" className="welcome-logo" />
                <h2>Welcome to FramerAI</h2>
                <p>A multimodal AI that can generate text, code, images, video, and audio.</p>
                <nav
                  ref={suggestionsRef}
                  className="suggestions"
                  aria-label="Suggested prompts"
                  onKeyDown={handleSuggestionKeyDown}
                >
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
                </nav>
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
              <div className="message assistant" role="status" aria-label="FramerAI is typing">
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
      <form className="input-container" onSubmit={handleSubmit} aria-label="Send a message">
        <div className={`input-wrapper ${isBusy ? "busy" : ""}`}>
          <div
            className="input-modes"
            role="group"
            aria-label="Message type"
            ref={inputModesRef}
            onKeyDown={handleInputModesKeyDown}
          >
            <button type="button" className={`mode-btn ${messageType === "text" ? "active" : ""}`}
              onClick={() => setMessageType("text")} aria-label="Text mode" aria-pressed={messageType === "text"}>
              <MessageIcon size={16} aria-hidden="true" />
            </button>
            <button type="button" className={`mode-btn ${messageType === "code" ? "active" : ""}`}
              onClick={() => setMessageType("code")} aria-label="Code mode" aria-pressed={messageType === "code"}>
              <Code size={16} aria-hidden="true" />
            </button>
            <button type="button" className={`mode-btn ${messageType === "image" ? "active" : ""}`}
              onClick={() => setMessageType("image")} aria-label="Image generation mode" aria-pressed={messageType === "image"}>
              <Image size={16} aria-hidden="true" />
            </button>
            <button type="button" className={`mode-btn ${messageType === "video" ? "active" : ""}`}
              onClick={() => setMessageType("video")} aria-label="Video generation mode" aria-pressed={messageType === "video"}>
              <Video size={16} aria-hidden="true" />
            </button>
            <button type="button" className={`mode-btn ${messageType === "audio" ? "active" : ""}`}
              onClick={() => setMessageType("audio")} aria-label="Audio generation mode" aria-pressed={messageType === "audio"}>
              <AudioLines size={16} aria-hidden="true" />
            </button>
            <button type="button" className="mode-btn"
              onClick={() => audioInputRef.current?.click()}
              aria-label="Upload audio to transcribe"
              disabled={transcribing} aria-busy={transcribing}>
              {transcribing ? <Loader2 size={16} className="spin" aria-hidden="true" /> : <Mic size={16} aria-hidden="true" />}
            </button>
            <input ref={audioInputRef} type="file" accept="audio/*"
              onChange={handleAudioUpload} style={{ display: "none" }} aria-hidden="true" tabIndex={-1} />
          </div>

          <textarea
            ref={textareaRef}
            id="chat-input"
            className="chat-input"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={loadingMessages ? "Loading conversation…" : "Message FramerAI…"}
            rows={1}
            disabled={isBusy}
            aria-label="Message input"
            aria-multiline="true"
          />

          <button
            type="submit"
            ref={sendBtnRef}
            className="send-btn"
            disabled={!input.trim() || isBusy}
            aria-label="Send message"
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft") {
                e.preventDefault();
                textareaRef.current?.focus();
              } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
                e.preventDefault();
                chatSettingsBtnRef.current?.focus();
              }
            }}
          >
            {loading || streaming
              ? <Loader2 size={20} className="spin" aria-hidden="true" />
              : <Send size={20} aria-hidden="true" />
            }
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
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}
