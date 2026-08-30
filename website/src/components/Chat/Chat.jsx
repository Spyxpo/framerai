import { useState, useRef, useEffect, useCallback } from "react";
import { Send, PanelLeft, Image, Video, Code, AudioLines, Mic, MicOff, Paperclip, Loader2, X, AlertTriangle, SlidersHorizontal, FileText } from "lucide-react";
import MessageBubble from "./MessageBubble";
import { api } from "../../services/api";

export default function Chat({
  messages,
  loading,
  streaming,
  loadingMessages,
  error,
  pendingApproval,
  onApproveCommand,
  onDenyCommand,
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
  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [attachments, setAttachments] = useState([]);
  const [attaching, setAttaching] = useState(false);
  const [attachError, setAttachError] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const audioInputRef = useRef(null);
  const attachInputRef = useRef(null);
  const suggestionsRef = useRef(null);
  const inputModesRef = useRef(null);
  const sendBtnRef = useRef(null);
  const chatSettingsBtnRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingTimerRef = useRef(null);

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

  // Transcribe an audio File/Blob and insert text into the input
  const transcribeAudio = useCallback(async (file) => {
    setTranscribing(true);
    setTranscribeError(null);
    try {
      const result = await api.transcribe(file);
      // result.text may be a placeholder when the model isn't trained yet —
      // insert it anyway so the user sees something
      const text = result?.text?.trim();
      if (text) {
        setInput((prev) => (prev ? `${prev} ${text}` : text));
        textareaRef.current?.focus();
      } else {
        setTranscribeError("Transcription returned empty text. Make sure the model is trained.");
      }
    } catch (err) {
      setTranscribeError(`Transcription failed: ${err.message}`);
    } finally {
      setTranscribing(false);
    }
  }, []);

  // Store a picked file and keep the path it came back with. Uploading here
  // rather than at send time means the attachment is visible, and removable,
  // before the message goes anywhere.
  const handleAttach = async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (!files.length) return;

    setAttaching(true);
    setAttachError(null);
    try {
      for (const file of files) {
        const stored = await api.uploadAttachment(file);
        setAttachments((prev) => [...prev, { ...stored, name: stored.name || file.name }]);
      }
    } catch (err) {
      setAttachError(`Could not attach the file: ${err.message}`);
    } finally {
      setAttaching(false);
    }
  };

  const removeAttachment = (attachmentPath) => {
    setAttachments((prev) => prev.filter((a) => a.path !== attachmentPath));
  };

  // File-upload fallback handler
  const handleAudioUpload = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    await transcribeAudio(file);
  };

  // Start / stop live microphone recording
  const handleMicClick = useCallback(async () => {
    // ── STOP ──────────────────────────────────────────────
    if (recording) {
      // Guard against calling stop() on an already-inactive recorder
      // (e.g. double-click) which would throw InvalidStateError
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      return;
    }

    // ── START ─────────────────────────────────────────────
    setTranscribeError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setTranscribeError("Microphone access is not supported in this browser.");
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setTranscribeError("Microphone permission denied. Allow access in your browser settings.");
      } else if (err.name === "NotFoundError") {
        setTranscribeError("No microphone found. Plug one in and try again.");
      } else {
        setTranscribeError(`Could not access microphone: ${err.message}`);
      }
      return;
    }

    // Pick a supported MIME type — always fall back to audio/webm so the
    // backend mimeFilter("audio/") never sees an empty content-type
    const PREFERRED = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    const mimeType = PREFERRED.find((t) => MediaRecorder.isTypeSupported(t)) ?? "audio/webm";

    // Reset chunks for this recording session (supports record-twice)
    audioChunksRef.current = [];

    const recorder = new MediaRecorder(stream, { mimeType });
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      // Release the mic — browser indicator disappears here
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(recordingTimerRef.current);
      setRecording(false);
      setRecordingSeconds(0);

      // Derive extension from the actual mimeType used
      const ext = mimeType.includes("ogg") ? "ogg" : mimeType.includes("mp4") ? "mp4" : "webm";
      const blob = new Blob(audioChunksRef.current, { type: mimeType });
      const file = new File([blob], `recording.${ext}`, { type: mimeType });
      await transcribeAudio(file);
    };

    recorder.onerror = () => {
      // Always release the stream on error so mic indicator clears
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(recordingTimerRef.current);
      setRecording(false);
      setRecordingSeconds(0);
      setTranscribeError("Recording failed. Please try again.");
    };

    recorder.start(250); // collect chunks every 250 ms
    setRecording(true);
    setRecordingSeconds(0);
    recordingTimerRef.current = setInterval(() => {
      setRecordingSeconds((s) => s + 1);
    }, 1000);
  }, [recording, transcribeAudio]);

  // Clean up recorder on unmount
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      clearInterval(recordingTimerRef.current);
    };
  }, []);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading || streaming || loadingMessages) return;
    onSend(input, messageType, attachments.map((a) => a.path));
    setInput("");
    setMessageType("text");
    setAttachments([]);
    setAttachError(null);
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

      {/* CLI Command Approval Prompt */}
      {pendingApproval && (
        <div className="cli-approval-card" role="alert" aria-live="assertive" data-testid="cli-approval-card">
          <div className="cli-approval-header">
            <AlertTriangle size={18} className="cli-approval-icon" aria-hidden="true" />
            <span>CLI Command Approval Required</span>
          </div>
          <div className="cli-approval-body">
            <p className="cli-approval-prompt">
              The model requests approval to run the following shell command:
            </p>
            <pre className="cli-approval-cmd">
              <code>{pendingApproval.command}</code>
            </pre>
            <div className="cli-approval-meta">
              <div><strong>Working Root:</strong> {pendingApproval.root}</div>
              {Array.isArray(pendingApproval.argv) && (
                <div><strong>argv:</strong> {JSON.stringify(pendingApproval.argv)}</div>
              )}
            </div>
            <div className="cli-approval-actions">
              <button
                type="button"
                className="btn btn-primary approve-btn"
                onClick={() => onApproveCommand?.(pendingApproval.approvalId)}
              >
                Approve
              </button>
              <button
                type="button"
                className="btn btn-secondary deny-btn"
                onClick={() => onDenyCommand?.(pendingApproval.approvalId, false)}
              >
                Deny
              </button>
              <button
                type="button"
                className="btn btn-danger deny-all-btn"
                onClick={() => onDenyCommand?.(pendingApproval.approvalId, true)}
              >
                Deny all future commands
              </button>
            </div>
          </div>
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
        {attachments.length > 0 && (
          <ul className="attachment-list" aria-label="Attachments on this message">
            {attachments.map((attachment) => (
              <li key={attachment.path} className="attachment-chip">
                <FileText size={14} aria-hidden="true" />
                <span className="attachment-name">{attachment.name}</span>
                <button
                  type="button"
                  className="attachment-remove"
                  onClick={() => removeAttachment(attachment.path)}
                  aria-label={`Remove ${attachment.name}`}
                >
                  <X size={12} aria-hidden="true" />
                </button>
              </li>
            ))}
          </ul>
        )}
        {attachError && (
          <p className="attachment-error" role="alert">{attachError}</p>
        )}
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

            {/* Attach an image or document to the message */}
            <button
              type="button"
              className="mode-btn"
              onClick={() => attachInputRef.current?.click()}
              aria-label="Attach an image or document"
              disabled={attaching}
              title="Attach an image or document"
            >
              {attaching
                ? <Loader2 size={16} className="spin" aria-hidden="true" />
                : <FileText size={16} aria-hidden="true" />}
            </button>

            <input ref={attachInputRef} type="file" multiple
              accept="image/*,application/pdf,text/plain,text/markdown"
              onChange={handleAttach} style={{ display: "none" }} aria-hidden="true" tabIndex={-1} />

            {/* Upload audio file button */}
            <button
              type="button"
              className="mode-btn"
              onClick={() => audioInputRef.current?.click()}
              aria-label="Upload audio file to transcribe"
              disabled={transcribing || recording}
              title="Upload audio file"
            >
              <Paperclip size={16} aria-hidden="true" />
            </button>

            {/* Live mic capture button */}
            <button
              type="button"
              className={`mode-btn mic-btn ${recording ? "recording" : ""}`}
              onClick={handleMicClick}
              aria-label={recording ? "Stop recording" : "Record from mic"}
              aria-pressed={recording}
              disabled={transcribing}
              title={recording ? "Stop recording" : "Record from mic"}
            >
              {transcribing
                ? <Loader2 size={16} className="spin" aria-hidden="true" />
                : recording
                  ? <MicOff size={16} aria-hidden="true" />
                  : <Mic size={16} aria-hidden="true" />
              }
            </button>

            {/* Recording timer badge */}
            {recording && (
              <span className="recording-timer" aria-live="off" aria-hidden="true">
                {String(Math.floor(recordingSeconds / 60)).padStart(2, "0")}:{String(recordingSeconds % 60).padStart(2, "0")}
              </span>
            )}

            {/* File upload fallback (hidden) */}
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
