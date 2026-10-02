import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { Search, X, MessageSquare, GitBranch, ChevronUp, ChevronDown, User, Bot, CornerDownLeft } from "lucide-react";
import { SEARCH_SCOPES, searchConversations, highlightMatches } from "../../utils/search";

export default function SearchModal({
  open,
  conversations,
  activeConversationId,
  onClose,
  onNavigate,
}) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState(SEARCH_SCOPES.ALL);
  const [rawSelectedIndex, setRawSelectedIndex] = useState(0);

  const inputRef = useRef(null);
  const resultsContainerRef = useRef(null);
  const activeItemRef = useRef(null);

  // Focus input when modal opens
  useEffect(() => {
    if (open) {
      inputRef.current?.focus();
    }
  }, [open]);

  // Compute search results deterministically
  const results = useMemo(() => {
    if (!query.trim()) return [];
    return searchConversations({
      conversations,
      activeConversationId,
      query,
      scope,
    });
  }, [conversations, activeConversationId, query, scope]);

  // Derive active selectedIndex safely within bounds
  const selectedIndex = results.length === 0 ? 0 : Math.min(rawSelectedIndex, results.length - 1);

  // Scroll active item into view
  useEffect(() => {
    if (activeItemRef.current && resultsContainerRef.current) {
      activeItemRef.current.scrollIntoView({
        block: "nearest",
        behavior: "smooth",
      });
    }
  }, [selectedIndex]);

  const handleClose = useCallback(() => {
    setQuery("");
    setRawSelectedIndex(0);
    onClose?.();
  }, [onClose]);

  const handleSelectResult = useCallback((result) => {
    if (!result) return;
    onNavigate?.(result);
    handleClose();
  }, [onNavigate, handleClose]);

  const handlePrev = useCallback(() => {
    if (results.length === 0) return;
    setRawSelectedIndex((prev) => {
      const current = Math.min(prev, results.length - 1);
      return current > 0 ? current - 1 : results.length - 1;
    });
  }, [results.length]);

  const handleNext = useCallback(() => {
    if (results.length === 0) return;
    setRawSelectedIndex((prev) => {
      const current = Math.min(prev, results.length - 1);
      return current < results.length - 1 ? current + 1 : 0;
    });
  }, [results.length]);

  const handleKeyDown = (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      handleClose();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      handleNext();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      handlePrev();
    } else if (e.key === "Enter") {
      if (results.length > 0 && results[selectedIndex]) {
        e.preventDefault();
        handleSelectResult(results[selectedIndex]);
      }
    }
  };

  if (!open) return null;

  return (
    <div
      className="search-modal-backdrop"
      onClick={handleClose}
      role="dialog"
      aria-modal="true"
      aria-label="Global message search"
    >
      <div
        className="search-modal"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Search Input Bar */}
        <div className="search-input-wrapper">
          <Search size={18} className="search-input-icon" aria-hidden="true" />
          <input
            ref={inputRef}
            type="search"
            className="search-input"
            placeholder="Search conversations, titles, and messages..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setRawSelectedIndex(0);
            }}
            aria-label="Search query"
          />
          {query && (
            <button
              type="button"
              className="search-clear-btn"
              onClick={() => {
                setQuery("");
                setRawSelectedIndex(0);
                inputRef.current?.focus();
              }}
              aria-label="Clear search query"
            >
              <X size={16} aria-hidden="true" />
            </button>
          )}
          <button
            type="button"
            className="search-close-btn"
            onClick={handleClose}
            aria-label="Close search"
          >
            <span className="search-key-badge" aria-hidden="true">Esc</span>
          </button>
        </div>

        {/* Scope Selector Bar */}
        <div className="search-scopes" role="tablist" aria-label="Search scope">
          <button
            type="button"
            role="tab"
            aria-selected={scope === SEARCH_SCOPES.ALL}
            className={`search-scope-btn ${scope === SEARCH_SCOPES.ALL ? "active" : ""}`}
            onClick={() => setScope(SEARCH_SCOPES.ALL)}
          >
            All
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={scope === SEARCH_SCOPES.CONVERSATIONS}
            className={`search-scope-btn ${scope === SEARCH_SCOPES.CONVERSATIONS ? "active" : ""}`}
            onClick={() => setScope(SEARCH_SCOPES.CONVERSATIONS)}
          >
            Conversations
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={scope === SEARCH_SCOPES.MESSAGES}
            className={`search-scope-btn ${scope === SEARCH_SCOPES.MESSAGES ? "active" : ""}`}
            onClick={() => setScope(SEARCH_SCOPES.MESSAGES)}
          >
            Messages
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={scope === SEARCH_SCOPES.CURRENT}
            className={`search-scope-btn ${scope === SEARCH_SCOPES.CURRENT ? "active" : ""}`}
            onClick={() => setScope(SEARCH_SCOPES.CURRENT)}
            disabled={!activeConversationId}
            title={!activeConversationId ? "No active conversation" : undefined}
          >
            Current Chat
          </button>
        </div>

        {/* Results summary header with Previous / Next navigation */}
        {query.trim() && (
          <div className="search-results-header">
            <span className="search-results-count" role="status" aria-live="polite">
              {results.length === 0
                ? "No results found"
                : `${results.length} result${results.length === 1 ? "" : "s"} found`}
            </span>
            {results.length > 0 && (
              <div className="search-nav-controls">
                <span className="search-nav-index" aria-label={`Result ${selectedIndex + 1} of ${results.length}`}>
                  {selectedIndex + 1} of {results.length}
                </span>
                <button
                  type="button"
                  className="search-nav-btn"
                  onClick={handlePrev}
                  aria-label="Previous result"
                  title="Previous result (↑)"
                >
                  <ChevronUp size={15} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  className="search-nav-btn"
                  onClick={handleNext}
                  aria-label="Next result"
                  title="Next result (↓)"
                >
                  <ChevronDown size={15} aria-hidden="true" />
                </button>
              </div>
            )}
          </div>
        )}

        {/* Results list */}
        <div
          ref={resultsContainerRef}
          className="search-results-list"
          role="listbox"
          aria-label="Search results"
        >
          {!query.trim() ? (
            <div className="search-empty-state">
              <Search size={32} className="search-empty-icon" aria-hidden="true" />
              <p className="search-empty-title">Search FramerAI</p>
              <p className="search-empty-desc">
                Find messages, code, prompts, and conversation titles across all your chats.
              </p>
            </div>
          ) : results.length === 0 ? (
            <div className="search-empty-state">
              <p className="search-empty-title">No matches found</p>
              <p className="search-empty-desc">
                No results found for &ldquo;{query}&rdquo;. Try another term or switch search scope.
              </p>
            </div>
          ) : (
            results.map((result, idx) => {
              const isSelected = idx === selectedIndex;
              const isTitle = result.type === "title";
              const isAssistant = result.role === "assistant";

              return (
                <div
                  key={result.id}
                  ref={isSelected ? activeItemRef : null}
                  role="option"
                  aria-selected={isSelected}
                  className={`search-result-item ${isSelected ? "selected" : ""}`}
                  onClick={() => handleSelectResult(result)}
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      handleSelectResult(result);
                    }
                  }}
                >
                  <div className="search-result-top">
                    <div className="search-result-conv">
                      {result.isBranch ? (
                        <GitBranch size={14} className="search-branch-icon" aria-hidden="true" />
                      ) : (
                        <MessageSquare size={14} aria-hidden="true" />
                      )}
                      <span className="search-conv-title">{result.conversationTitle}</span>
                      {result.isBranch && (
                        <span className="branch-badge search-badge" aria-label="Branch">branch</span>
                      )}
                    </div>
                    <div className="search-result-meta">
                      {isTitle ? (
                        <span className="search-role-badge title">Title</span>
                      ) : isAssistant ? (
                        <span className="search-role-badge assistant">
                          <Bot size={11} aria-hidden="true" /> Assistant
                        </span>
                      ) : (
                        <span className="search-role-badge user">
                          <User size={11} aria-hidden="true" /> User
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="search-result-snippet">
                    {highlightMatches(result.snippet, query).map((segment, segIdx) =>
                      segment.isMatch ? (
                        <mark key={segIdx} className="search-match-text">
                          {segment.text}
                        </mark>
                      ) : (
                        <span key={segIdx}>{segment.text}</span>
                      )
                    )}
                  </div>

                  {isSelected && (
                    <div className="search-result-hint" aria-hidden="true">
                      <span>Jump to message</span>
                      <CornerDownLeft size={12} />
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer shortcuts helper */}
        <div className="search-footer">
          <span className="search-footer-hint">
            <kbd className="search-kbd">↑</kbd> <kbd className="search-kbd">↓</kbd> Navigate
          </span>
          <span className="search-footer-hint">
            <kbd className="search-kbd">↵</kbd> Select
          </span>
          <span className="search-footer-hint">
            <kbd className="search-kbd">Esc</kbd> Close
          </span>
        </div>
      </div>
    </div>
  );
}
