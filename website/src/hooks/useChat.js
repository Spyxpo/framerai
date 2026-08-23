import { useState, useCallback, useEffect, useRef } from "react";
import { api } from "../services/api";
import { WebSocketClient } from "../services/websocket";
import {
  loadConversationsFromStorage,
  saveConversationsToStorage,
  clearConversationsFromStorage,
} from "../utils/storage";

export function useChat(settings) {
  const [conversations, setConversations] = useState(() => {
    const stored = loadConversationsFromStorage();
    return stored.conversations;
  });
  const [activeConversation, setActiveConversation] = useState(() => {
    const stored = loadConversationsFromStorage();
    return stored.activeConversationId;
  });
  const [messages, setMessages] = useState(() => {
    const stored = loadConversationsFromStorage();
    return stored.messages;
  });
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [loadingConversations, setLoadingConversations] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [error, setError] = useState(null); // global banner error
  const wsRef = useRef(null);

  // Read through a ref so sendMessage always sees the current settings without
  // being rebuilt every time a slider moves.
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  // Keep active conversation's messages updated in conversations list
  useEffect(() => {
    if (!activeConversation) return;
    setConversations((prev) => {
      const idx = prev.findIndex((c) => c.id === activeConversation);
      if (idx === -1) return prev;
      const currentConv = prev[idx];
      if (currentConv.messages === messages) return prev;

      const updatedConv = {
        ...currentConv,
        messages,
        updatedAt: new Date().toISOString(),
      };
      const updatedList = [...prev];
      updatedList[idx] = updatedConv;
      return updatedList;
    });
  }, [activeConversation, messages]);

  // Persist state to localStorage whenever conversations or activeConversation updates
  useEffect(() => {
    saveConversationsToStorage(conversations, activeConversation);
  }, [conversations, activeConversation]);

  // Initialize WebSocket
  useEffect(() => {
    const ws = new WebSocketClient();
    wsRef.current = ws;

    ws.connect()
      .then(() => {})
      .catch(() => {
        // Non-fatal: REST fallback will be used. No banner needed.
      });

    ws.on("stream", (data) => {
      if (data.type === "error") {
        // Server sent an error event mid-stream
        setStreaming(false);
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.role === "assistant") {
            last.content = data.message || "An error occurred while generating the response.";
            last.type = "error";
          }
          return [...updated];
        });
        return;
      }

      // Handle audio streaming chunks
      if (data.responseType === "audio") {
        if (data.done) {
          setStreaming(false);
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant") {
              last.type = "audio";
              last.content = data.content || last.content;
              // Push final chunk data if present (create new array for reactivity)
              if (data.metadata?.chunkData) {
                const existingChunks = last.audioChunks || [];
                last.audioChunks = [...existingChunks, data.metadata.chunkData];
              }
              last.metadata = data.metadata;
              last.audioComplete = true;
            }
            return [...updated];
          });
        } else {
          // Accumulate audio chunks (create new array for reactivity)
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant") {
              last.type = "audio";
              last.content = data.content || last.content;
              const existingChunks = last.audioChunks || [];
              last.audioChunks = [...existingChunks, data.metadata.chunkData];
              last.audioMetadata = {
                sampleRate: data.metadata.sampleRate,
                channels: data.metadata.channels,
                bitsPerSample: data.metadata.bitsPerSample,
                totalChunks: data.metadata.totalChunks,
              };
            }
            return [...updated];
          });
        }
        return;
      }

      if (data.done) {
        setStreaming(false);
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.role === "assistant") {
            last.content = data.content;
            last.type = data.responseType || "text";
            last.metadata = data.metadata;
          }
          return [...updated];
        });
      } else {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.role === "assistant") {
            last.content = data.content;
          }
          return [...updated];
        });
      }
    });

    ws.on("typing", () => setStreaming(true));

    // Server-side error frame, for example a rate limit rejection. Without
    // this the placeholder bubble would sit there empty with no explanation.
    ws.on("error", (data) => {
      setStreaming(false);
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "assistant" && !last.content) {
          last.content = data?.message || "Something went wrong. Please try again.";
          last.type = "error";
        }
        return [...updated];
      });
    });

    return () => ws.disconnect();
  }, []);

  // Load conversations on mount
  useEffect(() => {
    setLoadingConversations(true);
    api
      .listConversations()
      .then((remoteConvs) => {
        if (Array.isArray(remoteConvs) && remoteConvs.length > 0) {
          setConversations((prev) => {
            if (prev.length === 0) return remoteConvs;
            const existingIds = new Set(prev.map((c) => c.id));
            const newFromRemote = remoteConvs.filter((c) => !existingIds.has(c.id));
            return [...prev, ...newFromRemote];
          });
        }
      })
      .catch(() => {
        setConversations((prev) => {
          if (prev.length === 0) {
            setError("Unable to load conversations. Make sure the backend is running.");
          }
          return prev;
        });
      })
      .finally(() => setLoadingConversations(false));
  }, []);

  const createConversation = useCallback(async () => {
    setError(null);
    try {
      const conv = await api.createConversation();
      const newConv = { ...conv, messages: conv.messages || [] };
      setConversations((prev) => [newConv, ...prev]);
      setActiveConversation(newConv.id);
      setMessages([]);
    } catch {
      // Offline fallback — create locally and let the user keep working
      const id = crypto.randomUUID();
      const conv = { id, title: "New Chat", messages: [], updatedAt: new Date().toISOString() };
      setConversations((prev) => [conv, ...prev]);
      setActiveConversation(id);
      setMessages([]);
    }
  }, []);

  const selectConversation = useCallback(async (id) => {
    setActiveConversation(id);
    setError(null);

    // Set messages immediately from local state if available
    setConversations((prev) => {
      const found = prev.find((c) => c.id === id);
      if (found && Array.isArray(found.messages)) {
        setMessages(found.messages);
      }
      return prev;
    });

    setLoadingMessages(true);
    try {
      const conv = await api.getConversation(id);
      if (conv && Array.isArray(conv.messages)) {
        setMessages(conv.messages);
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? { ...c, messages: conv.messages, title: conv.title || c.title } : c))
        );
      }
    } catch (err) {
      setConversations((prev) => {
        const found = prev.find((c) => c.id === id);
        if (!found || !found.messages || found.messages.length === 0) {
          setError(`Could not load conversation: ${err.message}`);
        }
        return prev;
      });
    } finally {
      setLoadingMessages(false);
    }
  }, []);

  const deleteConversation = useCallback(
    async (id) => {
      setError(null);
      try {
        await api.deleteConversation(id);
      } catch {
        // Continue anyway — remove from local list regardless
      }
      setConversations((prev) => {
        const remaining = prev.filter((c) => c.id !== id);
        if (activeConversation === id) {
          const nextConv = remaining[0] || null;
          const nextId = nextConv ? nextConv.id : null;
          const nextMsgs = nextConv ? nextConv.messages || [] : [];
          setActiveConversation(nextId);
          setMessages(nextMsgs);
          saveConversationsToStorage(remaining, nextId);
        } else {
          saveConversationsToStorage(remaining, activeConversation);
        }
        return remaining;
      });
    },
    [activeConversation]
  );

  const clearAllConversations = useCallback(() => {
    setError(null);
    setConversations([]);
    setActiveConversation(null);
    setMessages([]);
    clearConversationsFromStorage();
  }, []);

  const dismissError = useCallback(() => setError(null), []);

  const sendMessage = useCallback(
    async (content, type = "text") => {
      if (!content.trim()) return;
      setError(null);

      let convId = activeConversation;
      if (!convId) {
        try {
          const conv = await api.createConversation();
          convId = conv.id;
          const newConv = { ...conv, messages: conv.messages || [] };
          setConversations((prev) => [newConv, ...prev]);
          setActiveConversation(convId);
        } catch {
          convId = crypto.randomUUID();
          const conv = { id: convId, title: content.slice(0, 30) || "New Chat", messages: [] };
          setConversations((prev) => [conv, ...prev]);
          setActiveConversation(convId);
        }
      } else {
        setConversations((prev) =>
          prev.map((c) =>
            c.id === convId && (c.title === "New Chat" || !c.title)
              ? { ...c, title: content.length > 30 ? content.slice(0, 30) + "..." : content }
              : c
          )
        );
      }

      const userMsg = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        type,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);

      const assistantMsg = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        type: "text",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Try WebSocket streaming first
      if (wsRef.current?.ws?.readyState === WebSocket.OPEN) {
        setStreaming(true);
        wsRef.current.send({
          type: "chat",
          content,
          conversationId: convId,
          messageType: type,
          settings: settingsRef.current,
        });
        return;
      }

      // Fallback to REST API
      setLoading(true);
      try {
        const response = await api.sendMessage(convId, content, type, [], settingsRef.current);
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...assistantMsg,
            content: response.content,
            type: response.type,
            metadata: response.metadata,
          };
          return updated;
        });
      } catch (err) {
        // Write the error into the assistant message bubble so context is preserved
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...assistantMsg,
            content: err.message || "Something went wrong. Please try again.",
            type: "error",
          };
          return updated;
        });
      } finally {
        setLoading(false);
      }
    },
    [activeConversation]
  );

  return {
    conversations,
    activeConversation,
    messages,
    loading,
    streaming,
    loadingConversations,
    loadingMessages,
    error,
    createConversation,
    selectConversation,
    deleteConversation,
    clearAllConversations,
    sendMessage,
    dismissError,
  };
}
