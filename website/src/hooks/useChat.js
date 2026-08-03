import { useState, useCallback, useEffect, useRef } from "react";
import { api } from "../services/api";
import { WebSocketClient } from "../services/websocket";

export function useChat() {
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [loadingConversations, setLoadingConversations] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [error, setError] = useState(null); // global banner error
  const wsRef = useRef(null);

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

    ws.on("error", () => {
      setStreaming(false);
    });

    return () => ws.disconnect();
  }, []);

  // Load conversations on mount
  useEffect(() => {
    setLoadingConversations(true);
    api
      .listConversations()
      .then(setConversations)
      .catch(() => {
        setError("Unable to load conversations. Make sure the backend is running.");
      })
      .finally(() => setLoadingConversations(false));
  }, []);

  const createConversation = useCallback(async () => {
    setError(null);
    try {
      const conv = await api.createConversation();
      setConversations((prev) => [conv, ...prev]);
      setActiveConversation(conv.id);
      setMessages([]);
    } catch {
      // Offline fallback — create locally and let the user keep working
      const id = crypto.randomUUID();
      const conv = { id, title: "New Chat", messages: [] };
      setConversations((prev) => [conv, ...prev]);
      setActiveConversation(id);
      setMessages([]);
    }
  }, []);

  const selectConversation = useCallback(async (id) => {
    setActiveConversation(id);
    setError(null);
    setLoadingMessages(true);
    try {
      const conv = await api.getConversation(id);
      setMessages(conv.messages || []);
    } catch (err) {
      setMessages([]);
      setError(`Could not load conversation: ${err.message}`);
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
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConversation === id) {
        setActiveConversation(null);
        setMessages([]);
      }
    },
    [activeConversation]
  );

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
          setConversations((prev) => [conv, ...prev]);
          setActiveConversation(convId);
        } catch {
          convId = crypto.randomUUID();
          setActiveConversation(convId);
        }
      }

      const userMsg = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        type,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);

      // Add placeholder for assistant response
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
        });
        return;
      }

      // Fallback to REST API
      setLoading(true);
      try {
        const response = await api.sendMessage(convId, content, type);
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
    sendMessage,
    dismissError,
  };
}
