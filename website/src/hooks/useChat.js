import { useState, useCallback, useEffect, useRef } from "react";
import { api } from "../services/api";
import { WebSocketClient } from "../services/websocket";

export function useChat() {
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const wsRef = useRef(null);

  // Initialize WebSocket
  useEffect(() => {
    const ws = new WebSocketClient();
    wsRef.current = ws;

    ws.connect().catch(() => {
      console.warn("WebSocket connection failed, falling back to REST API");
    });

    ws.on("stream", (data) => {
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
          return updated;
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

    return () => ws.disconnect();
  }, []);

  // Load conversations on mount
  useEffect(() => {
    api.listConversations().then(setConversations).catch(() => {});
  }, []);

  const createConversation = useCallback(async () => {
    try {
      const conv = await api.createConversation();
      setConversations((prev) => [conv, ...prev]);
      setActiveConversation(conv.id);
      setMessages([]);
    } catch {
      // Offline fallback
      const id = crypto.randomUUID();
      const conv = { id, title: "New Chat", messages: [] };
      setConversations((prev) => [conv, ...prev]);
      setActiveConversation(id);
      setMessages([]);
    }
  }, []);

  const selectConversation = useCallback(async (id) => {
    setActiveConversation(id);
    try {
      const conv = await api.getConversation(id);
      setMessages(conv.messages || []);
    } catch {
      setMessages([]);
    }
  }, []);

  const deleteConversation = useCallback(
    async (id) => {
      try {
        await api.deleteConversation(id);
      } catch {
        // continue anyway
      }
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConversation === id) {
        setActiveConversation(null);
        setMessages([]);
      }
    },
    [activeConversation]
  );

  const sendMessage = useCallback(
    async (content, type = "text") => {
      if (!content.trim()) return;

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
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...assistantMsg,
            content: `Error: ${err.message}. Make sure the backend is running.`,
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
    createConversation,
    selectConversation,
    deleteConversation,
    sendMessage,
  };
}
