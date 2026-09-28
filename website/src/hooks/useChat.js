import { useState, useCallback, useEffect, useRef } from "react";
import { api } from "../services/api";
import { WebSocketClient } from "../services/websocket";
import {
  loadConversationsFromStorage,
  saveConversationsToStorage,
  clearConversationsFromStorage,
} from "../utils/storage";

export function useChat(settings) {
  const [initialStorage] = useState(loadConversationsFromStorage);
  const [conversations, setConversations] = useState(initialStorage.conversations);
  const [activeConversation, setActiveConversation] = useState(initialStorage.activeConversationId);
  const [messages, setMessages] = useState(initialStorage.messages);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  // Track ALL currently-streaming conversation IDs. setStreaming(false) only fires
  // when the Set becomes empty — fixes concurrent-streaming bug (#253).
  const streamingConversationIdsRef = useRef(new Set());
  // Track ALL in-flight REST generation IDs (#354).
  const loadingConversationIdsRef = useRef(new Set());
  // Track active in-flight assistant message ID per conversation (#366)
  const activeAssistantIdByConvRef = useRef(new Map());
  // Track completed assistant message IDs to guarantee idempotent completion handling (#366)
  const completedMessageIdsRef = useRef(new Set());

  const findTargetAssistantIndex = (msgs, targetMsgId, convId) => {
    if (!Array.isArray(msgs) || msgs.length === 0) return -1;
    if (targetMsgId) {
      const idx = msgs.findIndex((m) => m.id === targetMsgId);
      if (idx !== -1) return idx;
    }
    const inFlightId = convId ? activeAssistantIdByConvRef.current.get(convId) : null;
    if (inFlightId) {
      const idx = msgs.findIndex((m) => m.id === inFlightId);
      if (idx !== -1) return idx;
    }
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i]?.role === "assistant" && !msgs[i]?.completed) {
        return i;
      }
    }
    return -1;
  };
  const [loadingConversations, setLoadingConversations] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [error, setError] = useState(null); // global banner error
  const [pendingApproval, setPendingApproval] = useState(null);
  const [denyEverything, setDenyEverything] = useState(false);
  const wsRef = useRef(null);

  // Add a conversation to the active-streaming set; flip global streaming on when first one starts.
  const markStreamingStart = (convId) => {
    if (convId) streamingConversationIdsRef.current.add(convId);
    setStreaming(true);
  };

  // Remove a conversation from the active-streaming set; flip global streaming off only when empty.
  const markStreamingEnd = (convId) => {
    if (convId) streamingConversationIdsRef.current.delete(convId);
    if (streamingConversationIdsRef.current.size === 0) {
      setStreaming(false);
    }
  };

  const markLoadingStart = (convId) => {
    if (convId) loadingConversationIdsRef.current.add(convId);
    setLoading(true);
  };

  const markLoadingEnd = (convId) => {
    if (convId) loadingConversationIdsRef.current.delete(convId);
    if (loadingConversationIdsRef.current.size === 0) {
      setLoading(false);
    }
  };

  // Track active conversation and conversations in refs so async handlers see current values
  const activeConversationRef = useRef(activeConversation);
  activeConversationRef.current = activeConversation;
  useEffect(() => {
    activeConversationRef.current = activeConversation;
  }, [activeConversation]);

  const conversationsRef = useRef(conversations);
  conversationsRef.current = conversations;

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

    ws.on("approval_request", (data) => {
      setPendingApproval({
        approvalId: data.approvalId,
        conversationId: data.conversationId,
        command: data.command,
        argv: data.argv,
        root: data.root,
      });
    });

    ws.on("stream", (data) => {
      // ISSUE #241 & #354 FIX: Route stream frames to the originating conversation.
      // Update the target conversation's state even if it's not currently active.
      const inFlightConvId = streamingConversationIdsRef.current.size === 1
        ? Array.from(streamingConversationIdsRef.current)[0]
        : null;
      const targetConvId = data.conversationId || inFlightConvId || activeConversationRef.current;
      const isActiveConv = Boolean(targetConvId && targetConvId === activeConversationRef.current);
      const targetMsgId = data.messageId || data.id || (targetConvId ? activeAssistantIdByConvRef.current.get(targetConvId) : null);

      // If this message has already completed, ignore duplicate completion/stream events safely (#366)
      if (targetMsgId && completedMessageIdsRef.current.has(targetMsgId)) {
        if (data.done && targetConvId) {
          markStreamingEnd(targetConvId);
        }
        return;
      }

      if (data.type === "error") {
        // Server sent an error event mid-stream
        // Update the correct conversation's messages
        if (targetConvId) markStreamingEnd(targetConvId);
        const applyError = (msgs) => {
          const idx = findTargetAssistantIndex(msgs, targetMsgId, targetConvId);
          if (idx === -1) return msgs;
          const target = msgs[idx];
          if (target.completed) return msgs;
          const updated = [...msgs];
          const newMsg = {
            ...target,
            content: data.message || "An error occurred while generating the response.",
            type: "error",
            completed: true,
          };
          if (newMsg.id) completedMessageIdsRef.current.add(newMsg.id);
          if (targetConvId) activeAssistantIdByConvRef.current.delete(targetConvId);
          updated[idx] = newMsg;
          return updated;
        };

        if (isActiveConv) {
          setMessages(applyError);
        } else if (targetConvId) {
          setConversations((prev) =>
            prev.map((c) => (c.id === targetConvId ? { ...c, messages: applyError(c.messages || []) } : c))
          );
        }
        return;
      }

      // Handle audio streaming chunks
      if (data.responseType === "audio") {
        if (data.done) {
          if (targetConvId) markStreamingEnd(targetConvId);
          const applyAudioDone = (msgs) => {
            const idx = findTargetAssistantIndex(msgs, targetMsgId, targetConvId);
            if (idx === -1) return msgs;
            const target = msgs[idx];
            if (target.completed || target.audioComplete) return msgs;
            const updated = [...msgs];
            const newMsg = {
              ...target,
              type: "audio",
              content: data.content || target.content,
              audioChunks: data.metadata?.chunkData
                ? [...(target.audioChunks || []), data.metadata.chunkData]
                : target.audioChunks,
              metadata: data.metadata,
              audioComplete: true,
              completed: true,
            };
            if (newMsg.id) completedMessageIdsRef.current.add(newMsg.id);
            if (targetConvId) activeAssistantIdByConvRef.current.delete(targetConvId);
            updated[idx] = newMsg;
            return updated;
          };

          if (isActiveConv) {
            setMessages(applyAudioDone);
          } else if (targetConvId) {
            setConversations((prev) =>
              prev.map((c) => (c.id === targetConvId ? { ...c, messages: applyAudioDone(c.messages || []) } : c))
            );
          }
        } else {
          // Accumulate audio chunks
          const applyAudioChunk = (msgs) => {
            const idx = findTargetAssistantIndex(msgs, targetMsgId, targetConvId);
            if (idx === -1) return msgs;
            const target = msgs[idx];
            if (target.completed || target.audioComplete) return msgs;
            const updated = [...msgs];
            const newMsg = {
              ...target,
              type: "audio",
              content: data.content || target.content,
              audioChunks: [...(target.audioChunks || []), data.metadata.chunkData],
              audioMetadata: {
                sampleRate: data.metadata?.sampleRate,
                channels: data.metadata?.channels,
                bitsPerSample: data.metadata?.bitsPerSample,
                totalChunks: data.metadata?.totalChunks,
              },
            };
            updated[idx] = newMsg;
            return updated;
          };

          if (isActiveConv) {
            setMessages(applyAudioChunk);
          } else if (targetConvId) {
            setConversations((prev) =>
              prev.map((c) => (c.id === targetConvId ? { ...c, messages: applyAudioChunk(c.messages || []) } : c))
            );
          }
        }
        return;
      }

      if (data.done) {
        if (targetConvId) markStreamingEnd(targetConvId);
        const applyDone = (msgs) => {
          const idx = findTargetAssistantIndex(msgs, targetMsgId, targetConvId);
          if (idx === -1) return msgs;
          const target = msgs[idx];
          if (target.completed) return msgs;
          const updated = [...msgs];
          const newMsg = {
            ...target,
            content: data.content !== undefined ? data.content : target.content,
            type: data.responseType || target.type || "text",
            metadata: data.metadata !== undefined ? data.metadata : target.metadata,
            completed: true,
          };
          if (newMsg.id) completedMessageIdsRef.current.add(newMsg.id);
          if (targetConvId) activeAssistantIdByConvRef.current.delete(targetConvId);
          updated[idx] = newMsg;
          return updated;
        };

        if (isActiveConv) {
          setMessages(applyDone);
        } else if (targetConvId) {
          setConversations((prev) =>
            prev.map((c) =>
              c.id === targetConvId
                ? { ...c, messages: applyDone(c.messages || []), updatedAt: new Date().toISOString() }
                : c
            )
          );
        }
      } else {
        const applyChunk = (msgs) => {
          const idx = findTargetAssistantIndex(msgs, targetMsgId, targetConvId);
          if (idx === -1) return msgs;
          const target = msgs[idx];
          if (target.completed) return msgs;
          const updated = [...msgs];
          updated[idx] = { ...target, content: data.content };
          return updated;
        };

        if (isActiveConv) {
          setMessages(applyChunk);
        } else if (targetConvId) {
          setConversations((prev) =>
            prev.map((c) => (c.id === targetConvId ? { ...c, messages: applyChunk(c.messages || []) } : c))
          );
        }
      }
    });

    ws.on("typing", (data) => {
      const targetConvId = data?.conversationId;
      const isActiveConv = !targetConvId || targetConvId === activeConversationRef.current;

      if (isActiveConv) {
        markStreamingStart(targetConvId || activeConversationRef.current);
      }
    });

    // Server-side error frame, for example a rate limit rejection. Without
    // this the placeholder bubble would sit there empty with no explanation.
    ws.on("error", (data) => {
      const inFlightConvId = streamingConversationIdsRef.current.size === 1
        ? Array.from(streamingConversationIdsRef.current)[0]
        : null;
      const targetConvId = data?.conversationId || inFlightConvId || activeConversationRef.current;
      const isActiveConv = Boolean(targetConvId && targetConvId === activeConversationRef.current);
      const targetMsgId = data?.messageId || data?.id || (targetConvId ? activeAssistantIdByConvRef.current.get(targetConvId) : null);

      if (targetConvId) markStreamingEnd(targetConvId);

      const applyWsError = (msgs) => {
        const idx = findTargetAssistantIndex(msgs, targetMsgId, targetConvId);
        if (idx === -1) return msgs;
        const target = msgs[idx];
        if (target.completed || target.content) return msgs;
        const updated = [...msgs];
        const newMsg = {
          ...target,
          content: data?.message || "Something went wrong. Please try again.",
          type: "error",
          completed: true,
        };
        if (newMsg.id) completedMessageIdsRef.current.add(newMsg.id);
        if (targetConvId) activeAssistantIdByConvRef.current.delete(targetConvId);
        updated[idx] = newMsg;
        return updated;
      };

      if (isActiveConv) {
        setMessages(applyWsError);
      } else if (targetConvId) {
        setConversations((prev) =>
          prev.map((c) => (c.id === targetConvId ? { ...c, messages: applyWsError(c.messages || []) } : c))
        );
      }
    });

    // WebSocket closed unexpectedly (network drop, server restart, etc.).
    // Any conversation whose stream was in flight will never receive a done/error
    // frame, so we must drain those IDs from the Set now — otherwise the
    // composer stays disabled until the page is reloaded.
    // Additionally, any empty assistant placeholder must be converted to an
    // error state so the user knows the request did not complete (#332).
    ws.on("close", () => {
      const orphaned = [...streamingConversationIdsRef.current];
      for (const convId of orphaned) {
        const inFlightId = activeAssistantIdByConvRef.current.get(convId);
        const applyClose = (msgs) => {
          const idx = findTargetAssistantIndex(msgs, inFlightId, convId);
          if (idx === -1) return msgs;
          const target = msgs[idx];
          if (target.completed || target.content) return msgs;
          const updated = [...msgs];
          const newMsg = {
            ...target,
            content: "Connection lost. Please retry.",
            type: "error",
            completed: true,
          };
          if (newMsg.id) completedMessageIdsRef.current.add(newMsg.id);
          activeAssistantIdByConvRef.current.delete(convId);
          updated[idx] = newMsg;
          return updated;
        };
        if (convId === activeConversationRef.current) {
          setMessages(applyClose);
        } else {
          setConversations((prev) =>
            prev.map((c) => (c.id === convId ? { ...c, messages: applyClose(c.messages || []) } : c))
          );
        }
        markStreamingEnd(convId);
      }
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
      activeConversationRef.current = newConv.id;
      setConversations((prev) => [newConv, ...prev]);
      setActiveConversation(newConv.id);
      setMessages([]);
    } catch {
      // Offline fallback — create locally and let the user keep working
      const id = crypto.randomUUID();
      const conv = { id, title: "New Chat", messages: [], updatedAt: new Date().toISOString() };
      activeConversationRef.current = id;
      setConversations((prev) => [conv, ...prev]);
      setActiveConversation(id);
      setMessages([]);
    }
  }, []);

  const selectConversation = useCallback(async (id) => {
    activeConversationRef.current = id;
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
        const seen = new Set();
        const deduped = conv.messages.filter((m) => {
          if (!m?.id) return true;
          if (seen.has(m.id)) return false;
          seen.add(m.id);
          return true;
        });
        if (activeConversationRef.current === id) {
          setMessages(deduped);
        }
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? { ...c, messages: deduped, title: conv.title || c.title } : c))
        );
      }
    } catch (err) {
      if (activeConversationRef.current === id) {
        setConversations((prev) => {
          const f = prev.find((c) => c.id === id);
          if (!f || !f.messages || f.messages.length === 0) {
            setError(`Could not load conversation: ${err.message}`);
          }
          return prev;
        });
      }
    } finally {
      if (activeConversationRef.current === id) {
        setLoadingMessages(false);
      }
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
      // If this conversation was actively streaming or loading, remove it from the Sets
      markStreamingEnd(id);
      markLoadingEnd(id);
      activeAssistantIdByConvRef.current.delete(id);
      setConversations((prev) => {
        const remaining = prev.filter((c) => c.id !== id);
        if (activeConversationRef.current === id) {
          const nextConv = remaining[0] || null;
          const nextId = nextConv ? nextConv.id : null;
          const nextMsgs = nextConv ? nextConv.messages || [] : [];
          activeConversationRef.current = nextId;
          setActiveConversation(nextId);
          setMessages(nextMsgs);
          saveConversationsToStorage(remaining, nextId);
        } else {
          saveConversationsToStorage(remaining, activeConversationRef.current);
        }
        return remaining;
      });
    },
    []
  );

  const clearAllConversations = useCallback(() => {
    setError(null);
    streamingConversationIdsRef.current.clear();
    setStreaming(false);
    loadingConversationIdsRef.current.clear();
    setLoading(false);
    activeAssistantIdByConvRef.current.clear();
    completedMessageIdsRef.current.clear();
    activeConversationRef.current = null;
    setConversations([]);
    setActiveConversation(null);
    setMessages([]);
    clearConversationsFromStorage();
  }, []);

  const dismissError = useCallback(() => setError(null), []);

  const sendMessage = useCallback(
    async (content, type = "text", attachments = []) => {
      if (!content.trim()) return;
      setError(null);

      let convId = activeConversationRef.current || activeConversation;
      if (!convId) {
        try {
          const conv = await api.createConversation();
          convId = conv.id;
          const newConv = { ...conv, messages: conv.messages || [] };
          activeConversationRef.current = convId;
          setConversations((prev) => [newConv, ...prev]);
          setActiveConversation(convId);
        } catch {
          convId = crypto.randomUUID();
          const conv = { id: convId, title: content.slice(0, 30) || "New Chat", messages: [] };
          activeConversationRef.current = convId;
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
        attachments,
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
      activeAssistantIdByConvRef.current.set(convId, assistantMsg.id);
      setMessages((prev) => [...prev, assistantMsg]);

      // Try WebSocket streaming first
      if (wsRef.current?.ws?.readyState === WebSocket.OPEN) {
        markStreamingStart(convId);
        wsRef.current.send({
          type: "chat",
          content,
          conversationId: convId,
          messageType: type,
          attachments,
          settings: settingsRef.current,
        });
        return;
      }

      markLoadingStart(convId);
      try {
        const response = await api.sendMessage(convId, content, type, attachments, settingsRef.current);
        const isActiveConv = convId === activeConversationRef.current;
        completedMessageIdsRef.current.add(assistantMsg.id);
        activeAssistantIdByConvRef.current.delete(convId);

        const applyRestSuccess = (msgs) => {
          const idx = msgs.findIndex((m) => m.id === assistantMsg.id);
          const targetIdx = idx !== -1 ? idx : (msgs[msgs.length - 1]?.role === "assistant" && !msgs[msgs.length - 1]?.completed ? msgs.length - 1 : -1);
          if (targetIdx === -1) return msgs;
          if (msgs[targetIdx].completed) return msgs;
          const updated = [...msgs];
          updated[targetIdx] = {
            ...updated[targetIdx],
            ...assistantMsg,
            content: response.content,
            type: response.type,
            metadata: response.metadata,
            completed: true,
          };
          return updated;
        };

        if (isActiveConv) {
          setMessages(applyRestSuccess);
        } else {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              return { ...c, messages: applyRestSuccess(c.messages || []), updatedAt: new Date().toISOString() };
            })
          );
        }
      } catch (err) {
        const isActiveConv = convId === activeConversationRef.current;
        completedMessageIdsRef.current.add(assistantMsg.id);
        activeAssistantIdByConvRef.current.delete(convId);

        const applyRestError = (msgs) => {
          const idx = msgs.findIndex((m) => m.id === assistantMsg.id);
          const targetIdx = idx !== -1 ? idx : (msgs[msgs.length - 1]?.role === "assistant" && !msgs[msgs.length - 1]?.completed ? msgs.length - 1 : -1);
          if (targetIdx === -1) return msgs;
          if (msgs[targetIdx].completed) return msgs;
          const updated = [...msgs];
          updated[targetIdx] = {
            ...updated[targetIdx],
            ...assistantMsg,
            content: err.message || "Something went wrong. Please try again.",
            type: "error",
            completed: true,
          };
          return updated;
        };

        if (isActiveConv) {
          setMessages(applyRestError);
        } else {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              return { ...c, messages: applyRestError(c.messages || []) };
            })
          );
        }
      } finally {
        markLoadingEnd(convId);
      }
    },
    [activeConversation]
  );

  const approveCommand = useCallback((approvalId) => {
    if (wsRef.current) {
      wsRef.current.sendApprovalResponse(approvalId, true, false);
    }
    setPendingApproval((prev) => (prev?.approvalId === approvalId ? null : prev));
  }, []);

  const denyCommand = useCallback((approvalId, shouldDenyEverything = false) => {
    if (shouldDenyEverything) {
      setDenyEverything(true);
    }
    if (wsRef.current) {
      wsRef.current.sendApprovalResponse(approvalId, false, shouldDenyEverything);
    }
    setPendingApproval((prev) => (prev?.approvalId === approvalId ? null : prev));
  }, []);

  return {
    conversations,
    activeConversation,
    messages,
    loading,
    streaming,
    loadingConversations,
    loadingMessages,
    error,
    pendingApproval,
    denyEverything,
    createConversation,
    selectConversation,
    deleteConversation,
    clearAllConversations,
    sendMessage,
    dismissError,
    approveCommand,
    denyCommand,
  };
}
