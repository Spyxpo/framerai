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

      if (data.type === "error") {
        // Server sent an error event mid-stream
        // Update the correct conversation's messages
        if (isActiveConv) {
          markStreamingEnd(targetConvId);
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant") {
              const newLast = {
                ...last,
                content: data.message || "An error occurred while generating the response.",
                type: "error",
              };
              updated[updated.length - 1] = newLast;
            }
            return updated;
          });
        } else if (targetConvId) {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== targetConvId) return c;
              const msgs = c.messages || [];
              const updated = [...msgs];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                const newLast = {
                  ...last,
                  content: data.message || "An error occurred while generating the response.",
                  type: "error",
                };
                updated[updated.length - 1] = newLast;
              }
              return { ...c, messages: updated };
            })
          );

          // Stop streaming for this specific conversation
          if (targetConvId) markStreamingEnd(targetConvId);
        }
        return;
      }

      // Handle audio streaming chunks
      if (data.responseType === "audio") {
        if (data.done) {
          if (isActiveConv) {
            markStreamingEnd(targetConvId);
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                const newLast = {
                  ...last,
                  type: "audio",
                  content: data.content || last.content,
                  audioChunks: data.metadata?.chunkData
                    ? [...(last.audioChunks || []), data.metadata.chunkData]
                    : last.audioChunks,
                  metadata: data.metadata,
                  audioComplete: true,
                };
                updated[updated.length - 1] = newLast;
              }
              return updated;
            });
          } else if (targetConvId) {
            setConversations((prev) =>
              prev.map((c) => {
                if (c.id !== targetConvId) return c;
                const msgs = c.messages || [];
                const updated = [...msgs];
                const last = updated[updated.length - 1];
                if (last?.role === "assistant") {
                  const newLast = {
                    ...last,
                    type: "audio",
                    content: data.content || last.content,
                    audioChunks: data.metadata?.chunkData
                      ? [...(last.audioChunks || []), data.metadata.chunkData]
                      : last.audioChunks,
                    metadata: data.metadata,
                    audioComplete: true,
                  };
                  updated[updated.length - 1] = newLast;
                }
                return { ...c, messages: updated };
              })
            );

            if (targetConvId) markStreamingEnd(targetConvId);
          }
        } else {
          // Accumulate audio chunks
          if (isActiveConv) {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                const newLast = {
                  ...last,
                  type: "audio",
                  content: data.content || last.content,
                  audioChunks: [...(last.audioChunks || []), data.metadata.chunkData],
                  audioMetadata: {
                    sampleRate: data.metadata.sampleRate,
                    channels: data.metadata.channels,
                    bitsPerSample: data.metadata.bitsPerSample,
                    totalChunks: data.metadata.totalChunks,
                  },
                };
                updated[updated.length - 1] = newLast;
              }
              return updated;
            });
          } else if (targetConvId) {
            setConversations((prev) =>
              prev.map((c) => {
                if (c.id !== targetConvId) return c;
                const msgs = c.messages || [];
                const updated = [...msgs];
                const last = updated[updated.length - 1];
                if (last?.role === "assistant") {
                  const newLast = {
                    ...last,
                    type: "audio",
                    content: data.content || last.content,
                    audioChunks: [...(last.audioChunks || []), data.metadata.chunkData],
                    audioMetadata: {
                      sampleRate: data.metadata.sampleRate,
                      channels: data.metadata.channels,
                      bitsPerSample: data.metadata.bitsPerSample,
                      totalChunks: data.metadata.totalChunks,
                    },
                  };
                  updated[updated.length - 1] = newLast;
                }
                return { ...c, messages: updated };
              })
            );
          }
        }
        return;
      }

      if (data.done) {
        if (isActiveConv) {
          markStreamingEnd(targetConvId);
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant") {
              const newLast = {
                ...last,
                content: data.content,
                type: data.responseType || "text",
                metadata: data.metadata,
              };
              updated[updated.length - 1] = newLast;
            }
            return updated;
          });
        } else if (targetConvId) {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== targetConvId) return c;
              const msgs = c.messages || [];
              const updated = [...msgs];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                const newLast = {
                  ...last,
                  content: data.content,
                  type: data.responseType || "text",
                  metadata: data.metadata,
                };
                updated[updated.length - 1] = newLast;
              }
              return { ...c, messages: updated, updatedAt: new Date().toISOString() };
            })
          );

          if (targetConvId) markStreamingEnd(targetConvId);
        }
      } else {
        if (isActiveConv) {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant") {
              const newLast = { ...last, content: data.content };
              updated[updated.length - 1] = newLast;
            }
            return updated;
          });
        } else if (targetConvId) {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== targetConvId) return c;
              const msgs = c.messages || [];
              const updated = [...msgs];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                const newLast = { ...last, content: data.content };
                updated[updated.length - 1] = newLast;
              }
              return { ...c, messages: updated };
            })
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

      if (isActiveConv) {
        markStreamingEnd(targetConvId);
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.role === "assistant" && !last.content) {
            const newLast = {
              ...last,
              content: data?.message || "Something went wrong. Please try again.",
              type: "error",
            };
            updated[updated.length - 1] = newLast;
          }
          return updated;
        });
      } else if (targetConvId) {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== targetConvId) return c;
            const msgs = c.messages || [];
            const updated = [...msgs];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant" && !last.content) {
              const newLast = {
                ...last,
                content: data?.message || "Something went wrong. Please try again.",
                type: "error",
              };
              updated[updated.length - 1] = newLast;
            }
            return { ...c, messages: updated };
          })
        );

        if (targetConvId) markStreamingEnd(targetConvId);
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
        if (convId === activeConversationRef.current) {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant" && !last.content) {
              updated[updated.length - 1] = {
                ...last,
                content: "Connection lost. Please retry.",
                type: "error",
              };
            }
            return updated;
          });
        } else {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              const msgs = c.messages || [];
              const updated = [...msgs];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant" && !last.content) {
                updated[updated.length - 1] = {
                  ...last,
                  content: "Connection lost. Please retry.",
                  type: "error",
                };
              }
              return { ...c, messages: updated };
            })
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
        if (activeConversationRef.current === id) {
          setMessages(conv.messages);
        }
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? { ...c, messages: conv.messages, title: conv.title || c.title } : c))
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
        if (isActiveConv) {
          setMessages((prev) => {
            const updated = [...prev];
            const idx = updated.findIndex((m) => m.id === assistantMsg.id);
            const targetIdx = idx !== -1 ? idx : (updated[updated.length - 1]?.role === "assistant" ? updated.length - 1 : -1);
            if (targetIdx >= 0) {
              updated[targetIdx] = {
                ...updated[targetIdx],
                ...assistantMsg,
                content: response.content,
                type: response.type,
                metadata: response.metadata,
              };
            }
            return updated;
          });
        } else {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              const msgs = c.messages || [];
              const updated = [...msgs];
              const idx = updated.findIndex((m) => m.id === assistantMsg.id);
              const targetIdx = idx !== -1 ? idx : (updated[updated.length - 1]?.role === "assistant" ? updated.length - 1 : -1);
              if (targetIdx >= 0) {
                updated[targetIdx] = {
                  ...updated[targetIdx],
                  ...assistantMsg,
                  content: response.content,
                  type: response.type,
                  metadata: response.metadata,
                };
              }
              return { ...c, messages: updated, updatedAt: new Date().toISOString() };
            })
          );
        }
      } catch (err) {
        const isActiveConv = convId === activeConversationRef.current;
        if (isActiveConv) {
          setMessages((prev) => {
            const updated = [...prev];
            const idx = updated.findIndex((m) => m.id === assistantMsg.id);
            const targetIdx = idx !== -1 ? idx : (updated[updated.length - 1]?.role === "assistant" ? updated.length - 1 : -1);
            if (targetIdx >= 0) {
              updated[targetIdx] = {
                ...updated[targetIdx],
                ...assistantMsg,
                content: err.message || "Something went wrong. Please try again.",
                type: "error",
              };
            }
            return updated;
          });
        } else {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              const msgs = c.messages || [];
              const updated = [...msgs];
              const idx = updated.findIndex((m) => m.id === assistantMsg.id);
              const targetIdx = idx !== -1 ? idx : (updated[updated.length - 1]?.role === "assistant" ? updated.length - 1 : -1);
              if (targetIdx >= 0) {
                updated[targetIdx] = {
                  ...updated[targetIdx],
                  ...assistantMsg,
                  content: err.message || "Something went wrong. Please try again.",
                  type: "error",
                };
              }
              return { ...c, messages: updated };
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
