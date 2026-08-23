export const STORAGE_KEY = "framerai:conversations:v1";
export const STORAGE_VERSION = 1;
export const DEFAULT_MAX_BYTES = 2 * 1024 * 1024; // 2 MB

/**
 * Sanitizes a single message object, preserving standard serializable properties
 * while stripping transient audio/streaming chunks or non-serializable fields.
 */
export function sanitizeMessage(msg) {
  if (!msg || typeof msg !== "object") return null;
  const { id, role, content, type, timestamp, metadata } = msg;

  if (!id || typeof id !== "string") return null;
  if (!role || typeof role !== "string") return null;

  return {
    id,
    role,
    content: typeof content === "string" ? content : "",
    type: typeof type === "string" ? type : "text",
    timestamp: typeof timestamp === "string" ? timestamp : new Date().toISOString(),
    ...(metadata && typeof metadata === "object" ? { metadata } : {}),
  };
}

/**
 * Sanitizes a conversation object and its messages.
 */
export function sanitizeConversation(conv) {
  if (!conv || typeof conv !== "object") return null;
  const id = typeof conv.id === "string" ? conv.id : String(conv.id || "");
  if (!id) return null;

  const title = typeof conv.title === "string" && conv.title.trim() ? conv.title : "New Chat";
  const updatedAt = typeof conv.updatedAt === "string" ? conv.updatedAt : new Date().toISOString();

  const rawMessages = Array.isArray(conv.messages) ? conv.messages : [];
  const messages = rawMessages.map(sanitizeMessage).filter(Boolean);

  return {
    id,
    title,
    updatedAt,
    messages,
  };
}

/**
 * Safely reads and validates conversations state from localStorage.
 */
export function loadConversationsFromStorage(storage = typeof window !== "undefined" ? window.localStorage : null) {
  const fallback = { conversations: [], activeConversationId: null, messages: [] };

  if (!storage) return fallback;

  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return fallback;

    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return fallback;
    }

    if (parsed.version !== STORAGE_VERSION) {
      return fallback;
    }

    if (!Array.isArray(parsed.conversations)) {
      return fallback;
    }

    const conversations = parsed.conversations.map(sanitizeConversation).filter(Boolean);
    if (conversations.length === 0) {
      return fallback;
    }

    let activeConversationId = null;
    if (typeof parsed.activeConversationId === "string" && conversations.some((c) => c.id === parsed.activeConversationId)) {
      activeConversationId = parsed.activeConversationId;
    } else {
      activeConversationId = conversations[0].id;
    }

    const activeConv = conversations.find((c) => c.id === activeConversationId);
    const messages = activeConv ? activeConv.messages : [];

    return {
      conversations,
      activeConversationId,
      messages,
    };
  } catch {
    return fallback;
  }
}

/**
 * Evicts the oldest conversations until payload fits maxBytes.
 * Preserves the active conversation unless storage limit requires removing everything.
 */
export function evictOldestConversations(conversations, activeId, maxBytes = DEFAULT_MAX_BYTES) {
  let list = [...conversations];

  while (list.length > 0) {
    const payload = {
      version: STORAGE_VERSION,
      conversations: list,
      activeConversationId: list.some((c) => c.id === activeId) ? activeId : (list[0]?.id || null),
    };

    const serialized = JSON.stringify(payload);
    if (serialized.length <= maxBytes) {
      return { conversations: list, activeConversationId: payload.activeConversationId };
    }

    if (list.length === 1) {
      // If even a single conversation exceeds maxBytes, return empty list or trimmed conv
      return { conversations: [], activeConversationId: null };
    }

    // Evict oldest conversation (looking for one that is not active first)
    const oldestNonActiveIdx = list.findIndex((c) => c.id !== activeId);
    if (oldestNonActiveIdx !== -1) {
      list.splice(oldestNonActiveIdx, 1);
    } else {
      // If all remaining are active (e.g. only 1 left), remove oldest
      list.pop();
    }
  }

  return { conversations: [], activeConversationId: null };
}

/**
 * Safely serializes and persists conversations state to localStorage.
 */
export function saveConversationsToStorage(
  conversations,
  activeConversationId,
  options = {}
) {
  const { maxBytes = DEFAULT_MAX_BYTES, storage = typeof window !== "undefined" ? window.localStorage : null } = options;

  if (!storage) return false;

  try {
    const sanitizedList = Array.isArray(conversations)
      ? conversations.map(sanitizeConversation).filter(Boolean)
      : [];

    if (sanitizedList.length === 0) {
      return clearConversationsFromStorage(storage);
    }

    const { conversations: evictedList, activeConversationId: finalActiveId } = evictOldestConversations(
      sanitizedList,
      activeConversationId,
      maxBytes
    );

    if (evictedList.length === 0) {
      return clearConversationsFromStorage(storage);
    }

    const payload = {
      version: STORAGE_VERSION,
      conversations: evictedList,
      activeConversationId: finalActiveId,
    };

    storage.setItem(STORAGE_KEY, JSON.stringify(payload));
    return true;
  } catch {
    // Catch QuotaExceededError, SecurityError, or any write error
    return false;
  }
}

/**
 * Safely removes conversations from localStorage.
 */
export function clearConversationsFromStorage(storage = typeof window !== "undefined" ? window.localStorage : null) {
  if (!storage) return false;
  try {
    storage.removeItem(STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}
