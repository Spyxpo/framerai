import { dedupeMessages } from "./dedupe";

export const SEARCH_SCOPES = {
  ALL: "all",
  CONVERSATIONS: "conversations",
  MESSAGES: "messages",
  CURRENT: "current",
};

/**
 * Extracts a relevant snippet from text centered around the search query.
 *
 * @param {string} content - Full text content
 * @param {string} query - Search query
 * @param {number} maxLength - Target maximum snippet length
 * @returns {string} Truncated snippet with ellipsis if truncated
 */
export function extractSnippet(content, query, maxLength = 100) {
  if (!content || typeof content !== "string") return "";
  if (!query || typeof query !== "string" || content.length <= maxLength) {
    return content;
  }

  const lowerContent = content.toLowerCase();
  const lowerQuery = query.toLowerCase().trim();
  const matchIndex = lowerContent.indexOf(lowerQuery);

  if (matchIndex === -1) {
    return content.length > maxLength ? content.slice(0, maxLength) + "..." : content;
  }

  const queryLen = lowerQuery.length;
  const contextLen = Math.max(10, Math.floor((maxLength - queryLen) / 2));

  let start = Math.max(0, matchIndex - contextLen);
  let end = Math.min(content.length, matchIndex + queryLen + contextLen);

  // Adjust if hitting boundaries
  if (start === 0) {
    end = Math.min(content.length, maxLength);
  } else if (end === content.length) {
    start = Math.max(0, content.length - maxLength);
  }

  let snippet = content.slice(start, end);
  if (start > 0) snippet = "..." + snippet;
  if (end < content.length) snippet = snippet + "...";

  return snippet;
}

/**
 * Breaks a text into segments of matching and non-matching parts for safe rendering.
 *
 * @param {string} text - Text to highlight
 * @param {string} query - Query to match
 * @returns {Array<{text: string, isMatch: boolean}>} Array of text segments
 */
export function highlightMatches(text, query) {
  if (!text || typeof text !== "string") return [];
  if (!query || typeof query !== "string" || !query.trim()) {
    return [{ text, isMatch: false }];
  }

  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase().trim();
  const segments = [];
  let lastIndex = 0;

  let matchIndex = lowerText.indexOf(lowerQuery, lastIndex);
  while (matchIndex !== -1) {
    if (matchIndex > lastIndex) {
      segments.push({
        text: text.slice(lastIndex, matchIndex),
        isMatch: false,
      });
    }
    segments.push({
      text: text.slice(matchIndex, matchIndex + lowerQuery.length),
      isMatch: true,
    });
    lastIndex = matchIndex + lowerQuery.length;
    matchIndex = lowerText.indexOf(lowerQuery, lastIndex);
  }

  if (lastIndex < text.length) {
    segments.push({
      text: text.slice(lastIndex),
      isMatch: false,
    });
  }

  return segments;
}

/**
 * Deterministically searches conversations and messages according to scope.
 *
 * @param {Object} options
 * @param {Array<Object>} options.conversations - All conversations
 * @param {string|null} [options.activeConversationId] - ID of current active conversation
 * @param {string} options.query - User search query
 * @param {string} [options.scope] - Search scope ('all', 'conversations', 'messages', 'current')
 * @returns {Array<Object>} Deterministic list of search results
 */
export function searchConversations({
  conversations,
  activeConversationId = null,
  query = "",
  scope = SEARCH_SCOPES.ALL,
}) {
  if (!query || typeof query !== "string" || !query.trim()) {
    return [];
  }
  if (!Array.isArray(conversations) || conversations.length === 0) {
    return [];
  }

  const trimmedQuery = query.trim();
  const lowerQuery = trimmedQuery.toLowerCase();
  const targetScope = scope || SEARCH_SCOPES.ALL;

  // Filter conversations according to scope
  let targetConversations = conversations;
  if (targetScope === SEARCH_SCOPES.CURRENT) {
    if (!activeConversationId) return [];
    targetConversations = conversations.filter((c) => c.id === activeConversationId);
  }

  const results = [];

  for (const conv of targetConversations) {
    if (!conv || !conv.id) continue;
    const convTitle = conv.title || "New Chat";
    const isBranch = Boolean(conv.parentConversationId);

    // 1. Conversation Title Search
    if (targetScope !== SEARCH_SCOPES.MESSAGES) {
      if (convTitle.toLowerCase().includes(lowerQuery)) {
        results.push({
          id: `conv-title-${conv.id}`,
          type: "title",
          conversationId: conv.id,
          conversationTitle: convTitle,
          parentConversationId: conv.parentConversationId || null,
          branchedFromMessageId: conv.branchedFromMessageId || null,
          isBranch,
          messageId: null,
          role: null,
          content: convTitle,
          snippet: convTitle,
          timestamp: conv.updatedAt || conv.createdAt || null,
        });
      }
    }

    // 2. Messages Search
    if (targetScope !== SEARCH_SCOPES.CONVERSATIONS) {
      if (Array.isArray(conv.messages) && conv.messages.length > 0) {
        // Reuse deduplication to guarantee no duplicate message results
        const dedupedMessages = dedupeMessages(conv.messages);

        for (const msg of dedupedMessages) {
          if (!msg || typeof msg.content !== "string") continue;
          if (msg.content.toLowerCase().includes(lowerQuery)) {
            const messageId = msg.id || msg.clientId || null;
            results.push({
              id: `msg-${conv.id}-${messageId || Math.random()}`,
              type: "message",
              conversationId: conv.id,
              conversationTitle: convTitle,
              parentConversationId: conv.parentConversationId || null,
              branchedFromMessageId: conv.branchedFromMessageId || null,
              isBranch,
              messageId,
              role: msg.role || "user",
              content: msg.content,
              snippet: extractSnippet(msg.content, trimmedQuery),
              timestamp: msg.timestamp || null,
            });
          }
        }
      }
    }
  }

  return results;
}
