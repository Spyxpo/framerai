/**
 * Shared message deduplication and reconciliation utilities.
 *
 * Ensures consistent message deduplication across conversation selection,
 * background sync, message merging, and local storage persistence.
 */

/**
 * Deduplicates an array of message objects by their message ID, preserving
 * the first occurrence of each unique ID. Messages without an ID are preserved.
 *
 * @param {Array<Object>} messages - Array of message objects
 * @returns {Array<Object>} Deduplicated array of message objects
 */
export function dedupeMessages(messages) {
  if (!Array.isArray(messages)) return [];
  const seen = new Set();
  return messages.filter((m) => {
    if (!m?.id) return true;
    if (seen.has(m.id)) return false;
    seen.add(m.id);
    return true;
  });
}

/**
 * Merges local conversation messages with remote backend messages deterministically.
 *
 * - Preserves any in-flight or newly created local messages that the remote snapshot does not yet contain.
 * - Updates existing messages with remote server timestamps/metadata/content without losing local status.
 * - Retains any remote messages that are new to the local conversation.
 * - Deduplicates the final message list by message ID.
 *
 * @param {Array<Object>} localMsgs - Current messages in local frontend state
 * @param {Array<Object>} remoteMsgs - Messages returned from remote backend API
 * @returns {Array<Object>} Deterministically merged and deduplicated message list
 */
export function mergeMessages(localMsgs = [], remoteMsgs = []) {
  if (!Array.isArray(localMsgs) || localMsgs.length === 0) {
    return dedupeMessages(remoteMsgs);
  }
  if (!Array.isArray(remoteMsgs) || remoteMsgs.length === 0) {
    return dedupeMessages(localMsgs);
  }

  const remoteById = new Map();
  for (const m of remoteMsgs) {
    if (m?.id) {
      remoteById.set(m.id, m);
    }
  }

  const merged = [];
  const seenIds = new Set();

  // First, walk through local messages. For any message that also exists in remote,
  // take the remote message updated with local properties (or remote content if local was placeholder).
  for (const local of localMsgs) {
    if (!local?.id) {
      merged.push(local);
      continue;
    }
    if (seenIds.has(local.id)) continue;
    seenIds.add(local.id);

    const remote = remoteById.get(local.id);
    if (remote) {
      merged.push({
        ...remote,
        ...local,
        content: remote.content || local.content || "",
        completed: remote.completed !== undefined ? remote.completed : local.completed,
        metadata: { ...(remote.metadata || {}), ...(local.metadata || {}) },
      });
    } else {
      // Local message not yet in remote snapshot (e.g., in-flight user or assistant turn)
      merged.push(local);
    }
  }

  // Then append any remote messages that weren't in local state
  for (const remote of remoteMsgs) {
    if (remote?.id && !seenIds.has(remote.id)) {
      seenIds.add(remote.id);
      merged.push(remote);
    }
  }

  return dedupeMessages(merged);
}

/**
 * Reconciles an existing local conversation with an incoming remote snapshot.
 * Preserves the newest valid title according to local modification timestamps,
 * deterministically dedupes messages, and retains local fields.
 *
 * @param {Object} existingConv - Existing local conversation state
 * @param {Object} incomingConv - Incoming conversation snapshot (e.g. from backend API)
 * @param {number} [localTitleUpdatedAt=0] - Monotonic timestamp of local title modification
 * @returns {Object} Reconciled conversation object
 */
export function reconcileConversation(existingConv, incomingConv, localTitleUpdatedAt = 0) {
  if (!existingConv && !incomingConv) return null;
  if (!existingConv) {
    return {
      ...incomingConv,
      messages: dedupeMessages(incomingConv?.messages),
    };
  }
  if (!incomingConv) return existingConv;

  const dedupedMessages = dedupeMessages(
    Array.isArray(incomingConv.messages) && incomingConv.messages.length > 0
      ? incomingConv.messages
      : existingConv.messages || []
  );

  const effectiveTitleUpdatedAt = Math.max(
    localTitleUpdatedAt || 0,
    existingConv.titleUpdatedAt || 0
  );
  const incomingTitleUpdatedAt = incomingConv.titleUpdatedAt || 0;

  // The latest valid conversation title must win:
  // If local modification is newer than remote or remote has no title update timestamp,
  // preserve existingConv.title.
  let title;
  let titleUpdatedAt;
  if (effectiveTitleUpdatedAt > incomingTitleUpdatedAt && existingConv.title) {
    title = existingConv.title;
    titleUpdatedAt = effectiveTitleUpdatedAt;
  } else if (incomingConv.title) {
    title = incomingConv.title;
    titleUpdatedAt = incomingTitleUpdatedAt || effectiveTitleUpdatedAt;
  } else {
    title = existingConv.title || "New Chat";
    titleUpdatedAt = effectiveTitleUpdatedAt;
  }

  return {
    ...existingConv,
    ...incomingConv,
    title,
    ...(titleUpdatedAt > 0 ? { titleUpdatedAt } : {}),
    messages: dedupedMessages,
  };
}
