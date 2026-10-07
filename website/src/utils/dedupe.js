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
  const remoteByClientId = new Map();
  for (const m of remoteMsgs) {
    if (m?.id) remoteById.set(m.id, m);
    if (m?.clientId) remoteByClientId.set(m.clientId, m);
  }

  const localMatchedToRemote = new Map();
  const remoteMatchedToLocal = new Map();
  const matchedRemoteSet = new Set();

  // Pass 1: Match by exact ID or clientId
  for (const local of localMsgs) {
    if (!local) continue;
    let match = null;
    if (local.id && remoteById.has(local.id)) {
      match = remoteById.get(local.id);
    } else if (local.clientId && remoteById.has(local.clientId)) {
      match = remoteById.get(local.clientId);
    } else if (local.id && remoteByClientId.has(local.id)) {
      match = remoteByClientId.get(local.id);
    }
    if (match && !matchedRemoteSet.has(match)) {
      localMatchedToRemote.set(local, match);
      remoteMatchedToLocal.set(match, local);
      matchedRemoteSet.add(match);
    }
  }

  // Pass 2: Correlate unacknowledged in-flight turns (e.g. disconnected before ack)
  for (let i = 0; i < localMsgs.length; i++) {
    const local = localMsgs[i];
    if (!local || localMatchedToRemote.has(local)) continue;

    if (local.role === "user") {
      // Find the first unmatched remote user message with same content
      const match = remoteMsgs.find(
        (r) => !matchedRemoteSet.has(r) && r.role === "user" && r.content === local.content
      );
      if (match) {
        localMatchedToRemote.set(local, match);
        remoteMatchedToLocal.set(match, local);
        matchedRemoteSet.add(match);
      }
    } else if (local.role === "assistant" && i > 0) {
      const prevLocal = localMsgs[i - 1];
      const prevRemote = localMatchedToRemote.get(prevLocal);
      if (prevRemote) {
        const remoteUserIdx = remoteMsgs.indexOf(prevRemote);
        if (remoteUserIdx !== -1 && remoteUserIdx + 1 < remoteMsgs.length) {
          const nextRemote = remoteMsgs[remoteUserIdx + 1];
          if (nextRemote.role === "assistant" && !matchedRemoteSet.has(nextRemote)) {
            localMatchedToRemote.set(local, nextRemote);
            remoteMatchedToLocal.set(nextRemote, local);
            matchedRemoteSet.add(nextRemote);
          }
        }
      }
    }
  }

  const merged = [];
  const seenIds = new Set();

  // Construct merged array based on remote messages, updated with local details
  for (const remote of remoteMsgs) {
    if (!remote?.id) {
      merged.push(remote);
      continue;
    }
    const local = remoteMatchedToLocal.get(remote);
    if (local) {
      const isAssistant = remote.role === "assistant";
      const content =
        remote.content !== undefined && remote.content !== ""
          ? remote.content
          : local.content || "";
      const type =
        remote.type && remote.type !== "text"
          ? remote.type
          : local.type && local.type !== "error"
          ? local.type
          : remote.type || local.type || "text";
      const completed = isAssistant
        ? true
        : remote.completed !== undefined
        ? remote.completed
        : local.completed;
      const clientId =
        local.clientId || (local.id !== remote.id ? local.id : remote.clientId || remote.id);

      const combined = {
        ...local,
        ...remote,
        id: remote.id,
        clientId,
        content,
        type,
        ...(completed !== undefined ? { completed } : {}),
        metadata: { ...(local.metadata || {}), ...(remote.metadata || {}) },
      };
      seenIds.add(combined.id);
      if (combined.clientId) seenIds.add(combined.clientId);
      merged.push(combined);
    } else {
      const isAssistant = remote.role === "assistant";
      const combined = {
        ...remote,
        ...(isAssistant && remote.completed === undefined ? { completed: true } : {}),
      };
      seenIds.add(combined.id);
      merged.push(combined);
    }
  }

  // Preserve any local in-flight messages that have not reached the server yet
  for (const local of localMsgs) {
    if (!local) continue;
    if (localMatchedToRemote.has(local)) continue;
    if (local.id && seenIds.has(local.id)) continue;
    if (local.clientId && seenIds.has(local.clientId)) continue;

    if (local.id) seenIds.add(local.id);
    if (local.clientId) seenIds.add(local.clientId);
    merged.push(local);
  }

  return dedupeMessages(merged);
}

/**
 * Reconciles an existing local conversation with an incoming remote snapshot.
 * Preserves the newest valid title according to local modification timestamps,
 * deterministically dedupes messages, retains local fields, and resolves versioning.
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
      ? mergeMessages(existingConv.messages || [], incomingConv.messages)
      : existingConv.messages || []
  );

  const localVersion = Number.isInteger(existingConv?.version) && existingConv.version >= 1 ? existingConv.version : 1;
  const remoteVersion = Number.isInteger(incomingConv?.version) && incomingConv.version >= 1 ? incomingConv.version : null;
  // The authoritative server version must prevail when synchronizing with a remote snapshot.
  // A local uncommitted optimistic version bump must not overwrite authoritative server state (#438 / PR #441 follow-up).
  const authoritativeVersion = remoteVersion !== null ? remoteVersion : localVersion;

  const effectiveTitleUpdatedAt = Math.max(
    localTitleUpdatedAt || 0,
    existingConv.titleUpdatedAt || 0
  );
  const incomingTitleUpdatedAt = incomingConv.titleUpdatedAt || 0;

  // The latest valid conversation title must win:
  // If remote version is strictly newer (conflict reconciliation), remote title wins.
  // Otherwise if local modification is newer than remote or remote has no title update timestamp,
  // preserve existingConv.title.
  let title;
  let titleUpdatedAt;
  if (remoteVersion > localVersion && incomingConv.title) {
    title = incomingConv.title;
    titleUpdatedAt = incomingTitleUpdatedAt || effectiveTitleUpdatedAt;
  } else if (effectiveTitleUpdatedAt > incomingTitleUpdatedAt && existingConv.title) {
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
    ...(Number.isInteger(existingConv.version) || Number.isInteger(incomingConv.version) ? { version: authoritativeVersion } : {}),
    messages: dedupedMessages,
    updatedAt: incomingConv.updatedAt || existingConv.updatedAt || new Date().toISOString(),
  };
}
