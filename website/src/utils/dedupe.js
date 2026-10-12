/**
 * Shared message deduplication and differential reconciliation utilities.
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
 * Checks whether two message objects represent identical content and metadata.
 *
 * @param {Object} a - First message
 * @param {Object} b - Second message
 * @returns {boolean} True if messages are functionally identical
 */
export function isMessageEqual(a, b) {
  if (!a || !b) return false;
  if (a.role !== b.role) return false;
  if ((a.content || "") !== (b.content || "")) return false;
  if ((a.type || "text") !== (b.type || "text")) return false;
  if (Boolean(a.completed) !== Boolean(b.completed)) return false;

  const aAtt = Array.isArray(a.attachments) ? a.attachments : [];
  const bAtt = Array.isArray(b.attachments) ? b.attachments : [];
  if (aAtt.length !== bAtt.length) return false;
  for (let i = 0; i < aAtt.length; i++) {
    if (aAtt[i] !== bAtt[i]) return false;
  }

  const aMeta = a.metadata && typeof a.metadata === "object" ? a.metadata : {};
  const bMeta = b.metadata && typeof b.metadata === "object" ? b.metadata : {};
  const aKeys = Object.keys(aMeta);
  const bKeys = Object.keys(bMeta);
  if (aKeys.length !== bKeys.length) return false;
  for (const k of aKeys) {
    if (JSON.stringify(aMeta[k]) !== JSON.stringify(bMeta[k])) return false;
  }

  return true;
}

/**
 * Merges local conversation messages with remote backend messages deterministically.
 *
 * - Preserves any in-flight or newly created local messages that the remote snapshot does not yet contain.
 * - Updates existing messages with remote server timestamps/metadata/content without losing local status.
 * - Retains any remote messages that are new to the local conversation.
 * - Filters out deleted message IDs.
 * - Deduplicates the final message list by message ID.
 *
 * @param {Array<Object>} localMsgs - Current messages in local frontend state
 * @param {Array<Object>} remoteMsgs - Messages returned from remote backend API
 * @param {Object} [options={}] - Options such as deletedMessageIds
 * @returns {Array<Object>} Deterministically merged and deduplicated message list
 */
export function mergeMessages(localMsgs = [], remoteMsgs = [], options = {}) {
  const deletedIds = new Set();
  const addDeleted = (list) => {
    if (Array.isArray(list) || list instanceof Set) {
      for (const id of list) {
        if (id && typeof id === "string") deletedIds.add(id);
      }
    }
  };
  addDeleted(options?.deletedMessageIds);

  const cleanLocal = Array.isArray(localMsgs)
    ? localMsgs.filter((m) => m && (!m.id || !deletedIds.has(m.id)) && (!m.clientId || !deletedIds.has(m.clientId)))
    : [];
  const cleanRemote = Array.isArray(remoteMsgs)
    ? remoteMsgs.filter((m) => m && (!m.id || !deletedIds.has(m.id)) && (!m.clientId || !deletedIds.has(m.clientId)))
    : [];

  if (cleanLocal.length === 0) {
    return dedupeMessages(cleanRemote);
  }
  if (cleanRemote.length === 0) {
    return dedupeMessages(cleanLocal);
  }

  const remoteById = new Map();
  const remoteByClientId = new Map();
  for (const m of cleanRemote) {
    if (m?.id) remoteById.set(m.id, m);
    if (m?.clientId) remoteByClientId.set(m.clientId, m);
  }

  const localMatchedToRemote = new Map();
  const remoteMatchedToLocal = new Map();
  const matchedRemoteSet = new Set();

  // Pass 1: Match by exact ID or clientId
  for (const local of cleanLocal) {
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
  for (let i = 0; i < cleanLocal.length; i++) {
    const local = cleanLocal[i];
    if (!local || localMatchedToRemote.has(local)) continue;

    if (local.role === "user") {
      const match = cleanRemote.find(
        (r) => !matchedRemoteSet.has(r) && r.role === "user" && r.content === local.content
      );
      if (match) {
        localMatchedToRemote.set(local, match);
        remoteMatchedToLocal.set(match, local);
        matchedRemoteSet.add(match);
      }
    } else if (local.role === "assistant" && i > 0) {
      const prevLocal = cleanLocal[i - 1];
      const prevRemote = localMatchedToRemote.get(prevLocal);
      if (prevRemote) {
        const remoteUserIdx = cleanRemote.indexOf(prevRemote);
        if (remoteUserIdx !== -1 && remoteUserIdx + 1 < cleanRemote.length) {
          const nextRemote = cleanRemote[remoteUserIdx + 1];
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
  for (const remote of cleanRemote) {
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

  // Preserve any local in-flight or newly created messages that have not reached the server yet
  for (const local of cleanLocal) {
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
 * Computes deterministic differences between client and server conversation states.
 *
 * @param {Object} clientConv - Client conversation state
 * @param {Object} serverConv - Server conversation state
 * @param {Object} [options={}] - Sync options
 * @returns {Object} Structured diff representation
 */
export function computeConversationDiff(clientConv = {}, serverConv = {}, options = {}) {
  const clientVersion =
    Number.isInteger(clientConv?.version) && clientConv.version >= 1
      ? clientConv.version
      : Number.isInteger(clientConv?.clientVersion) && clientConv.clientVersion >= 1
      ? clientConv.clientVersion
      : 1;

  const serverVersion =
    Number.isInteger(serverConv?.version) && serverConv.version >= 1
      ? serverConv.version
      : 1;

  const clientMessages = Array.isArray(clientConv?.messages) ? clientConv.messages : [];
  const serverMessages = Array.isArray(serverConv?.messages) ? serverConv.messages : [];

  const deletedIds = new Set();
  const addDeleted = (list) => {
    if (Array.isArray(list) || list instanceof Set) {
      for (const id of list) {
        if (id && typeof id === "string") deletedIds.add(id);
      }
    }
  };
  addDeleted(clientConv?.deletedMessageIds);
  addDeleted(serverConv?.deletedMessageIds);
  addDeleted(options?.deletedMessageIds);

  const serverById = new Map();
  const serverByClientId = new Map();
  for (const s of serverMessages) {
    if (!s || deletedIds.has(s.id)) continue;
    if (s.id) serverById.set(s.id, s);
    if (s.clientId) serverByClientId.set(s.clientId, s);
  }

  const clientOnlyMessages = [];
  const updatedMessages = [];
  const identicalMessageIds = [];
  const matchedServerSet = new Set();

  for (const c of clientMessages) {
    if (!c) continue;
    if (c.id && deletedIds.has(c.id)) continue;
    if (c.clientId && deletedIds.has(c.clientId)) continue;

    let match = null;
    if (c.id && serverById.has(c.id)) {
      match = serverById.get(c.id);
    } else if (c.clientId && serverById.has(c.clientId)) {
      match = serverById.get(c.clientId);
    } else if (c.id && serverByClientId.has(c.id)) {
      match = serverByClientId.get(c.id);
    }

    if (match) {
      matchedServerSet.add(match);
      if (isMessageEqual(c, match)) {
        identicalMessageIds.push(match.id);
      } else {
        const isAssistant = match.role === "assistant" || c.role === "assistant";
        const completed = isAssistant
          ? true
          : match.completed !== undefined
          ? match.completed
          : c.completed;

        let content = match.content;
        let type = match.type || c.type || "text";

        if (clientVersion > serverVersion && c.content !== undefined && c.content !== "") {
          content = c.content;
          type = c.type || type;
        } else if (match.content === undefined || match.content === "") {
          content = c.content || "";
        }

        const resolved = {
          ...c,
          ...match,
          id: match.id,
          clientId: c.clientId || (c.id !== match.id ? c.id : match.clientId || match.id),
          content,
          type,
          ...(completed !== undefined ? { completed } : {}),
          metadata: { ...(c.metadata || {}), ...(match.metadata || {}) },
        };

        updatedMessages.push({
          id: match.id,
          client: c,
          server: match,
          resolved,
        });
      }
    } else {
      clientOnlyMessages.push(c);
    }
  }

  const serverOnlyMessages = [];
  for (const s of serverMessages) {
    if (!s || deletedIds.has(s.id)) continue;
    if (!matchedServerSet.has(s)) {
      serverOnlyMessages.push(s);
    }
  }

  // Title reconciliation
  const clientTitle = typeof clientConv?.title === "string" ? clientConv.title.trim() : undefined;
  const serverTitle = typeof serverConv?.title === "string" ? serverConv.title.trim() : undefined;
  const effectiveTitleUpdatedAt = Math.max(
    Number(options?.localTitleUpdatedAt) || 0,
    Number(clientConv?.titleUpdatedAt) || 0
  );
  const incomingTitleUpdatedAt = Number(serverConv?.titleUpdatedAt) || 0;
  let resolvedTitle;
  let resolvedTitleUpdatedAt;
  if (effectiveTitleUpdatedAt > incomingTitleUpdatedAt && clientTitle) {
    resolvedTitle = clientTitle;
    resolvedTitleUpdatedAt = effectiveTitleUpdatedAt;
  } else if (incomingTitleUpdatedAt > effectiveTitleUpdatedAt && serverTitle) {
    resolvedTitle = serverTitle;
    resolvedTitleUpdatedAt = incomingTitleUpdatedAt;
  } else if (serverVersion > clientVersion && serverTitle) {
    resolvedTitle = serverTitle;
    resolvedTitleUpdatedAt = incomingTitleUpdatedAt || effectiveTitleUpdatedAt;
  } else if (serverTitle) {
    resolvedTitle = serverTitle;
    resolvedTitleUpdatedAt = incomingTitleUpdatedAt || effectiveTitleUpdatedAt;
  } else {
    resolvedTitle = clientTitle || "New Chat";
    resolvedTitleUpdatedAt = effectiveTitleUpdatedAt;
  }
  const titleChanged = Boolean(clientTitle && serverTitle && resolvedTitle !== serverTitle);

  const hasDifferences =
    clientOnlyMessages.length > 0 ||
    serverOnlyMessages.length > 0 ||
    updatedMessages.length > 0 ||
    deletedIds.size > 0 ||
    titleChanged ||
    clientVersion !== serverVersion;

  let status = "in_sync";
  if (hasDifferences) {
    if (serverOnlyMessages.length > 0 && clientOnlyMessages.length === 0 && updatedMessages.length === 0 && !titleChanged) {
      status = "server_ahead";
    } else if (clientOnlyMessages.length > 0 && serverOnlyMessages.length === 0 && updatedMessages.length === 0 && !titleChanged) {
      status = "client_ahead";
    } else {
      status = "diverged";
    }
  }

  return {
    status,
    hasDifferences,
    clientVersion,
    serverVersion,
    messages: {
      clientOnly: clientOnlyMessages,
      serverOnly: serverOnlyMessages,
      updated: updatedMessages,
      deletedIds: [...deletedIds],
      identicalCount: identicalMessageIds.length,
      identicalIds: identicalMessageIds,
    },
    title: {
      client: clientTitle,
      server: serverTitle,
      resolved: resolvedTitle,
      titleUpdatedAt: resolvedTitleUpdatedAt,
      changed: titleChanged,
    },
    branch: {
      parentConversationId: serverConv?.parentConversationId || clientConv?.parentConversationId,
      parentVersion: serverConv?.parentVersion || clientConv?.parentVersion,
      branchedFromMessageId: serverConv?.branchedFromMessageId || clientConv?.branchedFromMessageId,
    },
  };
}

/**
 * Reconciles conversation states using calculated differential model.
 *
 * @param {Object} clientConv - Client conversation state
 * @param {Object} serverConv - Server conversation state
 * @param {Object} [options={}] - Sync options
 * @returns {Object} { diff, reconciled }
 */
export function reconcileConversationDiff(clientConv = {}, serverConv = {}, options = {}) {
  const diff = computeConversationDiff(clientConv, serverConv, options);
  const deletedIds = new Set(diff.messages.deletedIds);

  const updatedMap = new Map();
  for (const u of diff.messages.updated) {
    updatedMap.set(u.id, u.resolved);
  }

  const serverMessages = Array.isArray(serverConv?.messages) ? serverConv.messages : [];
  const reconciledMessages = [];
  const seenIds = new Set();
  const seenClientIds = new Set();

  // 1. Process server messages in authoritative order, applying updates
  for (const s of serverMessages) {
    if (!s || deletedIds.has(s.id)) continue;
    const msg = updatedMap.has(s.id) ? updatedMap.get(s.id) : s;
    if (deletedIds.has(msg.id)) continue;
    if (seenIds.has(msg.id)) continue;

    seenIds.add(msg.id);
    if (msg.clientId) seenClientIds.add(msg.clientId);
    reconciledMessages.push(msg);
  }

  // 2. Insert client-only messages deterministically
  for (const c of diff.messages.clientOnly) {
    if (!c || deletedIds.has(c.id)) continue;
    if (c.id && seenIds.has(c.id)) continue;
    if (c.clientId && (seenIds.has(c.clientId) || seenClientIds.has(c.clientId))) continue;

    if (c.id) seenIds.add(c.id);
    if (c.clientId) seenClientIds.add(c.clientId);

    const replyToId = c.replyToId;
    if (replyToId) {
      const parentIdx = reconciledMessages.findIndex((m) => m.id === replyToId || m.clientId === replyToId);
      if (parentIdx !== -1) {
        let insertIdx = parentIdx + 1;
        while (
          insertIdx < reconciledMessages.length &&
          reconciledMessages[insertIdx].role === "assistant"
        ) {
          insertIdx++;
        }
        reconciledMessages.splice(insertIdx, 0, c);
        continue;
      }
    }

    reconciledMessages.push(c);
  }

  const clientVersion =
    Number.isInteger(clientConv?.version) && clientConv.version >= 1
      ? clientConv.version
      : Number.isInteger(clientConv?.clientVersion) && clientConv.clientVersion >= 1
      ? clientConv.clientVersion
      : 1;

  const remoteVersion =
    Number.isInteger(serverConv?.version) && serverConv.version >= 1
      ? serverConv.version
      : null;

  // Authoritative server version prevails when reconciling with remote snapshot (#438)
  const authoritativeVersion = remoteVersion !== null ? remoteVersion : clientVersion;

  const clientMadeChanges =
    diff.messages.clientOnly.length > 0 ||
    (diff.title.changed && diff.title.resolved === diff.title.client) ||
    (Array.isArray(clientConv?.deletedMessageIds) && clientConv.deletedMessageIds.length > 0);

  const resolvedVersion =
    options?.advanceVersionIfChanged && diff.hasDifferences && clientMadeChanges
      ? Math.max(diff.serverVersion, diff.clientVersion) + 1
      : authoritativeVersion;

  const baseConv = serverConv?.id ? serverConv : clientConv;
  const reconciled = {
    ...clientConv,
    ...serverConv,
    id: baseConv?.id || clientConv?.id || serverConv?.id,
    title: diff.title.resolved,
    ...(diff.title.titleUpdatedAt > 0 ? { titleUpdatedAt: diff.title.titleUpdatedAt } : {}),
    ...(Number.isInteger(clientConv?.version) || Number.isInteger(serverConv?.version) ? { version: resolvedVersion } : {}),
    messages: reconciledMessages,
    ...(diff.branch.parentConversationId ? { parentConversationId: diff.branch.parentConversationId } : {}),
    ...(diff.branch.parentVersion ? { parentVersion: diff.branch.parentVersion } : {}),
    ...(diff.branch.branchedFromMessageId ? { branchedFromMessageId: diff.branch.branchedFromMessageId } : {}),
    ...(deletedIds.size > 0 ? { deletedMessageIds: [...deletedIds] } : {}),
    updatedAt: serverConv?.updatedAt || clientConv?.updatedAt || new Date().toISOString(),
    // A creation time is only invented when neither side has one; otherwise a
    // second reconcile against a server copy without one minted a new time, so
    // repeated synchronization was not idempotent.
    createdAt: baseConv?.createdAt || clientConv?.createdAt || serverConv?.createdAt || new Date().toISOString(),
  };

  return { diff, reconciled };
}

/**
 * Applies a diff to a base conversation.
 *
 * @param {Object} baseConv - Base conversation
 * @param {Object} diff - Computed diff
 * @returns {Object} Reconciled conversation
 */
export function applyConversationDiff(baseConv, diff) {
  if (!baseConv) return null;
  if (!diff || !diff.hasDifferences) return baseConv;
  return reconcileConversationDiff(diff, baseConv).reconciled;
}

/**
 * Reconciles an existing local conversation with an incoming remote snapshot.
 * Preserves the newest valid title according to local modification timestamps,
 * deterministically dedupes messages, retains local fields, and resolves versioning.
 *
 * @param {Object} existingConv - Existing local conversation state
 * @param {Object} incomingConv - Incoming conversation snapshot (e.g. from backend API)
 * @param {number} [localTitleUpdatedAt=0] - Monotonic timestamp of local title modification
 * @param {Object} [options={}] - Optional reconciliation configuration
 * @returns {Object} Reconciled conversation object
 */
export function reconcileConversation(existingConv, incomingConv, localTitleUpdatedAt = 0, options = {}) {
  if (!existingConv && !incomingConv) return null;
  if (!existingConv) {
    return {
      ...incomingConv,
      messages: dedupeMessages(incomingConv?.messages),
    };
  }
  if (!incomingConv) return existingConv;

  const { reconciled } = reconcileConversationDiff(existingConv, incomingConv, {
    localTitleUpdatedAt,
    ...options,
  });

  return reconciled;
}
