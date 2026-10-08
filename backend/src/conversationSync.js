/**
 * Differential Conversation Synchronization and Conflict Resolution engine.
 *
 * Provides deterministic calculation of differences between client and server
 * conversation states and resolves conflicts according to authoritative versioning,
 * monotonic ordering, stable message identity, and deletion tracking.
 */

/**
 * Checks whether two message objects represent identical content and metadata.
 */
function isMessageEqual(a, b) {
  if (!a || !b) return false;
  if (a.role !== b.role) return false;
  if ((a.content || "") !== (b.content || "")) return false;
  if ((a.type || "text") !== (b.type || "text")) return false;
  if (Boolean(a.completed) !== Boolean(b.completed)) return false;

  // Compare attachments length and content
  const aAtt = Array.isArray(a.attachments) ? a.attachments : [];
  const bAtt = Array.isArray(b.attachments) ? b.attachments : [];
  if (aAtt.length !== bAtt.length) return false;
  for (let i = 0; i < aAtt.length; i++) {
    if (aAtt[i] !== bAtt[i]) return false;
  }

  // Compare metadata keys and values
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
 * Computes deterministic differences between client and server conversation states.
 *
 * @param {Object} clientConv - Client conversation snapshot or state summary
 * @param {Object} serverConv - Server conversation snapshot
 * @param {Object} [options={}] - Additional sync options (e.g. deletedMessageIds)
 * @returns {Object} Structured diff representation
 */
function computeConversationDiff(clientConv = {}, serverConv = {}, options = {}) {
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

  // Assemble all deleted message IDs from client, server, and options
  const deletedIds = new Set();
  const addDeleted = (list) => {
    if (Array.isArray(list)) {
      for (const id of list) {
        if (id && typeof id === "string") deletedIds.add(id);
      }
    } else if (list instanceof Set) {
      for (const id of list) {
        if (id && typeof id === "string") deletedIds.add(id);
      }
    }
  };
  addDeleted(clientConv?.deletedMessageIds);
  addDeleted(serverConv?.deletedMessageIds);
  addDeleted(options?.deletedMessageIds);

  // Map server messages by id and clientId for O(N) lookup
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
        // Resolve conflicting fields deterministically
        const isAssistant = match.role === "assistant" || c.role === "assistant";
        const completed = isAssistant
          ? true
          : match.completed !== undefined
          ? match.completed
          : c.completed;

        let content = match.content;
        let type = match.type || c.type || "text";

        // Authoritative resolution: if server is strictly newer, server wins.
        // If client is newer or has uncommitted edit, preserve client content.
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
function reconcileConversationDiff(clientConv = {}, serverConv = {}, options = {}) {
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

  // 1. Process server messages in their authoritative order, applying updates
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

    // Default placement: appended at end maintaining arrival order
    reconciledMessages.push(c);
  }

  // Calculate authoritative synchronized version:
  // If client contributed new changes accepted by the server, advance version.
  let resolvedVersion = Math.max(diff.serverVersion, diff.clientVersion);
  const clientMadeChanges =
    diff.messages.clientOnly.length > 0 ||
    (diff.title.changed && diff.title.resolved === diff.title.client) ||
    (Array.isArray(clientConv?.deletedMessageIds) && clientConv.deletedMessageIds.length > 0);

  if (diff.hasDifferences && clientMadeChanges) {
    resolvedVersion = Math.max(diff.serverVersion, diff.clientVersion) + 1;
  }

  const baseConv = serverConv?.id ? serverConv : clientConv;
  const reconciled = {
    ...clientConv,
    ...serverConv,
    id: baseConv?.id || clientConv?.id || serverConv?.id,
    title: diff.title.resolved,
    titleUpdatedAt: diff.title.titleUpdatedAt > 0 ? diff.title.titleUpdatedAt : undefined,
    version: resolvedVersion,
    messages: reconciledMessages,
    ...(diff.branch.parentConversationId ? { parentConversationId: diff.branch.parentConversationId } : {}),
    ...(diff.branch.parentVersion ? { parentVersion: diff.branch.parentVersion } : {}),
    ...(diff.branch.branchedFromMessageId ? { branchedFromMessageId: diff.branch.branchedFromMessageId } : {}),
    ...(deletedIds.size > 0 ? { deletedMessageIds: [...deletedIds] } : {}),
    updatedAt: new Date().toISOString(),
    createdAt: baseConv?.createdAt || new Date().toISOString(),
  };

  return { diff, reconciled };
}

/**
 * Applies a diff to a base conversation.
 */
function applyConversationDiff(baseConv, diff) {
  if (!baseConv) return null;
  if (!diff || !diff.hasDifferences) return baseConv;
  return reconcileConversationDiff(diff, baseConv).reconciled;
}

module.exports = {
  isMessageEqual,
  computeConversationDiff,
  reconcileConversationDiff,
  applyConversationDiff,
};
