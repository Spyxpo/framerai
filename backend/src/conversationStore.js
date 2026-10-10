/**
 * The in-memory conversation store.
 *
 * It lived inside the chat route, which meant the WebSocket service could not
 * see it: a streamed turn was built as a one-message array and the model was
 * given no prior context, however much of it the route had recorded. Both
 * paths now read and write the same store, so a conversation is a conversation
 * whichever transport carried it.
 *
 * Still in memory, so it is lost on restart and not shared across replicas.
 * That was already true; moving it here does not make it worse, and it puts
 * the eventual swap for a real store in one place.
 *
 * Growth bounds
 * -------------
 * At most _max conversations are kept at any time. On each create():
 *  1. Any entry whose lastAccessedAtMs is older than _ttl ms is swept.
 *  2. If the store is still at or above the cap, the least-recently-accessed
 *     conversation is evicted to make room.
 *
 * Within each conversation, messages are capped at _maxMessages. When a new
 * message would exceed the cap, the oldest messages are trimmed from the front
 * so that the newest _maxMessages messages are retained in order.
 *
 * LRU order uses a monotonic sequence number (_accessSeq) so entries accessed
 * within the same millisecond still have a strict, deterministic ordering.
 * TTL checks use Date.now() (wall-clock milliseconds).
 *
 * Eviction is lazy (triggered by create(), not a background timer) so behaviour
 * is deterministic and tests can drive it without fake timers.
 */

let _max = Number(process.env.FRAMER_MAX_CONVERSATIONS) || 1000;
let _ttl = Number(process.env.FRAMER_CONVERSATION_TTL_MS) || 24 * 60 * 60 * 1000;
let _maxMessages = Number(process.env.FRAMER_MAX_MESSAGES_PER_CONVERSATION) || 1000;
let _seq = 0; // monotonic counter — strictly increasing across every access

const conversations = new Map();

/**
 * Stamp the bookkeeping a conversation needs to be ordered and expired.
 *
 * The fields are defined non-enumerable because GET /chat/conversations/:id
 * serialises the stored object as-is. Plain assignment would put _lruSeq and
 * _accessedAtMs in that response, where they are neither documented by the
 * OpenAPI Conversation schema nor of any use to a client. Hiding them here
 * keeps the fix in one place rather than asking every consumer of the store
 * to remember to strip them.
 */
function _touch(conv) {
  if (!Object.prototype.hasOwnProperty.call(conv, "_lruSeq")) {
    const hidden = { value: 0, writable: true, enumerable: false, configurable: true };
    Object.defineProperty(conv, "_lruSeq", hidden);
    Object.defineProperty(conv, "_accessedAtMs", hidden);
  }
  conv._lruSeq = ++_seq;           // LRU ordering (monotonic, no ties)
  conv._accessedAtMs = Date.now(); // TTL expiry (wall-clock)
}

/**
 * Sweep TTL-expired entries, then evict the LRU entry if still at or above cap.
 * Called once per create() so memory is reclaimed at a deterministic point.
 */
function _evict() {
  const now = Date.now();
  for (const [id, conv] of conversations) {
    if (now - conv._accessedAtMs > _ttl) {
      conversations.delete(id);
    }
  }
  if (conversations.size >= _max) {
    let lruId = null;
    let lruSeq = Infinity;
    for (const [id, conv] of conversations) {
      if (conv._lruSeq < lruSeq) {
        lruSeq = conv._lruSeq;
        lruId = id;
      }
    }
    if (lruId !== null) conversations.delete(lruId);
  }
}

function create(conversation) {
  // Ensure versioning safely initialized (#438)
  if (!Object.prototype.hasOwnProperty.call(conversation, "version")) {
    let _v = 1;
    Object.defineProperty(conversation, "version", {
      get() { return _v; },
      set(val) { _v = val; },
      enumerable: false,
      configurable: true,
    });
  } else if (!Number.isInteger(conversation.version) || conversation.version < 1) {
    conversation.version = 1;
  }

  // If conversation already exists and incoming conversation has an older version or fewer messages,
  // do not let a stale snapshot overwrite newer state (#438).
  const existing = conversations.get(conversation.id);
  if (existing) {
    const existingVer = existing.version || 1;
    const incomingVer = conversation.version || 1;
    if (incomingVer < existingVer) {
      _touch(existing);
      return existing;
    }
    if (incomingVer === existingVer && Array.isArray(existing.messages) && Array.isArray(conversation.messages)) {
      if (conversation.messages.length < existing.messages.length) {
        _touch(existing);
        return existing;
      }
    }
  }

  // Re-creating an existing id refreshes it without consuming a slot.
  if (!conversations.has(conversation.id)) {
    _evict();
  }
  _touch(conversation);
  conversations.set(conversation.id, conversation);
  return conversation;
}

function get(id) {
  const conv = conversations.get(id) || null;
  if (conv) {
    _touch(conv);
  }
  return conv;
}

function has(id) {
  return conversations.has(id);
}

function remove(id) {
  return conversations.delete(id);
}

function list() {
  return [...conversations.values()];
}

/** The turns recorded for a conversation, or an empty list if it is unknown. */
function messages(id) {
  const conv = conversations.get(id);
  if (conv) _touch(conv);
  return conv ? conv.messages : [];
}

/**
 * Record a turn against a conversation, if it exists.
 * Supports optional expectedVersion for optimistic concurrency control.
 */
function append(id, message, options = {}) {
  const conv = conversations.get(id);
  if (!conv) return false;
  if (!Number.isInteger(conv.version) || conv.version < 1) conv.version = 1;

  if (options.expectedVersion !== undefined && conv.version !== options.expectedVersion) {
    if (options.throwOnConflict) {
      const err = new Error("Conversation version mismatch");
      err.code = "VERSION_CONFLICT";
      err.currentVersion = conv.version;
      err.expectedVersion = options.expectedVersion;
      throw err;
    }
    return false;
  }

  _touch(conv);
  if (message?.id && conv.messages.some((m) => m.id === message.id)) {
    return true;
  }

  const replyToId = options.replyToId || message.replyToId;
  if (replyToId) {
    const parentIndex = conv.messages.findIndex((m) => m.id === replyToId);
    if (parentIndex !== -1) {
      let insertIndex = parentIndex + 1;
      while (
        insertIndex < conv.messages.length &&
        conv.messages[insertIndex].role === "assistant"
      ) {
        insertIndex++;
      }

      // Ensure monotonic timestamp ordering:
      // message timestamp must not be earlier than the parent message,
      // and if inserted before a subsequent message, must not exceed that next message.
      const parentMsg = conv.messages[parentIndex];
      const parentTime = new Date(parentMsg.timestamp || parentMsg.createdAt || 0).getTime();
      const nextMsg = insertIndex < conv.messages.length ? conv.messages[insertIndex] : null;
      const nextTime = nextMsg ? new Date(nextMsg.timestamp || nextMsg.createdAt || 0).getTime() : Infinity;

      let msgTime = new Date(message.timestamp || message.createdAt || 0).getTime();
      if (isNaN(msgTime) || msgTime < parentTime) {
        msgTime = parentTime;
      }
      if (nextTime !== Infinity && !isNaN(nextTime) && msgTime > nextTime) {
        msgTime = nextTime;
      }

      const isoTime = new Date(msgTime).toISOString();
      if (message.timestamp) message.timestamp = isoTime;
      if (message.createdAt) message.createdAt = isoTime;
      if (!message.timestamp && !message.createdAt) message.timestamp = isoTime;

      conv.messages.splice(insertIndex, 0, message);
      if (conv.messages.length > _maxMessages) {
        conv.messages.splice(0, conv.messages.length - _maxMessages);
      }
      conv.version = (conv.version || 1) + 1;
      if (conv.updatedAt !== undefined) {
        conv.updatedAt = message.timestamp || message.createdAt || new Date().toISOString();
      }
      return true;
    }
  }

  // Monotonic timestamp check against last message if appended at end
  if (conv.messages.length > 0) {
    const lastMsg = conv.messages[conv.messages.length - 1];
    const lastTime = new Date(lastMsg.timestamp || lastMsg.createdAt || 0).getTime();
    let msgTime = new Date(message.timestamp || message.createdAt || 0).getTime();
    if (!isNaN(lastTime) && (!isNaN(msgTime) && msgTime < lastTime)) {
      const isoTime = lastMsg.timestamp || lastMsg.createdAt || new Date(lastTime).toISOString();
      if (message.timestamp) message.timestamp = isoTime;
      if (message.createdAt) message.createdAt = isoTime;
      if (!message.timestamp && !message.createdAt) message.timestamp = isoTime;
    }
  }

  conv.messages.push(message);
  if (conv.messages.length > _maxMessages) {
    conv.messages.splice(0, conv.messages.length - _maxMessages);
  }
  conv.version = (conv.version || 1) + 1;
  if (conv.updatedAt !== undefined) {
    conv.updatedAt = message.timestamp || message.createdAt || new Date().toISOString();
  }
  return true;
}

/**
 * Update conversation attributes (e.g. title) with optional expectedVersion check.
 */
function update(id, fields = {}, options = {}) {
  const conv = conversations.get(id);
  if (!conv) return null;
  if (!Number.isInteger(conv.version) || conv.version < 1) conv.version = 1;

  if (options.expectedVersion !== undefined && conv.version !== options.expectedVersion) {
    if (options.throwOnConflict) {
      const err = new Error("Conversation version mismatch");
      err.code = "VERSION_CONFLICT";
      err.currentVersion = conv.version;
      err.expectedVersion = options.expectedVersion;
      throw err;
    }
    return { success: false, conflict: true, currentVersion: conv.version };
  }

  _touch(conv);
  if (typeof fields.title === "string") {
    conv.title = fields.title;
  }
  conv.version = (conv.version || 1) + 1;
  conv.updatedAt = new Date().toISOString();
  return { success: true, conversation: conv, version: conv.version };
}

function clear() {
  conversations.clear();
}

const conversationSync = require("./conversationSync");

// Deleted ids are kept so a stale client cannot bring a message back, but only
// the most recent ones: the list would otherwise grow for as long as the
// conversation lives.
const MAX_DELETED_IDS = 1000;

function _trackDeleted(conv, ids) {
  const list = Array.isArray(conv.deletedMessageIds) ? conv.deletedMessageIds : [];
  const seen = new Set(list);
  for (const id of ids) {
    if (id && !seen.has(id)) {
      seen.add(id);
      list.push(id);
    }
  }
  if (list.length > MAX_DELETED_IDS) list.splice(0, list.length - MAX_DELETED_IDS);
  conv.deletedMessageIds = list;
}

function maxMessages() {
  return _maxMessages;
}

/**
 * Truncate all messages in a conversation strictly after the specified message ID.
 */
function truncateAfter(id, messageId) {
  const conv = conversations.get(id);
  if (!conv || !Array.isArray(conv.messages)) return false;
  _touch(conv);
  const idx = conv.messages.findIndex((m) => m.id === messageId);
  if (idx === -1) return false;
  const removed = conv.messages.splice(idx + 1);
  if (removed.length > 0) {
    _trackDeleted(conv, removed.map((m) => m?.id));
  }
  conv.version = (conv.version || 1) + 1;
  conv.updatedAt = new Date().toISOString();
  return true;
}

/**
 * Truncate all messages from the specified message ID onward (inclusive).
 */
function truncateFrom(id, messageId) {
  const conv = conversations.get(id);
  if (!conv || !Array.isArray(conv.messages)) return false;
  _touch(conv);
  const idx = conv.messages.findIndex((m) => m.id === messageId);
  if (idx === -1) return false;
  const removed = conv.messages.splice(idx);
  if (removed.length > 0) {
    _trackDeleted(conv, removed.map((m) => m?.id));
  }
  conv.version = (conv.version || 1) + 1;
  conv.updatedAt = new Date().toISOString();
  return true;
}

/**
 * Delete a specific message from a conversation and track its deletion.
 */
function deleteMessage(id, messageId) {
  const conv = conversations.get(id);
  if (!conv || !Array.isArray(conv.messages)) return false;
  _touch(conv);
  const idx = conv.messages.findIndex((m) => m.id === messageId);
  if (idx === -1) return false;
  conv.messages.splice(idx, 1);
  _trackDeleted(conv, [messageId]);
  conv.version = (conv.version || 1) + 1;
  conv.updatedAt = new Date().toISOString();
  return true;
}

function _fingerprint(conv) {
  return JSON.stringify([conv.messages, conv.title, conv.deletedMessageIds || []]);
}

/**
 * Perform differential synchronization and conflict resolution against stored conversation.
 */
function sync(id, clientState = {}, options = {}) {
  const conv = conversations.get(id);
  if (!conv) return null;
  _touch(conv);

  const serverVersion = conv.version || 1;
  // A client message is the stored one if the diff would match it: by its id,
  // by its clientId, or by a stored message's clientId.
  const serverIds = new Set(conv.messages.map((m) => m?.id).filter(Boolean));
  const serverClientIds = new Set(conv.messages.map((m) => m?.clientId).filter(Boolean));
  const storedHere = (m) =>
    serverIds.has(m.id) || (m.clientId && serverIds.has(m.clientId)) || serverClientIds.has(m.id);
  const client = {
    ...clientState,
    // The server owns the version. A client claiming one ahead of it would
    // otherwise win every conflict and could set the counter to anything.
    clientVersion: Math.min(clientState.clientVersion || 1, serverVersion),
    // Every reply is stored by the server, under its own id, once it completes,
    // so a client only ever holds a copy of one. An assistant message the server
    // does not have is a reply it has not finished: one cut off by a dropped
    // socket arrives marked completed, under the very id the whole reply will
    // be stored under, and storing it made append() skip that reply as already
    // there (Issue #456). Sync updates replies the server has and never adds one.
    messages: (clientState.messages || []).filter((m) => m.role !== "assistant" || storedHere(m)),
  };

  const { diff, reconciled } = conversationSync.reconcileConversationDiff(client, conv, options);

  const before = _fingerprint(conv);
  if (diff.hasDifferences) {
    conv.messages = reconciled.messages;
    if (conv.messages.length > _maxMessages) {
      conv.messages.splice(0, conv.messages.length - _maxMessages);
    }
    conv.title = reconciled.title;
    if (reconciled.titleUpdatedAt) {
      conv.titleUpdatedAt = reconciled.titleUpdatedAt;
    }
    if (Array.isArray(reconciled.deletedMessageIds)) {
      conv.deletedMessageIds = [];
      _trackDeleted(conv, reconciled.deletedMessageIds);
    }
  }
  // Only a sync that changed something is a new revision, and it is one step on.
  if (_fingerprint(conv) !== before) {
    conv.version = serverVersion + 1;
    conv.updatedAt = new Date().toISOString();
  }

  let responseStatus = "synchronized";
  if (!diff.hasDifferences) {
    responseStatus = "up_to_date";
  } else if (diff.status === "server_ahead") {
    responseStatus = "server_ahead";
  } else if (diff.status === "client_ahead") {
    responseStatus = "client_ahead";
  }

  return {
    status: responseStatus,
    version: conv.version,
    serverVersion: conv.version,
    clientVersion: clientState.clientVersion,
    diff,
    conversation: conv,
  };
}

/**
 * Update properties of an existing message in a conversation.
 */
function updateMessage(id, messageId, updates = {}) {
  const conv = conversations.get(id);
  if (!conv || !Array.isArray(conv.messages)) return false;
  _touch(conv);
  const idx = conv.messages.findIndex((m) => m.id === messageId);
  if (idx === -1) return false;
  conv.messages[idx] = {
    ...conv.messages[idx],
    ...updates,
  };
  conv.version = (conv.version || 1) + 1;
  conv.updatedAt = new Date().toISOString();
  return true;
}

/**
 * Override growth limits. Pass no arguments to restore production defaults.
 * For tests only — not part of the public API.
 */
function _resetLimits({ max, ttl, maxMessages } = {}) {
  _max = max !== undefined ? max : (Number(process.env.FRAMER_MAX_CONVERSATIONS) || 1000);
  _ttl = ttl !== undefined ? ttl : (Number(process.env.FRAMER_CONVERSATION_TTL_MS) || 24 * 60 * 60 * 1000);
  _maxMessages = maxMessages !== undefined ? maxMessages : (Number(process.env.FRAMER_MAX_MESSAGES_PER_CONVERSATION) || 1000);
}

module.exports = {
  create, get, has, remove, list, messages, append, update, clear,
  truncateAfter, truncateFrom, updateMessage, deleteMessage, sync,
  maxMessages, MAX_DELETED_IDS,
  _map: conversations,
  _evict,
  _resetLimits,
  get _maxMessages() { return _maxMessages; },
};
