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
  if (conv) _touch(conv);
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
 *
 * An unknown id is ignored rather than created: the WebSocket path may name a
 * conversation the REST path never opened, and inventing one there would build
 * a second history that nothing else can see.
 */
function append(id, message) {
  const conv = conversations.get(id);
  if (!conv) return false;
  _touch(conv);
  conv.messages.push(message);
  if (conv.messages.length > _maxMessages) {
    conv.messages.splice(0, conv.messages.length - _maxMessages);
  }
  return true;
}

function clear() {
  conversations.clear();
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
  create, get, has, remove, list, messages, append, clear,
  _map: conversations,
  _evict,
  _resetLimits,
  get _maxMessages() { return _maxMessages; },
};
