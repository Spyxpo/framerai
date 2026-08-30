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
 */

const conversations = new Map();

function create(conversation) {
  conversations.set(conversation.id, conversation);
  return conversation;
}

function get(id) {
  return conversations.get(id) || null;
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
  const conversation = conversations.get(id);
  return conversation ? conversation.messages : [];
}

/**
 * Record a turn against a conversation, if it exists.
 *
 * An unknown id is ignored rather than created: the WebSocket path may name a
 * conversation the REST path never opened, and inventing one there would build
 * a second history that nothing else can see.
 */
function append(id, message) {
  const conversation = conversations.get(id);
  if (!conversation) return false;
  conversation.messages.push(message);
  return true;
}

function clear() {
  conversations.clear();
}

module.exports = { create, get, has, remove, list, messages, append, clear, _map: conversations };
