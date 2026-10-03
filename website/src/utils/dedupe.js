/**
 * Deduplicates an array of messages by message `id`.
 * Messages without an `id` are preserved.
 * The first occurrence of each unique `id` is retained.
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
