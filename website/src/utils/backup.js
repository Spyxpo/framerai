/**
 * Conversation Backup, Export, Import, and Restore System (Issue #411).
 *
 * Provides structured, versioned, and atomic backup export and restoration for
 * single and multiple conversations while preserving complete parent/child branch
 * relationships, metadata, timestamps, ordering, and message IDs.
 */

import { sanitizeConversation, sanitizeMessage } from "./storage";
import { dedupeMessages } from "./dedupe";

export const BACKUP_VERSION = 1;
export const SUPPORTED_BACKUP_VERSIONS = [1];
export const MAX_IMPORT_FILE_SIZE = 10 * 1024 * 1024; // 10MB limit

function generateUuid() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `import-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Creates a versioned backup payload from an array of conversations (or a single conversation).
 *
 * @param {Array<Object>|Object} conversations - List of conversations or a single conversation object
 * @returns {Object} Structured and versioned backup object
 */
export function createBackup(conversations) {
  const convList = Array.isArray(conversations)
    ? conversations
    : conversations && typeof conversations === "object"
    ? [conversations]
    : [];

  const sanitized = convList.map(sanitizeConversation).filter(Boolean);

  return {
    version: BACKUP_VERSION,
    framerai_backup: true,
    exportedAt: new Date().toISOString(),
    conversationCount: sanitized.length,
    conversations: sanitized,
  };
}

/**
 * Exports a single conversation into human-readable Markdown format.
 *
 * @param {Object} conversation - Conversation object
 * @returns {string} Markdown text representation of the conversation
 */
export function exportConversationToMarkdown(conversation) {
  if (!conversation || typeof conversation !== "object") return "";
  const title = conversation.title || "Conversation";
  const messages = Array.isArray(conversation.messages) ? conversation.messages : [];

  const lines = [
    `# ${title}`,
    "",
    `*Exported on ${new Date().toISOString()}*`,
  ];

  if (conversation.parentConversationId) {
    lines.push(`*Parent Conversation: \`${conversation.parentConversationId}\`*`);
  }
  if (conversation.branchedFromMessageId) {
    lines.push(`*Branched from Message: \`${conversation.branchedFromMessageId}\`*`);
  }

  lines.push("", "---", "");

  for (const m of messages) {
    const roleLabel =
      m.role === "user"
        ? "User"
        : m.role === "assistant"
        ? "Assistant"
        : m.role === "system"
        ? "System"
        : m.role
        ? m.role.charAt(0).toUpperCase() + m.role.slice(1)
        : "Unknown";
    const timeStr = m.timestamp ? ` (${m.timestamp})` : "";
    lines.push(`### ${roleLabel}${timeStr}`);
    lines.push("");
    if (m.content) {
      lines.push(m.content);
    }
    if (Array.isArray(m.attachments) && m.attachments.length > 0) {
      lines.push("");
      lines.push(`*Attachments: ${m.attachments.join(", ")}*`);
    }
    lines.push("", "---", "");
  }

  return lines.join("\n");
}

/**
 * Exports a single conversation into plain text format.
 *
 * @param {Object} conversation - Conversation object
 * @returns {string} Plain text representation
 */
export function exportConversationToPlainText(conversation) {
  if (!conversation || typeof conversation !== "object") return "";
  const title = conversation.title || "Conversation";
  const messages = Array.isArray(conversation.messages) ? conversation.messages : [];

  const lines = [
    `TITLE: ${title}`,
    `DATE: ${new Date().toISOString()}`,
  ];
  if (conversation.parentConversationId) {
    lines.push(`PARENT: ${conversation.parentConversationId}`);
  }
  lines.push("=".repeat(40), "");

  for (const m of messages) {
    const roleLabel = (m.role || "UNKNOWN").toUpperCase();
    const timeStr = m.timestamp ? ` [${m.timestamp}]` : "";
    lines.push(`${roleLabel}${timeStr}:`);
    lines.push(m.content || "");
    lines.push("-".repeat(40));
  }

  return lines.join("\n");
}

/**
 * Parses a Markdown-formatted conversation into a structured conversation object.
 * Preserves message ordering, roles, timestamps, attachments, code blocks,
 * multiline text, Unicode, and branching metadata.
 *
 * @param {string} markdownText - Raw markdown text
 * @returns {{ success: boolean, conversation?: Object, error?: string }}
 */
export function parseMarkdownConversation(markdownText) {
  if (typeof markdownText !== "string") {
    return { success: false, error: "Invalid Markdown content: expected string." };
  }
  const trimmed = markdownText.trim();
  if (!trimmed) {
    return { success: false, error: "Empty Markdown content: no content provided." };
  }

  const lines = markdownText.split(/\r?\n/);
  let title = "Imported Conversation";
  let parentConversationId = undefined;
  let branchedFromMessageId = undefined;
  const messages = [];

  let currentRole = null;
  let currentTimestamp = null;
  let currentLines = [];
  let currentAttachments = [];
  let inCodeBlock = false;

  const flushMessage = () => {
    if (currentRole) {
      const content = currentLines.join("\n").trim();
      const msg = {
        id: generateUuid(),
        role: currentRole,
        content: content,
        type: "text",
      };
      if (currentTimestamp) {
        msg.timestamp = currentTimestamp;
      }
      if (currentAttachments.length > 0) {
        msg.attachments = [...currentAttachments];
      }
      messages.push(msg);
    }
    currentRole = null;
    currentTimestamp = null;
    currentLines = [];
    currentAttachments = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Code block fences toggle code block mode
    if (/^\s*```/.test(line)) {
      inCodeBlock = !inCodeBlock;
      if (currentRole) {
        currentLines.push(line);
      }
      continue;
    }

    if (inCodeBlock) {
      if (currentRole) {
        currentLines.push(line);
      }
      continue;
    }

    // Header metadata before the first message begins
    if (!currentRole && messages.length === 0) {
      const titleMatch = line.match(/^#\s+(.+)$/);
      if (titleMatch) {
        title = titleMatch[1].trim();
        continue;
      }
      const parentMatch = line.match(/^\*Parent Conversation:\s*`?([^`*]+)`?\*/i);
      if (parentMatch) {
        parentConversationId = parentMatch[1].trim();
        continue;
      }
      const branchMatch = line.match(/^\*Branched from Message:\s*`?([^`*]+)`?\*/i);
      if (branchMatch) {
        branchedFromMessageId = branchMatch[1].trim();
        continue;
      }
    }

    // Role header matches:
    // ### User (2026-08-01T12:00:00.000Z)
    // ### Assistant
    // ## User
    // #### Assistant
    // **User**:
    // **User** (2026-08-01T10:00:00.000Z):
    // User:
    const roleMatch =
      line.match(/^(?:#{1,6}\s+|\*\*)?(User|Assistant|System|Human|AI)(?:\*\*)?(?:\s*\(([^)]+)\))?\s*:?\s*$/i);

    if (roleMatch) {
      flushMessage();
      const rawRole = roleMatch[1].toLowerCase();
      currentRole = rawRole === "human" ? "user" : rawRole === "ai" ? "assistant" : rawRole;
      const rawTime = roleMatch[2]?.trim();
      if (rawTime) {
        currentTimestamp = rawTime;
      }
      continue;
    }

    if (currentRole) {
      // Horizontal rule line separating messages
      if (/^---+$/.test(line.trim())) {
        flushMessage();
        continue;
      }

      // Attachments metadata
      const attachMatch = line.match(/^\*Attachments:\s*(.+)\*$/i);
      if (attachMatch) {
        currentAttachments = attachMatch[1]
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
        continue;
      }

      currentLines.push(line);
    }
  }

  flushMessage();

  const convId = generateUuid();
  const conversation = {
    id: convId,
    title,
    messages,
    createdAt: messages[0]?.timestamp || new Date().toISOString(),
    updatedAt: messages[messages.length - 1]?.timestamp || new Date().toISOString(),
    ...(parentConversationId ? { parentConversationId } : {}),
    ...(branchedFromMessageId ? { branchedFromMessageId } : {}),
  };

  return {
    success: true,
    conversation,
  };
}

/**
 * Validates a backup payload or raw JSON/Markdown string before any mutations occur.
 *
 * @param {string|Object} backupData - Raw JSON/Markdown string or parsed backup object
 * @param {Object} [options={}] - Optional metadata such as filename
 * @returns {{ valid: boolean, error?: string, backup?: Object, format?: string }} Validation result
 */
export function validateBackup(backupData, options = {}) {
  if (backupData === null || backupData === undefined) {
    return { valid: false, error: "Empty backup data: no content provided." };
  }

  // Size limit validation
  if (backupData && typeof backupData === "object" && typeof backupData.size === "number") {
    if (backupData.size > MAX_IMPORT_FILE_SIZE) {
      return { valid: false, error: "File exceeds maximum allowed size of 10MB." };
    }
  }
  if (typeof Blob !== "undefined" && backupData instanceof Blob) {
    if (backupData.size > MAX_IMPORT_FILE_SIZE) {
      return { valid: false, error: "File exceeds maximum allowed size of 10MB." };
    }
  }
  if (typeof backupData === "string") {
    let stringBytes;
    if (typeof Blob !== "undefined") {
      stringBytes = new Blob([backupData]).size;
    } else if (typeof globalThis !== "undefined" && globalThis.Buffer) {
      stringBytes = globalThis.Buffer.byteLength(backupData, "utf8");
    } else {
      stringBytes = backupData.length;
    }
    if (stringBytes > MAX_IMPORT_FILE_SIZE) {
      return { valid: false, error: "File exceeds maximum allowed size of 10MB." };
    }
  }

  let parsed = backupData;
  let isMarkdown = false;

  if (typeof backupData === "string") {
    const trimmed = backupData.trim();
    if (!trimmed) {
      return { valid: false, error: "Empty backup file: no content provided." };
    }

    const filename = options?.filename || "";
    const isMdExt = /\.(md|markdown)$/i.test(filename);

    if (isMdExt) {
      const mdResult = parseMarkdownConversation(backupData);
      if (mdResult.success) {
        parsed = {
          version: BACKUP_VERSION,
          framerai_backup: true,
          exportedAt: new Date().toISOString(),
          conversationCount: 1,
          conversations: [mdResult.conversation],
        };
        isMarkdown = true;
      } else {
        return { valid: false, error: mdResult.error || "Failed to parse Markdown file." };
      }
    } else {
      try {
        parsed = JSON.parse(backupData);
      } catch {
        const looksLikeJson = /^\s*[[{]/.test(trimmed);
        if (!looksLikeJson) {
          const mdResult = parseMarkdownConversation(backupData);
          if (mdResult.success && mdResult.conversation.messages.length > 0) {
            parsed = {
              version: BACKUP_VERSION,
              framerai_backup: true,
              exportedAt: new Date().toISOString(),
              conversationCount: 1,
              conversations: [mdResult.conversation],
            };
            isMarkdown = true;
          } else {
            return { valid: false, error: "Malformed JSON file: failed to parse backup data." };
          }
        } else {
          return { valid: false, error: "Malformed JSON file: failed to parse backup data." };
        }
      }
    }
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { valid: false, error: "Invalid backup format: root must be a JSON object." };
  }

  // Handle single-conversation JSON export format
  if (!Array.isArray(parsed.conversations) && Array.isArray(parsed.messages)) {
    parsed = {
      version: typeof parsed.version === "number" ? parsed.version : BACKUP_VERSION,
      framerai_backup: true,
      exportedAt: parsed.exportedAt || new Date().toISOString(),
      conversationCount: 1,
      conversations: [
        {
          id: typeof parsed.id === "string" && parsed.id.trim() ? parsed.id : generateUuid(),
          title: typeof parsed.title === "string" ? parsed.title : "Imported Conversation",
          messages: parsed.messages,
          ...(parsed.parentConversationId ? { parentConversationId: parsed.parentConversationId } : {}),
          ...(parsed.branchedFromMessageId ? { branchedFromMessageId: parsed.branchedFromMessageId } : {}),
          ...(parsed.createdAt ? { createdAt: parsed.createdAt } : {}),
          ...(parsed.updatedAt ? { updatedAt: parsed.updatedAt } : {}),
        },
      ],
    };
  }

  if (typeof parsed.version !== "number") {
    return { valid: false, error: "Invalid backup: missing required 'version' field." };
  }

  if (!SUPPORTED_BACKUP_VERSIONS.includes(parsed.version)) {
    return {
      valid: false,
      error: `Unsupported backup version ${parsed.version}. Supported versions: ${SUPPORTED_BACKUP_VERSIONS.join(", ")}.`,
    };
  }

  if (!Array.isArray(parsed.conversations)) {
    return { valid: false, error: "Invalid backup: 'conversations' field must be an array." };
  }

  if (parsed.conversations.length === 0) {
    return { valid: false, error: "Backup contains no conversations." };
  }

  for (let i = 0; i < parsed.conversations.length; i++) {
    const conv = parsed.conversations[i];
    if (!conv || typeof conv !== "object" || Array.isArray(conv)) {
      return { valid: false, error: `Invalid conversation at index ${i}: must be an object.` };
    }
    if (typeof conv.id !== "string" || !conv.id.trim()) {
      return { valid: false, error: `Invalid conversation at index ${i}: missing or invalid 'id'.` };
    }
    if (conv.title !== undefined && typeof conv.title !== "string") {
      return { valid: false, error: `Invalid conversation '${conv.id}': 'title' must be a string.` };
    }
    if (!Array.isArray(conv.messages)) {
      return { valid: false, error: `Invalid conversation '${conv.id}': 'messages' must be an array.` };
    }
    if (conv.parentConversationId !== undefined && typeof conv.parentConversationId !== "string") {
      return {
        valid: false,
        error: `Invalid conversation '${conv.id}': 'parentConversationId' must be a string.`,
      };
    }
    if (conv.branchedFromMessageId !== undefined && typeof conv.branchedFromMessageId !== "string") {
      return {
        valid: false,
        error: `Invalid conversation '${conv.id}': 'branchedFromMessageId' must be a string.`,
      };
    }

    // Validate messages
    for (let mIdx = 0; mIdx < conv.messages.length; mIdx++) {
      const msg = conv.messages[mIdx];
      if (!msg || typeof msg !== "object" || Array.isArray(msg)) {
        return {
          valid: false,
          error: `Invalid message at index ${mIdx} in conversation '${conv.id}': must be an object.`,
        };
      }
      if (typeof msg.id !== "string" || !msg.id.trim()) {
        return {
          valid: false,
          error: `Invalid message at index ${mIdx} in conversation '${conv.id}': missing or invalid 'id'.`,
        };
      }
      if (typeof msg.role !== "string" || !msg.role.trim()) {
        return {
          valid: false,
          error: `Invalid message at index ${mIdx} in conversation '${conv.id}': missing or invalid 'role'.`,
        };
      }
      if (msg.content !== undefined && typeof msg.content !== "string") {
        return {
          valid: false,
          error: `Invalid message at index ${mIdx} in conversation '${conv.id}': 'content' must be a string.`,
        };
      }
      if (msg.metadata !== undefined && (typeof msg.metadata !== "object" || Array.isArray(msg.metadata))) {
        return {
          valid: false,
          error: `Invalid message at index ${mIdx} in conversation '${conv.id}': 'metadata' must be an object.`,
        };
      }
    }
  }

  return { valid: true, backup: parsed, format: isMarkdown ? "markdown" : "json" };
}

/**
 * Restores conversations from a validated backup payload, resolving ID collisions
 * without overwriting existing conversations while preserving branch hierarchies.
 *
 * @param {string|Object} backupData - Raw JSON/Markdown string or parsed backup object
 * @param {Array<Object>} existingConversations - Currently existing conversations
 * @param {Object} [options={}] - Optional metadata such as filename
 * @returns {{ success: boolean, error?: string, conversations?: Array<Object>, count?: number }}
 */
export function restoreBackup(backupData, existingConversations = [], options = {}) {
  const validation = validateBackup(backupData, options);
  if (!validation.valid) {
    return { success: false, error: validation.error };
  }

  const backup = validation.backup;
  const existingConvIds = new Set((existingConversations || []).map((c) => c?.id).filter(Boolean));
  const existingMsgIds = new Set();
  for (const c of existingConversations || []) {
    if (Array.isArray(c?.messages)) {
      for (const m of c.messages) {
        if (m?.id) existingMsgIds.add(m.id);
      }
    }
  }

  // Detect and remap collisions
  const convIdMap = new Map();
  const msgIdMap = new Map();
  const seenImportConvIds = new Set();
  const seenImportMsgIds = new Set();

  for (const conv of backup.conversations) {
    if (existingConvIds.has(conv.id) || seenImportConvIds.has(conv.id)) {
      const newId = generateUuid();
      convIdMap.set(conv.id, newId);
      seenImportConvIds.add(newId);
    } else {
      convIdMap.set(conv.id, conv.id);
      seenImportConvIds.add(conv.id);
    }

    for (const msg of conv.messages) {
      if (!msg?.id) continue;
      if (existingMsgIds.has(msg.id) || seenImportMsgIds.has(msg.id)) {
        const newMsgId = generateUuid();
        msgIdMap.set(msg.id, newMsgId);
        seenImportMsgIds.add(newMsgId);
      } else {
        msgIdMap.set(msg.id, msg.id);
        seenImportMsgIds.add(msg.id);
      }
    }
  }

  const restoredConversations = [];
  const seenAllMsgIds = new Set(existingMsgIds);

  for (const conv of backup.conversations) {
    const targetConvId = convIdMap.get(conv.id) || conv.id;
    const targetParentId = conv.parentConversationId
      ? convIdMap.get(conv.parentConversationId) || conv.parentConversationId
      : undefined;
    const targetBranchedMsgId = conv.branchedFromMessageId
      ? msgIdMap.get(conv.branchedFromMessageId) || conv.branchedFromMessageId
      : undefined;

    const mappedMessages = [];
    const seenInConv = new Set();
    for (const m of conv.messages) {
      let targetMsgId = msgIdMap.get(m.id) || m.id;
      if (!targetMsgId || seenInConv.has(targetMsgId) || seenAllMsgIds.has(targetMsgId)) {
        targetMsgId = generateUuid();
      }
      seenInConv.add(targetMsgId);
      seenAllMsgIds.add(targetMsgId);

      const sanitized = sanitizeMessage({
        ...m,
        id: targetMsgId,
      });
      if (sanitized) {
        mappedMessages.push(sanitized);
      }
    }

    // Reuse dedupeMessages to eliminate duplicate message IDs within the conversation
    const dedupedMessages = dedupeMessages(mappedMessages);

    const sanitizedConv = sanitizeConversation({
      ...conv,
      id: targetConvId,
      ...(targetParentId ? { parentConversationId: targetParentId } : {}),
      ...(targetBranchedMsgId ? { branchedFromMessageId: targetBranchedMsgId } : {}),
      messages: dedupedMessages,
    });

    if (sanitizedConv) {
      restoredConversations.push(sanitizedConv);
    }
  }

  return {
    success: true,
    conversations: restoredConversations,
    count: restoredConversations.length,
  };
}

/**
 * Triggers a client-side file download.
 *
 * @param {string} content - File content
 * @param {string} filename - Desired filename
 * @param {string} mimeType - MIME type
 */
export function downloadFile(content, filename, mimeType = "application/json") {
  if (typeof window === "undefined" || typeof document === "undefined") return;
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
