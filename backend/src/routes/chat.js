const express = require("express");
const router = express.Router();
const { randomUUID } = require("node:crypto");
const { processMessage, validateTrace, traceAllowed } = require("../services/model");
const { ApiError, asyncHandler } = require("../middleware/errors");
const { validator, Validator } = require("../middleware/validate");
const { generationLimiter } = require("../middleware/limiters");
const { readSettings } = require("../generationSettings");

// The store is shared with the WebSocket service, so a streamed turn sees the
// same history a posted one does.
const conversations = require("../conversationStore");

const MESSAGE_TYPES = ["text", "code", "image", "video", "audio"];
const SYNC_ROLES = ["user", "assistant"];
const modelLimits = require("../modelLimits");

// The documented floor. The accepted length rises with the window of the model
// actually loaded, so a preset with a million-token context is reachable
// instead of being held to a constant that fits neither end of the range.
const MAX_MESSAGE_LENGTH = modelLimits.BASE_MESSAGE_CHARS;

function getConversation(id) {
  const conv = conversations.get(id);
  if (!conv) throw ApiError.notFound("Conversation not found");
  return conv;
}

function conversationId(req) {
  const v = validator(req.params);
  const id = v.uuid("id");
  v.done();
  return id;
}

// Create new conversation
router.post("/conversations", (req, res) => {
  const id = randomUUID();
  const now = new Date().toISOString();
  const conv = {
    id,
    title: "New Chat",
    messages: [],
    version: 1,
    createdAt: now,
    updatedAt: now,
  };
  conversations.create(conv);
  res.json(conv);
});

// List conversations
router.get("/conversations", (req, res) => {
  const list = conversations.list()
    .map(({ id, title, createdAt, updatedAt, version, messages, parentConversationId, parentVersion, branchedFromMessageId }) => ({
      id,
      title,
      createdAt,
      updatedAt: updatedAt || createdAt,
      version: version || 1,
      messageCount: messages.length,
      ...(parentConversationId ? { parentConversationId, branchedFromMessageId, ...(parentVersion ? { parentVersion } : {}) } : {}),
    }))
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  res.json(list);
});

// Get conversation
router.get("/conversations/:id", (req, res) => {
  res.json(getConversation(conversationId(req)));
});

// Update conversation (e.g. rename) with optimistic concurrency support
router.patch("/conversations/:id", (req, res) => {
  const id = conversationId(req);
  const conv = getConversation(id);

  const v = validator(req.body);
  const title = v.string("title", { min: 1, max: 200, optional: true });
  const expectedVersion = req.body.expectedVersion !== undefined ? Number(req.body.expectedVersion) : undefined;
  v.done();

  if (expectedVersion !== undefined && (conv.version || 1) !== expectedVersion) {
    throw ApiError.conflict("Conversation version mismatch", {
      currentVersion: conv.version || 1,
      expectedVersion,
      conversation: conv,
    });
  }

  const updates = {};
  if (title) updates.title = title;
  const result = conversations.update(id, updates);
  res.json(result.conversation);
});

// Branch conversation
router.post("/conversations/:id/branch", (req, res) => {
  const parentId = conversationId(req);
  const conv = getConversation(parentId);

  const v = validator(req.body);
  const messageId = v.uuid("messageId");
  const expectedVersion = req.body.expectedVersion !== undefined ? Number(req.body.expectedVersion) : undefined;
  v.done();

  if (expectedVersion !== undefined && (conv.version || 1) !== expectedVersion) {
    throw ApiError.conflict("Conversation version mismatch", {
      currentVersion: conv.version || 1,
      expectedVersion,
      conversation: conv,
    });
  }

  const messageIndex = conv.messages.findIndex((m) => m.id === messageId);
  if (messageIndex === -1) {
    throw ApiError.badRequest("Message not found in conversation");
  }

  // Slice history up to and including the branched message, deep cloning messages
  const branchedMessages = conv.messages.slice(0, messageIndex + 1).map((m) => ({
    ...m,
    ...(m.attachments ? { attachments: [...m.attachments] } : {}),
    ...(m.metadata ? { metadata: { ...m.metadata } } : {}),
  }));

  const branchId = randomUUID();
  const branchTitle = `${conv.title} (Branch)`;
  const now = new Date().toISOString();
  const branchConv = {
    id: branchId,
    title: branchTitle,
    parentConversationId: parentId,
    parentVersion: conv.version || 1,
    branchedFromMessageId: messageId,
    messages: branchedMessages,
    version: 1,
    createdAt: now,
    updatedAt: now,
  };

  conversations.create(branchConv);
  res.json(branchConv);
});

// The deleted ids a sync may carry. The store keeps the same number, so a list
// longer than this could never be honoured anyway.
const MAX_SYNC_DELETED_IDS = conversations.MAX_DELETED_IDS;

// A synced message is held to the same rules as one posted to /messages: the
// stored history is what the model reads, so a sync must not be a way around
// the role, type or length checks. Only known fields are kept.
function readSyncMessages(v) {
  const raw = v.array("messages", { max: conversations.maxMessages() });
  const limit = modelLimits.messageChars();
  const out = [];
  raw.forEach((item, i) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      v.fail(`messages.${i}`, "must be an object");
      return;
    }
    const m = new Validator(item, { prefix: `messages.${i}.`, errors: v.errors });
    const id = m.uuid("id");
    const role = m.oneOf("role", SYNC_ROLES, { required: true });
    const type = m.oneOf("type", MESSAGE_TYPES, { fallback: "text" });
    const attachments = m.array("attachments", { max: 10, fallback: undefined });
    const content = item.content === undefined ? "" : item.content;
    if (typeof content !== "string") m.fail("content", "must be a string");
    else if (content.length > limit) m.fail("content", `must be at most ${limit} characters`);
    const clientId = item.clientId !== undefined ? m.uuid("clientId") : undefined;
    const replyToId = item.replyToId !== undefined ? m.uuid("replyToId") : undefined;
    const timestamp = m.string("timestamp", { max: 64 });
    if (item.completed !== undefined && typeof item.completed !== "boolean") {
      m.fail("completed", "must be a boolean");
    }

    out.push({
      id,
      role,
      content: typeof content === "string" ? content : "",
      type,
      ...(attachments ? { attachments } : {}),
      ...(clientId ? { clientId } : {}),
      ...(replyToId ? { replyToId } : {}),
      ...(timestamp ? { timestamp } : {}),
      ...(typeof item.completed === "boolean" ? { completed: item.completed } : {}),
    });
  });
  return out;
}

function readDeletedIds(v) {
  const raw = v.array("deletedMessageIds", { max: MAX_SYNC_DELETED_IDS });
  const ids = new Validator(raw, { prefix: "deletedMessageIds.", errors: v.errors });
  return raw.map((_, i) => ids.uuid(i));
}

// Differential conversation synchronization and conflict resolution
router.post("/conversations/:id/sync", (req, res) => {
  const id = conversationId(req);
  getConversation(id);

  const v = validator(req.body);
  const title = v.string("title", { min: 1, max: 200, optional: true });
  const clientVersion = v.integer("clientVersion", { min: 1 });
  // A clock ahead of the server's would win every later rename, so it is held to now.
  const titleUpdatedAt = v.integer("titleUpdatedAt", { min: 0, max: Date.now() + 60_000 });
  const messages = readSyncMessages(v);
  const deletedMessageIds = readDeletedIds(v);
  v.done();

  const syncResult = conversations.sync(id, {
    clientVersion,
    messages,
    deletedMessageIds,
    title,
    titleUpdatedAt,
  });

  res.json(syncResult);
});

// Delete message from conversation
router.delete("/conversations/:id/messages/:messageId", (req, res) => {
  const id = conversationId(req);
  getConversation(id);

  const v = validator(req.params);
  const messageId = v.uuid("messageId");
  v.done();

  const success = conversations.deleteMessage(id, messageId);
  if (!success) {
    throw ApiError.notFound("Message not found in conversation");
  }
  const conv = conversations.get(id);
  res.json({ success: true, version: conv.version, messageId });
});

// Delete conversation
router.delete("/conversations/:id", (req, res) => {
  conversations.remove(conversationId(req));
  res.json({ success: true });
});

// Send message. This runs the model, so it shares the generation rate limit.
router.post(
  "/conversations/:id/messages",
  generationLimiter,
  asyncHandler(async (req, res) => {
    const conv = getConversation(conversationId(req));

    const v = validator(req.body);
    const isRegenerate = Boolean(req.body.regenerateMessageId || req.body.regenerate);
    const content = v.string("content", { required: !isRegenerate, max: modelLimits.messageChars() });
    const type = v.oneOf("type", MESSAGE_TYPES, { fallback: "text" });
    const attachments = v.array("attachments", { max: 10 });
    const expectedVersion = req.body.expectedVersion !== undefined ? Number(req.body.expectedVersion) : undefined;
    const settings = readSettings(v);
    let editMessageId;
    if (req.body.editMessageId !== undefined) {
      editMessageId = v.uuid("editMessageId");
    }
    let regenerateMessageId;
    if (req.body.regenerateMessageId !== undefined) {
      regenerateMessageId = v.uuid("regenerateMessageId");
    }
    v.done();

    if (expectedVersion !== undefined && (conv.version || 1) !== expectedVersion) {
      throw ApiError.conflict("Conversation version mismatch", {
        currentVersion: conv.version || 1,
        expectedVersion,
        conversation: conv,
      });
    }

    let userMessage;
    if (editMessageId) {
      const editIdx = conv.messages ? conv.messages.findIndex((m) => m.id === editMessageId) : -1;
      if (editIdx !== -1) {
        conv.messages[editIdx] = {
          ...conv.messages[editIdx],
          content,
          type,
          attachments,
          timestamp: new Date().toISOString(),
        };
        conversations.truncateAfter(conv.id, editMessageId);
        userMessage = conv.messages[editIdx];
      }
    } else if (regenerateMessageId) {
      const regenIdx = conv.messages ? conv.messages.findIndex((m) => m.id === regenerateMessageId) : -1;
      if (regenIdx !== -1) {
        conversations.truncateFrom(conv.id, regenerateMessageId);
        for (let i = conv.messages.length - 1; i >= 0; i--) {
          if (conv.messages[i]?.role === "user") {
            userMessage = conv.messages[i];
            break;
          }
        }
      }
    }

    if (!userMessage) {
      userMessage = {
        id: randomUUID(),
        role: "user",
        content,
        type,
        attachments,
        timestamp: new Date().toISOString(),
      };
      conversations.append(conv.id, userMessage);
    }

    // Update title from first message
    if (conv.messages.length === 1 && content) {
      conv.title = content.substring(0, 50) + (content.length > 50 ? "..." : "");
    }

    const operatorCtx = { operator: req.headers["x-operator"] === "true" };
    const response = await processMessage(conv.messages, type, settings, req.requestId, operatorCtx);

    // Privacy: validate and strip trace from response if not allowed (defense in depth).
    // processMessage already filters, but this ensures traces can't leak even
    // if mocked/bypassed for testing or if implementation changes. The gate is
    // the same one processMessage uses, so there is one rule, not two.
    if (response.metadata?.trace) {
      const validated = traceAllowed(operatorCtx) ? validateTrace(response.metadata.trace) : null;
      if (validated) {
        response.metadata.trace = validated;
      } else {
        delete response.metadata.trace;
      }
    }

    const assistantMessage = {
      id: randomUUID(),
      role: "assistant",
      content: response.content,
      type: response.type,
      metadata: response.metadata || {},
      timestamp: new Date().toISOString(),
    };
    conversations.append(conv.id, assistantMessage, { replyToId: userMessage.id });
    // The reply is the response, so the user's message would otherwise never
    // learn the id it was stored under, and a client holding a different one
    // could not ask for it again (branching). Added to the response only: the
    // stored reply stays a plain message.
    res.json({ ...assistantMessage, userMessageId: userMessage.id, version: conv.version });
  })
);

router.MESSAGE_TYPES = MESSAGE_TYPES;
router.MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH;

module.exports = router;
