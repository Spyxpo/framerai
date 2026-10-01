const express = require("express");
const router = express.Router();
const { randomUUID } = require("node:crypto");
const { processMessage, validateTrace, traceAllowed } = require("../services/model");
const { ApiError, asyncHandler } = require("../middleware/errors");
const { validator } = require("../middleware/validate");
const { generationLimiter } = require("../middleware/limiters");
const { readSettings } = require("../generationSettings");

// The store is shared with the WebSocket service, so a streamed turn sees the
// same history a posted one does.
const conversations = require("../conversationStore");

const MESSAGE_TYPES = ["text", "code", "image", "video", "audio"];
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
  conversations.create({
    id,
    title: "New Chat",
    messages: [],
    createdAt: new Date().toISOString(),
  });
  res.json({ id, title: "New Chat", messages: [] });
});

// List conversations
router.get("/conversations", (req, res) => {
  const list = conversations.list()
    .map(({ id, title, createdAt, messages, parentConversationId, branchedFromMessageId }) => ({
      id,
      title,
      createdAt,
      messageCount: messages.length,
      ...(parentConversationId ? { parentConversationId, branchedFromMessageId } : {}),
    }))
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  res.json(list);
});

// Get conversation
router.get("/conversations/:id", (req, res) => {
  res.json(getConversation(conversationId(req)));
});

// Branch conversation
router.post("/conversations/:id/branch", (req, res) => {
  const parentId = conversationId(req);
  const conv = getConversation(parentId);

  const v = validator(req.body);
  const messageId = v.uuid("messageId");
  v.done();

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
  const branchConv = {
    id: branchId,
    title: branchTitle,
    parentConversationId: parentId,
    branchedFromMessageId: messageId,
    messages: branchedMessages,
    createdAt: new Date().toISOString(),
  };

  conversations.create(branchConv);
  res.json(branchConv);
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
    const content = v.string("content", { required: true, max: modelLimits.messageChars() });
    const type = v.oneOf("type", MESSAGE_TYPES, { fallback: "text" });
    const attachments = v.array("attachments", { max: 10 });
    const settings = readSettings(v);
    v.done();

    const userMessage = {
      id: randomUUID(),
      role: "user",
      content,
      type,
      attachments,
      timestamp: new Date().toISOString(),
    };
    conversations.append(conv.id, userMessage);

    // Update title from first message
    if (conv.messages.length === 1) {
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
    conversations.append(conv.id, assistantMessage);
    // The reply is the response, so the user's message would otherwise never
    // learn the id it was stored under, and a client holding a different one
    // could not ask for it again (branching). Added to the response only: the
    // stored reply stays a plain message.
    res.json({ ...assistantMessage, userMessageId: userMessage.id });
  })
);

router.MESSAGE_TYPES = MESSAGE_TYPES;
router.MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH;

module.exports = router;
