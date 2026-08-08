const express = require("express");
const router = express.Router();
const { randomUUID } = require("node:crypto");
const { processMessage } = require("../services/model");
const { ApiError, asyncHandler } = require("../middleware/errors");
const { validator } = require("../middleware/validate");
const { generationLimiter } = require("../middleware/limiters");
const { readSettings } = require("../generationSettings");

// In-memory conversation store
const conversations = new Map();

const MESSAGE_TYPES = ["text", "code", "image", "video", "audio"];
const MAX_MESSAGE_LENGTH = 8000;

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
  conversations.set(id, {
    id,
    title: "New Chat",
    messages: [],
    createdAt: new Date().toISOString(),
  });
  res.json({ id, title: "New Chat", messages: [] });
});

// List conversations
router.get("/conversations", (req, res) => {
  const list = Array.from(conversations.values())
    .map(({ id, title, createdAt, messages }) => ({
      id,
      title,
      createdAt,
      messageCount: messages.length,
    }))
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  res.json(list);
});

// Get conversation
router.get("/conversations/:id", (req, res) => {
  res.json(getConversation(conversationId(req)));
});

// Delete conversation
router.delete("/conversations/:id", (req, res) => {
  conversations.delete(conversationId(req));
  res.json({ success: true });
});

// Send message. This runs the model, so it shares the generation rate limit.
router.post(
  "/conversations/:id/messages",
  generationLimiter,
  asyncHandler(async (req, res) => {
    const conv = getConversation(conversationId(req));

    const v = validator(req.body);
    const content = v.string("content", { required: true, max: MAX_MESSAGE_LENGTH });
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
    conv.messages.push(userMessage);

    // Update title from first message
    if (conv.messages.length === 1) {
      conv.title = content.substring(0, 50) + (content.length > 50 ? "..." : "");
    }

    const response = await processMessage(conv.messages, type, settings);
    const assistantMessage = {
      id: randomUUID(),
      role: "assistant",
      content: response.content,
      type: response.type,
      metadata: response.metadata || {},
      timestamp: new Date().toISOString(),
    };
    conv.messages.push(assistantMessage);
    res.json(assistantMessage);
  })
);

module.exports = router;
