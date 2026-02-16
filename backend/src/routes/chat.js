const express = require("express");
const router = express.Router();
const { v4: uuidv4 } = require("uuid");
const { processMessage } = require("../services/model");

// In-memory conversation store
const conversations = new Map();

// Create new conversation
router.post("/conversations", (req, res) => {
  const id = uuidv4();
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
  const conv = conversations.get(req.params.id);
  if (!conv) return res.status(404).json({ error: "Conversation not found" });
  res.json(conv);
});

// Delete conversation
router.delete("/conversations/:id", (req, res) => {
  conversations.delete(req.params.id);
  res.json({ success: true });
});

// Send message
router.post("/conversations/:id/messages", async (req, res) => {
  const conv = conversations.get(req.params.id);
  if (!conv) return res.status(404).json({ error: "Conversation not found" });

  const { content, type = "text", attachments = [] } = req.body;

  const userMessage = {
    id: uuidv4(),
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

  try {
    const response = await processMessage(conv.messages, type);
    const assistantMessage = {
      id: uuidv4(),
      role: "assistant",
      content: response.content,
      type: response.type,
      metadata: response.metadata || {},
      timestamp: new Date().toISOString(),
    };
    conv.messages.push(assistantMessage);
    res.json(assistantMessage);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
