/**
 * One id per message, shared by the client and the server (Issue #394).
 *
 * The server stores every turn under an id of its own. The client minted a
 * different one for the same message and never learned the server's, so "Branch
 * from here" asked for an id the server had never seen and got "Message not
 * found in conversation" until the page was reloaded. Over WebSocket it was
 * worse: the turns were stored with no id at all, and the ack carried a random
 * id that matched nothing.
 *
 * The server's id is the authoritative one, so the server has to say what it is.
 * The REST send response names the user message as well as the reply, and over
 * WebSocket the ack names the user message while the last frame of the reply
 * names the reply. These tests pin that contract, and that the branch lookup
 * finds a message by exactly the id it was told about.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");
const fs = require("node:fs");
const path = require("node:path");
const { randomUUID } = require("node:crypto");

const { mockModel, loadApp, startServer, createTestWav } = require("./helpers");

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const AUDIO_URL = "/uploads/generated/message-id-consistency.wav";

// Streams the way the worker does when asked to: the deltas arrive through
// onStream and the whole reply is returned afterwards. A prompt that does not
// start with "stream" is answered in one piece, which the WebSocket service
// then streams itself.
mockModel({
  processMessage: async (messages, type = "text", settings = {}, options = {}) => {
    const content = messages[messages.length - 1].content;
    if (type === "audio") {
      // A prompt that mentions "no file" gets an audio reply that names none.
      const named = content.includes("no file") ? {} : { url: AUDIO_URL };
      return {
        type: "audio",
        content: "Here is your audio",
        metadata: { model: "test-model", ...named },
      };
    }
    const reply = `reply to: ${content}`;
    if (typeof options?.onStream === "function" && content.startsWith("stream")) {
      for (const delta of reply.match(/.{1,6}/g)) options.onStream({ delta });
    }
    return { type, content: reply, metadata: { model: "test-model" } };
  },
});

const app = loadApp();
const { generationCounter, apiCounter } = require("../src/middleware/limiters");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
});

async function newConversation(app) {
  const res = await request(app).post("/api/chat/conversations");
  return res.body.id;
}

async function storedMessages(app, conversationId) {
  const res = await request(app).get(`/api/chat/conversations/${conversationId}`);
  assert.equal(res.status, 200);
  return res.body.messages;
}

function branchFrom(app, conversationId, messageId) {
  return request(app).post(`/api/chat/conversations/${conversationId}/branch`).send({ messageId });
}

function openSocket(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.on("open", () => resolve(ws));
    ws.on("error", reject);
  });
}

/** Send one chat frame and collect every frame of the turn, up to its last. */
function runTurn(ws, frame, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const frames = [];
    const timer = setTimeout(() => {
      ws.off("message", onMessage);
      reject(new Error(`the turn did not finish; saw ${JSON.stringify(frames.map((f) => f.type))}`));
    }, timeoutMs);

    function onMessage(data) {
      const message = JSON.parse(data);
      frames.push(message);
      if (message.type === "error" || (message.type === "stream" && message.done)) {
        clearTimeout(timer);
        ws.off("message", onMessage);
        resolve(frames);
      }
    }

    ws.on("message", onMessage);
    ws.send(JSON.stringify({ type: "chat", ...frame }));
  });
}

// ── REST ──────────────────────────────────────────────────────────────────

test("the send response names the user message as well as the reply", async () => {
  const conversationId = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${conversationId}/messages`)
    .send({ content: "hello" });

  assert.equal(res.status, 200);
  assert.match(res.body.id, UUID, "the reply's own id");
  assert.match(res.body.userMessageId, UUID, "the id the user message is stored under");
  assert.notEqual(res.body.userMessageId, res.body.id);

  const stored = await storedMessages(app, conversationId);
  assert.deepEqual(
    stored.map((m) => [m.role, m.id]),
    [
      ["user", res.body.userMessageId],
      ["assistant", res.body.id],
    ]
  );
});

test("userMessageId describes the exchange and is not stored on the reply", async () => {
  const conversationId = await newConversation(app);

  await request(app).post(`/api/chat/conversations/${conversationId}/messages`).send({ content: "hello" });

  const stored = await storedMessages(app, conversationId);
  for (const message of stored) {
    assert.equal(Object.hasOwn(message, "userMessageId"), false);
  }
});

test("branching finds the user message by the id the send response gave", async () => {
  const conversationId = await newConversation(app);
  const sent = await request(app)
    .post(`/api/chat/conversations/${conversationId}/messages`)
    .send({ content: "first question" });

  const branch = await branchFrom(app, conversationId, sent.body.userMessageId);

  assert.equal(branch.status, 200);
  assert.equal(branch.body.branchedFromMessageId, sent.body.userMessageId);
  assert.deepEqual(
    branch.body.messages.map((m) => [m.role, m.id]),
    [["user", sent.body.userMessageId]]
  );
});

test("branching finds the reply by the id the send response gave", async () => {
  const conversationId = await newConversation(app);
  const sent = await request(app)
    .post(`/api/chat/conversations/${conversationId}/messages`)
    .send({ content: "first question" });

  const branch = await branchFrom(app, conversationId, sent.body.id);

  assert.equal(branch.status, 200);
  assert.deepEqual(
    branch.body.messages.map((m) => [m.role, m.id]),
    [
      ["user", sent.body.userMessageId],
      ["assistant", sent.body.id],
    ]
  );
});

// ── WebSocket ─────────────────────────────────────────────────────────────

test("the ack names the stored user message and the last frame names the stored reply", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const conversationId = await newConversation(server.app);
  const ws = await openSocket(server.wsUrl);
  t.after(() => ws.close());

  const frames = await runTurn(ws, { content: "hello", conversationId });

  const ack = frames.find((f) => f.type === "ack");
  const last = frames[frames.length - 1];
  assert.match(ack.messageId, UUID);
  assert.equal(last.type, "stream");
  assert.equal(last.done, true);
  assert.match(last.messageId, UUID);
  assert.notEqual(ack.messageId, last.messageId);

  const stored = await storedMessages(server.app, conversationId);
  assert.deepEqual(
    stored.map((m) => [m.role, m.id]),
    [
      ["user", ack.messageId],
      ["assistant", last.messageId],
    ],
    "the ids the client was told are the ids the server stored"
  );
});

test("branching finds a WebSocket turn by the ids it was told about", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const conversationId = await newConversation(server.app);
  const ws = await openSocket(server.wsUrl);
  t.after(() => ws.close());

  const frames = await runTurn(ws, { content: "hello", conversationId });
  const userId = frames.find((f) => f.type === "ack").messageId;
  const replyId = frames[frames.length - 1].messageId;

  const fromUser = await branchFrom(server.app, conversationId, userId);
  assert.equal(fromUser.status, 200);
  assert.deepEqual(fromUser.body.messages.map((m) => m.id), [userId]);

  const fromReply = await branchFrom(server.app, conversationId, replyId);
  assert.equal(fromReply.status, 200);
  assert.deepEqual(fromReply.body.messages.map((m) => m.id), [userId, replyId]);
});

test("a reply streamed in pieces is one message under one id", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const conversationId = await newConversation(server.app);
  const ws = await openSocket(server.wsUrl);
  t.after(() => ws.close());

  const frames = await runTurn(ws, { content: "stream this reply", conversationId });

  const streamFrames = frames.filter((f) => f.type === "stream");
  assert.ok(streamFrames.length > 2, "the reply arrives in several frames");
  const last = streamFrames[streamFrames.length - 1];
  assert.equal(last.done, true);

  // A frame that names the reply must name the one that is stored, so the same
  // message is never known by two ids while it streams.
  const named = new Set(streamFrames.map((f) => f.messageId).filter(Boolean));
  assert.deepEqual([...named], [last.messageId]);

  const stored = await storedMessages(server.app, conversationId);
  assert.equal(stored.length, 2, "one user message and one reply, no duplicate");
  assert.equal(stored.filter((m) => m.id === last.messageId).length, 1);
  assert.equal(stored[1].content, "reply to: stream this reply");
});

test("each turn gets its own ids, in the order the turns were sent", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const conversationId = await newConversation(server.app);
  const ws = await openSocket(server.wsUrl);
  t.after(() => ws.close());

  const first = await runTurn(ws, { content: "one", conversationId });
  const second = await runTurn(ws, { content: "two", conversationId });

  const told = [
    first.find((f) => f.type === "ack").messageId,
    first[first.length - 1].messageId,
    second.find((f) => f.type === "ack").messageId,
    second[second.length - 1].messageId,
  ];
  assert.equal(new Set(told).size, 4, "four messages, four different ids");

  const stored = await storedMessages(server.app, conversationId);
  assert.deepEqual(stored.map((m) => m.id), told);
  assert.deepEqual(stored.map((m) => m.content), ["one", "reply to: one", "two", "reply to: two"]);
});

test("a WebSocket turn leaves the ids of earlier messages alone", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const conversationId = await newConversation(server.app);
  const sent = await request(server.app)
    .post(`/api/chat/conversations/${conversationId}/messages`)
    .send({ content: "over REST first" });
  const ws = await openSocket(server.wsUrl);
  t.after(() => ws.close());

  await runTurn(ws, { content: "then over WebSocket", conversationId });

  const stored = await storedMessages(server.app, conversationId);
  assert.equal(stored.length, 4);
  assert.deepEqual(stored.slice(0, 2).map((m) => m.id), [sent.body.userMessageId, sent.body.id]);
  assert.ok(stored.every((m) => UUID.test(m.id)), "every stored message has an id");
});

test("a turn for a conversation the server does not hold still gets an ack id and a reply", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const ws = await openSocket(server.wsUrl);
  t.after(() => ws.close());
  const unknown = randomUUID();

  const frames = await runTurn(ws, { content: "hello", conversationId: unknown });

  assert.match(frames.find((f) => f.type === "ack").messageId, UUID);
  assert.equal(frames[frames.length - 1].done, true);
  const res = await request(server.app).get(`/api/chat/conversations/${unknown}`);
  assert.equal(res.status, 404, "nothing was recorded, as before");
});

test("the audio reply is named by its last frame, chunked or not", async (t) => {
  const wavPath = path.join(__dirname, "..", "uploads", "generated", path.basename(AUDIO_URL));
  fs.mkdirSync(path.dirname(wavPath), { recursive: true });
  fs.writeFileSync(wavPath, createTestWav(0.6, 24000));
  t.after(() => fs.rmSync(wavPath, { force: true }));

  const server = await startServer();
  t.after(() => server.stop());
  const conversationId = await newConversation(server.app);
  const ws = await openSocket(server.wsUrl);
  t.after(() => ws.close());

  const frames = await runTurn(ws, { content: "make some audio", messageType: "audio", conversationId });

  const chunks = frames.filter((f) => f.type === "stream");
  assert.ok(chunks.length > 1, "the audio streams in chunks");
  const last = chunks[chunks.length - 1];
  assert.equal(last.done, true);
  assert.match(last.messageId, UUID);
  for (const chunk of chunks.slice(0, -1)) {
    assert.ok(!chunk.messageId || chunk.messageId === last.messageId);
  }

  const stored = await storedMessages(server.app, conversationId);
  assert.equal(stored[1].id, last.messageId);
});

test("an audio reply that names no file is named too", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const conversationId = await newConversation(server.app);
  const ws = await openSocket(server.wsUrl);
  t.after(() => ws.close());

  const frames = await runTurn(ws, { content: "make audio, no file", messageType: "audio", conversationId });

  const last = frames[frames.length - 1];
  assert.equal(last.done, true);
  assert.match(last.messageId, UUID);
  const stored = await storedMessages(server.app, conversationId);
  assert.equal(last.messageId, stored[1].id);
});

test("the audio fallback frame names the reply too", async (t) => {
  // No file on disk, so the service falls back to one non-streaming frame.
  fs.rmSync(path.join(__dirname, "..", "uploads", "generated", path.basename(AUDIO_URL)), { force: true });
  const server = await startServer();
  t.after(() => server.stop());
  const conversationId = await newConversation(server.app);
  const ws = await openSocket(server.wsUrl);
  t.after(() => ws.close());

  const frames = await runTurn(ws, { content: "make some audio", messageType: "audio", conversationId });

  const last = frames[frames.length - 1];
  assert.equal(last.done, true);
  assert.match(last.messageId, UUID);
  const stored = await storedMessages(server.app, conversationId);
  assert.equal(last.messageId, stored[1].id);
});
