const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");
const { randomUUID } = require("node:crypto");

const { mockModel, loadApp, newConversation, startServer } = require("./helpers");

const calls = mockModel();
const app = loadApp();
const store = require("../src/conversationStore");
const { generationCounter, apiCounter } = require("../src/middleware/limiters");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
});

function connectWebSocket(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.on("open", () => resolve(ws));
    ws.on("error", reject);
  });
}

function collectFrames(ws, count, condition, timeoutMs = 4000) {
  return new Promise((resolve) => {
    const frames = [];
    const onMessage = (data) => {
      const msg = JSON.parse(data);
      frames.push(msg);
      if (condition && condition(msg, frames)) {
        cleanup();
        resolve(frames);
      } else if (count && frames.length >= count) {
        cleanup();
        resolve(frames);
      }
    };
    const timer = setTimeout(() => {
      cleanup();
      resolve(frames);
    }, timeoutMs);

    function cleanup() {
      clearTimeout(timer);
      ws.off("message", onMessage);
    }

    ws.on("message", onMessage);
  });
}

// ---------------------------------------------------------------------------
// 1. conversationStore primitives: truncateAfter, truncateFrom, updateMessage
// ---------------------------------------------------------------------------

test("store.truncateAfter removes messages strictly after the target message", () => {
  const convId = randomUUID();
  const m1 = { id: randomUUID(), role: "user", content: "m1" };
  const m2 = { id: randomUUID(), role: "assistant", content: "m2" };
  const m3 = { id: randomUUID(), role: "user", content: "m3" };
  const m4 = { id: randomUUID(), role: "assistant", content: "m4" };

  store.create({ id: convId, title: "Test", messages: [m1, m2, m3, m4] });

  const res = store.truncateAfter(convId, m2.id);
  assert.equal(res, true);

  const updated = store.get(convId);
  assert.equal(updated.messages.length, 2);
  assert.equal(updated.messages[0].id, m1.id);
  assert.equal(updated.messages[1].id, m2.id);

  // Non-existent message returns false and doesn't mutate
  assert.equal(store.truncateAfter(convId, randomUUID()), false);
  assert.equal(store.truncateAfter(randomUUID(), m1.id), false);
});

test("store.truncateFrom removes messages starting from the target message (inclusive)", () => {
  const convId = randomUUID();
  const m1 = { id: randomUUID(), role: "user", content: "m1" };
  const m2 = { id: randomUUID(), role: "assistant", content: "m2" };
  const m3 = { id: randomUUID(), role: "user", content: "m3" };

  store.create({ id: convId, title: "Test", messages: [m1, m2, m3] });

  const res = store.truncateFrom(convId, m2.id);
  assert.equal(res, true);

  const updated = store.get(convId);
  assert.equal(updated.messages.length, 1);
  assert.equal(updated.messages[0].id, m1.id);

  assert.equal(store.truncateFrom(convId, randomUUID()), false);
});

test("store.updateMessage updates existing message in place", () => {
  const convId = randomUUID();
  const m1 = { id: randomUUID(), role: "user", content: "original content" };
  store.create({ id: convId, title: "Test", messages: [m1] });

  const ok = store.updateMessage(convId, m1.id, { content: "edited content" });
  assert.equal(ok, true);

  const updated = store.get(convId);
  assert.equal(updated.messages[0].content, "edited content");
  assert.equal(store.updateMessage(convId, randomUUID(), { content: "nope" }), false);
});

// ---------------------------------------------------------------------------
// 2. REST API: Edit user message and Regenerate assistant response
// ---------------------------------------------------------------------------

test("REST: editing an earlier user message updates content and truncates subsequent turns", async () => {
  const convId = await newConversation(app);

  // Turn 1
  const res1 = await request(app)
    .post(`/api/chat/conversations/${convId}/messages`)
    .send({ content: "First question" });
  assert.equal(res1.status, 200);

  // Turn 2
  const res2 = await request(app)
    .post(`/api/chat/conversations/${convId}/messages`)
    .send({ content: "Second question" });
  assert.equal(res2.status, 200);

  const convBefore = await request(app).get(`/api/chat/conversations/${convId}`);
  assert.equal(convBefore.body.messages.length, 4);
  const firstUserMsg = convBefore.body.messages[0];

  const beforeCalls = calls.length;

  // Edit Turn 1 user message
  const editRes = await request(app)
    .post(`/api/chat/conversations/${convId}/messages`)
    .send({
      content: "Edited first question",
      editMessageId: firstUserMsg.id,
    });

  assert.equal(editRes.status, 200);
  assert.equal(editRes.body.role, "assistant");
  assert.equal(editRes.body.content, "reply to: Edited first question");

  // Verify conversation in store: obsolete Turn 2 is removed, Turn 1 is updated
  const convAfter = await request(app).get(`/api/chat/conversations/${convId}`);
  assert.equal(convAfter.body.messages.length, 2);
  assert.equal(convAfter.body.messages[0].id, firstUserMsg.id);
  assert.equal(convAfter.body.messages[0].content, "Edited first question");
  assert.equal(convAfter.body.messages[1].role, "assistant");

  // Verify the model received only the updated context, not obsolete turn 2
  const [lastCall] = calls.slice(beforeCalls);
  assert.equal(lastCall.name, "processMessage");
  const modelMessages = lastCall.args[0];
  assert.equal(modelMessages[0].content, "Edited first question");
  assert.equal(modelMessages.some((m) => m.content === "Second question"), false);
});

test("REST: regenerating an assistant response replaces it without duplicating", async () => {
  const convId = await newConversation(app);

  // Turn 1
  await request(app)
    .post(`/api/chat/conversations/${convId}/messages`)
    .send({ content: "Question to regenerate" });

  const conv = await request(app).get(`/api/chat/conversations/${convId}`);
  assert.equal(conv.body.messages.length, 2);
  const assistantMsg = conv.body.messages[1];

  const regenRes = await request(app)
    .post(`/api/chat/conversations/${convId}/messages`)
    .send({
      regenerateMessageId: assistantMsg.id,
    });

  assert.equal(regenRes.status, 200);
  assert.equal(regenRes.body.role, "assistant");
  assert.equal(regenRes.body.content, "reply to: Question to regenerate");

  const convAfter = await request(app).get(`/api/chat/conversations/${convId}`);
  assert.equal(convAfter.body.messages.length, 2);
  assert.notEqual(convAfter.body.messages[1].id, assistantMsg.id);
  assert.equal(convAfter.body.messages[1].content, "reply to: Question to regenerate");
});

// ---------------------------------------------------------------------------
// 3. WebSocket: Edit user message and Regenerate assistant response
// ---------------------------------------------------------------------------

test("WebSocket: editing an earlier user message updates frame and history", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const convId = randomUUID();
  const user1 = { id: randomUUID(), role: "user", content: "Original prompt 1" };
  const ast1 = { id: randomUUID(), role: "assistant", content: "Original answer 1" };
  const user2 = { id: randomUUID(), role: "user", content: "Followup 2" };
  const ast2 = { id: randomUUID(), role: "assistant", content: "Followup answer 2" };

  store.create({
    id: convId,
    title: "Edit Test",
    messages: [user1, ast1, user2, ast2],
  });

  const ws = await connectWebSocket(server.wsUrl);
  t.after(() => ws.close());

  const framesPromise = collectFrames(ws, null, (f) => f.type === "stream" && f.done);

  ws.send(
    JSON.stringify({
      type: "chat",
      conversationId: convId,
      editMessageId: user1.id,
      content: "Rewritten prompt 1",
    })
  );

  const frames = await framesPromise;
  const ack = frames.find((f) => f.type === "ack");
  assert.ok(ack, "Expected ack frame");
  assert.equal(ack.messageId, user1.id);
  assert.equal(ack.conversationId, convId);

  const doneFrame = frames.find((f) => f.type === "stream" && f.done);
  assert.ok(doneFrame, "Expected done stream frame");

  const conv = store.get(convId);
  assert.equal(conv.messages.length, 2);
  assert.equal(conv.messages[0].id, user1.id);
  assert.equal(conv.messages[0].content, "Rewritten prompt 1");
  assert.equal(conv.messages[1].role, "assistant");
  assert.equal(conv.messages[1].id, doneFrame.messageId);
});

test("WebSocket: regenerating an assistant response truncates from target and streams replacement", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const convId = randomUUID();
  const user1 = { id: randomUUID(), role: "user", content: "Prompt to regenerate" };
  const ast1 = { id: randomUUID(), role: "assistant", content: "First answer" };

  store.create({
    id: convId,
    title: "Regenerate Test",
    messages: [user1, ast1],
  });

  const ws = await connectWebSocket(server.wsUrl);
  t.after(() => ws.close());

  const framesPromise = collectFrames(ws, null, (f) => f.type === "stream" && f.done);

  ws.send(
    JSON.stringify({
      type: "chat",
      conversationId: convId,
      regenerateMessageId: ast1.id,
    })
  );

  const frames = await framesPromise;
  const ack = frames.find((f) => f.type === "ack");
  assert.ok(ack);
  assert.equal(ack.messageId, user1.id);

  const doneFrame = frames.find((f) => f.type === "stream" && f.done);
  assert.ok(doneFrame);

  const conv = store.get(convId);
  assert.equal(conv.messages.length, 2);
  assert.equal(conv.messages[0].id, user1.id);
  assert.equal(conv.messages[1].role, "assistant");
  assert.notEqual(conv.messages[1].id, ast1.id);
  assert.equal(conv.messages[1].id, doneFrame.messageId);
});

// ---------------------------------------------------------------------------
// 4. Editing a message the server never stored (Issue #447)
//
// A send can fail before the server stores it (rate limited, lost with the
// socket) while the client keeps the message under its own temporary id, with
// Edit still offered on it. Over REST that id matches nothing and the edited
// text is stored as a new turn. Over WebSocket the edit branch was entered
// anyway: nothing was stored, the ack named an id that was never stored, and
// the model was handed the last stored turn instead of the edit.
//
// The stub answers "reply to: <last turn it was given>", and the stored turns
// below read differently, so the reply says which turn reached the model.
// ---------------------------------------------------------------------------

const FRANCE = "What is the capital of France?";
const PARIS = "The capital of France is Paris.";
const PORTUGAL = "What is the capital of Portugal?";

function seedConversation() {
  const convId = randomUUID();
  const user1 = { id: randomUUID(), role: "user", content: FRANCE };
  const ast1 = { id: randomUUID(), role: "assistant", content: PARIS };
  store.create({ id: convId, title: "Unknown edit target", messages: [user1, ast1] });
  return { convId, user1, ast1 };
}

function sendChat(ws, frame) {
  const frames = collectFrames(ws, null, (f) => (f.type === "stream" && f.done) || f.type === "error");
  ws.send(JSON.stringify({ type: "chat", ...frame }));
  return frames;
}

test("WebSocket: editing a stored message still rewrites it in place (Issue #447)", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const ws = await connectWebSocket(server.wsUrl);
  t.after(() => ws.close());
  const { convId, user1 } = seedConversation();

  const frames = await sendChat(ws, { conversationId: convId, editMessageId: user1.id, content: PORTUGAL });

  const ack = frames.find((f) => f.type === "ack");
  const done = frames.find((f) => f.type === "stream" && f.done);
  assert.equal(ack.messageId, user1.id);
  assert.equal(done.content, `reply to: ${PORTUGAL}`);
  const stored = store.get(convId).messages;
  assert.deepEqual(stored.map((m) => [m.role, m.content]), [
    ["user", PORTUGAL],
    ["assistant", `reply to: ${PORTUGAL}`],
  ]);
  assert.equal(stored[0].id, user1.id);
  assert.equal(stored[1].id, done.messageId);
});

test("WebSocket: editing a message the server never stored answers the edit, not the last stored turn (Issue #447)", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const ws = await connectWebSocket(server.wsUrl);
  t.after(() => ws.close());
  const { convId, user1, ast1 } = seedConversation();

  const frames = await sendChat(ws, { conversationId: convId, editMessageId: randomUUID(), content: PORTUGAL });

  const done = frames.find((f) => f.type === "stream" && f.done);
  assert.ok(done, `expected a reply, got ${JSON.stringify(frames)}`);
  assert.equal(done.content, `reply to: ${PORTUGAL}`, "the edit reached the model, not the stored assistant turn");

  const stored = store.get(convId).messages;
  assert.deepEqual(stored.map((m) => [m.role, m.content]), [
    ["user", FRANCE],
    ["assistant", PARIS],
    ["user", PORTUGAL],
    ["assistant", `reply to: ${PORTUGAL}`],
  ]);
  assert.equal(stored[0].id, user1.id, "the stored turns are left as they were");
  assert.equal(stored[1].id, ast1.id);
  assert.equal(stored[3].id, done.messageId);
});

test("WebSocket: the ack for an unknown edit target names the stored message, which can then be branched and edited (Issue #447)", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const ws = await connectWebSocket(server.wsUrl);
  t.after(() => ws.close());
  const { convId } = seedConversation();
  const clientOnlyId = randomUUID();

  const frames = await sendChat(ws, { conversationId: convId, editMessageId: clientOnlyId, content: PORTUGAL });

  const ack = frames.find((f) => f.type === "ack");
  assert.notEqual(ack.messageId, clientOnlyId, "an id the server never stored is not adopted");
  const stored = store.get(convId).messages;
  const named = stored.filter((m) => m.id === ack.messageId);
  assert.equal(named.length, 1, "the ack names a message the server stored");
  assert.equal(named[0].role, "user");
  assert.equal(named[0].content, PORTUGAL);

  const branch = await request(app).post(`/api/chat/conversations/${convId}/branch`).send({ messageId: ack.messageId });
  assert.equal(branch.status, 200, JSON.stringify(branch.body));

  // The acknowledged id is a real edit target from now on.
  const again = await sendChat(ws, { conversationId: convId, editMessageId: ack.messageId, content: "What is the capital of Spain?" });
  assert.equal(again.find((f) => f.type === "ack").messageId, ack.messageId);
  assert.deepEqual(store.get(convId).messages.map((m) => m.content), [
    FRANCE,
    PARIS,
    "What is the capital of Spain?",
    "reply to: What is the capital of Spain?",
  ]);
});

test("REST: editing a message the server never stored stores the edit as a new turn (the contract WebSocket now follows)", async () => {
  const { convId, user1, ast1 } = seedConversation();

  const res = await request(app)
    .post(`/api/chat/conversations/${convId}/messages`)
    .send({ editMessageId: randomUUID(), content: PORTUGAL });

  assert.equal(res.status, 200);
  assert.equal(res.body.content, `reply to: ${PORTUGAL}`);
  const stored = store.get(convId).messages;
  assert.deepEqual(stored.map((m) => [m.role, m.content]), [
    ["user", FRANCE],
    ["assistant", PARIS],
    ["user", PORTUGAL],
    ["assistant", `reply to: ${PORTUGAL}`],
  ]);
  assert.equal(stored[0].id, user1.id);
  assert.equal(stored[1].id, ast1.id);
  assert.equal(res.body.userMessageId, stored[2].id);
});

// Regenerate has its own unknown-target case. It is out of scope here and left
// as it was; this only checks the change above did not reach it.
test("WebSocket: an unknown regenerate target still leaves the stored turns alone", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  const ws = await connectWebSocket(server.wsUrl);
  t.after(() => ws.close());
  const { convId, user1, ast1 } = seedConversation();

  await sendChat(ws, { conversationId: convId, regenerateMessageId: randomUUID(), content: FRANCE });

  const stored = store.get(convId).messages;
  assert.equal(stored[0].id, user1.id);
  assert.equal(stored[0].content, FRANCE);
  assert.equal(stored[1].id, ast1.id);
  assert.equal(stored[1].content, PARIS);
});
