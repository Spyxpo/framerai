/**
 * Regenerating a reply the server is still generating (Issue #477).
 *
 * A WebSocket ack names the reply before it is stored. When the connection
 * drops mid-stream the client keeps what had arrived under that id, the server
 * keeps generating, and Regenerate targets an id the server has not stored. The
 * regenerate branch found nothing, generated another reply anyway, and the
 * original was stored too when it finished: two answers to one question. Over
 * REST the unknown target was even stored as a new question first.
 *
 * Now a regeneration of a reply still being generated replaces it (the original
 * is not stored when it finishes), and one of an id the conversation does not
 * have at all is refused on both transports.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");
const { randomUUID } = require("node:crypto");

const { mockModel, startServer } = require("./helpers");

// Every prompt the model was asked, in order.
const asked = [];
// A prompt starting with "hold" streams one word, then waits for the test to
// release it, as a worker still generating does. Each held answer is numbered,
// so the original and its replacement can be told apart. "fail" throws.
let held = [];
mockModel({
  processMessage: async (messages, type = "text", settings = {}, options = {}) => {
    const content = messages[messages.length - 1].content;
    asked.push(content);
    if (content.startsWith("fail")) throw new Error("the model failed");
    if (!content.startsWith("hold")) {
      return { type, content: `reply to: ${content}`, metadata: { model: "test-model" } };
    }
    const n = held.length + 1;
    if (typeof options?.onStream === "function") options.onStream({ delta: `partial ${n} ` });
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    let markReturned;
    const returned = new Promise((resolve) => {
      markReturned = resolve;
    });
    held.push({ release, returned });
    await gate;
    markReturned();
    return { type, content: `answer ${n}`, metadata: { model: "test-model" } };
  },
});

const store = require("../src/conversationStore");
const { generationCounter, apiCounter } = require("../src/middleware/limiters");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
  held = [];
  asked.length = 0;
});

async function waitFor(check, what, timeoutMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await check();
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 10));
  }
}

/** Let held answer `i` (0-based) finish, and wait until its turn has recorded it. */
async function finish(i) {
  held[i].release();
  await held[i].returned;
  await new Promise((resolve) => setImmediate(resolve));
}

function openSocket(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.on("open", () => resolve(ws));
    ws.on("error", reject);
  });
}

/** Send a chat frame and resolve with the first frame `until` accepts. */
function send(ws, frame, until) {
  return new Promise((resolve, reject) => {
    const frames = [];
    const timer = setTimeout(() => reject(new Error(`no matching frame; saw ${JSON.stringify(frames.map((f) => f.type))}`)), 3000);
    ws.on("message", function onMessage(data) {
      const f = JSON.parse(data);
      frames.push(f);
      if (until(f, frames)) {
        clearTimeout(timer);
        ws.off("message", onMessage);
        resolve({ frames, frame: f });
      }
    });
    ws.send(JSON.stringify({ type: "chat", messageType: "text", attachments: [], settings: {}, ...frame }));
  });
}

const ackOf = ({ frames }) => frames.find((f) => f.type === "ack");

/** A held turn whose connection drops once the first words have streamed. */
async function droppedTurn(server, conversationId, content) {
  const ws = await openSocket(server.wsUrl);
  const result = await send(ws, { content, conversationId }, (f) => f.type === "stream" && !f.done);
  ws.terminate();
  return ackOf(result);
}

async function newConversation(app) {
  return (await request(app).post("/api/chat/conversations")).body.id;
}

const replies = (id) => store.get(id).messages.filter((m) => m.role === "assistant");
const shape = (id) => store.get(id).messages.map((m) => (m.role === "user" ? `Q:${m.content}` : `A:${m.content}`));

test("#477: regenerating a stored reply still replaces it, on both transports", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const first = await request(server.app).post(`/api/chat/conversations/${conversationId}/messages`).send({ content: "hello" });

    const rest = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/messages`)
      .send({ regenerateMessageId: first.body.id, content: "" });
    assert.equal(rest.status, 200);
    assert.deepEqual(shape(conversationId), ["Q:hello", "A:reply to: hello"]);
    assert.notEqual(replies(conversationId)[0].id, first.body.id);

    const ws = await openSocket(server.wsUrl);
    const regen = await send(ws, { content: "hello", conversationId, regenerateMessageId: rest.body.id }, (f) => f.type === "stream" && f.done);
    ws.close();
    assert.deepEqual(shape(conversationId), ["Q:hello", "A:reply to: hello"]);
    assert.equal(replies(conversationId)[0].id, ackOf(regen).assistantMessageId);
  } finally {
    await server.stop();
  }
});

test("#477: WebSocket — a reply cut off mid-stream and regenerated leaves only the replacement", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const ack1 = await droppedTurn(server, conversationId, "hold: tell me a story");

    // Regenerate on a new connection while the original is still generating.
    const ws = await openSocket(server.wsUrl);
    const regen = send(ws, { content: "hold: tell me a story", conversationId, regenerateMessageId: ack1.assistantMessageId }, (f) => f.type === "stream" && f.done);
    await waitFor(() => held.length === 2, "the replacement to start");

    await finish(0); // the original finishes first
    await finish(1);
    const ack2 = ackOf(await regen);
    ws.close();

    assert.equal(ack2.messageId, ack1.messageId, "the replacement answers the same question");
    assert.deepEqual(shape(conversationId), ["Q:hold: tell me a story", "A:answer 2"]);
    assert.equal(replies(conversationId)[0].id, ack2.assistantMessageId);
    // Only the replacement was a new revision.
    assert.equal(store.get(conversationId).version, ack1.version + 1);
  } finally {
    held.forEach((h) => h.release());
    await server.stop();
  }
});

test("#477: a replaced reply that finishes after its replacement is still not stored", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const ack1 = await droppedTurn(server, conversationId, "hold: tell me a story");

    const ws = await openSocket(server.wsUrl);
    const regen = send(ws, { content: "hold: tell me a story", conversationId, regenerateMessageId: ack1.assistantMessageId }, (f) => f.type === "stream" && f.done);
    await waitFor(() => held.length === 2, "the replacement to start");

    await finish(1); // the replacement finishes first
    await regen;
    ws.close();
    assert.deepEqual(shape(conversationId), ["Q:hold: tell me a story", "A:answer 2"]);

    await finish(0); // the original, late
    assert.deepEqual(shape(conversationId), ["Q:hold: tell me a story", "A:answer 2"]);
    assert.equal(store.pendingReply(conversationId, ack1.assistantMessageId), null);
  } finally {
    held.forEach((h) => h.release());
    await server.stop();
  }
});

test("#477: REST — regenerating a reply still being generated replaces it, without storing the question again", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const ack1 = await droppedTurn(server, conversationId, "hold: tell me a story");

    const pending = request(server.app)
      .post(`/api/chat/conversations/${conversationId}/messages`)
      .send({ regenerateMessageId: ack1.assistantMessageId, content: "hold: tell me a story" })
      .then((res) => res);
    await waitFor(() => held.length === 2, "the replacement to start");
    await finish(0);
    await finish(1);
    const res = await pending;

    assert.equal(res.status, 200);
    assert.equal(res.body.userMessageId, ack1.messageId);
    assert.equal(res.body.content, "answer 2");
    assert.deepEqual(shape(conversationId), ["Q:hold: tell me a story", "A:answer 2"]);
  } finally {
    held.forEach((h) => h.release());
    await server.stop();
  }
});

test("#477: regenerating an id that is neither stored nor being generated is refused, and nothing is asked", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    await request(server.app).post(`/api/chat/conversations/${conversationId}/messages`).send({ content: "hello" });
    const before = store.get(conversationId);
    const version = before.version;
    asked.length = 0;

    const res = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/messages`)
      .send({ regenerateMessageId: randomUUID(), content: "hello" });
    assert.equal(res.status, 404);
    assert.equal(res.body.code, "NOT_FOUND");

    const ws = await openSocket(server.wsUrl);
    const refused = await send(ws, { content: "hello", conversationId, regenerateMessageId: randomUUID() }, (f) => f.type === "error");
    assert.equal(refused.frame.code, "NOT_FOUND");
    assert.equal(refused.frame.conversationId, conversationId);
    // A conversation the server does not have at all is refused too, as one it
    // does not hold, before the regeneration is looked at (Issue #503).
    const unknownConversation = await send(ws, { content: "hello", conversationId: randomUUID(), regenerateMessageId: randomUUID() }, (f) => f.type === "error");
    assert.equal(unknownConversation.frame.code, "CONVERSATION_NOT_FOUND");
    ws.close();

    assert.deepEqual(shape(conversationId), ["Q:hello", "A:reply to: hello"]);
    assert.equal(store.get(conversationId).version, version);
    assert.deepEqual(asked, [], "the model was not asked");
  } finally {
    await server.stop();
  }
});

test("#477: a reply can be replaced once: a second regeneration of the same reply is refused", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const ack1 = await droppedTurn(server, conversationId, "hold: tell me a story");

    const ws = await openSocket(server.wsUrl);
    const regen = send(ws, { content: "hold: tell me a story", conversationId, regenerateMessageId: ack1.assistantMessageId }, (f) => f.type === "stream" && f.done);
    await waitFor(() => held.length === 2, "the replacement to start");

    const ws2 = await openSocket(server.wsUrl);
    const again = await send(ws2, { content: "hold: tell me a story", conversationId, regenerateMessageId: ack1.assistantMessageId }, (f) => f.type === "error");
    ws2.close();
    assert.equal(again.frame.code, "NOT_FOUND");

    await finish(0);
    await finish(1);
    await regen;
    ws.close();
    assert.equal(held.length, 2, "no third generation");
    assert.deepEqual(shape(conversationId), ["Q:hold: tell me a story", "A:answer 2"]);
  } finally {
    held.forEach((h) => h.release());
    await server.stop();
  }
});

test("#477: replies being generated are tracked per conversation", async () => {
  const server = await startServer();
  try {
    const a = await newConversation(server.app);
    const b = await newConversation(server.app);
    const ackA = await droppedTurn(server, a, "hold: conversation a");
    const ackB = await droppedTurn(server, b, "hold: conversation b");

    // B's reply id means nothing in A.
    const ws = await openSocket(server.wsUrl);
    const crossed = await send(ws, { content: "hold: conversation a", conversationId: a, regenerateMessageId: ackB.assistantMessageId }, (f) => f.type === "error");
    assert.equal(crossed.frame.code, "NOT_FOUND");
    const rest = await request(server.app)
      .post(`/api/chat/conversations/${a}/messages`)
      .send({ regenerateMessageId: ackB.assistantMessageId, content: "hold: conversation a" });
    assert.equal(rest.status, 404);

    // Replacing A's reply leaves B's alone.
    const regen = send(ws, { content: "hold: conversation a", conversationId: a, regenerateMessageId: ackA.assistantMessageId }, (f) => f.type === "stream" && f.done);
    await waitFor(() => held.length === 3, "A's replacement to start");
    await finish(0);
    await finish(1);
    await finish(2);
    await regen;
    ws.close();

    assert.deepEqual(shape(a), ["Q:hold: conversation a", "A:answer 3"]);
    assert.deepEqual(shape(b), ["Q:hold: conversation b", "A:answer 2"]);
  } finally {
    held.forEach((h) => h.release());
    await server.stop();
  }
});

test("#477: the #457 sync rule and the replacement work together", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const ack1 = await droppedTurn(server, conversationId, "hold: tell me a story");

    // The reconnect sync carries the cut-off reply; it is still left out.
    const now = new Date().toISOString();
    const synced = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/sync`)
      .send({
        clientVersion: ack1.version,
        messages: [
          { id: ack1.messageId, role: "user", content: "hold: tell me a story", type: "text", attachments: [], timestamp: now },
          { id: ack1.assistantMessageId, role: "assistant", content: "partial 1", type: "text", completed: true, timestamp: now },
        ],
      });
    assert.equal(synced.status, 200);
    assert.deepEqual(shape(conversationId), ["Q:hold: tell me a story"]);

    const ws = await openSocket(server.wsUrl);
    const regen = send(ws, { content: "hold: tell me a story", conversationId, regenerateMessageId: ack1.assistantMessageId }, (f) => f.type === "stream" && f.done);
    await waitFor(() => held.length === 2, "the replacement to start");
    await finish(0);
    await finish(1);
    await regen;
    ws.close();
    assert.deepEqual(shape(conversationId), ["Q:hold: tell me a story", "A:answer 2"]);
  } finally {
    held.forEach((h) => h.release());
    await server.stop();
  }
});

test("#477: a stale version still wins over a regeneration of a reply being generated", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const ack1 = await droppedTurn(server, conversationId, "hold: tell me a story");

    const ws = await openSocket(server.wsUrl);
    const stale = await send(ws, {
      content: "hold: tell me a story", conversationId, regenerateMessageId: ack1.assistantMessageId, expectedVersion: ack1.version - 1,
    }, (f) => f.type === "error");
    ws.close();
    assert.equal(stale.frame.code, "VERSION_CONFLICT");

    // Nothing was replaced: the original is still the answer.
    assert.ok(store.pendingReply(conversationId, ack1.assistantMessageId));
    await finish(0);
    assert.deepEqual(shape(conversationId), ["Q:hold: tell me a story", "A:answer 1"]);
  } finally {
    held.forEach((h) => h.release());
    await server.stop();
  }
});

test("#477: nothing is left tracked once a turn finishes, whether it succeeded or failed", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const ws = await openSocket(server.wsUrl);
    const ok = await send(ws, { content: "hello", conversationId }, (f) => f.type === "stream" && f.done);
    const failed = await send(ws, { content: "fail: please", conversationId }, (f) => f.type === "error");
    ws.close();

    assert.equal(store.pendingReply(conversationId, ackOf(ok).assistantMessageId), null);
    assert.equal(store.pendingReply(conversationId, ackOf(failed).assistantMessageId), null);
    // A finished reply is a stored one: regenerating it takes the stored path.
    assert.equal(replies(conversationId)[0].id, ackOf(ok).assistantMessageId);
  } finally {
    await server.stop();
  }
});
