/**
 * A reply cut off by a dropped socket must not stand in for the finished one
 * (Issue #456).
 *
 * The server stores every reply under its own id once it completes, and the
 * ack tells the client that id. When the socket drops mid-stream the client
 * keeps what had arrived, marks it complete, and on reconnect syncs it under
 * the id the server is still generating the whole reply for. Sync stored that
 * copy, so when the reply finished append() found the id already taken and
 * dropped the reply, and the conversation kept the cut-off text for good.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");
const { randomUUID } = require("node:crypto");

const { mockModel, startServer } = require("./helpers");

const FULL_REPLY = Array.from({ length: 40 }, (_, i) => `w${String(i).padStart(3, "0")}`).join(" ");
const STREAMED_BEFORE_DROP = 10;

// A prompt starting with "hold" streams the first words of its reply, then waits
// for the test to release it, as a worker still generating does. Anything else
// is answered at once.
let held = null;
mockModel({
  processMessage: async (messages, type = "text", settings = {}, options = {}) => {
    const content = messages[messages.length - 1].content;
    if (!content.startsWith("hold") || typeof options?.onStream !== "function") {
      return { type, content: `reply to: ${content}`, metadata: { model: "test-model" } };
    }
    for (const word of FULL_REPLY.split(" ").slice(0, STREAMED_BEFORE_DROP)) {
      options.onStream({ delta: `${word} ` });
    }
    let markReturned;
    const returned = new Promise((resolve) => {
      markReturned = resolve;
    });
    await new Promise((resolve) => {
      held = { release: resolve, returned };
    });
    markReturned();
    return { type, content: FULL_REPLY, metadata: { model: "test-model" } };
  },
});

const { generationCounter, apiCounter } = require("../src/middleware/limiters");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
  held = null;
});

/** Let the held reply finish, and wait until its turn has stored what it got. */
async function finishHeldReply() {
  held.release();
  await held.returned;
  // The turn picks the reply up a few microtasks later; by the next macrotask
  // it has recorded it.
  await new Promise((resolve) => setImmediate(resolve));
}

const words = (text) => (text.match(/w\d{3}/g) || []).length;

function openSocket(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.on("open", () => resolve(ws));
    ws.on("error", reject);
  });
}

async function waitFor(check, what, timeoutMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await check();
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 10));
  }
}

async function stored(app, conversationId) {
  const res = await request(app).get(`/api/chat/conversations/${conversationId}`);
  assert.equal(res.status, 200);
  return res.body;
}

/**
 * Start a held turn and cut the socket once part of the reply has streamed.
 * Returns the ack and the text the client held when the socket went.
 */
async function turnCutOffMidStream(wsUrl, conversationId, content) {
  const ws = await openSocket(wsUrl);
  let ack = null;
  const partial = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("the reply never streamed")), 3000);
    ws.on("message", (data) => {
      const frame = JSON.parse(data);
      if (frame.type === "ack") ack = frame;
      if (frame.type === "stream" && !frame.done && words(frame.content) === STREAMED_BEFORE_DROP) {
        clearTimeout(timer);
        resolve(frame.content);
      }
    });
    ws.send(JSON.stringify({ type: "chat", content, conversationId, messageType: "text", attachments: [], expectedVersion: 1, settings: {} }));
  });
  // No close handshake: this is what a dropped connection looks like.
  ws.terminate();
  await waitFor(() => held, "the turn to be waiting on the model");
  return { ack, partial };
}

/**
 * The body the website's reconnect handler sends for that conversation: the
 * user message and the reply under the ids the ack gave them, the reply holding
 * what had streamed, marked complete by the close handler.
 */
function reconnectSync(ack, content, replyContent) {
  const now = new Date().toISOString();
  return {
    clientVersion: ack.version,
    messages: [
      { id: ack.messageId, clientId: randomUUID(), role: "user", content, type: "text", attachments: [], timestamp: now },
      { id: ack.assistantMessageId, clientId: randomUUID(), role: "assistant", content: replyContent, type: "text", completed: true, timestamp: now },
    ],
    deletedMessageIds: [],
    title: "New Chat",
    titleUpdatedAt: 0,
  };
}

async function newConversation(app) {
  return (await request(app).post("/api/chat/conversations")).body.id;
}

test("#456: a reply cut off mid-stream and synced on reconnect does not replace the finished reply", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const content = "hold: tell me a long story";
    const { ack, partial } = await turnCutOffMidStream(server.wsUrl, conversationId, content);
    assert.equal(words(partial), STREAMED_BEFORE_DROP);

    const sync = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/sync`)
      .send(reconnectSync(ack, content, partial));
    assert.equal(sync.status, 200);
    const atSync = await stored(server.app, conversationId);

    // The worker finishes after the client has reconnected and synced.
    await finishHeldReply();
    const after = await stored(server.app, conversationId);

    const replies = after.messages.filter((m) => m.role === "assistant");
    assert.equal(replies.length, 1);
    assert.equal(replies[0].id, ack.assistantMessageId);
    assert.equal(words(replies[0].content), 40, "the whole reply is stored, not the cut-off copy");
    assert.equal(replies[0].content, FULL_REPLY);
    assert.deepEqual(after.messages.map((m) => m.role), ["user", "assistant"]);

    // The sync added nothing (the reply was not the client's to add), so it was
    // not a new revision; the stored reply is the one revision after it.
    assert.deepEqual(atSync.messages.map((m) => m.role), ["user"]);
    assert.equal(atSync.version, ack.version);
    assert.equal(after.version, ack.version + 1);
  } finally {
    if (held) held.release();
    await server.stop();
  }
});

test("#456: a stale partial copy synced after the reply finished leaves the finished reply alone", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const content = "hold: tell me a long story";
    const { ack, partial } = await turnCutOffMidStream(server.wsUrl, conversationId, content);

    // This time the reply finishes before the client is back.
    await finishHeldReply();
    assert.equal((await stored(server.app, conversationId)).messages.filter((m) => m.role === "assistant").length, 1);

    const sync = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/sync`)
      .send(reconnectSync(ack, content, partial));
    assert.equal(sync.status, 200);

    const replies = (await stored(server.app, conversationId)).messages.filter((m) => m.role === "assistant");
    assert.equal(replies.length, 1);
    assert.equal(replies[0].content, FULL_REPLY);
    // The client is told the whole reply, under the id it already holds.
    const synced = sync.body.conversation.messages.find((m) => m.id === ack.assistantMessageId);
    assert.equal(synced.content, FULL_REPLY);
  } finally {
    if (held) held.release();
    await server.stop();
  }
});

test("#456: syncing what the server holds keeps every stored message, replies included", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    await request(server.app).post(`/api/chat/conversations/${conversationId}/messages`).send({ content: "hello" });
    const before = await stored(server.app, conversationId);

    const sync = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/sync`)
      .send({ clientVersion: before.version, messages: before.messages, title: before.title });

    assert.equal(sync.status, 200);
    const shape = (messages) => messages.map((m) => [m.id, m.role, m.content]);
    assert.deepEqual(shape((await stored(server.app, conversationId)).messages), shape(before.messages));
    assert.equal(sync.body.diff.messages.clientOnly.length, 0);
    // The client's copy of the reply is matched to the stored one, not ignored.
    assert.equal(sync.body.diff.messages.serverOnly.length, 0);
  } finally {
    await server.stop();
  }
});

test("#456: sync never adds an assistant turn the server did not store, and still adds the client's own messages", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    await request(server.app).post(`/api/chat/conversations/${conversationId}/messages`).send({ content: "hello" });
    const before = await stored(server.app, conversationId);
    const [u1, a1] = before.messages;

    const offlineQuestion = { id: randomUUID(), role: "user", content: "written while offline", type: "text" };
    const strayReply = { id: randomUUID(), role: "assistant", content: "a reply the server never wrote", type: "text", completed: true };
    const sync = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/sync`)
      .send({ clientVersion: before.version, messages: [u1, a1, offlineQuestion, strayReply] });

    assert.equal(sync.status, 200);
    const after = await stored(server.app, conversationId);
    assert.deepEqual(after.messages.map((m) => m.id), [u1.id, a1.id, offlineQuestion.id]);
    assert.deepEqual(sync.body.diff.messages.clientOnly.map((m) => m.id), [offlineQuestion.id]);
    assert.equal(after.version, before.version + 1);
  } finally {
    await server.stop();
  }
});

test("#456: a reply the server holds is still matched by the client's other id", async () => {
  const server = await startServer();
  try {
    const conversationStore = require("../src/conversationStore");
    const conversationId = await newConversation(server.app);
    const clientId = randomUUID();
    const reply = { id: randomUUID(), clientId, role: "assistant", content: "stored reply", type: "text" };
    conversationStore.append(conversationId, { id: randomUUID(), role: "user", content: "question", type: "text" });
    conversationStore.append(conversationId, reply);
    const before = await stored(server.app, conversationId);

    // The same reply, known to this client only by the id it minted for it.
    const sync = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/sync`)
      .send({
        clientVersion: before.version,
        messages: [before.messages[0], { id: clientId, role: "assistant", content: "stored reply", type: "text" }],
      });

    assert.equal(sync.status, 200);
    assert.equal(sync.body.diff.messages.identicalCount, 2);
    assert.equal(sync.body.diff.messages.clientOnly.length, 0);
    assert.equal(sync.body.version, before.version);
    assert.deepEqual((await stored(server.app, conversationId)).messages, before.messages);

    // And the other way round: a copy under another id that names the stored
    // reply's id as its clientId.
    const again = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/sync`)
      .send({
        clientVersion: before.version,
        messages: [before.messages[0], { id: randomUUID(), clientId: reply.id, role: "assistant", content: "stored reply", type: "text" }],
      });

    assert.equal(again.status, 200);
    assert.equal(again.body.diff.messages.identicalCount, 2);
    assert.equal(again.body.diff.messages.clientOnly.length, 0);
    assert.deepEqual((await stored(server.app, conversationId)).messages, before.messages);
  } finally {
    await server.stop();
  }
});
