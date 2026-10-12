/**
 * A question the server already holds must not be stored again by a reconnect
 * sync (Issue #471).
 *
 * The server stores a new user message under an id of its own and tells the
 * client which in the ack (WebSocket) or the response (REST). When that answer
 * is lost to a dropped connection, the client still holds the message under the
 * id it minted, and the sync it sends on reconnect carries that id. The server
 * did not know it, so the sync stored the question a second time, and the model
 * read it twice from then on. A send now carries the client's id, the server
 * keeps it as the message's clientId, and the sync matches it there.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");
const { randomUUID } = require("node:crypto");

const { mockModel, startServer } = require("./helpers");

const FULL_REPLY = Array.from({ length: 40 }, (_, i) => `w${String(i).padStart(3, "0")}`).join(" ");

// A prompt starting with "hold" waits for the test to release it, as a worker
// still generating does; over WebSocket it streams a few words first. Anything
// else is answered at once.
let held = null;
mockModel({
  processMessage: async (messages, type = "text", settings = {}, options = {}) => {
    const content = messages[messages.length - 1].content;
    if (!content.startsWith("hold")) {
      return { type, content: `reply to: ${content}`, metadata: { model: "test-model" } };
    }
    if (typeof options?.onStream === "function") {
      for (const word of FULL_REPLY.split(" ").slice(0, 10)) options.onStream({ delta: `${word} ` });
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

async function waitFor(check, what, timeoutMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await check();
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 10));
  }
}

/** Let the held reply finish, and wait until its turn has stored what it got. */
async function finishHeldReply() {
  held.release();
  await held.returned;
  await new Promise((resolve) => setImmediate(resolve));
}

async function stored(app, conversationId) {
  const res = await request(app).get(`/api/chat/conversations/${conversationId}`);
  assert.equal(res.status, 200);
  return res.body;
}

const questions = (conv, content) => conv.messages.filter((m) => m.role === "user" && m.content === content);
const replies = (conv) => conv.messages.filter((m) => m.role === "assistant");

async function newConversation(app) {
  return (await request(app).post("/api/chat/conversations")).body.id;
}

function sendRest(app, conversationId, content, clientMessageId) {
  return request(app)
    .post(`/api/chat/conversations/${conversationId}/messages`)
    .send({ content, type: "text", attachments: [], ...(clientMessageId ? { clientMessageId } : {}) });
}

function sync(app, conversationId, messages, clientVersion = 1) {
  return request(app)
    .post(`/api/chat/conversations/${conversationId}/sync`)
    .send({ clientVersion, messages, deletedMessageIds: [], title: "New Chat", titleUpdatedAt: 0 });
}

/** The user message as the website holds it before it learns the server's id. */
function local(id, content) {
  return { id, role: "user", content, type: "text", attachments: [], timestamp: new Date().toISOString() };
}

function openSocket(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.on("open", () => resolve(ws));
    ws.on("error", reject);
  });
}

test("#471: a sent message is stored once, under the server's id, keeping the id the client gave it", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const clientId = randomUUID();
    const res = await sendRest(server.app, conversationId, "hello", clientId);
    assert.equal(res.status, 200);
    assert.notEqual(res.body.userMessageId, clientId);

    const conv = await stored(server.app, conversationId);
    const [question] = questions(conv, "hello");
    assert.equal(questions(conv, "hello").length, 1);
    assert.equal(question.id, res.body.userMessageId);
    assert.equal(question.clientId, clientId);

    // Over WebSocket the same: the ack names the server's id, the client's is kept.
    const ws = await openSocket(server.wsUrl);
    const wsClientId = randomUUID();
    const ack = await new Promise((resolve) => {
      ws.on("message", (data) => {
        const frame = JSON.parse(data);
        if (frame.type === "ack") resolve(frame);
      });
      ws.send(JSON.stringify({ type: "chat", content: "again", conversationId, clientMessageId: wsClientId }));
    });
    ws.close();
    await waitFor(async () => replies(await stored(server.app, conversationId)).length === 2, "the second reply");
    const after = await stored(server.app, conversationId);
    const [second] = questions(after, "again");
    assert.equal(second.id, ack.messageId);
    assert.equal(second.clientId, wsClientId);
  } finally {
    await server.stop();
  }
});

test("#471: REST — a reconnect sync that still carries the client's id does not store the question again", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const content = "hold: what is the tallest lighthouse?";
    const clientId = randomUUID();
    // Started, not awaited: supertest sends only once something waits on it.
    const pending = sendRest(server.app, conversationId, content, clientId).then((res) => res);
    await waitFor(() => held, "the turn to be waiting on the model");

    // The website reconnected while the turn was in flight: it never had the
    // server's id, so it syncs the question under its own, with the empty reply
    // placeholder beside it.
    const placeholder = { id: randomUUID(), role: "assistant", content: "", type: "text", timestamp: new Date().toISOString() };
    const during = await sync(server.app, conversationId, [local(clientId, content), placeholder]);
    assert.equal(during.status, 200);
    assert.equal(during.body.diff.messages.clientOnly.length, 0);

    await finishHeldReply();
    const res = await pending;
    assert.equal(res.status, 200);

    // The response was lost too: the next sync still carries the client's id.
    const after = await sync(server.app, conversationId, [local(clientId, content)], res.body.version);
    assert.equal(after.status, 200);

    const conv = await stored(server.app, conversationId);
    assert.equal(questions(conv, content).length, 1, "the question is stored once");
    assert.deepEqual(conv.messages.map((m) => m.role), ["user", "assistant"]);
    assert.equal(replies(conv)[0].content, FULL_REPLY);
    assert.equal(conv.version, res.body.version, "neither sync was a new revision");
  } finally {
    if (held) held.release();
    await server.stop();
  }
});

test("#471: WebSocket — the ack lost to a dropped socket does not let the reconnect sync store the question again", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const content = "hold: what is the tallest lighthouse?";
    const clientId = randomUUID();
    const ws = await openSocket(server.wsUrl);
    ws.send(JSON.stringify({ type: "chat", content, conversationId, messageType: "text", attachments: [], clientMessageId: clientId }));
    // Gone before the ack is read.
    ws.terminate();
    await waitFor(() => held, "the turn to be waiting on the model");

    const synced = await sync(server.app, conversationId, [local(clientId, content)]);
    assert.equal(synced.status, 200);
    assert.equal(synced.body.diff.messages.clientOnly.length, 0);

    await finishHeldReply();
    const conv = await stored(server.app, conversationId);
    assert.equal(questions(conv, content).length, 1, "the question is stored once");
    assert.deepEqual(conv.messages.map((m) => m.role), ["user", "assistant"]);
    assert.equal(replies(conv)[0].content, FULL_REPLY);
  } finally {
    if (held) held.release();
    await server.stop();
  }
});

test("#471: a message the server never received still syncs as an offline message", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const offline = local(randomUUID(), "written while offline");

    const synced = await sync(server.app, conversationId, [offline]);
    assert.equal(synced.status, 200);
    assert.deepEqual(synced.body.diff.messages.clientOnly.map((m) => m.id), [offline.id]);

    const conv = await stored(server.app, conversationId);
    assert.deepEqual(conv.messages.map((m) => [m.id, m.role, m.content]), [[offline.id, "user", offline.content]]);
  } finally {
    await server.stop();
  }
});

test("#471: a send that arrives after a sync already stored the message answers that message", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const clientId = randomUUID();
    const content = "which came first?";
    await sync(server.app, conversationId, [local(clientId, content)]);

    // REST: the request carrying the same message lands after the sync.
    const res = await sendRest(server.app, conversationId, content, clientId);
    assert.equal(res.status, 200);
    assert.equal(res.body.userMessageId, clientId);
    let conv = await stored(server.app, conversationId);
    assert.equal(questions(conv, content).length, 1);
    assert.deepEqual(conv.messages.map((m) => m.role), ["user", "assistant"]);

    // WebSocket: the same, with the ack naming the message already stored.
    const wsClientId = randomUUID();
    const wsContent = "and over the socket?";
    await sync(server.app, conversationId, [...conv.messages, local(wsClientId, wsContent)], conv.version);
    const ws = await openSocket(server.wsUrl);
    const ack = await new Promise((resolve) => {
      ws.on("message", (data) => {
        const frame = JSON.parse(data);
        if (frame.type === "ack") resolve(frame);
      });
      ws.send(JSON.stringify({ type: "chat", content: wsContent, conversationId, clientMessageId: wsClientId }));
    });
    assert.equal(ack.messageId, wsClientId);
    await waitFor(async () => replies(await stored(server.app, conversationId)).length === 2, "the second reply");
    ws.close();
    conv = await stored(server.app, conversationId);
    assert.equal(questions(conv, wsContent).length, 1);
    assert.deepEqual(conv.messages.map((m) => m.role), ["user", "assistant", "user", "assistant"]);
  } finally {
    await server.stop();
  }
});

test("#471: several messages around one reconnect: each is stored once", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    // A: sent and answered; the client learned the server's ids.
    const a = randomUUID();
    const sentA = await sendRest(server.app, conversationId, "first", a);
    // B: sent and answered, but the response was lost.
    const b = randomUUID();
    await sendRest(server.app, conversationId, "second", b);
    // C: never reached the server.
    const c = randomUUID();

    const now = new Date().toISOString();
    const synced = await sync(server.app, conversationId, [
      { id: sentA.body.userMessageId, clientId: a, role: "user", content: "first", type: "text", attachments: [], timestamp: now },
      { id: sentA.body.id, role: "assistant", content: sentA.body.content, type: "text", completed: true, timestamp: now },
      local(b, "second"),
      local(c, "third, offline"),
    ]);
    assert.equal(synced.status, 200);
    assert.deepEqual(synced.body.diff.messages.clientOnly.map((m) => m.id), [c]);

    const conv = await stored(server.app, conversationId);
    for (const content of ["first", "second", "third, offline"]) {
      assert.equal(questions(conv, content).length, 1, `"${content}" is stored once`);
    }
    assert.equal(replies(conv).length, 2);
    assert.equal(new Set(conv.messages.map((m) => m.id)).size, conv.messages.length);
  } finally {
    await server.stop();
  }
});

test("#471: the #457 rule still holds beside it: a cut-off reply in the same sync is left out", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const content = "hold: tell me a long story";
    const clientId = randomUUID();
    const ws = await openSocket(server.wsUrl);
    let ack = null;
    const partial = await new Promise((resolve) => {
      ws.on("message", (data) => {
        const frame = JSON.parse(data);
        if (frame.type === "ack") ack = frame;
        if (frame.type === "stream" && !frame.done && (frame.content.match(/w\d{3}/g) || []).length === 10) {
          resolve(frame.content);
        }
      });
      ws.send(JSON.stringify({ type: "chat", content, conversationId, clientMessageId: clientId }));
    });
    ws.terminate();
    await waitFor(() => held, "the turn to be waiting on the model");

    const now = new Date().toISOString();
    const synced = await sync(server.app, conversationId, [
      { id: ack.messageId, clientId, role: "user", content, type: "text", attachments: [], timestamp: now },
      { id: ack.assistantMessageId, role: "assistant", content: partial, type: "text", completed: true, timestamp: now },
    ], ack.version);
    assert.equal(synced.status, 200);

    await finishHeldReply();
    const conv = await stored(server.app, conversationId);
    assert.equal(questions(conv, content).length, 1);
    assert.equal(replies(conv).length, 1);
    assert.equal(replies(conv)[0].content, FULL_REPLY);
  } finally {
    if (held) held.release();
    await server.stop();
  }
});

test("#471: a stored message is matched by the server's id and by the client's", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const clientId = randomUUID();
    const res = await sendRest(server.app, conversationId, "hello", clientId);
    const before = await stored(server.app, conversationId);
    const [question, reply] = before.messages;
    const replyCopy = { id: reply.id, role: "assistant", content: reply.content, type: reply.type };

    const byServerId = await sync(server.app, conversationId, [
      { id: res.body.userMessageId, clientId, role: "user", content: "hello", type: "text", attachments: [] },
      replyCopy,
    ], before.version);
    assert.equal(byServerId.status, 200);
    assert.equal(byServerId.body.diff.messages.clientOnly.length, 0);

    const byClientId = await sync(server.app, conversationId, [local(clientId, "hello"), replyCopy], before.version);
    assert.equal(byClientId.status, 200);
    assert.equal(byClientId.body.diff.messages.clientOnly.length, 0);

    const after = await stored(server.app, conversationId);
    assert.deepEqual(after.messages.map((m) => [m.id, m.role]), [[question.id, "user"], [reply.id, "assistant"]]);
  } finally {
    await server.stop();
  }
});

test("#471: a client message id must be a valid id on both transports", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const res = await sendRest(server.app, conversationId, "hello", "not-a-uuid");
    assert.equal(res.status, 400);

    const ws = await openSocket(server.wsUrl);
    const frame = await new Promise((resolve) => {
      ws.on("message", (data) => resolve(JSON.parse(data)));
      ws.send(JSON.stringify({ type: "chat", content: "hello", conversationId, clientMessageId: "not-a-uuid" }));
    });
    ws.close();
    assert.equal(frame.type, "error");
    assert.match(frame.message, /clientMessageId/);
    assert.equal((await stored(server.app, conversationId)).messages.length, 0);
  } finally {
    await server.stop();
  }
});
