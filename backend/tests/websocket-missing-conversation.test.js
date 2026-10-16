/**
 * A WebSocket turn for a conversation the server does not hold (Issue #503).
 *
 * The website keeps its conversations after the server has lost them, to a
 * restart or to the idle sweep, and goes on sending turns for them. The turn
 * used to be answered from its last message alone and stored nowhere, and the
 * reply looked like any other: the earlier conversation was silently gone.
 * REST has always refused such a turn with 404. Now the WebSocket refuses it
 * too, with a CONVERSATION_NOT_FOUND error frame that names the message it
 * refuses, before any model work and before any ack, so the website can restore
 * the conversation (POST /chat/conversations/:id/restore) and send it again.
 * A frame with no conversation id is still a one-off chat.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");
const { randomUUID } = require("node:crypto");

const { mockModel, startServer } = require("./helpers");

const modelCalls = [];
mockModel({
  processMessage: async (messages, type = "text") => {
    modelCalls.push(messages.map((m) => `${m.role}: ${m.content}`));
    return { type, content: `reply to: ${messages[messages.length - 1].content}`, metadata: { model: "test-model" } };
  },
});

const store = require("../src/conversationStore");
const { generationCounter, apiCounter } = require("../src/middleware/limiters");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
  modelCalls.length = 0;
  store._resetLimits();
});

/**
 * Send one frame and collect every reply until done or error, then wait a
 * little longer so a frame that should not come at all would still be seen.
 */
function exchange(wsUrl, frame) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const frames = [];
    let settle = null;
    const finish = () => {
      clearTimeout(timer);
      ws.close();
      resolve(frames);
    };
    const timer = setTimeout(finish, 3000);
    ws.on("open", () => ws.send(JSON.stringify({ type: "chat", messageType: "text", attachments: [], ...frame })));
    ws.on("error", reject);
    ws.on("message", (data) => {
      const f = JSON.parse(data);
      frames.push(f);
      if ((f.type === "error" || (f.type === "stream" && f.done)) && !settle) {
        settle = setTimeout(finish, 150);
      }
    });
  });
}

async function conversationWithATurn(server) {
  const id = (await request(server.app).post("/api/chat/conversations")).body.id;
  const frames = await exchange(server.wsUrl, { conversationId: id, content: "My name is Ada." });
  assert.ok(frames.some((f) => f.type === "stream" && f.done));
  return { id, version: store.get(id).version, messages: store.get(id).messages.map((m) => ({ ...m })) };
}

/** Assert the frames are exactly one refusal of `clientMessageId` in `id`, and nothing ran. */
function assertRefused(frames, id, clientMessageId) {
  assert.deepEqual(frames.map((f) => f.type), ["error"], "a refusal alone: no ack, typing or stream frame");
  const [error] = frames;
  assert.equal(error.code, "CONVERSATION_NOT_FOUND");
  assert.equal(error.conversationId, id);
  if (clientMessageId) assert.equal(error.clientMessageId, clientMessageId, "it names the message it refuses");
  assert.equal(typeof error.message, "string");
  assert.equal(modelCalls.length, 0, "the model is not run");
  assert.equal(store.has(id), false, "and nothing is created");
}

for (const [lost, lose] of [
  ["a backend restart", async () => store.clear()],
  [
    "the idle sweep",
    async (server) => {
      store._resetLimits({ ttl: 1 });
      await new Promise((r) => setTimeout(r, 20));
      await request(server.app).post("/api/chat/conversations");
      store._resetLimits();
    },
  ],
]) {
  test(`#503: a turn for a conversation lost to ${lost} is refused, not answered from its last message`, async () => {
    const server = await startServer();
    try {
      const { id, version } = await conversationWithATurn(server);
      await lose(server);
      assert.equal(store.has(id), false);
      modelCalls.length = 0;

      const clientMessageId = randomUUID();
      const frames = await exchange(server.wsUrl, {
        conversationId: id,
        content: "What is my name?",
        expectedVersion: version,
        clientMessageId,
      });
      assertRefused(frames, id, clientMessageId);
    } finally {
      await server.stop();
    }
  });
}

test("#503: a turn for an id the server never had is refused the same way", async () => {
  const server = await startServer();
  try {
    const id = randomUUID();
    const clientMessageId = randomUUID();
    assertRefused(await exchange(server.wsUrl, { conversationId: id, content: "hello", clientMessageId }), id, clientMessageId);
  } finally {
    await server.stop();
  }
});

test("#503: an edit or a regeneration in a lost conversation is refused before anything runs", async () => {
  const server = await startServer();
  try {
    const { id, messages } = await conversationWithATurn(server);
    store.clear();
    modelCalls.length = 0;
    const [question, reply] = messages;

    assertRefused(await exchange(server.wsUrl, { conversationId: id, content: "Edited", editMessageId: question.id }), id);
    assertRefused(await exchange(server.wsUrl, { conversationId: id, content: "", regenerateMessageId: reply.id }), id);
  } finally {
    await server.stop();
  }
});

test("#503: a refused turn uses no generation allowance", async () => {
  const server = await startServer();
  try {
    const id = randomUUID();
    await exchange(server.wsUrl, { conversationId: id, content: "hello" });
    assert.equal(generationCounter.buckets.size, 0, "the rate limiter was not charged");
  } finally {
    await server.stop();
  }
});

test("#503 control: a frame with no conversation id is still a one-off chat", async () => {
  const server = await startServer();
  try {
    for (const frame of [{ content: "no id here" }, { content: "null id", conversationId: null }]) {
      modelCalls.length = 0;
      const frames = await exchange(server.wsUrl, frame);
      assert.ok(frames.some((f) => f.type === "ack"), "acknowledged");
      assert.ok(frames.some((f) => f.type === "stream" && f.done), "answered");
      assert.deepEqual(modelCalls, [[`user: ${frame.content}`]]);
    }
  } finally {
    await server.stop();
  }
});

test("#503 control: a turn for a conversation the server holds runs as before", async () => {
  const server = await startServer();
  try {
    const { id, version } = await conversationWithATurn(server);
    modelCalls.length = 0;
    const frames = await exchange(server.wsUrl, { conversationId: id, content: "What is my name?", expectedVersion: version });
    assert.equal(frames[0].type, "ack");
    assert.ok(frames.some((f) => f.type === "stream" && f.done));
    assert.deepEqual(modelCalls, [["user: My name is Ada.", "assistant: reply to: My name is Ada.", "user: What is my name?"]]);
    assert.equal(store.get(id).messages.length, 4);
  } finally {
    await server.stop();
  }
});

test("#503 control: REST still answers 404 for the same lost conversation", async () => {
  const server = await startServer();
  try {
    const { id } = await conversationWithATurn(server);
    store.clear();
    modelCalls.length = 0;
    const res = await request(server.app).post(`/api/chat/conversations/${id}/messages`).send({ content: "What is my name?" });
    assert.equal(res.status, 404);
    assert.equal(modelCalls.length, 0);
  } finally {
    await server.stop();
  }
});
