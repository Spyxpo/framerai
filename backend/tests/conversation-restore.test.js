/**
 * Restoring a conversation the server lost (Issue #503).
 *
 * The store lives in memory: a restart empties it, and a conversation left idle
 * past its TTL is swept on the next create(). The website keeps its own copy of
 * every conversation, so after either one it holds a conversation the server
 * does not. POST /chat/conversations/:id/restore puts that copy back under its
 * own id, assistant replies included, which /sync deliberately never adds
 * (Issue #456). It only ever creates: a conversation the server holds is
 * answered with 409 and left exactly as it is.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");
const { randomUUID } = require("node:crypto");

const { mockModel, startServer } = require("./helpers");

// Every model run is logged with the history it was given, copied at call time:
// the store hands over its live array, which grows once the reply is stored.
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

function restore(app, id, body) {
  return request(app).post(`/api/chat/conversations/${id}/restore`).send(body);
}

/** One WebSocket chat turn; resolves with every frame up to done or error. */
function turn(wsUrl, frame) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const frames = [];
    const timer = setTimeout(() => {
      ws.close();
      reject(new Error(`no done or error frame; saw ${frames.map((f) => f.type).join(",")}`));
    }, 3000);
    ws.on("open", () => ws.send(JSON.stringify({ type: "chat", messageType: "text", attachments: [], ...frame })));
    ws.on("error", reject);
    ws.on("message", (data) => {
      const f = JSON.parse(data);
      frames.push(f);
      if (f.type === "error" || (f.type === "stream" && f.done)) {
        clearTimeout(timer);
        ws.close();
        resolve(frames);
      }
    });
  });
}

/** A conversation with two finished exchanges, as the website holds it afterwards. */
async function conversationWithHistory(server) {
  const id = (await request(server.app).post("/api/chat/conversations")).body.id;
  await turn(server.wsUrl, { conversationId: id, content: "My name is Ada." });
  await turn(server.wsUrl, { conversationId: id, content: "I live in London.", expectedVersion: store.get(id).version });
  const held = (await request(server.app).get(`/api/chat/conversations/${id}`)).body;
  assert.equal(held.messages.length, 4);
  // What the website sends: its own copy, every finished reply marked completed.
  const local = held.messages.map((m) => (m.role === "assistant" ? { ...m, completed: true } : m));
  return { id, held, local };
}

/** The idle sweep, as production runs it: on the next create(), past the TTL. */
async function sweepIdle(app) {
  store._resetLimits({ ttl: 1 });
  await new Promise((r) => setTimeout(r, 20));
  await request(app).post("/api/chat/conversations");
  store._resetLimits();
}

const shape = (messages) => messages.map((m) => [m.id, m.role, m.content]);

for (const [lost, lose] of [
  ["a backend restart", async () => store.clear()],
  ["the idle sweep", async (server) => sweepIdle(server.app)],
]) {
  test(`#503: a conversation lost to ${lost} is restored under its own id, with its replies, and answers with its history`, async () => {
    const server = await startServer();
    try {
      const { id, held, local } = await conversationWithHistory(server);
      await lose(server);
      assert.equal(store.has(id), false, `${lost} removed the conversation`);

      const res = await restore(server.app, id, { title: "About Ada", messages: local });
      assert.equal(res.status, 201);
      assert.equal(res.body.id, id, "the original id is kept");
      assert.equal(res.body.title, "About Ada");
      assert.deepEqual(shape(res.body.messages), shape(held.messages), "every message, in order, under its own id");

      const stored = (await request(server.app).get(`/api/chat/conversations/${id}`)).body;
      assert.deepEqual(shape(stored.messages), shape(held.messages));
      assert.ok(stored.messages.every((m) => !("completed" in m)), "the client's completed flag is not stored");

      // The next turn joins the restored history rather than starting over.
      modelCalls.length = 0;
      const frames = await turn(server.wsUrl, { conversationId: id, content: "What is my name?", expectedVersion: stored.version });
      assert.ok(frames.some((f) => f.type === "stream" && f.done), "the turn completes");
      assert.deepEqual(modelCalls[0], [
        "user: My name is Ada.",
        "assistant: reply to: My name is Ada.",
        "user: I live in London.",
        "assistant: reply to: I live in London.",
        "user: What is my name?",
      ]);
      assert.equal(store.get(id).messages.length, 6);
    } finally {
      await server.stop();
    }
  });
}

test("#503: restoring a conversation the server holds is refused with 409 and changes nothing", async () => {
  const server = await startServer();
  try {
    const { id, held } = await conversationWithHistory(server);
    const before = JSON.stringify(store.get(id));
    const versionBefore = store.get(id).version;

    const res = await restore(server.app, id, {
      title: "Overwritten?",
      messages: [{ id: randomUUID(), role: "user", content: "a different history" }],
    });

    assert.equal(res.status, 409);
    assert.equal(res.body.code, "CONVERSATION_EXISTS");
    assert.equal(JSON.stringify(store.get(id)), before, "messages, title and timestamps are untouched");
    assert.equal(store.get(id).version, versionBefore, "and so is the version");
    assert.deepEqual(shape(store.get(id).messages), shape(held.messages));
  } finally {
    await server.stop();
  }
});

test("#503: an assistant reply not marked completed is left out of a restore", async () => {
  const server = await startServer();
  try {
    const id = randomUUID();
    const [q1, a1, q2, a2, a3] = Array.from({ length: 5 }, () => randomUUID());
    const res = await restore(server.app, id, {
      messages: [
        { id: q1, role: "user", content: "first" },
        { id: a1, role: "assistant", content: "finished reply", completed: true },
        { id: q2, role: "user", content: "second" },
        { id: a2, role: "assistant", content: "cut off mid-str", completed: false },
        { id: a3, role: "assistant", content: "never marked" },
      ],
    });
    assert.equal(res.status, 201);
    assert.deepEqual(shape(res.body.messages), [
      [q1, "user", "first"],
      [a1, "assistant", "finished reply"],
      [q2, "user", "second"],
    ]);
  } finally {
    await server.stop();
  }
});

test("#503: a restore is validated by the rules /sync applies, and a refused one stores nothing", async () => {
  const server = await startServer();
  try {
    const ok = () => ({ id: randomUUID(), role: "user", content: "fine" });
    const cases = [
      ["an unknown role", [{ ...ok(), role: "system" }], "messages.0.role"],
      ["an unknown type", [{ ...ok(), type: "error" }], "messages.0.type"],
      ["a message that is not an object", ["just text"], "messages.0"],
      ["a missing message id", [{ role: "user", content: "no id" }], "messages.0.id"],
      ["content that is not a string", [{ ...ok(), content: 42 }], "messages.0.content"],
      ["content past the message limit", [{ ...ok(), content: "x".repeat(8001) }], "messages.0.content"],
      ["a completed flag that is not a boolean", [{ ...ok(), role: "assistant", completed: "yes" }], "messages.0.completed"],
      ["the same id twice", (() => { const m = ok(); return [m, { ...m, content: "again" }]; })(), "messages.1.id"],
    ];
    for (const [label, messages, field] of cases) {
      const id = randomUUID();
      const res = await restore(server.app, id, { messages });
      assert.equal(res.status, 400, label);
      assert.equal(res.body.code, "VALIDATION_ERROR", label);
      assert.ok(res.body.details.some((d) => d.field === field), `${label}: expected an error on ${field}, got ${JSON.stringify(res.body.details)}`);
      assert.equal(store.has(id), false, `${label}: nothing was created`);
    }

    // More messages than a conversation may hold, the same ceiling /sync uses.
    store._resetLimits({ maxMessages: 3 });
    const id = randomUUID();
    const res = await restore(server.app, id, { messages: [ok(), ok(), ok(), ok()] });
    assert.equal(res.status, 400);
    assert.ok(res.body.details.some((d) => d.field === "messages"));
    assert.equal(store.has(id), false);

    const badId = await restore(server.app, "not-a-uuid", { messages: [ok()] });
    assert.equal(badId.status, 400);
  } finally {
    await server.stop();
  }
});

test("#503: two restores of one id at once create it once, and the second gets 409", async () => {
  const server = await startServer();
  try {
    const id = randomUUID();
    const first = { messages: [{ id: randomUUID(), role: "user", content: "from the first tab" }] };
    const second = { messages: [{ id: randomUUID(), role: "user", content: "from the second tab" }] };

    const results = await Promise.all([restore(server.app, id, first), restore(server.app, id, second)]);
    const statuses = results.map((r) => r.status).sort();
    assert.deepEqual(statuses, [201, 409]);
    const winner = results.find((r) => r.status === 201);
    assert.deepEqual(store.get(id).messages.map((m) => m.content), winner.body.messages.map((m) => m.content));
    assert.equal(store.get(id).messages.length, 1);
  } finally {
    await server.stop();
  }
});

test("#503: an empty history restores an empty conversation the next turn can use", async () => {
  const server = await startServer();
  try {
    const id = randomUUID();
    const res = await restore(server.app, id, {});
    assert.equal(res.status, 201);
    assert.deepEqual(res.body.messages, []);
    assert.equal(res.body.title, "New Chat");

    const frames = await turn(server.wsUrl, { conversationId: id, content: "hello", expectedVersion: res.body.version });
    assert.ok(frames.some((f) => f.type === "stream" && f.done));
    assert.equal(store.get(id).messages.length, 2);
  } finally {
    await server.stop();
  }
});

test("#503: /sync on a restored conversation still never adds an assistant reply (#456)", async () => {
  const server = await startServer();
  try {
    const { id, local } = await conversationWithHistory(server);
    store.clear();
    const restored = (await restore(server.app, id, { messages: local })).body;

    const res = await request(server.app)
      .post(`/api/chat/conversations/${id}/sync`)
      .send({
        clientVersion: restored.version,
        messages: [...local, { id: randomUUID(), role: "assistant", content: "a reply the server never stored", completed: true }],
      });
    assert.equal(res.status, 200);
    assert.deepEqual(shape(res.body.conversation.messages), shape(restored.messages));
  } finally {
    await server.stop();
  }
});
