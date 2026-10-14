/**
 * Editing a question whose reply is still being generated (Issue #483).
 *
 * When the connection drops mid-stream the server keeps generating, and the
 * client, no longer streaming, lets the user edit the question. The edit
 * rewrote the question and asked for a new answer, but the reply still being
 * generated for the old wording was stored too when it finished, under the
 * edited question: two answers, one of them to a question nobody asked any more.
 *
 * Now an edit replaces any reply still being generated for that question, as a
 * regeneration of it does (#477), on both transports.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");

const { mockModel, startServer } = require("./helpers");

// A question starting with "hold" streams one word, then waits for the test to
// release it (or fail it), as a worker still generating does. Every answer
// names the question it was generated for.
let held = [];
mockModel({
  processMessage: async (messages, type = "text", settings = {}, options = {}) => {
    const question = messages[messages.length - 1].content;
    if (!question.startsWith("hold")) {
      return { type, content: `answer to: ${question}`, metadata: { model: "test-model" } };
    }
    if (typeof options?.onStream === "function") options.onStream({ delta: "partial " });
    let settle;
    const gate = new Promise((resolve) => {
      settle = resolve;
    });
    let markReturned;
    const returned = new Promise((resolve) => {
      markReturned = resolve;
    });
    held.push({ release: () => settle(false), fail: () => settle(true), returned });
    const failed = await gate;
    markReturned();
    if (failed) throw new Error("the model failed");
    return { type, content: `answer to: ${question}`, metadata: { model: "test-model" } };
  },
});

const store = require("../src/conversationStore");
const { generationCounter, apiCounter } = require("../src/middleware/limiters");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
  held = [];
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

/** Let held answer `i` (0-based) finish or fail, and wait until its turn has dealt with it. */
async function settle(i, how = "release") {
  held[i][how]();
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

/** Send a chat frame and resolve with the frames up to the first one `until` accepts. */
function send(ws, frame, until) {
  return new Promise((resolve, reject) => {
    const frames = [];
    const timer = setTimeout(() => reject(new Error(`no matching frame; saw ${JSON.stringify(frames.map((f) => f.type))}`)), 3000);
    ws.on("message", function onMessage(data) {
      const f = JSON.parse(data);
      frames.push(f);
      if (until(f)) {
        clearTimeout(timer);
        ws.off("message", onMessage);
        resolve(frames);
      }
    });
    ws.send(JSON.stringify({ type: "chat", messageType: "text", attachments: [], settings: {}, ...frame }));
  });
}

/** A held turn whose connection drops once the first words have streamed. */
async function droppedTurn(server, conversationId, content) {
  const ws = await openSocket(server.wsUrl);
  const frames = await send(ws, { content, conversationId }, (f) => f.type === "stream" && !f.done);
  ws.terminate();
  return frames.find((f) => f.type === "ack");
}

/** Edit a question over one transport; resolves once the edit's reply is done. */
async function edit(server, transport, conversationId, editMessageId, content) {
  if (transport === "REST") {
    return request(server.app)
      .post(`/api/chat/conversations/${conversationId}/messages`)
      .send({ editMessageId, content })
      .then((res) => res);
  }
  const ws = await openSocket(server.wsUrl);
  const frames = await send(ws, { content, conversationId, editMessageId }, (f) => (f.type === "stream" && f.done) || f.type === "error");
  ws.close();
  return frames;
}

async function newConversation(app) {
  return (await request(app).post("/api/chat/conversations")).body.id;
}

const shape = (id) => store.get(id).messages.map((m) => `${m.role}: ${m.content}`);

const TALLEST = "hold: What is the tallest lighthouse?";
const OLDEST = "hold: What is the oldest lighthouse?";

for (const transport of ["WebSocket", "REST"]) {
  for (const order of ["original first", "edit's reply first"]) {
    test(`#483: ${transport} — editing the question of a reply still being generated leaves one answer, to the edit (${order})`, async () => {
      const server = await startServer();
      try {
        const conversationId = await newConversation(server.app);
        const ack1 = await droppedTurn(server, conversationId, TALLEST);

        const editing = edit(server, transport, conversationId, ack1.messageId, OLDEST);
        await waitFor(() => held.length === 2, "the edit's reply to start");

        if (order === "original first") {
          await settle(0);
          await settle(1);
          await editing;
        } else {
          await settle(1);
          await editing;
          assert.deepEqual(shape(conversationId), [`user: ${OLDEST}`, `assistant: answer to: ${OLDEST}`]);
          await settle(0); // the original, late
        }

        assert.deepEqual(shape(conversationId), [`user: ${OLDEST}`, `assistant: answer to: ${OLDEST}`]);
        // The edit and its reply are the only revisions: the replaced reply added none.
        assert.equal(store.get(conversationId).version, ack1.version + 2);
        assert.equal(store.pendingReply(conversationId, ack1.assistantMessageId), null);
      } finally {
        held.forEach((h) => h.release());
        await server.stop();
      }
    });
  }

  test(`#483: ${transport} — editing after the original reply was stored still replaces it`, async () => {
    const server = await startServer();
    try {
      const conversationId = await newConversation(server.app);
      const ack1 = await droppedTurn(server, conversationId, TALLEST);
      await settle(0);
      assert.deepEqual(shape(conversationId), [`user: ${TALLEST}`, `assistant: answer to: ${TALLEST}`]);

      await edit(server, transport, conversationId, ack1.messageId, "What is the oldest lighthouse?");

      assert.deepEqual(shape(conversationId), [
        "user: What is the oldest lighthouse?",
        "assistant: answer to: What is the oldest lighthouse?",
      ]);
    } finally {
      held.forEach((h) => h.release());
      await server.stop();
    }
  });
}

test("#483: a replaced reply that fails, and an edit's reply that fails, leave nothing behind", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const ack1 = await droppedTurn(server, conversationId, TALLEST);

    const ws = await openSocket(server.wsUrl);
    const editing = send(ws, { content: OLDEST, conversationId, editMessageId: ack1.messageId }, (f) => f.type === "error");
    await waitFor(() => held.length === 2, "the edit's reply to start");
    await settle(0, "fail");
    await settle(1, "fail");
    const frames = await editing;
    ws.close();

    const ack2 = frames.find((f) => f.type === "ack");
    assert.deepEqual(shape(conversationId), [`user: ${OLDEST}`]);
    assert.equal(store.pendingReply(conversationId, ack1.assistantMessageId), null);
    assert.equal(store.pendingReply(conversationId, ack2.assistantMessageId), null);
  } finally {
    held.forEach((h) => h.release());
    await server.stop();
  }
});

test("#483: an edit replaces only the replies to the question it edits", async () => {
  const server = await startServer();
  try {
    // Two questions in one conversation, both still being answered.
    const a = await newConversation(server.app);
    const first = await droppedTurn(server, a, "hold: first question");
    const second = await droppedTurn(server, a, "hold: second question");
    // And one in another conversation.
    const b = await newConversation(server.app);
    const other = await droppedTurn(server, b, "hold: other conversation");

    const editing = edit(server, "WebSocket", a, second.messageId, "hold: second question, edited");
    await waitFor(() => held.length === 4, "the edit's reply to start");
    for (let i = 0; i < 4; i++) await settle(i);
    await editing;

    assert.equal(store.pendingReply(a, first.assistantMessageId), null);
    assert.equal(store.pendingReply(b, other.assistantMessageId), null);
    assert.deepEqual(shape(a), [
      "user: hold: first question",
      "assistant: answer to: hold: first question",
      "user: hold: second question, edited",
      "assistant: answer to: hold: second question, edited",
    ]);
    assert.deepEqual(shape(b), ["user: hold: other conversation", "assistant: answer to: hold: other conversation"]);
  } finally {
    held.forEach((h) => h.release());
    await server.stop();
  }
});
