/**
 * conversationId on catch-all WebSocket error frames (Issue #358).
 *
 * Root cause: conversationId was declared with const inside the
 * `if (message.type === "chat")` block, so the handler's outer catch could not
 * name it. The rate-limit error frame carried the id, the catch-all did not,
 * and a client that routes frames by conversationId therefore could not tell
 * which conversation a failed turn belonged to.
 *
 * Fix: declare conversationId per message in the handler scope, populate it as
 * soon as parseChatFrame has validated one, and include it in the catch-all
 * frame. It stays undefined when parsing failed before an id was available, and
 * JSON.stringify drops undefined, so that case sends the frame unchanged rather
 * than a fabricated id.
 *
 * This is the backend half. #250 is frontend-scoped (useChat.js streaming
 * state) and does not put the id on this frame.
 *
 * These tests verify that:
 *   - a throw after the id is known reports that exact id,
 *   - a throw before an id is available adds no id at all,
 *   - an invalid id is never echoed back,
 *   - the rate-limit error path still carries the id,
 *   - the frame's type/message structure is otherwise unchanged,
 *   - the connection is still usable after the error.
 */

process.env.GENERATE_RATE_LIMIT_MAX = "500";

const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");

const { mockModel, startServer } = require("./helpers");
const { generationCounter } = require("../src/middleware/limiters");

mockModel();

const CONV_ID = "3fa85f64-5717-4562-b3fc-2c963f66afa6";

function exchange(wsUrl, frame, { timeoutMs = 4000 } = {}) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const received = [];

    const finish = () => {
      clearTimeout(timer);
      ws.close();
      resolve(received);
    };
    const timer = setTimeout(finish, timeoutMs);

    ws.on("open", () => ws.send(JSON.stringify(frame)));
    ws.on("error", reject);
    ws.on("message", (data) => {
      const message = JSON.parse(data);
      received.push(message);
      if (message.type === "error" || (message.type === "stream" && message.done)) finish();
    });
  });
}

/** A model whose turn throws once the frame has already been parsed. */
function throwingModel(message = "worker exploded") {
  return {
    processMessage: async () => {
      throw new Error(message);
    },
  };
}

// ---------------------------------------------------------------------------
// The fix
// ---------------------------------------------------------------------------

// DISCRIMINATING TEST: the throw happens well after parseChatFrame has run, so
// the id is known. Before the fix this frame arrived with no conversationId.
test("a throw after the id is known reports that exact conversationId", async (t) => {
  mockModel(throwingModel("worker exploded"));
  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "hello",
    conversationId: CONV_ID,
  });

  const error = messages.find((m) => m.type === "error");
  assert.ok(error, "an error frame must be emitted");
  assert.equal(
    error.conversationId,
    CONV_ID,
    "the error must name the conversation the failed turn belonged to"
  );
  assert.equal(error.message, "worker exploded", "the error message must be unchanged");
});

test("the error frame's structure is unchanged apart from the added id", async (t) => {
  mockModel(throwingModel("worker exploded"));
  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "hello",
    conversationId: CONV_ID,
  });

  const error = messages.find((m) => m.type === "error");
  assert.deepEqual(
    Object.keys(error).sort(),
    ["conversationId", "message", "type"],
    "no other field may be added to or removed from the frame"
  );
  assert.equal(error.type, "error");
});

// ---------------------------------------------------------------------------
// No id available: frame unchanged, nothing fabricated
// ---------------------------------------------------------------------------

test("a throw before an id is available adds no conversationId", async (t) => {
  // parseChatFrame checks content first and throws before returning, so no id
  // has been validated even though the frame carried one.
  mockModel();
  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, { type: "chat", conversationId: CONV_ID });

  const error = messages.find((m) => m.type === "error");
  assert.ok(error, "an error frame must still be emitted");
  assert.match(error.message, /content is required/, "the existing message is preserved");
  assert.ok(
    !Object.prototype.hasOwnProperty.call(error, "conversationId"),
    `no id may be fabricated, got ${JSON.stringify(error.conversationId)}`
  );
  assert.deepEqual(Object.keys(error).sort(), ["message", "type"], "frame unchanged in this case");
});

test("a malformed frame that is not JSON adds no conversationId", async (t) => {
  mockModel();
  const server = await startServer();
  t.after(() => server.stop());

  const ws = new WebSocket(server.wsUrl);
  const error = await new Promise((resolve, reject) => {
    ws.on("error", reject);
    ws.on("open", () => ws.send("definitely not json"));
    ws.on("message", (data) => resolve(JSON.parse(data)));
  });

  assert.equal(error.type, "error");
  assert.ok(
    !Object.prototype.hasOwnProperty.call(error, "conversationId"),
    "a frame that never parsed cannot name a conversation"
  );
  ws.close();
});

test("an invalid conversationId is never echoed back", async (t) => {
  mockModel();
  const server = await startServer();
  t.after(() => server.stop());

  for (const bad of ["not-a-uuid", "", 12345, { id: CONV_ID }]) {
    const messages = await exchange(server.wsUrl, {
      type: "chat",
      content: "hello",
      conversationId: bad,
    });

    const error = messages.find((m) => m.type === "error");
    assert.ok(error, `${JSON.stringify(bad)} should be rejected`);
    assert.match(error.message, /conversationId must be a valid id/);
    assert.ok(
      !Object.prototype.hasOwnProperty.call(error, "conversationId"),
      `an unvalidated id must not be echoed, got ${JSON.stringify(error.conversationId)}`
    );
  }
});

// ---------------------------------------------------------------------------
// The path that already carried the id keeps carrying it
// ---------------------------------------------------------------------------

test("the rate-limit error path still includes conversationId", async (t) => {
  mockModel();
  const server = await startServer();
  t.after(() => server.stop());

  // Shrink the shared counter rather than racing the default of 500.
  const originalMax = generationCounter.max;
  generationCounter.max = 1;
  generationCounter.buckets.clear();
  t.after(() => {
    generationCounter.max = originalMax;
    generationCounter.buckets.clear();
  });

  // First frame consumes the single token.
  await exchange(server.wsUrl, { type: "chat", content: "one", conversationId: CONV_ID });
  // Second is refused by the limiter.
  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "two",
    conversationId: CONV_ID,
  });

  const error = messages.find((m) => m.type === "error");
  assert.ok(error, "the limiter must answer with an error frame");
  assert.equal(error.code, "RATE_LIMITED", "the existing code is preserved");
  assert.equal(error.conversationId, CONV_ID, "this path already carried the id and must keep it");
});

// ---------------------------------------------------------------------------
// Connection lifecycle is untouched
// ---------------------------------------------------------------------------

test("the connection is still usable after a catch-all error", async (t) => {
  mockModel(throwingModel("worker exploded"));
  const server = await startServer();
  t.after(() => server.stop());

  const ws = new WebSocket(server.wsUrl);
  await new Promise((r) => ws.on("open", r));

  const error = await new Promise((resolve, reject) => {
    ws.on("error", reject);
    ws.on("message", (data) => {
      const m = JSON.parse(data);
      if (m.type === "error") resolve(m);
    });
    ws.send(JSON.stringify({ type: "chat", content: "hello", conversationId: CONV_ID }));
  });

  assert.equal(error.conversationId, CONV_ID);

  const pong = await new Promise((resolve, reject) => {
    ws.on("error", reject);
    ws.on("message", (data) => {
      const m = JSON.parse(data);
      if (m.type === "pong") resolve(m);
    });
    ws.send(JSON.stringify({ type: "ping" }));
  });

  assert.equal(pong.type, "pong", "the error must not take the connection down");
  ws.close();
});

// ---------------------------------------------------------------------------
// The id must not leak between messages on one connection
// ---------------------------------------------------------------------------

test("an id from an earlier frame does not leak into a later frame's error", async (t) => {
  mockModel(throwingModel("worker exploded"));
  const server = await startServer();
  t.after(() => server.stop());

  const ws = new WebSocket(server.wsUrl);
  await new Promise((r) => ws.on("open", r));

  const collect = (predicate) =>
    new Promise((resolve, reject) => {
      const onMessage = (data) => {
        const m = JSON.parse(data);
        if (predicate(m)) {
          ws.off("message", onMessage);
          resolve(m);
        }
      };
      ws.on("error", reject);
      ws.on("message", onMessage);
    });

  // A turn that fails with a known id.
  const first = collect((m) => m.type === "error");
  ws.send(JSON.stringify({ type: "chat", content: "hello", conversationId: CONV_ID }));
  assert.equal((await first).conversationId, CONV_ID);

  // A second frame on the SAME connection that fails before an id is available
  // must not inherit the previous one.
  const second = collect((m) => m.type === "error");
  ws.send(JSON.stringify({ type: "chat" }));
  const secondError = await second;

  assert.match(secondError.message, /content is required/);
  assert.ok(
    !Object.prototype.hasOwnProperty.call(secondError, "conversationId"),
    `the previous frame's id must not leak, got ${JSON.stringify(secondError.conversationId)}`
  );
  ws.close();
});
