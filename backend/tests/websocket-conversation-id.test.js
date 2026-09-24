/**
 * WebSocket conversationId validation (Issue #352).
 *
 * Root cause: parseChatFrame() returned message.conversationId untouched, so
 * any client-supplied value — including a non-string — reached
 * conversationStore.append()/messages() and was echoed back in every ack,
 * typing and stream frame. The REST route has always required a UUID through
 * v.uuid("id"), so the two transports disagreed about what an id even is.
 *
 * The expected format is the one the app already uses: conversations are
 * created with randomUUID() in routes/chat.js and the REST route validates with
 * Validator.uuid(). Nothing new is invented here.
 *
 * Fix: parseChatFrame() runs v.uuid("conversationId") whenever the field is
 * present, reusing the REST validator and the frame's existing error path, so a
 * bad id is answered with the established { type: "error" } frame before any
 * conversation or model work happens. The field stays optional — a frame
 * without one is a single-turn chat, which is long-standing behaviour.
 *
 * These tests verify that:
 *   - a valid UUID still acks, streams and completes as before,
 *   - an omitted or null id still works (single-turn chat),
 *   - empty, malformed and wrongly-typed ids are rejected,
 *   - a rejected id never reaches conversationStore or the model,
 *   - rejection uses the existing error frame, not a new protocol,
 *   - the connection survives a rejected frame.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");

const { mockModel, startServer } = require("./helpers");

const modelCalls = mockModel();
const conversations = require("../src/conversationStore");

const VALID_ID = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d";

/**
 * Open a connection, send one frame, and collect every reply until the stream
 * finishes, an error arrives, or the wait times out.
 */
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
      if (message.type === "pong") finish();
    });
  });
}

// ---------------------------------------------------------------------------
// Valid ids keep working
// ---------------------------------------------------------------------------

test("a valid UUID still acks, streams and completes", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "hello there",
    conversationId: VALID_ID,
  });

  const types = messages.map((m) => m.type);
  assert.ok(types.includes("ack"), "should still acknowledge the frame");
  assert.ok(types.includes("typing"), "should still send a typing indicator");

  const streamed = messages.filter((m) => m.type === "stream");
  assert.ok(streamed.length > 0, "should still stream");

  const final = streamed[streamed.length - 1];
  assert.equal(final.done, true);
  assert.equal(final.content, "reply to: hello there");
  // The id is echoed back untouched, exactly as before.
  assert.equal(final.conversationId, VALID_ID);
});

test("an omitted id is still a single-turn chat", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  // JSON.stringify drops an undefined value, so this frame carries no id at
  // all — the long-standing way to chat without joining a conversation.
  const messages = await exchange(server.wsUrl, { type: "chat", content: "no id here" });

  const streamed = messages.filter((m) => m.type === "stream");
  assert.ok(streamed.length > 0, "a frame with no id must still be answered");
  assert.equal(streamed[streamed.length - 1].done, true);
});

test("an explicit null id is treated as omitted, not as malformed", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "null id",
    conversationId: null,
  });

  const streamed = messages.filter((m) => m.type === "stream");
  assert.ok(streamed.length > 0, "null means 'no conversation', consistent with the other readers");
  assert.equal(streamed[streamed.length - 1].done, true);
});

// ---------------------------------------------------------------------------
// Invalid ids are rejected
// ---------------------------------------------------------------------------

// Each of these reached conversationStore unchecked before the fix.
const REJECTED = [
  ["an empty string", ""],
  ["a short non-uuid string", "abc"],
  ["a descriptive placeholder", "not-a-uuid"],
  ["the REST-style label the old tests used", "test-conv"],
  ["a uuid one character short", "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4"],
  ["a uuid one character long", "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4de"],
  ["a uuid with underscores for hyphens", "0a1b2c3d_4e5f_4a6b_8c7d_9e0f1a2b3c4d"],
  ["a uuid with no separators", "0a1b2c3d4e5f4a6b8c7d9e0f1a2b3c4d"],
  ["a uuid with non-hex characters", "gggggggg-4e5f-4a6b-8c7d-9e0f1a2b3c4d"],
  ["a uuid with trailing whitespace", `${VALID_ID} `],
  ["a uuid with leading whitespace", ` ${VALID_ID}`],
  ["a traversal-shaped id", "../../etc/passwd"],
  ["a path appended to a valid uuid", `${VALID_ID}/../other`],
  ["a number", 12345],
  ["a boolean", true],
  ["an object", { id: VALID_ID }],
  ["an array", [VALID_ID]],
];

test("empty, malformed and wrongly-typed ids are all rejected", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  for (const [label, conversationId] of REJECTED) {
    const [message] = await exchange(server.wsUrl, {
      type: "chat",
      content: "should not be processed",
      conversationId,
    });

    assert.equal(message.type, "error", `${label} should be rejected`);
    assert.match(
      message.message,
      /conversationId must be a valid id/,
      `${label} should report the id as the problem`
    );
  }
});

test("a rejected frame is answered with the existing error frame, not a close", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const ws = new WebSocket(server.wsUrl);
  await new Promise((r) => ws.on("open", r));

  const first = await new Promise((resolve, reject) => {
    ws.on("error", reject);
    ws.on("message", (data) => resolve(JSON.parse(data)));
    ws.send(JSON.stringify({ type: "chat", content: "bad id", conversationId: "abc" }));
  });

  assert.equal(first.type, "error", "rejection reuses the established error frame");

  // The connection must survive, the same way a malformed-JSON frame does.
  const pong = await new Promise((resolve, reject) => {
    ws.on("error", reject);
    ws.on("message", (data) => {
      const message = JSON.parse(data);
      if (message.type === "pong") resolve(message);
    });
    ws.send(JSON.stringify({ type: "ping" }));
  });

  assert.equal(pong.type, "pong", "a rejected id must not take the connection down");
  ws.close();
});

// ---------------------------------------------------------------------------
// Rejected ids never reach downstream conversation or model work
// ---------------------------------------------------------------------------

// This is the discriminating test: on the old code the bad id was handed to
// conversationStore.append() and messages(), and the model ran.
test("a rejected id reaches neither conversationStore nor the model", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  // websocket.js holds the conversationStore module object and looks the
  // methods up per call, so wrapping them here observes the real calls.
  const original = { append: conversations.append, messages: conversations.messages };
  const seen = [];
  conversations.append = (id, message) => {
    seen.push(["append", id]);
    return original.append(id, message);
  };
  conversations.messages = (id) => {
    seen.push(["messages", id]);
    return original.messages(id);
  };
  t.after(() => Object.assign(conversations, original));

  // A valid id first, to prove the spy is actually wired to the code under
  // test — otherwise the negative assertion below would pass on a blind spy.
  const callsBeforeValid = modelCalls.length;
  await exchange(server.wsUrl, { type: "chat", content: "good", conversationId: VALID_ID });

  assert.ok(
    seen.some(([, id]) => id === VALID_ID),
    "sanity check: a valid id must reach conversationStore, or this spy proves nothing"
  );
  assert.ok(modelCalls.length > callsBeforeValid, "sanity check: a valid id must run the model");

  // Now the invalid ones.
  seen.length = 0;
  const callsBeforeInvalid = modelCalls.length;

  for (const [, conversationId] of REJECTED) {
    await exchange(server.wsUrl, {
      type: "chat",
      content: "should not be processed",
      conversationId,
    });
  }

  assert.deepEqual(
    seen,
    [],
    `no rejected id may reach conversationStore, saw: ${JSON.stringify(seen)}`
  );
  assert.equal(
    modelCalls.length,
    callsBeforeInvalid,
    "a rejected frame must not run the model"
  );
});
