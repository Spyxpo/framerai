/**
 * WebSocket response content validation (Issue #356).
 *
 * Root cause: the chat handler recorded the assistant turn with
 * conversations.append() immediately after processMessage() returned, and only
 * afterwards used response.content. modelChat passes the worker's
 * result.content through verbatim, so it is not guaranteed to be a string; a
 * reply without content was therefore persisted as a malformed assistant
 * message and the non-streamed path then threw
 * "Cannot read properties of undefined (reading 'split')", which surfaced to
 * the client as a raw internal TypeError with no reply frame at all.
 *
 * Fix: normalise response.content to a string between processMessage() and
 * conversations.append(), falling back to whatever already streamed. Both the
 * stored message and the two streaming paths then see the type they expect.
 *
 * These tests verify that:
 *   - a normal string reply still streams and is stored unchanged,
 *   - undefined / non-string content never reaches .split() and never produces
 *     an error frame,
 *   - append() is never called with a non-string assistant content,
 *   - the streamedFromWorker path gets the same treatment and keeps the text
 *     the client already received,
 *   - valid streamedFromWorker behaviour is unchanged.
 *
 * The append() spy is the discriminating assertion: it fails if the validation
 * is moved back after conversations.append().
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");

const { mockModel, startServer } = require("./helpers");
const conversations = require("../src/conversationStore");

mockModel();

let idSeq = 0;
function freshConversation() {
  // Ids must be well-formed UUIDs (#352), and append() ignores unknown ids, so
  // the conversation has to exist before the turn is recorded against it.
  const id = `00000000-0000-4000-8000-${String(++idSeq).padStart(12, "0")}`;
  conversations.create({ id, title: "T", messages: [], createdAt: new Date().toISOString() });
  return id;
}

/**
 * Wrap conversations.append() so a test can see exactly what was recorded.
 * websocket.js holds the store module object and looks the method up per call,
 * so wrapping it here observes the real calls.
 */
function spyAppend(t) {
  const original = conversations.append;
  const calls = [];
  conversations.append = (id, message) => {
    calls.push({ id, message });
    return original.call(conversations, id, message);
  };
  t.after(() => {
    conversations.append = original;
  });
  return calls;
}

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

/** A model that returns `content` without streaming anything. */
function nonStreamingModel(content) {
  return {
    processMessage: async () => ({ type: "text", content, metadata: { model: "test-model" } }),
  };
}

/** A model that streams deltas and then returns `content` as its final value. */
function streamingModel(content, deltas = ["Hello", " world"]) {
  return {
    processMessage: async (messages, type, settings, options) => {
      for (const delta of deltas) options?.onStream?.({ delta });
      return { type: "text", content, metadata: { model: "test-model" } };
    },
  };
}

function assistantAppends(calls) {
  return calls.filter((c) => c.message?.role === "assistant").map((c) => c.message);
}

// ---------------------------------------------------------------------------
// Valid content is unaffected
// ---------------------------------------------------------------------------

test("a normal string reply still streams and is stored unchanged", async (t) => {
  mockModel(nonStreamingModel("hello there friend"));
  const server = await startServer();
  t.after(() => server.stop());
  const calls = spyAppend(t);
  const id = freshConversation();

  const messages = await exchange(server.wsUrl, { type: "chat", content: "hi", conversationId: id });

  const types = messages.map((m) => m.type);
  assert.ok(types.includes("ack"), "should still acknowledge");
  assert.ok(types.includes("typing"), "should still send a typing indicator");
  assert.ok(!types.includes("error"), "a valid reply must not produce an error frame");

  const streamed = messages.filter((m) => m.type === "stream");
  assert.ok(streamed.length > 0, "should still stream");
  const final = streamed[streamed.length - 1];
  assert.equal(final.done, true);
  assert.equal(final.content, "hello there friend", "the full reply must arrive unchanged");

  const stored = assistantAppends(calls);
  assert.equal(stored.length, 1, "exactly one assistant turn is recorded");
  assert.equal(stored[0].content, "hello there friend", "stored content must be untouched");
  assert.deepEqual(
    conversations.messages(id).map((m) => m.content),
    ["hi", "hello there friend"]
  );
});

// ---------------------------------------------------------------------------
// Invalid content: never reaches .split(), never persisted malformed
// ---------------------------------------------------------------------------

// DISCRIMINATING TEST. Before the fix this recorded {content: undefined} and
// then answered with "Cannot read properties of undefined (reading 'split')".
test("undefined content is handled safely and never persisted", async (t) => {
  mockModel(nonStreamingModel(undefined));
  const server = await startServer();
  t.after(() => server.stop());
  const calls = spyAppend(t);
  const id = freshConversation();

  const messages = await exchange(server.wsUrl, { type: "chat", content: "hi", conversationId: id });

  const error = messages.find((m) => m.type === "error");
  assert.equal(
    error,
    undefined,
    `no error frame expected, got ${error && JSON.stringify(error.message)}`
  );

  const streamed = messages.filter((m) => m.type === "stream");
  assert.ok(streamed.length > 0, "the turn must still be terminated for the client");
  assert.equal(streamed[streamed.length - 1].done, true, "client must receive done:true");

  const stored = assistantAppends(calls);
  assert.equal(stored.length, 1, "the assistant turn is still recorded");
  assert.equal(
    typeof stored[0].content,
    "string",
    "append() must never be called with a non-string content"
  );

  const contents = conversations.messages(id).map((m) => m.content);
  assert.ok(
    contents.every((c) => typeof c === "string"),
    `no null/undefined content may be stored, got ${JSON.stringify(contents)}`
  );
  assert.deepEqual(contents, ["hi", ""], 'must not produce a ["hi", null]-style entry');
});

test("non-string content values are all handled safely and never persisted", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  // Wrap the original once per case rather than nesting spies.
  const original = conversations.append;
  t.after(() => {
    conversations.append = original;
  });

  const CASES = [
    ["null", null],
    ["a number", 42],
    ["an object", { text: "nope" }],
    ["an array", ["a", "b"]],
    ["a boolean", true],
  ];

  for (const [label, value] of CASES) {
    mockModel(nonStreamingModel(value));
    const id = freshConversation();

    const calls = [];
    conversations.append = (cid, message) => {
      calls.push({ id: cid, message });
      return original.call(conversations, cid, message);
    };

    const messages = await exchange(server.wsUrl, { type: "chat", content: "hi", conversationId: id });

    const error = messages.find((m) => m.type === "error");
    assert.equal(error, undefined, `${label}: must not produce an error frame`);

    const stored = assistantAppends(calls);
    assert.equal(stored.length, 1, `${label}: assistant turn recorded once`);
    assert.equal(
      typeof stored[0].content,
      "string",
      `${label}: append() received ${JSON.stringify(stored[0].content)}`
    );

    const contents = conversations.messages(id).map((m) => m.content);
    assert.ok(
      contents.every((c) => typeof c === "string"),
      `${label}: stored ${JSON.stringify(contents)}`
    );
  }
});

// ---------------------------------------------------------------------------
// The streamedFromWorker path gets the same protection
// ---------------------------------------------------------------------------

test("streamedFromWorker with non-string content keeps the streamed text and stores a string", async (t) => {
  mockModel(streamingModel(undefined, ["Hello", " world"]));
  const server = await startServer();
  t.after(() => server.stop());
  const calls = spyAppend(t);
  const id = freshConversation();

  const messages = await exchange(server.wsUrl, { type: "chat", content: "hi", conversationId: id });

  assert.equal(messages.find((m) => m.type === "error"), undefined, "must not error");

  const streamed = messages.filter((m) => m.type === "stream");
  const final = streamed[streamed.length - 1];
  assert.equal(final.done, true);
  assert.equal(final.content, "Hello world", "the terminal frame keeps what was streamed");

  const stored = assistantAppends(calls);
  assert.equal(stored.length, 1);
  assert.equal(typeof stored[0].content, "string", "append() must receive a string");
  assert.equal(
    stored[0].content,
    "Hello world",
    "the stored reply must be the text the client actually received, not a malformed value"
  );

  const contents = conversations.messages(id).map((m) => m.content);
  assert.deepEqual(contents, ["hi", "Hello world"]);
});

test("valid streamedFromWorker behaviour is unchanged", async (t) => {
  mockModel(streamingModel("Hello world", ["Hello", " world"]));
  const server = await startServer();
  t.after(() => server.stop());
  const calls = spyAppend(t);
  const id = freshConversation();

  const messages = await exchange(server.wsUrl, { type: "chat", content: "hi", conversationId: id });

  assert.equal(messages.find((m) => m.type === "error"), undefined, "must not error");

  const streamed = messages.filter((m) => m.type === "stream");
  assert.ok(streamed.length >= 2, "incremental frames plus a terminal frame");
  // Incremental frames accumulate, exactly as before.
  assert.equal(streamed[0].content, "Hello");
  assert.equal(streamed[0].done, false);
  const final = streamed[streamed.length - 1];
  assert.equal(final.done, true);
  assert.equal(final.content, "Hello world");
  assert.deepEqual(final.metadata, { model: "test-model" });

  const stored = assistantAppends(calls);
  assert.equal(stored.length, 1);
  assert.equal(stored[0].content, "Hello world", "a valid reply is stored verbatim");
});
