/**
 * Regression tests for Issue #504:
 * WebSocket Generates Responses for Missing Conversations.
 *
 * When a generation request is sent over WebSocket with a conversationId that
 * does not exist in the server-side conversation store (e.g. after backend
 * restart, eviction, or deletion), the server must reject the request with a
 * structured error { type: "error", code: "NOT_FOUND", conversationId, message: "Conversation not found" }
 * rather than invoking the model or generating a response.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");
const { randomUUID } = require("node:crypto");

const { mockModel, startServer } = require("./helpers");
const conversations = require("../src/conversationStore");

const modelCalls = mockModel();

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

test("Issue #504: chat request for a missing conversation returns a structured NOT_FOUND error", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const missingId = randomUUID();
  assert.equal(conversations.has(missingId), false, "conversation must not exist in store");

  const callsBefore = modelCalls.length;
  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "hello in missing conversation",
    conversationId: missingId,
  });

  const error = messages.find((m) => m.type === "error");
  assert.ok(error, "server must return an error frame for a missing conversation");
  assert.equal(error.code, "NOT_FOUND", "error frame must specify code: NOT_FOUND");
  assert.equal(error.conversationId, missingId, "error frame must reference the missing conversationId");
  assert.equal(error.message, "Conversation not found", "error frame must contain clear message");

  // Confirm no ack or stream frames were sent
  const ack = messages.find((m) => m.type === "ack");
  const stream = messages.find((m) => m.type === "stream");
  assert.equal(ack, undefined, "server must not send an ack frame for a missing conversation");
  assert.equal(stream, undefined, "server must not stream a response for a missing conversation");

  // Confirm model was never invoked
  assert.equal(modelCalls.length, callsBefore, "model must not be invoked for a missing conversation");
});

test("Issue #504: chat request for an existing conversation continues to work normally", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const existingId = randomUUID();
  conversations.create({
    id: existingId,
    title: "Existing Conversation",
    messages: [],
    createdAt: new Date().toISOString(),
  });
  t.after(() => conversations.remove(existingId));

  const callsBefore = modelCalls.length;
  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "hello in existing conversation",
    conversationId: existingId,
  });

  const ack = messages.find((m) => m.type === "ack");
  assert.ok(ack, "server must send an ack for existing conversation");
  assert.equal(ack.conversationId, existingId);

  const finalStream = messages.find((m) => m.type === "stream" && m.done);
  assert.ok(finalStream, "server must stream a response to completion for existing conversation");
  assert.equal(finalStream.conversationId, existingId);
  assert.ok(modelCalls.length > callsBefore, "model must be invoked for existing conversation");
});
