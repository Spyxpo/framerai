/**
 * Cognition trace forwarding tests (Issue #164).
 *
 * Verify that result.trace from the Python worker is correctly forwarded through
 * both REST and WebSocket paths, respecting operator context privacy.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");

const { mockModel, loadApp, newConversation, startServer } = require("./helpers");

// Cache paths for clearing between tests
const APP_PATH = require.resolve("../src/app");
const MODEL_PATH = require.resolve("../src/services/model");
const CHAT_PATH = require.resolve("../src/routes/chat");
const WEBSOCKET_PATH = require.resolve("../src/services/websocket");

function clearCache() {
  delete require.cache[APP_PATH];
  delete require.cache[MODEL_PATH];
  delete require.cache[CHAT_PATH];
  delete require.cache[WEBSOCKET_PATH];
}

// ------------------------------------------------------------------
// Group 1: REST trace forwarding with operator context
// ------------------------------------------------------------------

test("REST: trace forwarded when operator context is set", async () => {
  clearCache();
  const calls = mockModel({
    processMessage: (messages, type, settings, requestId, operatorCtx) => ({
      type,
      content: "reply with trace",
      metadata: {
        model: "framerai-text",
        trace: operatorCtx?.operator === true ? {
          memories: [
            { text: "previous thing", score: 0.82 },
            { text: "another memory", score: 0.71 },
          ],
          affect: [0.2, -0.1, 0.5],
          affect_adj: 0.05,
          sampling: { temperature: 0.75, top_k: 50 },
        } : undefined,
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .set("x-operator", "true")
    .send({ content: "hello" });

  assert.equal(res.status, 200);
  assert.equal(res.body.content, "reply with trace");

  // Trace should be present
  assert.ok(res.body.metadata.trace, "trace should be forwarded");
  assert.equal(res.body.metadata.trace.memories.length, 2);
  assert.equal(res.body.metadata.trace.memories[0].text, "previous thing");
  assert.equal(res.body.metadata.trace.memories[0].score, 0.82);
  assert.deepEqual(res.body.metadata.trace.affect, [0.2, -0.1, 0.5]);
  assert.equal(res.body.metadata.trace.affect_adj, 0.05);
  assert.equal(res.body.metadata.trace.sampling.temperature, 0.75);
});

test("REST: trace omitted when operator context is not set", async () => {
  clearCache();
  const calls = mockModel({
    processMessage: () => ({
      type: "text",
      content: "reply without trace",
      metadata: {
        model: "framerai-text",
        // Simulate Python always returning trace (backend should strip it)
        trace: {
          memories: [{ text: "secret memory", score: 0.9 }],
          affect: [0.1, 0.2],
        },
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    // No x-operator header
    .send({ content: "hello" });

  assert.equal(res.status, 200);
  assert.equal(res.body.content, "reply without trace");

  // Trace should NOT be present
  assert.ok(!res.body.metadata.trace, "trace should be omitted for non-operator");
});

test("REST: other metadata preserved when trace omitted", async () => {
  clearCache();
  const calls = mockModel({
    processMessage: () => ({
      type: "text",
      content: "reply with tools",
      metadata: {
        model: "framerai-text",
        tools: [{ name: "web_search", input: {query: "test"}, output: {results: []} }],
        trace: { memories: [{ text: "secret", score: 0.9 }] },
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "hello" });

  assert.equal(res.status, 200);

  // Other metadata should be preserved
  assert.equal(res.body.metadata.model, "framerai-text");
  assert.ok(Array.isArray(res.body.metadata.tools), "tools should be preserved");
  assert.equal(res.body.metadata.tools.length, 1);

  // But trace should be omitted
  assert.ok(!res.body.metadata.trace, "trace should be omitted");
});

test("REST: validates and cleans trace structure", async () => {
  clearCache();
  const calls = mockModel({
    processMessage: (messages, type, settings, requestId, operatorCtx) => ({
      type: "text",
      content: "reply",
      metadata: {
        model: "framerai-text",
        trace: operatorCtx?.operator === true ? {
          memories: [
            { text: "memory", score: 0.5 },
            { text: "incomplete" }, // missing score
          ],
          affect: [0.1, "invalid", 0.3], // mixed types
          affect_adj: "0.05", // string instead of number
          sampling: { temperature: 0.7, invalid: "field" },
          extra_field: "should be ignored",
        } : undefined,
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .set("x-operator", "true")
    .send({ content: "hello" });

  assert.equal(res.status, 200);
  const trace = res.body.metadata.trace;

  assert.ok(trace, "trace should be present");

  // Memories should be cleaned
  assert.equal(trace.memories.length, 2);
  assert.equal(trace.memories[0].text, "memory");
  assert.equal(trace.memories[0].score, 0.5);
  assert.equal(trace.memories[1].text, "incomplete");
  assert.equal(trace.memories[1].score, 0); // default value

  // Affect should be numbers
  assert.ok(Array.isArray(trace.affect));
  assert.equal(typeof trace.affect[0], "number");

  // affect_adj should be number
  assert.equal(typeof trace.affect_adj, "number");
  assert.equal(trace.affect_adj, 0.05);

  // Sampling values should be numbers
  assert.equal(typeof trace.sampling.temperature, "number");

  // Extra fields should not appear
  assert.ok(!trace.extra_field);
});

test("REST: rejects completely invalid trace", async () => {
  clearCache();
  const calls = mockModel({
    processMessage: (messages, type, settings, requestId, operatorCtx) => ({
      type: "text",
      content: "reply",
      metadata: {
        model: "framerai-text",
        trace: operatorCtx?.operator === true ? "not an object" : undefined,
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .set("x-operator", "true")
    .send({ content: "hello" });

  assert.equal(res.status, 200);
  // Invalid trace should be omitted entirely
  assert.ok(!res.body.metadata.trace);
});

test("REST: omits empty trace", async () => {
  clearCache();
  const calls = mockModel({
    processMessage: (messages, type, settings, requestId, operatorCtx) => ({
      type: "text",
      content: "reply",
      metadata: {
        model: "framerai-text",
        trace: operatorCtx?.operator === true ? {} : undefined,
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .set("x-operator", "true")
    .send({ content: "hello" });

  assert.equal(res.status, 200);
  // Empty trace should be omitted
  assert.ok(!res.body.metadata.trace);
});

// ------------------------------------------------------------------
// WebSocket trace forwarding
// ------------------------------------------------------------------

test("WebSocket: trace in final chunk when operator context set", async (t) => {
  clearCache();
  const calls = mockModel({
    processMessage: (messages, type, settings, requestId, operatorCtx) => ({
      type,
      content: "streamed reply",
      metadata: {
        model: "framerai-text",
        trace: operatorCtx?.operator === true ? {
          memories: [{ text: "websocket memory", score: 0.88 }],
          affect: [0.3, -0.2],
          sampling: { temperature: 0.8 },
        } : undefined,
      },
    }),
  });

  const server = await startServer();
  t.after(() => server.stop());

  const ws = new WebSocket(server.wsUrl, {
    headers: { "x-operator": "true" },
  });

  const messages = await new Promise((resolve, reject) => {
    const received = [];
    ws.on("error", reject);
    ws.on("open", () => {
      ws.send(JSON.stringify({
        type: "chat",
        content: "test trace",
        conversationId: "test-conv",
      }));
    });
    ws.on("message", (data) => {
      const msg = JSON.parse(data);
      received.push(msg);
      if (msg.type === "stream" && msg.done) {
        ws.close();
        resolve(received);
      }
    });
  });

  const streams = messages.filter((m) => m.type === "stream");
  assert.ok(streams.length > 0, "should have stream messages");

  const final = streams[streams.length - 1];
  assert.equal(final.done, true);
  assert.ok(final.metadata, "final chunk should have metadata");
  assert.ok(final.metadata.trace, "trace should be forwarded");
  assert.equal(final.metadata.trace.memories[0].text, "websocket memory");
  assert.equal(final.metadata.trace.memories[0].score, 0.88);
});

test("WebSocket: trace omitted when operator context not set", async (t) => {
  clearCache();
  const calls = mockModel({
    processMessage: () => ({
      type: "text",
      content: "no trace reply",
      metadata: {
        model: "framerai-text",
        trace: { memories: [{ text: "should not appear", score: 0.9 }] },
      },
    }),
  });

  const server = await startServer();
  t.after(() => server.stop());

  const ws = new WebSocket(server.wsUrl);
  // No operator header

  const messages = await new Promise((resolve, reject) => {
    const received = [];
    ws.on("error", reject);
    ws.on("open", () => {
      ws.send(JSON.stringify({
        type: "chat",
        content: "test",
        conversationId: "test-conv",
      }));
    });
    ws.on("message", (data) => {
      const msg = JSON.parse(data);
      received.push(msg);
      if (msg.type === "stream" && msg.done) {
        ws.close();
        resolve(received);
      }
    });
  });

  const streams = messages.filter((m) => m.type === "stream");
  const final = streams[streams.length - 1];

  assert.ok(final.metadata, "metadata should exist");
  assert.ok(!final.metadata.trace, "trace should be omitted without operator context");
});
