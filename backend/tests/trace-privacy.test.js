/**
 * Cognition trace privacy tests (Issue #164).
 *
 * Verify that traces containing recalled memories (which may include previous
 * user content) are only exposed when the request context permits it.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");

const { mockModel, loadApp, newConversation } = require("./helpers");

// Cache paths for clearing between tests
const APP_PATH = require.resolve("../src/app");
const MODEL_PATH = require.resolve("../src/services/model");
const CHAT_PATH = require.resolve("../src/routes/chat");
const CONFIG_PATH = require.resolve("../src/config");

// The operator header is client-settable, so it is only honoured behind a
// trusted proxy. These tests run with TRUST_PROXY on; the untrusted case is
// covered separately below.
process.env.TRUST_PROXY = "true";

function clearCache() {
  delete require.cache[CONFIG_PATH];
  delete require.cache[APP_PATH];
  delete require.cache[MODEL_PATH];
  delete require.cache[CHAT_PATH];
}

test("trace with sensitive memories redacted for regular users", async () => {
  clearCache();
  const calls = mockModel({
    processMessage: () => ({
      type: "text",
      content: "reply",
      metadata: {
        model: "framerai-text",
        trace: {
          memories: [
            { text: "User said: my password is hunter2", score: 0.9 },
            { text: "User's credit card: 4111-1111-1111-1111", score: 0.85 },
          ],
          affect: [0.2, 0.1],
        },
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  // Request without operator header
  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "hello" });

  assert.equal(res.status, 200);

  // Trace should be completely omitted (privacy requirement)
  assert.ok(!res.body.metadata.trace, "trace with sensitive memories must be omitted");
});

test("trace exposed when operator context is true", async () => {
  clearCache();
  const calls = mockModel({
    processMessage: (messages, type, settings, requestId, operatorCtx) => ({
      type: "text",
      content: "reply",
      metadata: {
        model: "framerai-text",
        trace: operatorCtx?.operator === true ? {
          memories: [{ text: "sensitive memory", score: 0.9 }],
          affect: [0.1],
        } : undefined,
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  // Request WITH operator header
  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .set("x-operator", "true")
    .send({ content: "hello" });

  assert.equal(res.status, 200);

  // Trace should be present for operator
  assert.ok(res.body.metadata.trace, "trace should be exposed to operator");
  assert.ok(Array.isArray(res.body.metadata.trace.memories));
});

test("trace exposed when INCLUDE_TRACE env var is set", async () => {
  clearCache();
  const originalEnv = process.env.INCLUDE_TRACE;
  process.env.INCLUDE_TRACE = "true";

  try {
    const calls = mockModel({
      processMessage: () => ({
        type: "text",
        content: "reply",
        metadata: {
          model: "framerai-text",
          trace: {
            memories: [{ text: "memory", score: 0.8 }],
            affect: [0.2],
          },
        },
      }),
    });
    const app = loadApp();
    const id = await newConversation(app);

    // Request without operator header, but env var is set
    const res = await request(app)
      .post(`/api/chat/conversations/${id}/messages`)
      .send({ content: "hello" });

    assert.equal(res.status, 200);

    // Trace should be present due to env var
    assert.ok(res.body.metadata.trace, "trace should be exposed when INCLUDE_TRACE=true");
  } finally {
    // Restore original env
    if (originalEnv === undefined) {
      delete process.env.INCLUDE_TRACE;
    } else {
      process.env.INCLUDE_TRACE = originalEnv;
    }
  }
});

test("non-operator header values do not enable trace", async () => {
  clearCache();
  const calls = mockModel({
    processMessage: () => ({
      type: "text",
      content: "reply",
      metadata: {
        model: "framerai-text",
        trace: { memories: [{ text: "secret", score: 0.9 }] },
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  // Try various non-true values
  for (const value of ["false", "True", "1", "yes", "operator"]) {
    const res = await request(app)
      .post(`/api/chat/conversations/${id}/messages`)
      .set("x-operator", value)
      .send({ content: "hello" });

    assert.equal(res.status, 200);
    assert.ok(!res.body.metadata.trace, `trace should be omitted for x-operator: ${value}`);
  }
});

test("backend enforces trace privacy at boundary", async () => {
  clearCache();
  // This tests that the backend enforces privacy even if processMessage
  // mistakenly includes trace when it shouldn't
  const calls = mockModel({
    processMessage: () => ({
      type: "text",
      content: "reply",
      metadata: {
        model: "framerai-text",
        trace: {
          memories: [{ text: "should not leak", score: 0.95 }],
        },
      },
    }),
  });
  const app = loadApp();
  const id = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    // No operator header
    .send({ content: "hello" });

  assert.equal(res.status, 200);

  // Backend should strip the trace at the boundary
  assert.ok(!res.body.metadata.trace, "backend must enforce trace privacy at boundary");
});

test("operator header is ignored when no proxy is trusted", async (t) => {
  const original = process.env.TRUST_PROXY;
  process.env.TRUST_PROXY = "false";
  t.after(() => {
    if (original === undefined) delete process.env.TRUST_PROXY;
    else process.env.TRUST_PROXY = original;
    clearCache();
  });

  clearCache();
  mockModel({
    processMessage: () => ({
      type: "text",
      content: "reply",
      metadata: {
        model: "framerai-text",
        // The worker sends a trace regardless; the boundary has to strip it.
        trace: { memories: [{ text: "should not leak", score: 0.95 }] },
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
  assert.ok(
    !res.body.metadata.trace,
    "a client that sets the operator header for itself must not unlock the trace"
  );
});
