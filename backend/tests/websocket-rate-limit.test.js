/**
 * Regression tests for Issue #239: WebSocket rate limiting cannot be bypassed
 * by forging X-Forwarded-For headers.
 *
 * ROOT CAUSE: The old websocket.js read req.headers["x-forwarded-for"] directly
 * and used the leftmost entry as the client key. A client could send a different
 * XFF value on each connection to obtain a fresh rate-limit bucket, completely
 * bypassing the limit.
 *
 * FIX: resolveClientIp() in rateLimit.js uses proxy-addr with the same trust
 * semantics as Express req.ip, so the real socket address is always used when
 * trust proxy is disabled, and the correct hop-count is applied when enabled.
 */

// Use a very tight generation budget so tests run quickly.
process.env.RATE_LIMIT_WINDOW_MS = "60000";
process.env.RATE_LIMIT_MAX = "1000"; // keep API limit out of the way
process.env.GENERATE_RATE_LIMIT_MAX = "2";

const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");

const { mockModel, startServer } = require("./helpers");

mockModel();

// Reset the shared generation counter before each test so tests don't bleed
// into each other (the counter is a module-level singleton in this process).
function resetGenerationCounter() {
  const { generationCounter } = require("../src/middleware/limiters");
  generationCounter.buckets.clear();
  generationCounter.lastSweep = 0;
}

/**
 * Open a WS connection with optional extra HTTP headers (to simulate a forged
 * XFF), send one chat frame, and wait for the first error or completed stream.
 * Returns { rateLimited, messages }.
 */
function wsChat(wsUrl, extraHeaders = {}) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl, { headers: extraHeaders });
    const messages = [];
    const timer = setTimeout(() => {
      ws.close();
      resolve({ rateLimited: false, messages });
    }, 4000);

    ws.on("error", reject);
    ws.on("open", () =>
      ws.send(JSON.stringify({ type: "chat", content: "hi", conversationId: "c1" }))
    );
    ws.on("message", (data) => {
      const msg = JSON.parse(data);
      messages.push(msg);
      if (msg.type === "error") {
        clearTimeout(timer);
        ws.close();
        resolve({ rateLimited: msg.code === "RATE_LIMITED", messages });
      } else if (msg.type === "stream" && msg.done) {
        clearTimeout(timer);
        ws.close();
        resolve({ rateLimited: false, messages });
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. CORE REGRESSION: changing X-Forwarded-For does NOT reset the bucket
// ─────────────────────────────────────────────────────────────────────────────
test("REGRESSION #239: forging X-Forwarded-For cannot reset the WS rate-limit bucket", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  resetGenerationCounter();

  // Spend the 2-request budget from the real socket address (127.0.0.1)
  const r1 = await wsChat(server.wsUrl);
  const r2 = await wsChat(server.wsUrl);
  assert.equal(r1.rateLimited, false, "first request should be allowed");
  assert.equal(r2.rateLimited, false, "second request should be allowed");

  // Without the fix, sending a forged XFF header would cause the handler to
  // call splitXFF("9.9.9.9")[0] = "9.9.9.9" and get a fresh bucket.
  // With the fix, resolveClientIp() ignores XFF when trust proxy is disabled
  // and returns 127.0.0.1 — the budget is exhausted, this must be blocked.
  const r3 = await wsChat(server.wsUrl, { "x-forwarded-for": "9.9.9.9" });
  assert.equal(
    r3.rateLimited,
    true,
    "forging XFF must NOT bypass the rate limit — real socket address (127.0.0.1) is always used"
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Multiple different forged XFF values all hit the same bucket
// ─────────────────────────────────────────────────────────────────────────────
test("REGRESSION #239: multiple different forged XFF values do not each get a fresh bucket", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  resetGenerationCounter();

  // Spend the budget using two different forged XFF headers.
  // Old code: each different XFF → different key → unlimited fresh buckets.
  // Fixed code: all use socket address → same key → budget shared.
  const r1 = await wsChat(server.wsUrl, { "x-forwarded-for": "1.1.1.1" });
  const r2 = await wsChat(server.wsUrl, { "x-forwarded-for": "2.2.2.2" });
  assert.equal(r1.rateLimited, false, "first request (XFF=1.1.1.1) should be allowed");
  assert.equal(r2.rateLimited, false, "second request (XFF=2.2.2.2) should be allowed");

  // Third request, yet another forged XFF — should be blocked
  const r3 = await wsChat(server.wsUrl, { "x-forwarded-for": "3.3.3.3" });
  assert.equal(
    r3.rateLimited,
    true,
    "third request with yet another forged XFF must still be blocked (same real socket IP)"
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Without trust proxy, socket address is always used (no XFF)
// ─────────────────────────────────────────────────────────────────────────────
test("with trust proxy disabled, socket address is used as client key", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  resetGenerationCounter();

  // Without XFF: should use socket address and get rate limited on 3rd request
  const r1 = await wsChat(server.wsUrl);
  const r2 = await wsChat(server.wsUrl);
  const r3 = await wsChat(server.wsUrl);

  assert.equal(r1.rateLimited, false, "first request should be allowed");
  assert.equal(r2.rateLimited, false, "second request should be allowed");
  assert.equal(r3.rateLimited, true, "third request from same client should be blocked");
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Rate limit error frame has the expected shape
// ─────────────────────────────────────────────────────────────────────────────
test("rate-limited WS frame receives a well-formed error response", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  resetGenerationCounter();

  // Exhaust the budget
  await wsChat(server.wsUrl);
  await wsChat(server.wsUrl);

  const { messages } = await wsChat(server.wsUrl);
  const errMsg = messages.find((m) => m.type === "error");

  assert.ok(errMsg, "should receive an error frame");
  assert.equal(errMsg.code, "RATE_LIMITED");
  assert.match(errMsg.message, /Too many generation requests/);
  assert.match(errMsg.message, /Try again in \d+s/);
});
