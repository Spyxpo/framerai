/**
 * Regression tests for Issue #239: WebSocket rate limiting cannot be bypassed
 * by forging X-Forwarded-For headers.
 *
 * ROOT CAUSE: The old websocket.js resolved the client key as:
 *
 *   const forwarded = config.trustProxy
 *     ? (req?.headers["x-forwarded-for"] || "").split(",")[0].trim()
 *     : "";
 *   const clientKey = forwarded || req?.socket?.remoteAddress || clientId;
 *
 * When trustProxy is truthy (e.g. TRUST_PROXY=1), the leftmost X-Forwarded-For
 * value is used directly as the key. A client can forge a different leftmost IP
 * on each connection to obtain a fresh rate-limit bucket every time, completely
 * bypassing the limit.
 *
 * FIX: resolveClientIp() in rateLimit.js mirrors Express compileTrust semantics
 * exactly. With trustProxy=1 (hop-count), only the proxy directly adjacent to
 * the socket is trusted; the leftmost XFF entries beyond the trusted hop are
 * treated as client-controlled and all map to the same resolved key.
 *
 * The critical regression test (TRUST_PROXY=1) below fails with the old
 * implementation and passes with the fixed one.
 */

// ─── env vars must be set before any require() so config.js picks them up ───
process.env.RATE_LIMIT_WINDOW_MS = "60000";
process.env.RATE_LIMIT_MAX = "1000"; // keep API limiter out of the way
process.env.GENERATE_RATE_LIMIT_MAX = "2";
// TRUST_PROXY=1 activates the vulnerable code path: the old implementation
// used the leftmost XFF entry as the client key, making it forgeable.
process.env.TRUST_PROXY = "1";

const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");

const { mockModel, startServer } = require("./helpers");

mockModel();

function resetGenerationCounter() {
  const { generationCounter } = require("../src/middleware/limiters");
  generationCounter.buckets.clear();
  generationCounter.lastSweep = 0;
}

/**
 * Open a WS connection with the given headers, send one chat frame, and wait
 * for the first error frame or completed stream.
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
// 1. CORE E2E REGRESSION (TRUST_PROXY=1): forging the leftmost X-Forwarded-For
//    entry does NOT produce a fresh rate-limit bucket.
//
//    Test shape:
//      socket address = 127.0.0.1 (the loopback, acting as the trusted proxy)
//      X-Forwarded-For = "<forged-client-ip>, 10.0.0.1"
//        - 10.0.0.1 is the real downstream proxy (the hop adjacent to the socket)
//        - <forged-client-ip> changes across connections to simulate the attack
//
//    resolveClientIp(req, 1) with hop-count=1:
//      address list = [socket=127.0.0.1, XFF right-to-left: 10.0.0.1, forged]
//      trust(127.0.0.1, i=0): 0 < 1 → trusted proxy
//      trust(10.0.0.1,  i=1): 1 < 1 → NOT trusted → stop → resolved key = 10.0.0.1
//    All three connections share the same resolved key (10.0.0.1) regardless of
//    the forged leftmost entry, so the bucket is exhausted and the third is blocked.
//
//    Old implementation:
//      clientKey = xff.split(",")[0].trim() = forged-ip (changes each connection)
//      → each connection gets its own fresh bucket → bypass succeeds → test fails.
// ─────────────────────────────────────────────────────────────────────────────
test("REGRESSION #239 (TRUST_PROXY=1): forging leftmost XFF does not bypass rate limit", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  resetGenerationCounter();

  // All three connections come from socket 127.0.0.1 through a real proxy at
  // 10.0.0.1.  The leftmost entry is different each time — this is the forge.
  const r1 = await wsChat(server.wsUrl, { "x-forwarded-for": "9.9.9.1, 10.0.0.1" });
  const r2 = await wsChat(server.wsUrl, { "x-forwarded-for": "9.9.9.2, 10.0.0.1" });

  assert.equal(r1.rateLimited, false, "first request should be allowed");
  assert.equal(r2.rateLimited, false, "second request should be allowed");

  // Old code: key="9.9.9.3" (new bucket) → allowed → test would fail here.
  // New code: key="10.0.0.1" (same bucket, exhausted) → rate-limited → passes.
  const r3 = await wsChat(server.wsUrl, { "x-forwarded-for": "9.9.9.3, 10.0.0.1" });
  assert.equal(
    r3.rateLimited,
    true,
    "forging the leftmost XFF entry must NOT create a fresh bucket — " +
    "resolved key is always the hop-count-correct address (10.0.0.1)"
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Without a real proxy hop in XFF, socket address is the resolved key
// ─────────────────────────────────────────────────────────────────────────────
test("TRUST_PROXY=1: single-entry XFF still exhausts the same bucket", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  resetGenerationCounter();

  // With hop-count=1: socket(i=0) is trusted, XFF[0](i=1) is not trusted → stop.
  // Result = XFF[0].  Different forged values still produce different keys here,
  // but this test verifies the basic hop-count path completes without error.
  const r1 = await wsChat(server.wsUrl, { "x-forwarded-for": "5.5.5.5" });
  const r2 = await wsChat(server.wsUrl, { "x-forwarded-for": "5.5.5.5" });
  const r3 = await wsChat(server.wsUrl, { "x-forwarded-for": "5.5.5.5" });

  assert.equal(r1.rateLimited, false, "first request should be allowed");
  assert.equal(r2.rateLimited, false, "second request should be allowed");
  assert.equal(r3.rateLimited, true,  "third request from same resolved key is blocked");
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Rate limit error frame has the expected shape
// ─────────────────────────────────────────────────────────────────────────────
test("rate-limited WS frame receives a well-formed error response", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());
  resetGenerationCounter();

  // Exhaust the budget with consistent XFF so the bucket fills
  await wsChat(server.wsUrl, { "x-forwarded-for": "9.9.9.1, 10.0.0.1" });
  await wsChat(server.wsUrl, { "x-forwarded-for": "9.9.9.2, 10.0.0.1" });

  const { messages } = await wsChat(server.wsUrl, { "x-forwarded-for": "9.9.9.3, 10.0.0.1" });
  const errMsg = messages.find((m) => m.type === "error");

  assert.ok(errMsg, "should receive an error frame");
  assert.equal(errMsg.code, "RATE_LIMITED");
  assert.match(errMsg.message, /Too many generation requests/);
  assert.match(errMsg.message, /Try again in \d+s/);
});
