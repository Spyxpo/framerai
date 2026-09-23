/**
 * Regression tests for conversationStore growth bounds.
 *
 * Root cause: the store is a bare Map with no cap and no TTL, so conversations
 * accumulate indefinitely. Under the 300 req/min rate limit a server left
 * running for hours fills memory without bound.
 *
 * Fix: LRU eviction when the cap is reached, and TTL sweep on each create().
 *
 * These tests verify that:
 *  - the cap is enforced (size never exceeds MAX),
 *  - the LEAST-recently-accessed entry is evicted (not the newest),
 *  - conversations that are being actively appended to survive eviction,
 *  - TTL-expired entries are swept on the next create(),
 *  - explicit DELETE and clear() still work,
 *  - all existing store operations are unaffected.
 */

const test = require("node:test");
const assert = require("node:assert/strict");

const store = require("../src/conversationStore");

function makeConv(id) {
  return { id, title: `Conv ${id}`, messages: [], createdAt: new Date().toISOString() };
}

function setUp(opts = {}) {
  store.clear();
  // _resetLimits is only present after the fix; tests that call this will
  // TypeError on unfixed code, which is the intended "fails before fix" signal.
  store._resetLimits(opts);
}

function tearDown() {
  store.clear();
  store._resetLimits(); // restore production defaults
}

// ---------------------------------------------------------------------------
// Cap enforcement
// ---------------------------------------------------------------------------

test("store size never exceeds the cap", () => {
  setUp({ max: 3, ttl: Infinity });

  for (let i = 0; i < 5; i++) {
    store.create(makeConv(`conv-${i}`));
  }

  assert.ok(
    store._map.size <= 3,
    `Expected at most 3 conversations, got ${store._map.size}. ` +
      "The store has no cap — fix: implement LRU eviction on create()."
  );

  tearDown();
});

test("the cap leaves exactly MAX conversations, not MAX-1", () => {
  setUp({ max: 4, ttl: Infinity });

  for (let i = 0; i < 6; i++) {
    store.create(makeConv(`conv-exact-${i}`));
  }

  // Must be exactly 4, not fewer (over-eviction) or more (under-eviction).
  assert.equal(
    store._map.size,
    4,
    `Expected exactly 4 conversations, got ${store._map.size}`
  );

  tearDown();
});

// ---------------------------------------------------------------------------
// LRU ordering
// ---------------------------------------------------------------------------

test("the LEAST-recently-accessed conversation is evicted, not the newest", () => {
  setUp({ max: 3, ttl: Infinity });

  store.create(makeConv("a")); // created first
  store.create(makeConv("b"));
  store.create(makeConv("c")); // created last

  // Touch "a" so it is no longer the LRU — "b" becomes LRU.
  store.get("a");

  // Creating a 4th should evict "b" (LRU), not "a" or "c".
  store.create(makeConv("d"));

  assert.ok(store._map.size <= 3, `Cap not enforced: ${store._map.size}`);
  assert.notEqual(store.get("a"), null, '"a" should survive — it was accessed after "b"');
  assert.notEqual(store.get("c"), null, '"c" should survive — it was created after "b"');
  assert.notEqual(store.get("d"), null, '"d" should be present as the newest entry');
  assert.equal(store.get("b"), null, '"b" should have been evicted (it was the LRU)');

  tearDown();
});

test("append() counts as access: an active conversation is not evicted before idle ones", () => {
  setUp({ max: 2, ttl: Infinity });

  store.create(makeConv("idle"));
  store.create(makeConv("active"));

  // Simulate a streaming turn: the WebSocket path calls append().
  store.append("active", { role: "assistant", content: "…streaming…" });

  // Creating a third should evict "idle" (LRU), not "active".
  store.create(makeConv("third"));

  assert.equal(store.get("idle"), null, '"idle" should be evicted');
  assert.notEqual(store.get("active"), null, '"active" should survive — append() counts as access');
  assert.notEqual(store.get("third"), null, '"third" should be present');

  tearDown();
});

test("messages() counts as access: a conversation read by the WS path is not evicted", () => {
  setUp({ max: 2, ttl: Infinity });

  store.create(makeConv("cold"));
  store.create(makeConv("warm"));

  // WebSocket path calls messages() to build history.
  store.messages("warm");

  store.create(makeConv("incoming"));

  assert.equal(store.get("cold"), null, '"cold" should be evicted (not accessed)');
  assert.notEqual(store.get("warm"), null, '"warm" should survive — messages() counts as access');

  tearDown();
});

// ---------------------------------------------------------------------------
// TTL sweep
// ---------------------------------------------------------------------------

test("TTL-expired conversations are swept on the next create()", async () => {
  setUp({ max: 1000, ttl: 20 }); // 20 ms TTL

  store.create(makeConv("expiring-a"));
  store.create(makeConv("expiring-b"));
  assert.equal(store._map.size, 2, "Both should exist before TTL");

  // Wait for TTL to lapse.
  await new Promise((resolve) => setTimeout(resolve, 40));

  // A new create() triggers the sweep.
  store.create(makeConv("trigger"));

  assert.equal(store.get("expiring-a"), null, '"expiring-a" should be swept after TTL');
  assert.equal(store.get("expiring-b"), null, '"expiring-b" should be swept after TTL');
  assert.notEqual(store.get("trigger"), null, '"trigger" should be present');

  tearDown();
});

test("a recently-accessed conversation is not swept by TTL", async () => {
  setUp({ max: 1000, ttl: 40 }); // 40 ms TTL

  store.create(makeConv("old-idle"));
  store.create(makeConv("old-active"));

  // Wait so both are near expiry.
  await new Promise((resolve) => setTimeout(resolve, 20));

  // Touch "old-active" to reset its clock.
  store.get("old-active");

  // Wait for "old-idle" to expire (but "old-active" was refreshed).
  await new Promise((resolve) => setTimeout(resolve, 25));

  store.create(makeConv("trigger"));

  assert.equal(store.get("old-idle"), null, '"old-idle" should be swept (TTL lapsed)');
  assert.notEqual(store.get("old-active"), null, '"old-active" should survive (refreshed)');

  tearDown();
});

// ---------------------------------------------------------------------------
// Explicit delete and clear are unaffected
// ---------------------------------------------------------------------------

test("explicit remove() still deletes a conversation immediately", () => {
  setUp({ max: 100, ttl: Infinity });

  store.create(makeConv("to-delete"));
  assert.notEqual(store.get("to-delete"), null);

  store.remove("to-delete");
  assert.equal(store.get("to-delete"), null);

  tearDown();
});

test("clear() wipes all conversations regardless of cap and TTL", () => {
  setUp({ max: 3, ttl: Infinity });

  for (let i = 0; i < 3; i++) store.create(makeConv(`clr-${i}`));
  assert.equal(store._map.size, 3);

  store.clear();
  assert.equal(store._map.size, 0);

  tearDown();
});

// ---------------------------------------------------------------------------
// Existing behaviour is preserved
// ---------------------------------------------------------------------------

test("create / get / append / messages / has / list all work after the fix", () => {
  setUp({ max: 100, ttl: Infinity });

  const conv = store.create(makeConv("smoke"));
  assert.ok(conv);
  assert.notEqual(store.get("smoke"), null);
  assert.ok(store.has("smoke"));

  assert.ok(store.append("smoke", { role: "user", content: "hello" }));
  assert.equal(store.messages("smoke").length, 1);

  const listed = store.list();
  assert.ok(listed.some((c) => c.id === "smoke"));

  assert.equal(store.append("absent", { role: "user", content: "x" }), false);
  assert.deepEqual(store.messages("absent"), []);
  assert.equal(store.get("absent"), null);

  tearDown();
});

// ---------------------------------------------------------------------------
// Per-conversation message cap (issue #338)
// ---------------------------------------------------------------------------

test("messages per conversation never exceed the configured cap", () => {
  setUp({ max: 100, ttl: Infinity, maxMessages: 5 });

  store.create(makeConv("msg-cap"));
  for (let i = 0; i < 8; i++) {
    store.append("msg-cap", { role: "user", content: `message ${i}` });
  }

  const msgs = store.messages("msg-cap");
  assert.equal(
    msgs.length,
    5,
    `Expected at most 5 messages, got ${msgs.length}. ` +
      "The per-conversation message cap is not being enforced."
  );

  tearDown();
});

test("oldest messages are evicted when the cap is hit", () => {
  setUp({ max: 100, ttl: Infinity, maxMessages: 3 });

  store.create(makeConv("evict-order"));
  store.append("evict-order", { role: "user", content: "first" });
  store.append("evict-order", { role: "user", content: "second" });
  store.append("evict-order", { role: "user", content: "third" });
  // Fourth message — "first" should be evicted
  store.append("evict-order", { role: "user", content: "fourth" });

  const msgs = store.messages("evict-order");
  assert.equal(msgs.length, 3, "Should have exactly 3 messages");
  assert.equal(msgs[0].content, "second", "Oldest (first) message must be evicted");
  assert.equal(msgs[1].content, "third");
  assert.equal(msgs[2].content, "fourth", "Newest message must be retained");

  tearDown();
});

test("newest messages are retained in original order after eviction", () => {
  setUp({ max: 100, ttl: Infinity, maxMessages: 4 });

  store.create(makeConv("order-check"));
  const contents = ["a", "b", "c", "d", "e", "f"];
  for (const c of contents) {
    store.append("order-check", { role: "user", content: c });
  }

  const msgs = store.messages("order-check");
  assert.equal(msgs.length, 4);
  // Newest 4 of ["a","b","c","d","e","f"] are ["c","d","e","f"]
  assert.deepEqual(
    msgs.map((m) => m.content),
    ["c", "d", "e", "f"],
    "Messages must be the newest N in original ascending order"
  );

  tearDown();
});

test("conversations below the message cap are unaffected", () => {
  setUp({ max: 100, ttl: Infinity, maxMessages: 10 });

  store.create(makeConv("below-cap"));
  store.append("below-cap", { role: "user", content: "one" });
  store.append("below-cap", { role: "assistant", content: "two" });

  const msgs = store.messages("below-cap");
  assert.equal(msgs.length, 2, "Under-cap conversations must not lose any messages");
  assert.equal(msgs[0].content, "one");
  assert.equal(msgs[1].content, "two");

  tearDown();
});

test("_resetLimits restores default maxMessages", () => {
  setUp({ max: 100, ttl: Infinity, maxMessages: 2 });
  assert.equal(store._maxMessages, 2, "Override should take effect");

  tearDown(); // restores production defaults
  assert.equal(
    store._maxMessages,
    Number(process.env.FRAMER_MAX_MESSAGES_PER_CONVERSATION) || 1000,
    "Default should be restored"
  );
});

// ---------------------------------------------------------------------------
// Bookkeeping stays internal
// ---------------------------------------------------------------------------

test("LRU bookkeeping does not leak into a serialised conversation", () => {
  setUp({ max: 10, ttl: Infinity });

  // GET /chat/conversations/:id responds with the stored object as-is, so
  // anything enumerable on it reaches the client and drifts from the
  // documented OpenAPI Conversation schema.
  const conv = store.create(makeConv("serialised"));
  store.get("serialised");
  store.append("serialised", { role: "user", content: "hi" });

  assert.deepEqual(
    Object.keys(conv),
    ["id", "title", "messages", "createdAt"],
    "internal LRU/TTL fields must not be enumerable"
  );
  assert.deepEqual(Object.keys(JSON.parse(JSON.stringify(conv))), [
    "id",
    "title",
    "messages",
    "createdAt",
  ]);

  // Still readable internally — hiding them must not break eviction.
  assert.equal(typeof conv._lruSeq, "number");
  assert.equal(typeof conv._accessedAtMs, "number");

  tearDown();
});
