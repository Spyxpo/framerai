/**
 * Unit tests for resolveClientIp() in rateLimit.js.
 *
 * These tests are written against the Express reference implementation of
 * compileTrust so that any drift from Express semantics is immediately visible.
 *
 * Express compileTrust source (lib/utils.js):
 *   false / falsy  → proxyaddr.compile([])          → trust nobody
 *   true           → () => true                     → trust all hops
 *   number n       → (addr, i) => i < n             → trust n hops
 *   string/array   → proxyaddr.compile(val)          → named subnets / CIDRs
 *
 * The numeric case is what this suite is specifically targeting because
 * proxyaddr.compile() does not accept numbers and throws, which previously
 * caused the implementation to fall back to req.socket.remoteAddress instead
 * of applying the correct hop-count logic.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const proxyaddr = require("proxy-addr");
const { resolveClientIp } = require("../src/middleware/rateLimit");

// ── helpers ──────────────────────────────────────────────────────────────────

function makeReq(socketAddr, xffHeader) {
  return {
    socket: { remoteAddress: socketAddr },
    headers: xffHeader ? { "x-forwarded-for": xffHeader } : {},
  };
}

/**
 * Reference implementation that exactly mirrors Express compileTrust.
 * Used to generate expected values so the tests are self-documenting.
 */
function expressRef(req, trustProxy) {
  let trust;
  if (trustProxy === true) {
    trust = () => true;
  } else if (typeof trustProxy === "number") {
    trust = (addr, i) => i < trustProxy;
  } else {
    trust = proxyaddr.compile(trustProxy || []);
  }
  return proxyaddr(req, trust);
}

// ── trustProxy = false ───────────────────────────────────────────────────────

test("resolveClientIp: trustProxy=false ignores XFF, returns socket address", () => {
  const req = makeReq("203.0.113.5", "9.9.9.9");
  assert.equal(resolveClientIp(req, false), expressRef(req, false));
  assert.equal(resolveClientIp(req, false), "203.0.113.5");
});

test("resolveClientIp: trustProxy=false, no XFF, returns socket address", () => {
  const req = makeReq("203.0.113.5", null);
  assert.equal(resolveClientIp(req, false), expressRef(req, false));
  assert.equal(resolveClientIp(req, false), "203.0.113.5");
});

test("resolveClientIp: trustProxy=false, loopback socket, forged XFF ignored", () => {
  const req = makeReq("127.0.0.1", "9.9.9.9");
  assert.equal(resolveClientIp(req, false), expressRef(req, false));
  assert.equal(resolveClientIp(req, false), "127.0.0.1");
});

// ── trustProxy = true ────────────────────────────────────────────────────────

test("resolveClientIp: trustProxy=true, single XFF entry, returns XFF address", () => {
  const req = makeReq("127.0.0.1", "9.9.9.9");
  assert.equal(resolveClientIp(req, true), expressRef(req, true));
  assert.equal(resolveClientIp(req, true), "9.9.9.9");
});

test("resolveClientIp: trustProxy=true, multiple XFF entries, returns leftmost", () => {
  const req = makeReq("127.0.0.1", "9.9.9.9, 10.0.0.1");
  assert.equal(resolveClientIp(req, true), expressRef(req, true));
  assert.equal(resolveClientIp(req, true), "9.9.9.9");
});

test("resolveClientIp: trustProxy=true, no XFF, returns socket address", () => {
  const req = makeReq("203.0.113.5", null);
  assert.equal(resolveClientIp(req, true), expressRef(req, true));
  assert.equal(resolveClientIp(req, true), "203.0.113.5");
});

// ── trustProxy = 1 (numeric hop-count) ───────────────────────────────────────
//
// Express semantics: trust addresses at index i < 1, i.e. only the socket.
// Address list built by proxy-addr: [socket, ...XFF right-to-left]
//   socket=127.0.0.1, XFF="9.9.9.9, 10.0.0.1"
//   → addrs = [127.0.0.1, 10.0.0.1, 9.9.9.9]
//   trust(127.0.0.1, 0): 0 < 1 → trusted proxy
//   trust(10.0.0.1,  1): 1 < 1 → NOT trusted → stop; result = 10.0.0.1
//
//   socket=127.0.0.1, XFF="9.9.9.9"
//   → addrs = [127.0.0.1, 9.9.9.9]
//   trust(127.0.0.1, 0): 0 < 1 → trusted proxy
//   trust(9.9.9.9,   1): 1 < 1 → NOT trusted → stop; result = 9.9.9.9

test("resolveClientIp: trustProxy=1, single XFF, peels one hop (matches Express)", () => {
  const req = makeReq("127.0.0.1", "9.9.9.9");
  const expected = expressRef(req, 1); // 9.9.9.9
  assert.equal(resolveClientIp(req, 1), expected);
  assert.equal(resolveClientIp(req, 1), "9.9.9.9");
});

test("resolveClientIp: trustProxy=1, two XFF entries, peels one hop (matches Express)", () => {
  const req = makeReq("127.0.0.1", "9.9.9.9, 10.0.0.1");
  const expected = expressRef(req, 1); // 10.0.0.1
  assert.equal(resolveClientIp(req, 1), expected);
  assert.equal(resolveClientIp(req, 1), "10.0.0.1");
});

test("resolveClientIp: trustProxy=1, no XFF, returns socket address (matches Express)", () => {
  const req = makeReq("127.0.0.1", null);
  const expected = expressRef(req, 1); // 127.0.0.1
  assert.equal(resolveClientIp(req, 1), expected);
  assert.equal(resolveClientIp(req, 1), "127.0.0.1");
});

// ── trustProxy = 2 (numeric hop-count) ───────────────────────────────────────
//
//   socket=127.0.0.1, XFF="9.9.9.9, 10.0.0.1"
//   → addrs = [127.0.0.1, 10.0.0.1, 9.9.9.9]
//   trust(127.0.0.1, 0): 0 < 2 → trusted
//   trust(10.0.0.1,  1): 1 < 2 → trusted
//   trust(9.9.9.9,   2): 2 < 2 → NOT trusted → stop; result = 9.9.9.9

test("resolveClientIp: trustProxy=2, two XFF entries, peels two hops (matches Express)", () => {
  const req = makeReq("127.0.0.1", "9.9.9.9, 10.0.0.1");
  const expected = expressRef(req, 2); // 9.9.9.9
  assert.equal(resolveClientIp(req, 2), expected);
  assert.equal(resolveClientIp(req, 2), "9.9.9.9");
});

test("resolveClientIp: trustProxy=2, one XFF entry, peels available hops (matches Express)", () => {
  const req = makeReq("127.0.0.1", "9.9.9.9");
  const expected = expressRef(req, 2); // 9.9.9.9
  assert.equal(resolveClientIp(req, 2), expected);
  assert.equal(resolveClientIp(req, 2), "9.9.9.9");
});

// ── direct socket, no proxy ───────────────────────────────────────────────────

test("resolveClientIp: direct connection, no XFF, any trustProxy value returns socket", () => {
  const req = makeReq("203.0.113.5", null);
  assert.equal(resolveClientIp(req, false), "203.0.113.5");
  assert.equal(resolveClientIp(req, true),  "203.0.113.5");
  assert.equal(resolveClientIp(req, 1),     "203.0.113.5");
});

// ── exhaustive match against Express reference ────────────────────────────────

test("resolveClientIp: all cases match Express reference implementation exactly", () => {
  const cases = [
    { tp: false, socket: "127.0.0.1",   xff: null },
    { tp: false, socket: "127.0.0.1",   xff: "9.9.9.9" },
    { tp: false, socket: "203.0.113.5", xff: "9.9.9.9" },
    { tp: true,  socket: "127.0.0.1",   xff: "9.9.9.9" },
    { tp: true,  socket: "127.0.0.1",   xff: "9.9.9.9, 10.0.0.1" },
    { tp: true,  socket: "203.0.113.5", xff: null },
    { tp: 1,     socket: "127.0.0.1",   xff: "9.9.9.9" },
    { tp: 1,     socket: "127.0.0.1",   xff: "9.9.9.9, 10.0.0.1" },
    { tp: 1,     socket: "127.0.0.1",   xff: null },
    { tp: 2,     socket: "127.0.0.1",   xff: "9.9.9.9, 10.0.0.1" },
    { tp: 2,     socket: "127.0.0.1",   xff: "9.9.9.9" },
    { tp: 2,     socket: "127.0.0.1",   xff: null },
    { tp: false, socket: "203.0.113.5", xff: null },
    { tp: 1,     socket: "203.0.113.5", xff: null },
  ];

  for (const c of cases) {
    const req = makeReq(c.socket, c.xff);
    const got      = resolveClientIp(req, c.tp);
    const expected = expressRef(req, c.tp);
    assert.equal(
      got,
      expected,
      `trustProxy=${JSON.stringify(c.tp)} socket=${c.socket} xff=${c.xff} → got ${got}, want ${expected}`
    );
  }
});
