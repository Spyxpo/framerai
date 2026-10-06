/**
 * Regression tests for Issue #425: the WebSocket handshake is held to the same
 * origin allowlist as CORS.
 *
 * ROOT CAUSE: CORS_ORIGIN only decides what a browser may read from an HTTP
 * response. A browser never applies CORS to a WebSocket, and createServer()
 * built the WebSocketServer with no handshake check, so a web page on ANY origin
 * could open /ws on a locally running backend and drive generation through it.
 *
 * FIX: the handshake is checked against the origin CORS is configured with, read
 * from the one place both use.
 *
 * What "the same allowlist" means is what CORS already means for that setting:
 *   - one origin, compared exactly, the way a browser compares the origin it
 *     sends with Access-Control-Allow-Origin. No case folding, no trimming of a
 *     slash or a path, no prefix or substring match: a spelling that is not
 *     byte for byte the configured origin is a different origin.
 *   - "*" means every origin.
 *   - A handshake with no Origin header is not a browser (a browser always sends
 *     one on a WebSocket handshake) and is the same non-browser client CORS
 *     already lets through, so it is accepted. A header that is present but
 *     empty is not "absent" and is rejected.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const net = require("node:net");
const request = require("supertest");
const WebSocket = require("ws");

const { mockModel, startServer } = require("./helpers");

mockModel();

const ALLOWED = "https://app.example.test:8443";

/**
 * Point CORS_ORIGIN at `value` (or unset it) for the length of one test. Both CORS
 * and the handshake read it, so every test builds its server after calling this.
 */
function useCorsOrigin(t, value) {
  const saved = process.env.CORS_ORIGIN;
  if (value === undefined) delete process.env.CORS_ORIGIN;
  else process.env.CORS_ORIGIN = value;
  t.after(() => {
    if (saved === undefined) delete process.env.CORS_ORIGIN;
    else process.env.CORS_ORIGIN = saved;
  });
}

/**
 * Attempt a WebSocket handshake with the given headers. Resolves { accepted: true, ws }
 * when the upgrade succeeds, or { accepted: false, status, body } when the server
 * answers the handshake with an HTTP error instead.
 */
function handshake(wsUrl, headers = {}) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl, { headers });
    const timer = setTimeout(() => {
      ws.terminate();
      reject(new Error("the handshake was neither accepted nor rejected within 4s"));
    }, 4000);

    ws.on("open", () => {
      clearTimeout(timer);
      resolve({ accepted: true, ws });
    });
    ws.on("unexpected-response", (req, res) => {
      clearTimeout(timer);
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => (body += chunk));
      res.on("end", () => resolve({ accepted: false, status: res.statusCode, body }));
    });
    ws.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

/**
 * The same handshake written by hand over a plain socket, so a header can be sent
 * more than once. `originValues` become one Origin header line each.
 */
function rawHandshake(server, originValues) {
  const { port } = server.server.address();
  const upgradeRequest = [
    "GET /ws HTTP/1.1",
    `Host: 127.0.0.1:${port}`,
    "Connection: Upgrade",
    "Upgrade: websocket",
    "Sec-WebSocket-Version: 13",
    `Sec-WebSocket-Key: ${crypto.randomBytes(16).toString("base64")}`,
    ...originValues.map((value) => `Origin: ${value}`),
    "",
    "",
  ].join("\r\n");

  return new Promise((resolve, reject) => {
    const socket = net.connect(port, "127.0.0.1", () => socket.write(upgradeRequest));
    const timer = setTimeout(() => {
      socket.destroy();
      reject(new Error("the raw handshake timed out"));
    }, 4000);

    let received = "";
    socket.on("data", (chunk) => {
      received += chunk;
      const lineEnd = received.indexOf("\r\n");
      if (lineEnd === -1) return;
      clearTimeout(timer);
      socket.destroy();
      const status = Number(received.slice(0, lineEnd).split(" ")[1]);
      resolve({ accepted: status === 101, status });
    });
    socket.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

function ping(ws) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("no pong")), 4000);
    ws.once("message", (data) => {
      clearTimeout(timer);
      resolve(JSON.parse(data));
    });
    ws.send(JSON.stringify({ type: "ping" }));
  });
}

function chat(ws, content) {
  return new Promise((resolve, reject) => {
    const frames = [];
    const timer = setTimeout(() => reject(new Error("no completed reply")), 4000);
    ws.on("message", (data) => {
      const frame = JSON.parse(data);
      frames.push(frame);
      if (frame.type === "stream" && frame.done) {
        clearTimeout(timer);
        resolve(frames);
      }
    });
    ws.send(JSON.stringify({ type: "chat", content }));
  });
}

/** Run a handshake per origin and return the ones the server let in. */
async function originsLetIn(server, origins) {
  const letIn = [];
  for (const origin of origins) {
    const result = await handshake(server.wsUrl, { Origin: origin });
    if (result.accepted) {
      letIn.push(origin);
      result.ws.terminate();
    } else {
      assert.equal(result.status, 403, `Origin ${JSON.stringify(origin)} should be refused with 403, got ${result.status}`);
    }
  }
  return letIn;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. The configured origin still gets in, and what it gets is a working connection.
// ─────────────────────────────────────────────────────────────────────────────
test("an Origin in the CORS allowlist is accepted and the connection works", async (t) => {
  useCorsOrigin(t, ALLOWED);
  const server = await startServer();
  t.after(() => server.stop());

  const { accepted, ws } = await handshake(server.wsUrl, { Origin: ALLOWED });
  assert.equal(accepted, true, "the configured origin must be able to connect");
  t.after(() => ws.terminate());

  assert.deepEqual(await ping(ws), { type: "pong" });

  const frames = await chat(ws, "hello");
  assert.equal(frames[0].type, "ack");
  assert.equal(frames[1].type, "typing");
  const last = frames[frames.length - 1];
  assert.equal(last.type, "stream");
  assert.equal(last.done, true);
  assert.equal(last.content, "reply to: hello");
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. An origin outside the allowlist is refused in the handshake itself.
// ─────────────────────────────────────────────────────────────────────────────
test("an Origin outside the CORS allowlist is rejected during the handshake", async (t) => {
  useCorsOrigin(t, ALLOWED);
  const server = await startServer();
  t.after(() => server.stop());

  let connections = 0;
  server.wss.on("connection", () => connections++);

  const result = await handshake(server.wsUrl, { Origin: "https://evil.example" });

  assert.equal(result.accepted, false, "a page on another origin must not be able to open the WebSocket");
  assert.equal(result.status, 403);
  assert.match(result.body, /origin/i, "the refusal should say why");
  assert.equal(connections, 0, "no connection may be established for a refused handshake");
  assert.equal(server.wss.clients.size, 0);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Arbitrary origins: anything that is not the configured one is refused.
// ─────────────────────────────────────────────────────────────────────────────
test("arbitrary Origins are all rejected, and the server keeps serving afterwards", async (t) => {
  useCorsOrigin(t, ALLOWED);
  const server = await startServer();
  t.after(() => server.stop());

  const foreign = [
    "https://evil.example",
    "http://evil.example",
    "https://example.test",
    "http://localhost:3000",
    "http://127.0.0.1:8443",
    "null", // sandboxed iframe, file:// page, data: URL, redirect chain
    "file://",
    "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
    "*", // a literal asterisk in the header is not a wildcard
  ];
  assert.deepEqual(await originsLetIn(server, foreign), [], "these origins must not get in");

  // A refused handshake must not wedge anything: the real origin still connects.
  const { accepted, ws } = await handshake(server.wsUrl, { Origin: ALLOWED });
  assert.equal(accepted, true);
  assert.deepEqual(await ping(ws), { type: "pong" });
  ws.terminate();
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Casing and formatting: no spelling of the configured origin other than the
//    configured one itself, and no origin that merely contains it, gets in.
// ─────────────────────────────────────────────────────────────────────────────
test("changing the casing or formatting of an Origin does not get past the check", async (t) => {
  useCorsOrigin(t, ALLOWED);
  const server = await startServer();
  t.after(() => server.stop());

  const variants = [
    "HTTPS://APP.EXAMPLE.TEST:8443", // upper-case scheme and host
    "https://App.Example.Test:8443", // mixed-case host
    ALLOWED + "/", // trailing slash
    ALLOWED + "/ws", // path
    ALLOWED + "?x=1", // query
    ALLOWED + "#x", // fragment
    "https://app.example.test:8443@evil.example", // userinfo in front of the real host
    "https://app.example.test:8443.evil.example", // configured origin as a prefix
    "https://evil.example/https://app.example.test:8443", // configured origin embedded in a path
    "x" + ALLOWED, // text in front
    "http://app.example.test:8443", // different scheme
    "https://app.example.test:8444", // different port
    "https://app.example.test", // port omitted
    "https://app%2eexample.test:8443", // percent-encoded dot
    ALLOWED + ", https://evil.example", // a list: allowed first
    "https://evil.example, " + ALLOWED, // a list: allowed last
    "", // present but empty is not the same as absent
  ];
  assert.deepEqual(await originsLetIn(server, variants), [], "these spellings must not get in");
});

test("a repeated Origin header cannot smuggle the allowed origin past the check", async (t) => {
  useCorsOrigin(t, ALLOWED);
  const server = await startServer();
  t.after(() => server.stop());

  const control = await rawHandshake(server, [ALLOWED]);
  assert.equal(control.accepted, true, "the hand-written handshake must work for the real origin");

  const allowedFirst = await rawHandshake(server, [ALLOWED, "https://evil.example"]);
  assert.equal(allowedFirst.accepted, false, "Origin: <allowed> then Origin: <evil> must be refused");
  assert.equal(allowedFirst.status, 403);

  const allowedLast = await rawHandshake(server, ["https://evil.example", ALLOWED]);
  assert.equal(allowedLast.accepted, false, "Origin: <evil> then Origin: <allowed> must be refused");
  assert.equal(allowedLast.status, 403);
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. No Origin header: not a browser, so not what this check is for.
// ─────────────────────────────────────────────────────────────────────────────
test("a client that sends no Origin header (not a browser) is still accepted", async (t) => {
  useCorsOrigin(t, ALLOWED);
  const server = await startServer();
  t.after(() => server.stop());

  const { accepted, ws } = await handshake(server.wsUrl);
  assert.equal(accepted, true, "scripts, tests and other servers connect without an Origin header");
  t.after(() => ws.terminate());

  assert.deepEqual(await ping(ws), { type: "pong" });
  const frames = await chat(ws, "no origin here");
  assert.equal(frames[frames.length - 1].content, "reply to: no origin here");
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. CORS_ORIGIN=* means every origin for CORS, so it means every origin here too.
// ─────────────────────────────────────────────────────────────────────────────
test("CORS_ORIGIN=* lets every origin in, exactly as it does for CORS", async (t) => {
  useCorsOrigin(t, "*");
  const server = await startServer();
  t.after(() => server.stop());

  const rest = await request(server.app).get("/api/health").set("Origin", "https://evil.example");
  assert.equal(rest.headers["access-control-allow-origin"], "*");

  const letIn = await originsLetIn(server, ["https://evil.example", "null", ALLOWED]);
  assert.deepEqual(letIn, ["https://evil.example", "null", ALLOWED]);
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. One setting, not two: whatever CORS advertises is what the handshake accepts.
//    Nothing here names the default origin, so a second list cannot hide behind it.
// ─────────────────────────────────────────────────────────────────────────────
for (const configured of [undefined, ALLOWED, "http://127.0.0.1:8080"]) {
  test(`the handshake follows the origin CORS advertises (CORS_ORIGIN ${configured === undefined ? "unset" : configured})`, async (t) => {
    useCorsOrigin(t, configured);
    const server = await startServer();
    t.after(() => server.stop());

    // The CORS policy is untouched: REST still answers a foreign origin and
    // advertises the configured one, and the browser does the blocking.
    const rest = await request(server.app).get("/api/health").set("Origin", "https://evil.example");
    assert.equal(rest.status, 200);
    const advertised = rest.headers["access-control-allow-origin"];
    assert.ok(advertised && advertised !== "*", `CORS should advertise one origin, got ${advertised}`);

    const good = await handshake(server.wsUrl, { Origin: advertised });
    assert.equal(good.accepted, true, `the origin CORS advertises (${advertised}) must be accepted`);
    good.ws.terminate();

    const bad = await handshake(server.wsUrl, { Origin: "https://evil.example" });
    assert.equal(bad.accepted, false, "an origin CORS does not advertise must be refused");
    assert.equal(bad.status, 403);
  });
}
