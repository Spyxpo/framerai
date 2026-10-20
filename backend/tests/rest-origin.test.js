/**
 * State-changing REST requests from a page on another origin (Issue #513).
 *
 * CORS only stops a page reading the answer. Requests a browser sends without a
 * preflight (a multipart upload, a text/plain or empty POST) used to reach the
 * routes anyway, so any page a user visited could run the model, store files in
 * uploads/ and create conversations on their backend. The WebSocket has refused
 * other origins since #425/#426; REST now applies the same rule
 * (app.js originAllowed):
 *
 *   - the allowed origin (CORS_ORIGIN, or the Vite dev server) is served;
 *   - no Origin header is served: browsers send one on every state-changing and
 *     every cross-origin request, so its absence means a non-browser client
 *     (curl, a script). This is not authentication;
 *   - any other Origin, an empty one, or "null" is refused with 403
 *     ORIGIN_NOT_ALLOWED before the body is read or an upload written;
 *   - reads (GET, HEAD) and CORS preflights are unchanged.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { randomUUID } = require("node:crypto");
const request = require("supertest");

const { mockModel } = require("./helpers");

const calls = mockModel();
const store = require("../src/conversationStore");
const { generationCounter, apiCounter } = require("../src/middleware/limiters");

const ALLOWED = "http://localhost:5173"; // allowedOrigin() when CORS_ORIGIN is unset
const UPLOADS = path.join(__dirname, "..", "uploads");
const PNG = Buffer.from("89504e470d0a1a0a0000000d49484452", "hex");
const WAV = Buffer.from("RIFF0000WAVEfmt ", "latin1");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
  delete process.env.CORS_ORIGIN;
});
test.after(() => {
  delete process.env.CORS_ORIGIN;
});

function app() {
  return require("../src/app").createApp();
}

/** Every file under uploads/, so a test can tell whether a request wrote one. */
function uploadedFiles() {
  if (!fs.existsSync(UPLOADS)) return [];
  const out = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else out.push(full);
    }
  };
  walk(UPLOADS);
  return out.sort();
}

function seedConversation() {
  const id = randomUUID();
  const now = new Date().toISOString();
  const question = { id: randomUUID(), role: "user", content: "hello", type: "text", timestamp: now };
  store.create({ id, title: "Seeded", messages: [question], version: 1, createdAt: now, updatedAt: now });
  return { id, question };
}

/** Every state-changing route, as a function of the Origin to send (null: none). */
function stateChangingRequests() {
  const { id, question } = seedConversation();
  const multipart = (route, field, file, filename, contentType, fields = {}) => (a, origin) => {
    let r = a.post(route);
    if (origin !== null) r = r.set("Origin", origin);
    r = r.attach(field, file, { filename, contentType });
    for (const [k, v] of Object.entries(fields)) r = r.field(k, v);
    return r;
  };
  const json = (method, route, body) => (a, origin) => {
    let r = a[method](route);
    if (origin !== null) r = r.set("Origin", origin);
    return body === undefined ? r : r.send(body);
  };
  return {
    conversationId: id,
    requests: [
      ["POST /api/chat/conversations", json("post", "/api/chat/conversations")],
      ["POST /api/conversations (alias)", json("post", "/api/conversations")],
      ["PATCH /api/chat/conversations/:id", json("patch", `/api/chat/conversations/${id}`, { title: "renamed" })],
      ["POST /api/chat/conversations/:id/messages", json("post", `/api/chat/conversations/${id}/messages`, { content: "hi" })],
      ["POST /api/chat/conversations/:id/branch", json("post", `/api/chat/conversations/${id}/branch`, { messageId: question.id })],
      ["POST /api/chat/conversations/:id/sync", json("post", `/api/chat/conversations/${id}/sync`, { messages: [] })],
      ["POST /api/chat/conversations/:id/restore", json("post", `/api/chat/conversations/${randomUUID()}/restore`, { messages: [] })],
      ["DELETE /api/chat/conversations/:id/messages/:messageId", json("delete", `/api/chat/conversations/${id}/messages/${question.id}`)],
      ["DELETE /api/chat/conversations/:id", json("delete", `/api/chat/conversations/${id}`)],
      ["POST /api/generate/image", json("post", "/api/generate/image", { prompt: "a cat" })],
      ["POST /api/generate/video", json("post", "/api/generate/video", { prompt: "a cat" })],
      ["POST /api/generate/audio", json("post", "/api/generate/audio", { prompt: "hello" })],
      ["POST /api/generate/code", json("post", "/api/generate/code", { prompt: "fibonacci" })],
      ["POST /api/generate/upload", multipart("/api/generate/upload", "file", Buffer.from("notes"), "a.txt", "text/plain")],
      ["POST /api/generate/understand", multipart("/api/generate/understand", "image", PNG, "a.png", "image/png", { prompt: "describe" })],
      ["POST /api/generate/document", multipart("/api/generate/document", "document", Buffer.from("text"), "a.txt", "text/plain")],
      ["POST /api/generate/transcribe", multipart("/api/generate/transcribe", "audio", WAV, "a.wav", "audio/wav")],
      ["POST /api/generate/verify-audio", multipart("/api/generate/verify-audio", "audio", WAV, "a.wav", "audio/wav")],
      ["POST /api/generate/voice", multipart("/api/generate/voice", "reference", WAV, "a.wav", "audio/wav", { prompt: "hi", consent: "true" })],
      ["POST /api/generate/image-edit", multipart("/api/generate/image-edit", "image", PNG, "a.png", "image/png", { prompt: "redo" })],
      ["POST /api/generate/animate", multipart("/api/generate/animate", "image", PNG, "a.png", "image/png", { prompt: "move" })],
      ["POST /api/generate/understand-video", multipart("/api/generate/understand-video", "video", Buffer.from("not a video"), "a.mp4", "video/mp4")],
    ],
  };
}

for (const [label, origin] of [
  ["another origin", "https://evil.example"],
  ["another spelling of the allowed origin", "http://localhost:5173/"],
  ['the opaque origin "null"', "null"],
  ["an empty Origin header", ""],
]) {
  test(`#513: a state-changing request from ${label} is refused before it does anything`, async () => {
    const a = request(app());
    const { conversationId, requests } = stateChangingRequests();
    const conversationsBefore = store.list().length;
    const versionBefore = store.get(conversationId).version;
    const filesBefore = uploadedFiles();
    const callsBefore = calls.length;

    for (const [name, send] of requests) {
      const res = await send(a, origin);
      assert.equal(res.status, 403, `${name}: expected 403, got ${res.status}`);
      assert.equal(res.body.code, "ORIGIN_NOT_ALLOWED", name);
      assert.equal(res.headers["access-control-allow-origin"] === origin && origin !== "", false, `${name}: the page may not read it`);
    }

    assert.equal(calls.length, callsBefore, "no model call was made");
    assert.equal(store.list().length, conversationsBefore, "no conversation was created or deleted");
    assert.equal(store.get(conversationId).version, versionBefore, "the existing conversation is untouched");
    assert.deepEqual(uploadedFiles(), filesBefore, "no upload was written");
    assert.equal(generationCounter.buckets.size, 0, "a refused request uses no generation allowance");
  });
}

test("#513: a text/plain POST, the kind a page sends without a preflight, is refused", async () => {
  const before = store.list().length;
  const res = await request(app())
    .post("/api/chat/conversations")
    .set("Origin", "https://evil.example")
    .set("Content-Type", "text/plain")
    .send("");
  assert.equal(res.status, 403);
  assert.equal(store.list().length, before);
});

test("#513: the allowed origin is served, uploads included", async () => {
  const a = request(app());
  const created = await a.post("/api/chat/conversations").set("Origin", ALLOWED);
  assert.equal(created.status, 200);
  assert.ok(store.get(created.body.id), "the conversation exists");

  const callsBefore = calls.length;
  const reply = await a.post(`/api/chat/conversations/${created.body.id}/messages`).set("Origin", ALLOWED).send({ content: "hi" });
  assert.equal(reply.status, 200);
  assert.equal(calls.length, callsBefore + 1, "the model was asked");

  const uploaded = await a
    .post("/api/generate/upload")
    .set("Origin", ALLOWED)
    .attach("file", Buffer.from("notes"), { filename: "a.txt", contentType: "text/plain" });
  assert.equal(uploaded.status, 201);
  const written = path.join(UPLOADS, uploaded.body.path.replace(/^\/uploads\//, ""));
  assert.ok(fs.existsSync(written), "the upload was stored");
  fs.rmSync(written, { force: true });
});

test("#513: a request with no Origin is served, as non-browser clients send none (policy, not authentication)", async () => {
  const a = request(app());
  const created = await a.post("/api/chat/conversations");
  assert.equal(created.status, 200);
  const uploaded = await a.post("/api/generate/upload").attach("file", Buffer.from("notes"), { filename: "b.txt", contentType: "text/plain" });
  assert.equal(uploaded.status, 201);
  fs.rmSync(path.join(UPLOADS, uploaded.body.path.replace(/^\/uploads\//, "")), { force: true });
});

test("#513: reads and CORS preflights from another origin behave as before", async () => {
  const a = request(app());
  const { id } = seedConversation();
  for (const route of ["/api/health", "/api/chat/conversations", `/api/chat/conversations/${id}`, "/api/openapi.json"]) {
    const res = await a.get(route).set("Origin", "https://evil.example");
    assert.equal(res.status, 200, route);
    assert.notEqual(res.headers["access-control-allow-origin"], "https://evil.example", `${route}: still unreadable by that page`);
  }
  const preflight = await a
    .options("/api/chat/conversations")
    .set("Origin", "https://evil.example")
    .set("Access-Control-Request-Method", "POST")
    .set("Access-Control-Request-Headers", "content-type");
  assert.equal(preflight.status, 204);
  assert.notEqual(preflight.headers["access-control-allow-origin"], "https://evil.example");
});

test('#513: CORS_ORIGIN="*" serves every origin, as it does for CORS and the WebSocket', async () => {
  process.env.CORS_ORIGIN = "*";
  const res = await request(app()).post("/api/chat/conversations").set("Origin", "https://anywhere.example");
  assert.equal(res.status, 200);
});

test("#513: a configured CORS_ORIGIN is the one origin served", async () => {
  process.env.CORS_ORIGIN = "http://localhost:8080";
  const a = request(app());
  assert.equal((await a.post("/api/chat/conversations").set("Origin", "http://localhost:8080")).status, 200);
  assert.equal((await a.post("/api/chat/conversations").set("Origin", ALLOWED)).status, 403, "the dev default is not allowed once another is set");
});
