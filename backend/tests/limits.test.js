/**
 * Rate limiting and payload limits.
 *
 * The limits are read from the environment when the app module loads, so they
 * are set here before anything is required. `node --test` runs each file in
 * its own process, so this does not leak into the other suites.
 */

process.env.RATE_LIMIT_WINDOW_MS = "60000";
process.env.RATE_LIMIT_MAX = "50";
process.env.GENERATE_RATE_LIMIT_MAX = "3";
process.env.JSON_BODY_LIMIT = "1kb";

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");

const { mockModel, loadApp, newConversation } = require("./helpers");

mockModel();
const app = loadApp();

test("generation is cut off once the limit is spent", async () => {
  const statuses = [];
  for (let i = 0; i < 5; i++) {
    const res = await request(app).post("/api/generate/code").send({ prompt: "hi" });
    statuses.push(res.status);
  }

  assert.deepEqual(statuses, [200, 200, 200, 429, 429]);
});

test("a rejected request explains itself and says when to retry", async () => {
  const res = await request(app).post("/api/generate/code").send({ prompt: "hi" });

  assert.equal(res.status, 429);
  assert.equal(res.body.code, "RATE_LIMITED");
  assert.match(res.body.error, /Too many generation requests/);
  assert.equal(res.headers["ratelimit-limit"], "3");
  assert.equal(res.headers["ratelimit-remaining"], "0");
  assert.ok(Number(res.headers["retry-after"]) > 0);
});

test("chat messages draw on the same budget as the generation routes", async () => {
  const id = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "hi" });

  assert.equal(res.status, 429, "the generation budget was already spent above");
});

test("routes that do not run the model are unaffected", async () => {
  const res = await request(app).get("/api/health");
  assert.equal(res.status, 200);
});

test("an oversized body is refused before it is parsed", async () => {
  const res = await request(app)
    .post("/api/generate/audio")
    .send({ prompt: "x".repeat(5000) });

  assert.equal(res.status, 413);
  assert.equal(res.body.code, "PAYLOAD_TOO_LARGE");
});
