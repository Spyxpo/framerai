const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");

mockModel();
const app = loadApp();

test("health reports status and capabilities", async () => {
  const res = await request(app).get("/api/health");

  assert.equal(res.status, 200);
  assert.equal(res.body.status, "ok");
  assert.deepEqual(res.body.capabilities, ["text", "code", "image", "video", "audio"]);
  assert.ok(Date.parse(res.body.timestamp), "timestamp should be a date");
});

test("an unknown route returns the standard 404 body", async () => {
  const res = await request(app).get("/api/does-not-exist");

  assert.equal(res.status, 404);
  assert.equal(res.body.code, "NOT_FOUND");
  assert.match(res.body.error, /Cannot GET/);
});

test("a malformed JSON body is reported as such", async () => {
  const res = await request(app)
    .post("/api/generate/audio")
    .set("Content-Type", "application/json")
    .send("{not json");

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "INVALID_JSON");
});
